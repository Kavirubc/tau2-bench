"""
Base runner class for framework integration.

Provides common functionality for all framework runners.
"""

import time
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..task_adapter import Tau2TaskDefinition, get_task_query
from ..disruption_engine import get_disruption_engine

logger = logging.getLogger("tau2_integration.runners.base")


@dataclass
class RunnerResult:
    """Result from a framework runner execution."""
    task_id: str
    framework: str
    success: bool
    
    # Execution metrics
    execution_time: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    
    # Task completion
    achieved_goals: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Compensation tracking
    disruptions_triggered: List[Dict[str, Any]] = field(default_factory=list)
    compensation_actions: List[Dict[str, Any]] = field(default_factory=list)
    rollback_success: bool = False
    state_clean_after_error: bool = False
    
    # Error information
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Raw output
    raw_output: Optional[Any] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed Trace
    trace: Optional[Dict[str, Any]] = None


class BaseFrameworkRunner(ABC):
    """
    Abstract base class for framework runners.
    
    Subclasses implement run_task() for their specific framework.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        log_file: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the runner.

        Args:
            model: LLM model to use.
            log_file: Optional path to log file.
            verbose: Enable verbose logging.
        """
        self.model = model
        self.log_file = log_file
        self.verbose = verbose
        self.framework_name = self.__class__.__name__
        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
    
    @abstractmethod
    def run_task(
        self,
        task: Tau2TaskDefinition,
        tools: Dict[str, Any],
        policy: str,
    ) -> RunnerResult:
        """
        Run a task using this framework.

        Args:
            task: Task definition.
            tools: Dictionary of tool functions.
            policy: Policy text for the agent.

        Returns:
            RunnerResult with execution details.
        """
        pass
    
    def execute(
        self,
        task: Tau2TaskDefinition,
        tools: Dict[str, Any],
        policy: str,
        disruption_scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> RunnerResult:
        """
        Execute a task with optional disruption injection.

        Args:
            task: Task definition.
            tools: Dictionary of tool functions.
            policy: Policy text for the agent.
            disruption_scenarios: Optional list of disruption scenarios.

        Returns:
            RunnerResult with execution details.
        """
        # Reset and configure disruption engine
        engine = get_disruption_engine()
        engine.reset()
        
        if disruption_scenarios:
            engine.configure(disruption_scenarios)
            engine.enable()
            logger.info(f"Configured {len(disruption_scenarios)} disruption scenarios")
        else:
            engine.disable()
        
        start_time = time.time()
        
        try:
            result = self.run_task(task, tools, policy)
        except Exception as e:
            logger.error(f"Runner failed: {e}")
            result = RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=time.time() - start_time,
                error=str(e),
                error_traceback=traceback.format_exc(),
            )
        
        # Add disruption information to result
        result.disruptions_triggered = engine.get_triggered_disruptions()
        result.execution_time = time.time() - start_time
        
        return result
    
    def build_system_prompt(self, policy: str) -> str:
        """
        Build the system prompt for the agent.

        Args:
            policy: Policy text for the domain.

        Returns:
            Complete system prompt.
        """
        return f"""You are a customer service agent for an airline.

Your job is to help customers with their flight bookings, modifications, and cancellations.

IMPORTANT RULES:
- Always obtain user confirmation before making any changes
- Make only one tool call at a time
- Follow the policy guidelines strictly

POLICY:
{policy}
"""
    
    def extract_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from message history.

        Args:
            messages: List of messages from execution.

        Returns:
            List of tool call dictionaries.
        """
        tool_calls = []
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tool_calls.append({
                        "name": tc.get("name") or tc.get("function", {}).get("name"),
                        "arguments": tc.get("arguments") or tc.get("function", {}).get("arguments"),
                    })
        return tool_calls
