"""
React-Agent-Compensation (RAC) runner for τ²-bench.

Uses react-agent-compensation library for automatic rollback on failure.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseFrameworkRunner, RunnerResult
from ..task_adapter import Tau2TaskDefinition, get_task_query
from ..wrapped_tools import AIRLINE_COMPENSATION_MAPPING

logger = logging.getLogger("tau2_integration.runners.rac")


class RACRunner(BaseFrameworkRunner):
    """
    React-Agent-Compensation runner with automatic rollback.
    
    Uses CompensationMiddleware to track actions and automatically
    rollback on failure.
    """
    
    framework_name = "rac"
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_iterations: int = 25,
        auto_rollback: bool = True,
        **kwargs
    ):
        """
        Initialize the RAC runner.

        Args:
            model: LLM model to use.
            max_iterations: Maximum ReAct iterations.
            auto_rollback: Enable automatic rollback on failure.
        """
        super().__init__(model=model, **kwargs)
        self.max_iterations = max_iterations
        self.auto_rollback = auto_rollback
    
    def run_task(
        self,
        task: Tau2TaskDefinition,
        tools: Dict[str, Any],
        policy: str,
    ) -> RunnerResult:
        """
        Run a task using React-Agent-Compensation.

        Args:
            task: Task definition.
            tools: Dictionary of tool functions.
            policy: Policy text for the agent.

        Returns:
            RunnerResult with execution details.
        """
        try:
            from react_agent_compensation import (
                create_compensating_agent,
                CompensationMiddleware,
                RollbackFailure,
            )
            from react_agent_compensation.strategies import RetryStrategy
        except ImportError:
            logger.error("react-agent-compensation not installed")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error="react-agent-compensation not installed. Run: pip install react-agent-compensation",
            )
        
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            logger.error("langchain-openai not installed")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error="langchain-openai not installed",
            )
        
        # Create LLM
        llm = ChatOpenAI(model=self.model, temperature=0)
        
        # Convert tools to langchain format with compensation mapping
        langchain_tools = self._convert_tools_with_compensation(tools)
        
        # Create compensation middleware
        middleware = CompensationMiddleware(
            compensation_mapping=AIRLINE_COMPENSATION_MAPPING,
            retry_policy=RetryStrategy(max_retries=2),
            auto_recover=True,
            auto_rollback=self.auto_rollback,
        )
        
        # Create agent with compensation
        agent = create_compensating_agent(
            llm=llm,
            tools=langchain_tools,
            middleware=middleware,
        )
        
        # Build messages
        system_prompt = self.build_system_prompt(policy)
        user_query = get_task_query(task)
        
        # Execute agent
        tool_calls_made = []
        compensation_actions = []
        all_messages = []
        rollback_success = False
        
        try:
            result = agent.invoke({
                "input": user_query,
                "system": system_prompt,
            })
            
            all_messages = result.get("messages", [])
            tool_calls_made = self._extract_tool_calls(all_messages)
            compensation_actions = middleware.get_compensation_history()
            
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=True,
                execution_time=0,
                tool_calls=tool_calls_made,
                compensation_actions=compensation_actions,
                rollback_success=True,
                messages=[self._message_to_dict(m) for m in all_messages],
                raw_output=result,
            )
            
        except RollbackFailure as e:
            logger.error(f"Rollback failed: {e}")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                tool_calls=tool_calls_made,
                compensation_actions=middleware.get_compensation_history(),
                rollback_success=False,
                error=f"Rollback failed: {e}",
            )
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Try to get compensation history even on failure
            comp_history = []
            try:
                comp_history = middleware.get_compensation_history()
            except:
                pass
            
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                tool_calls=tool_calls_made,
                compensation_actions=comp_history,
                error=str(e),
            )
    
    def _convert_tools_with_compensation(
        self, 
        tools: Dict[str, Any]
    ) -> List[Any]:
        """Convert tools with compensation metadata."""
        from langchain_core.tools import tool as langchain_tool
        
        langchain_tools = []
        
        for name, func in tools.items():
            wrapped = langchain_tool(func)
            wrapped.name = name
            
            # Add compensation metadata
            if name in AIRLINE_COMPENSATION_MAPPING:
                wrapped.compensation_action = AIRLINE_COMPENSATION_MAPPING[name]
            
            langchain_tools.append(wrapped)
        
        return langchain_tools
    
    def _extract_tool_calls(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from message history."""
        tool_calls = []
        
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name"),
                        "arguments": tc.get("args", {}),
                    })
        
        return tool_calls
    
    def _message_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert a message to dictionary format."""
        if hasattr(msg, "dict"):
            return msg.dict()
        elif hasattr(msg, "content"):
            return {
                "role": getattr(msg, "type", "unknown"),
                "content": msg.content,
            }
        return {"content": str(msg)}
