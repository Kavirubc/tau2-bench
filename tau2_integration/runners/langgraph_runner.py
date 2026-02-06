"""
LangGraph runner for τ²-bench (vanilla ReAct - no automatic compensation).

Uses langchain create_react_agent with Gemini for basic tool calling.
Relies purely on LLM reasoning for error recovery.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Langtrace for tracing
try:
    from langtrace_python_sdk import langtrace
    langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))
except ImportError:
    pass
except Exception as e:
    logging.warning(f"Langtrace initialization failed: {e}")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool as langchain_tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import StructuredTool

from .base import BaseFrameworkRunner, RunnerResult
from ..task_adapter import Tau2TaskDefinition, get_task_query
from ..wrapped_tools import ToolExecutionError
from ..disruption_engine import get_disruption_engine

logger = logging.getLogger("tau2_integration.runners.langgraph")


class LangGraphRunner(BaseFrameworkRunner):
    """
    Vanilla LangGraph runner using ReAct pattern.
    
    No automatic compensation - relies on LLM reasoning for error recovery.
    Serves as baseline for comparison with RAC and SagaLLM.
    """
    
    framework_name = "langgraph"
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        max_iterations: int = 25,
        **kwargs
    ):
        """
        Initialize the LangGraph runner.

        Args:
            model: LLM model to use (Gemini model name).
            max_iterations: Maximum ReAct iterations.
        """
        super().__init__(model=model, **kwargs)
        self.max_iterations = max_iterations
        self._llm = None
    
    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            )
        return self._llm
    
    def run_task(
        self,
        task: Tau2TaskDefinition,
        tools: Dict[str, Any],
        policy: str,
    ) -> RunnerResult:
        """
        Run a task using vanilla LangGraph ReAct.

        Args:
            task: Task definition.
            tools: Dictionary of tool functions.
            policy: Policy text for the agent.

        Returns:
            RunnerResult with execution details.
        """
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            logger.error("langgraph not installed. Run: pip install langgraph")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error="langgraph not installed",
            )
        
        # Convert tools to langchain format
        langchain_tools = self._convert_tools(tools)
        
        # Create agent
        llm = self._get_llm()
        agent = create_react_agent(llm, langchain_tools)
        
        # Build messages
        system_prompt = self.build_system_prompt(policy)
        user_query = get_task_query(task)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]
        
        # Execute agent
        tool_calls_made = []
        all_messages = []
        
        try:
            # Configure run with Langsmith metadata
            run_config = {
                "run_name": f"LangGraph-Vanilla-Task-{task.task_id}",
                "tags": ["tau2-bench", "LangGraph", "framework:LangGraph-Vanilla", f"task-{task.task_id}"],
                "metadata": {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "framework": "LangGraph-Vanilla",
                    "framework_name": "LangGraph ReAct (No Compensation)",
                    "model": self.model,
                },
            }
            
            result = agent.invoke(
                {"messages": messages},
                config=run_config,
            )
            
            all_messages = result.get("messages", [])
            tool_calls_made = self._extract_tool_calls_from_messages(all_messages)
            
            # Construct synthetic trace for metrics
            trace_steps = []
            for msg in all_messages:
                if msg.type == "ai":
                    usage = msg.usage_metadata if hasattr(msg, "usage_metadata") else {}
                    if usage:
                        trace_steps.append({
                            "type": "llm",
                            "tokens": {
                                "input": usage.get("input_tokens", 0),
                                "output": usage.get("output_tokens", 0),
                                "total": usage.get("total_tokens", 0),
                            }
                        })
            
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=True,
                execution_time=0,  # Will be set by execute()
                tool_calls=tool_calls_made,
                messages=[self._message_to_dict(m) for m in all_messages],
                raw_output=result,
                trace={"steps": trace_steps}
            )
            
        except ToolExecutionError as e:
            logger.warning(f"Tool execution error (not auto-recovered): {e}")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                tool_calls=tool_calls_made,
                messages=[self._message_to_dict(m) for m in all_messages],
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error=str(e),
            )
    
    def _convert_tools(self, tools: Dict[str, Any]) -> List[Any]:
        """Convert tool dictionary to langchain tools with disruption checking."""
        import functools
        langchain_tools = []

        for name, func in tools.items():
            def create_wrapper(f, tool_name):
                @functools.wraps(f)
                def wrapped_func(*args, **kwargs):
                    disruption_engine = get_disruption_engine()
                    disruption_error = disruption_engine.check_disruption(tool_name, kwargs)
                    if disruption_error:
                        return f'{{"status": "failed", "error": "{disruption_error}"}}'
                    return f(*args, **kwargs)
                return wrapped_func

            wrapped = create_wrapper(func, name)
            tool = langchain_tool(wrapped)
            tool.name = name
            langchain_tools.append(tool)

        return langchain_tools
    
    def _extract_tool_calls_from_messages(
        self, 
        messages: List[Any]
    ) -> List[Dict[str, Any]]:
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
