"""
React-Agent-Compensation (RAC) runner for τ²-bench.

Uses react-agent-compensation library for automatic rollback on failure.
Leverages the langchain_adaptor module's create_compensated_agent.
Includes comprehensive Langsmith tracing.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Langsmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "tau2-rac-benchmark"))

from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import CallbackManager
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseFrameworkRunner, RunnerResult
from ..task_adapter import Tau2TaskDefinition, get_task_query
from ..wrapped_tools import AIRLINE_COMPENSATION_MAPPING

logger = logging.getLogger("tau2_integration.runners.rac")


class RACRunner(BaseFrameworkRunner):
    """
    React-Agent-Compensation runner with automatic rollback.
    
    Uses create_compensated_agent from react-agent-compensation library
    to create a LangGraph agent with compensation-aware tool wrapping.
    Includes comprehensive Langsmith tracing for observability.
    """
    
    framework_name = "rac"
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        max_iterations: int = 25,
        auto_rollback: bool = True,
        auto_recover: bool = True,
        enable_tracing: bool = True,
        **kwargs
    ):
        """
        Initialize the RAC runner.

        Args:
            model: LLM model to use (Gemini model name).
            max_iterations: Maximum ReAct iterations.
            auto_rollback: Enable automatic rollback on failure.
            auto_recover: Enable automatic recovery (retry/alternatives).
            enable_tracing: Enable Langsmith tracing.
        """
        super().__init__(model=model, **kwargs)
        self.max_iterations = max_iterations
        self.auto_rollback = auto_rollback
        self.auto_recover = auto_recover
        self.enable_tracing = enable_tracing
        self._tracer = None
    
    def _get_tracer(self) -> Optional[LangChainTracer]:
        """Get Langsmith tracer if enabled."""
        if not self.enable_tracing:
            return None
        
        if self._tracer is None:
            try:
                api_key = os.getenv("LANGSMITH_API_KEY")
                if api_key:
                    self._tracer = LangChainTracer(
                        project_name=os.getenv("LANGSMITH_PROJECT", "tau2-rac-benchmark")
                    )
                    logger.info("Langsmith tracing enabled")
                else:
                    logger.warning("LANGSMITH_API_KEY not set, tracing disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Langsmith tracer: {e}")
        
        return self._tracer
    
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
        # Import RAC components
        try:
            from react_agent_compensation.langchain_adaptor import (
                create_compensated_agent,
                get_compensation_middleware,
                CompensationMiddleware,
            )
            from react_agent_compensation.core import (
                CompensationSchema,
                RetryPolicy,
            )
        except ImportError as e:
            logger.error(f"react-agent-compensation not installed: {e}")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error="react-agent-compensation not installed. Run: pip install react-agent-compensation",
            )
        
        # Import LangChain components
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.error("langchain-google-genai not installed")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error="langchain-google-genai not installed",
            )
        
        # Initialize TraceRecorder
        from ..tracing import TraceRecorder
        from ..callbacks import TracingCallbackHandler
        
        recorder = TraceRecorder(task_id=task.task_id, framework="rac")
        tracing_handler = TracingCallbackHandler(recorder)
        
        # Setup callbacks for tracing
        callbacks = [tracing_handler]
        tracer = self._get_tracer()
        if tracer:
            callbacks.append(tracer)
        
        # Create LLM (Gemini) with tracing
        llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            callbacks=callbacks,
        )
        
        # Convert tools to LangChain StructuredTool format
        langchain_tools = self._convert_tools_to_langchain(tools)
        
        # Define compensation schemas for airline tools
        compensation_schemas = {
            "book_reservation": CompensationSchema(
                param_mapping={"reservation_id": "result.reservation_id"},
            ),
        }
        
        # Create retry policy
        retry_policy = RetryPolicy(
            max_retries=2,
            base_delay=0.5,
            max_delay=5.0,
        )
        
        # Build system prompt with task context
        system_prompt = self._build_traced_system_prompt(task, policy)
        
        # Create compensated agent using RAC's factory function
        try:
            agent = create_compensated_agent(
                model=llm,
                tools=langchain_tools,
                compensation_mapping=AIRLINE_COMPENSATION_MAPPING,
                retry_policy=retry_policy,
                compensation_schemas=compensation_schemas,
                auto_rollback=self.auto_rollback,
                auto_recover=self.auto_recover,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Failed to create compensated agent: {e}")
            recorder.finish(status="failed", metadata={"error": str(e)})
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error=f"Failed to create compensated agent: {e}",
                trace=recorder.trace.to_dict(),
            )
        
        # Get middleware for tracking
        middleware = get_compensation_middleware(agent)
        
        # Build messages
        user_query = get_task_query(task)
        
        # Log task start for tracing
        logger.info(f"[RAC] Starting task {task.task_id}: {task.name[:50]}...")
        logger.info(f"[RAC] Available tools: {list(tools.keys())}")
        
        # Execute agent
        tool_calls_made = []
        compensation_actions = []
        all_messages = []
        rollback_success = False
        
        try:
            # Configure run with metadata for Langsmith
            run_config = {
                "callbacks": callbacks,
                "run_name": f"RAC-Task-{task.task_id}",
                "tags": ["tau2-bench", "RAC", "framework:RAC", f"task-{task.task_id}"],
                "metadata": {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "framework": "RAC",
                    "framework_name": "React-Agent-Compensation",
                    "model": self.model,
                    "auto_rollback": self.auto_rollback,
                    "auto_recover": self.auto_recover,
                },
            }
            
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=run_config,
            )
            
            all_messages = result.get("messages", [])
            tool_calls_made = self._extract_tool_calls(all_messages)
            
            # Log tool calls for tracing
            logger.info(f"[RAC] Completed with {len(tool_calls_made)} tool calls")
            for i, tc in enumerate(tool_calls_made):
                logger.info(f"[RAC]   Tool {i+1}: {tc.get('name')} - {tc.get('arguments')}")
            
            # Get compensation history from middleware if available
            if middleware:
                try:
                    log_snapshot = middleware.transaction_log.snapshot()
                    for rid, record in log_snapshot.items():
                        status_str = str(record.status).lower() if hasattr(record, 'status') else ""
                        if 'compensated' in status_str:
                            compensation_actions.append({
                                "action": record.action,
                                "status": str(record.status),
                                "compensator": record.compensator,
                            })
                        logger.info(f"[RAC] Transaction: {record.action} -> {record.status}")
                except Exception as e:
                    logger.debug(f"Could not get compensation history: {e}")
            
            recorder.finish(status="success")
            
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
                trace=recorder.trace.to_dict(),
            )
            
        except Exception as e:
            logger.error(f"[RAC] Agent execution failed: {e}")
            recorder.finish(status="failed", metadata={"error": str(e)})
            
            # Try to get compensation history even on failure
            comp_history = []
            if middleware:
                try:
                    log_snapshot = middleware.transaction_log.snapshot()
                    comp_history = [
                        {"action": r.action, "status": str(r.status)}
                        for r in log_snapshot.values()
                    ]
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
                trace=recorder.trace.to_dict(),
            )
    
    def _build_traced_system_prompt(self, task: Tau2TaskDefinition, policy: str) -> str:
        """Build system prompt with task context for better tracing."""
        base_prompt = self.build_system_prompt(policy)
        
        # Add task context
        return f"""{base_prompt}

CURRENT TASK: {task.name}
TASK ID: {task.task_id}
CATEGORY: {task.category.value if hasattr(task.category, 'value') else task.category}

Think step by step and use the available tools to complete the user's request.
After each tool call, evaluate the result and decide on the next action.

IMPORTANT:
1. If a tool fails or constraints are not met, DO NOT GIVE UP.
2. Rollback any partial state (this happens automatically), then PLAN AN ALTERNATIVE.
3. You must continue trying different approaches (e.g., flight options, parameters) until the User's Goal is fully satisfied.
4. Only signal "Final Answer" when the request is TRULY complete.
"""
    
    def _convert_tools_to_langchain(
        self, 
        tools: Dict[str, Any]
    ) -> List[StructuredTool]:
        """Convert τ²-bench tools to LangChain StructuredTool format with normalization."""
        langchain_tools = []
        
        for name, func in tools.items():
            # Create a wrapper that normalizes output
            def create_wrapper(f, tool_name):
                def wrapped_func(**kwargs):
                    try:
                        result = f(**kwargs)
                        return self._normalize_tool_output(result, tool_name)
                    except Exception as e:
                        return f'{{"status": "failed", "error": "{str(e)}"}}'
                return wrapped_func
            
            wrapped = create_wrapper(func, name)
            
            # Create a StructuredTool from the method
            tool = StructuredTool.from_function(
                func=wrapped,
                name=name,
                description=func.__doc__ or f"Tool: {name}",
            )
            langchain_tools.append(tool)
            logger.debug(f"[RAC] Converted tool: {name}")
        
        return langchain_tools
    
    def _normalize_tool_output(self, result: Any, tool_name: str) -> str:
        """
        Normalize real-world tool output to clean JSON for RAC.
        
        REALM-Bench expects: {"status": "failed", "error": "..."} for failures.
        Real tools might return weird strings or partial dicts.
        """
        import json
        
        # If it's already a dict, check for failure signals
        if isinstance(result, dict):
            # Check for explicit failure keys
            if "error" in result and result["error"]:
                 result["status"] = "failed"
            return json.dumps(result)
            
        # If it's a string, try to parse it
        if isinstance(result, str):
            try:
                # Try to clean up common issues
                cleaned = result.strip()
                if not cleaned.startswith("{") and "error" in cleaned.lower():
                     return json.dumps({"status": "failed", "error": cleaned})
                     
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    if "error" in parsed and parsed["error"]:
                        parsed["status"] = "failed"
                        return json.dumps(parsed)
            except:
                # parsing failed, treat as raw string. 
                # If concise enough, it's fine. If too long/noisy, might confuse RAC.
                # Heuristic: if it contains "fail" or "error", force status=failed
                if "fail" in result.lower() or "error" in result.lower():
                    return json.dumps({"status": "failed", "error": result[:200]})
        
        # Default pass-through
        return str(result)
    
    def _extract_tool_calls(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from message history."""
        tool_calls = []
        
        for msg in messages:
            # Check for tool_calls attribute (AIMessage with tool calls)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name"),
                        "arguments": tc.get("args", {}),
                    })
            
            # Also check for tool messages (results)
            if hasattr(msg, "type") and msg.type == "tool":
                # This is a tool result message
                pass
        
        return tool_calls
    
    def _message_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert a message to dictionary format."""
        result = {}
        
        if hasattr(msg, "type"):
            result["type"] = msg.type
        if hasattr(msg, "content"):
            result["content"] = str(msg.content)[:500]  # Truncate long content
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
        if hasattr(msg, "name"):
            result["name"] = msg.name
        
        return result
