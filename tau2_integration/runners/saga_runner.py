"""
SagaLLM runner for τ²-bench.

Implements the 3-phase plan-execute-compensate workflow from SagaLLM.
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

from .base import BaseFrameworkRunner, RunnerResult
from ..task_adapter import Tau2TaskDefinition, get_task_query
from ..wrapped_tools import AIRLINE_COMPENSATION_MAPPING

logger = logging.getLogger("tau2_integration.runners.saga")


class SagaLLMRunner(BaseFrameworkRunner):
    """
    SagaLLM runner implementing 3-phase execution.
    
    Phases:
    1. Planning - Generate action plan
    2. Execution - Execute plan with state tracking
    3. Compensation - Rollback on failure using compensation actions
    """
    
    framework_name = "sagallm"
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        max_iterations: int = 25,
        enable_replanning: bool = True,
        **kwargs
    ):
        """
        Initialize the SagaLLM runner.

        Args:
            model: LLM model to use (Gemini model name).
            max_iterations: Maximum execution iterations.
            enable_replanning: Enable replanning after compensation.
        """
        super().__init__(model=model, **kwargs)
        self.max_iterations = max_iterations
        self.enable_replanning = enable_replanning
        self._executed_actions = []
    
    def run_task(
        self,
        task: Tau2TaskDefinition,
        tools: Dict[str, Any],
        policy: str,
    ) -> RunnerResult:
        """
        Run a task using SagaLLM 3-phase execution.

        Args:
            task: Task definition.
            tools: Dictionary of tool functions.
            policy: Policy text for the agent.

        Returns:
            RunnerResult with execution details.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            logger.error("langchain-google-genai not installed")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error="langchain-google-genai not installed",
            )
        
        self._executed_actions = []
        compensation_actions = []
        
        try:
            # Phase 1: Planning
            logger.info("Phase 1: Planning")
            
            # Setup Langsmith tracing
            callbacks = []
            try:
                from langchain_core.tracers import LangChainTracer
                api_key = os.getenv("LANGSMITH_API_KEY")
                if api_key:
                    tracer = LangChainTracer(
                        project_name=os.getenv("LANGSMITH_PROJECT", "tau2-saga-benchmark")
                    )
                    callbacks.append(tracer)
            except Exception as e:
                logger.debug(f"Langsmith tracer not available: {e}")
            
            # Configure run metadata for Langsmith
            run_config = {
                "run_name": f"SagaLLM-Task-{task.task_id}",
                "tags": ["tau2-bench", "SagaLLM", "framework:SagaLLM", f"task-{task.task_id}"],
                "metadata": {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "framework": "SagaLLM",
                    "framework_name": "SagaLLM 3-Phase (Plan-Execute-Compensate)",
                    "model": self.model,
                    "enable_replanning": self.enable_replanning,
                },
                "callbacks": callbacks,
            }
            
            llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
                callbacks=callbacks,
            )

            iteration = 0
            failure_context = []
            all_executed_actions = []
            all_compensation_actions = []
            
            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"=== SagaLLM Iteration {iteration}/{self.max_iterations} ===")
                
                self._executed_actions = [] # Reset for this iteration
                
                try:
                    # Phase 1: Planning (with context from previous failures)
                    logger.info("Phase 1: Planning")
                    
                    # Update run name for iteration
                    run_config["run_name"] = f"SagaLLM-Task-{task.task_id}-Iter{iteration}"
                    
                    plan = self._generate_plan(llm, task, tools, policy, run_config, failure_context)
                    
                    if not plan:
                        logger.warning("Planning failed, stopping.")
                        break
                    
                    # Phase 2: Execution
                    logger.info("Phase 2: Execution")
                    execution_result = self._execute_plan(llm, plan, tools, policy)
                    
                    # Capture messages for context
                    iter_messages = execution_result.get("messages", [])
                    
                    # Collect actions from this attempt
                    all_executed_actions.extend(self._executed_actions)
                    
                    if execution_result["success"]:
                        logger.info("Plan executed successfully!")
                        return RunnerResult(
                            task_id=task.task_id,
                            framework=self.framework_name,
                            success=True,
                            execution_time=0,
                            tool_calls=all_executed_actions,
                            messages=execution_result.get("messages", []),
                            raw_output=execution_result,
                        )
                    
                    # Phase 3: Compensation (if execution failed)
                    logger.info(f"Execution failed: {execution_result.get('error')}. Phase 3: Compensation")
                    
                    compensation_result = self._compensate(tools)
                    comp_actions_this_iter = compensation_result.get("actions", [])
                    all_compensation_actions.extend(comp_actions_this_iter)
                    
                    if not compensation_result["success"]:
                        logger.error("Critical: Compensation failed. Cannot safely replan.")
                        return RunnerResult(
                            task_id=task.task_id,
                            framework=self.framework_name,
                            success=False,
                            execution_time=0,
                            tool_calls=all_executed_actions,
                            compensation_actions=all_compensation_actions,
                            error=f"Compensation failed: {execution_result.get('error')}",
                        )

                    if self.enable_replanning:
                        logger.info("Compensation successful. Preparing to replan.")
                        failure_reason = execution_result.get("error", "Unknown error")
                        failed_step = execution_result.get("failed_step", "?")
                        
                        # Add to context
                        context_msg = f"Attempt {iteration} failed at step {failed_step}: {failure_reason}. " \
                                      f"The system has been rolled back. You must try a DIFFERENT approach."
                        failure_context.append(context_msg)
                        
                        # Also add the tool outputs from this attempt so the planner can see search results
                        # Filter for 'tool' messages
                        tool_outputs = [
                            f"Tool '{m['tool']}' output: {m['content']}" 
                            for m in iter_messages 
                            if m.get("role") == "tool"
                        ]
                        if tool_outputs:
                            failure_context.append("Information gathered in this attempt:\n" + "\n".join(tool_outputs))
                    else:
                        logger.info("Replanning disabled. Stopping.")
                        break
                        
                except Exception as e:
                    logger.error(f"Iteration {iteration} failed: {e}")
                    # Try last-ditch compensation
                    try:
                        self._compensate(tools)
                    except:
                        pass
                    return RunnerResult(
                        task_id=task.task_id,
                        framework=self.framework_name,
                        success=False,
                        execution_time=0,
                        error=str(e),
                    )
            
            # If we exited loop without success
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                tool_calls=all_executed_actions,
                compensation_actions=all_compensation_actions,
                error=f"Max iterations ({self.max_iterations}) reached without success.",
            )
            
        except Exception as e:
            logger.error(f"SagaLLM runner failed: {e}")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error=str(e),
            )
    
    def _generate_plan(
        self,
        llm: Any,
        task: Tau2TaskDefinition,
        tools: Dict[str, Any],
        policy: str,
        run_config: Optional[Dict[str, Any]] = None,
        failure_context: List[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate an action plan using the LLM.

        Args:
            llm: LLM instance.
            task: Task definition.
            tools: Available tools.
            policy: Policy text.
            run_config: Optional Langsmith run config.

        Returns:
            List of planned actions, or None if planning fails.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        tool_descriptions = self._format_tool_descriptions(tools)
        user_query = get_task_query(task)
        
        planning_prompt = f"""You are a planning agent. Generate a step-by-step action plan to accomplish the user's request.

Available tools:
{tool_descriptions}

Policy:
{policy}

User request:
{user_query}

Generate a JSON list of actions, where each action has:
- "tool": tool name to call
- "args": dictionary of arguments matching the tool schema EXACTLY.

IMPORTANT:
1. Use ONLY the arguments defined in the "Available tools" section above.
2. Do NOT invent new arguments (e.g., do not use "flight_number" if the schema says "flight_id").
3. Failure to follow the schema will cause the plan to fail.
4. If a value is unknown (like a flight ID from a future search), use a placeholder string like "{{flight_id_from_step_1}}".
5. If a tool takes a SINGLE string argument (like "passenger_name"), do NOT pass a list. Create multiple separate steps (one for each item).
- "args": arguments for the tool
- "reasoning": why this step is needed

Respond with ONLY the JSON list, no other text.
"""
        
        # Add failure context if available
        if failure_context:
            error_history = "\n".join([f"- {err}" for err in failure_context])
            planning_prompt += f"\n\nCRITICAL: Previous attempts failed. Improve your plan based on these errors:\n{error_history}\n"
        
        try:
            response = llm.invoke([
                SystemMessage(content="You are a planning assistant."),
                HumanMessage(content=planning_prompt),
            ])
            
            # Parse plan from response
            import json
            content = response.content.strip()
            
            # Try to extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            plan = json.loads(content)
            logger.info(f"Generated plan with {len(plan)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return None
    
    def _execute_plan(
        self,
        llm: Any,
        plan: List[Dict[str, Any]],
        tools: Dict[str, Any],
        policy: str,
    ) -> Dict[str, Any]:
        """
        Execute the generated plan.

        Args:
            llm: LLM instance.
            plan: List of planned actions.
            tools: Available tools.
            policy: Policy text.

        Returns:
            Execution result dictionary.
        """
        from ..wrapped_tools import ToolExecutionError
        
        messages = []
        step_results = {} # Map step number -> raw result for substitution
        
        for idx, step in enumerate(plan):
            tool_name = step.get("tool")
            tool_args = step.get("args", {})
            if step_num is None:
                step_num = idx + 1
            
            # Substitute placeholders with actual results using step_results
            for arg_key, arg_val in tool_args.items():
                if isinstance(arg_val, str):
                    try:
                        # New style: {{flight_id_from_step_1}}
                        if "from_step_" in arg_val:
                            import re
                            match = re.search(r"from_step_(\d+)", arg_val)
                            if match:
                                ref_step = int(match.group(1))
                                
                                if ref_step in step_results:
                                    prev_result = step_results[ref_step]
                                    
                                    # Handle different result types (list vs dict)
                                    if isinstance(prev_result, list) and len(prev_result) > 0:
                                        # If previous result is a list (like search_flights), pick the first item
                                        item = prev_result[0]
                                        if isinstance(item, dict):
                                            # Try to match key requested
                                            if "flight_id" in arg_key or "id" in arg_key:
                                                tool_args[arg_key] = item.get("id") or item.get("flight_id")
                                    elif isinstance(prev_result, dict):
                                        # If previous result is a dict (like book_flight)
                                        if "booking_id" in arg_key:
                                            tool_args[arg_key] = prev_result.get("booking_id")
                                        elif "payment_id" in arg_key:
                                            tool_args[arg_key] = prev_result.get("transaction_id")
                        
                        # Old style: RESULT_FROM_STEP_1
                        elif "RESULT_FROM_STEP_" in arg_val:
                             ref_str = arg_val.replace("RESULT_FROM_STEP_", "")
                             ref_step = int(ref_str)
                             pass 

                    except Exception as e:
                        logger.warning(f"Failed to substitute placeholder {arg_val}: {e}")
                             ref_str = arg_val.replace("RESULT_FROM_STEP_", "")
                             ref_step = int(ref_str)
                             # ... legacy logic if needed (omitted for now as we prefer new style) ...
                             pass 

                    except Exception as e:
                        logger.warning(f"Failed to substitute placeholder {arg_val}: {e}")
            
            if tool_name not in tools:
                logger.warning(f"Unknown tool: {tool_name}")
                continue
            
            try:
                # Execute the tool
                if hasattr(tools[tool_name], "invoke"):
                    result = tools[tool_name].invoke(tool_args)
                else:
                    result = tools[tool_name](**tool_args)
                
                # Store raw result for future steps
                if step_num is not None:
                    step_results[step_num] = result
                
                # Check for logical failure (e.g. status='failed')
                if isinstance(result, dict):
                    if result.get("status") == "failed":
                        error_msg = result.get("error", "Tool reported failure")
                        raise ToolExecutionError(error_msg)
                    if "error" in result and result["error"]:
                        raise ToolExecutionError(result["error"])
                
                # Track executed action
                self._executed_actions.append({
                    "step": step.get("step"),
                    "tool": tool_name,
                    "args": tool_args,
                    "result": str(result)[:500],  # Truncate for storage
                })
                
                messages.append({
                    "role": "tool",
                    "tool": tool_name,
                    "content": str(result),
                })
                
            except ToolExecutionError as e:
                logger.warning(f"Tool {tool_name} failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "failed_step": step.get("step"),
                    "messages": messages,
                }
            except Exception as e:
                logger.error(f"Tool {tool_name} raised exception: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "failed_step": step.get("step"),
                    "messages": messages,
                }
        
        return {
            "success": True,
            "messages": messages,
        }
    
    def _compensate(self, tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute compensation actions for executed actions (in reverse order).

        Args:
            tools: Available tools.

        Returns:
            Compensation result dictionary.
        """
        compensation_actions = []
        all_success = True
        
        # Process in reverse order
        for action in reversed(self._executed_actions):
            tool_name = action["tool"]
            comp_tool = AIRLINE_COMPENSATION_MAPPING.get(tool_name)
            
            if not comp_tool or comp_tool not in tools:
                logger.warning(f"No compensation for {tool_name}")
                continue
            
            try:
                # Build compensation arguments
                comp_args = self._build_compensation_args(action)
                
                if comp_args:
                    if hasattr(tools[comp_tool], "invoke"):
                         result = tools[comp_tool].invoke(comp_args)
                    else:
                         result = tools[comp_tool](**comp_args)
                    compensation_actions.append({
                        "original_tool": tool_name,
                        "compensation_tool": comp_tool,
                        "args": comp_args,
                        "success": True,
                    })
                    logger.info(f"Compensated {tool_name} with {comp_tool}")
                    
            except Exception as e:
                logger.error(f"Compensation for {tool_name} failed: {e}")
                compensation_actions.append({
                    "original_tool": tool_name,
                    "compensation_tool": comp_tool,
                    "error": str(e),
                    "success": False,
                })
                all_success = False
        
        return {
            "success": all_success,
            "actions": compensation_actions,
        }
    
    def _build_compensation_args(
        self, 
        action: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build arguments for compensation action."""
        tool_name = action["tool"]
        
        if tool_name == "book_reservation":
            # Extract reservation_id from result (legacy tau2 tool)
            result = action.get("result", "")
            if "reservation_id" in result:
                import re
                match = re.search(r"reservation_id['\"]?\s*[:=]\s*['\"]?(\w+)", result)
                if match:
                    return {"reservation_id": match.group(1)}
        
        elif tool_name == "book_flight":
            # Extract booking_id from result (new demo tool)
            result = action.get("result", "")
            if "booking_id" in result:
                import re
                match = re.search(r"booking_id['\"]?\s*[:=]\s*['\"]?([\w-]+)", result)
                if match:
                    return {"booking_id": match.group(1)}
        
        return None
    
    def _format_tool_descriptions(self, tools: Dict[str, Any]) -> str:
        """Format tool descriptions for the planning prompt."""
        descriptions = []
        
        for name, func in tools.items():
            doc = func.__doc__ or "No description"
            descriptions.append(f"- {name}: {doc[:200]}")
        
        return "\n".join(descriptions)
