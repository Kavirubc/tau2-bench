"""
SagaLLM runner for τ²-bench.

Implements Plan → ReAct Execute → Compensate workflow from SagaLLM.
Phase 1 (Planning) generates a structured plan via LLM.
Phase 2 (Execution) uses a ReAct agent to follow the plan.
Phase 3 (Compensation) rolls back on failure using compensation actions.
"""

import json
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
from langchain_core.tools import StructuredTool

from .base import BaseFrameworkRunner, RunnerResult
from ..task_adapter import Tau2TaskDefinition, get_task_query
from ..wrapped_tools import AIRLINE_COMPENSATION_MAPPING
from ..disruption_engine import get_disruption_engine

logger = logging.getLogger("tau2_integration.runners.saga")


class SagaLLMRunner(BaseFrameworkRunner):
    """
    SagaLLM runner implementing Plan → ReAct Execute → Compensate.

    Phases:
    1. Planning - Generate action plan via LLM
    2. Execution - ReAct agent follows the plan using tools
    3. Compensation - Rollback on failure using compensation actions
    """

    framework_name = "sagallm"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        max_iterations: int = 3,
        enable_replanning: bool = True,
        **kwargs
    ):
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
        Run a task using SagaLLM: Plan → ReAct Execute → Compensate.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langgraph.prebuilt import create_react_agent
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error=f"Required packages not installed: {e}",
            )

        self._executed_actions = []

        # Initialize TraceRecorder
        from ..tracing import TraceRecorder
        from ..callbacks import TracingCallbackHandler

        recorder = TraceRecorder(task_id=task.task_id, framework="sagallm")
        tracing_handler = TracingCallbackHandler(recorder)

        try:
            # Setup callbacks
            callbacks = [tracing_handler]
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

            run_config = {
                "run_name": f"SagaLLM-Task-{task.task_id}",
                "tags": ["tau2-bench", "SagaLLM", "framework:SagaLLM", f"task-{task.task_id}"],
                "metadata": {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "framework": "SagaLLM",
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

            # Convert tools to langchain format with disruption-aware wrappers
            langchain_tools = self._convert_tools(tools)

            iteration = 0
            all_tool_calls = []
            all_compensation_actions = []

            # Initialize Persistent Context
            from ..persistent_context import PersistentExecutionContext
            persistent_ctx = PersistentExecutionContext()

            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"=== SagaLLM Iteration {iteration}/{self.max_iterations} ===")

                try:
                    # ==========================================================
                    # PHASE 1: PLANNING (or REPLANNING)
                    # ==========================================================
                    logger.info("Phase 1: Planning")
                    recorder.push_context({"phase": "planning", "iteration": iteration})
                    run_config["run_name"] = f"SagaLLM-Task-{task.task_id}-Iter{iteration}"

                    plan = None
                    if iteration == 1:
                        persistent_ctx.original_plan = self._generate_plan(llm, task, tools, policy, run_config)
                        plan = persistent_ctx.original_plan
                    else:
                        logger.info(f"Replanning with failure history: {persistent_ctx.failures[-1]['reason']}")
                        recorder.record_event("saga_strategic_replan_start", {"failure": persistent_ctx.failures[-1]})
                        failure_msgs = [
                            f"Attempt {f['attempt']} failed at tool '{f['failed_tool']}' with error: {f['reason']}"
                            for f in persistent_ctx.failures
                        ]
                        plan = self._generate_plan(llm, task, tools, policy, run_config, failure_context=failure_msgs)
                        if plan:
                            recorder.record_event("saga_strategic_replan_success", {"new_plan_steps": len(plan)})
                        else:
                            recorder.record_event("saga_strategic_replan_failed")

                    recorder.pop_context()

                    if not plan:
                        logger.warning("Planning failed, stopping.")
                        break

                    if not persistent_ctx.original_plan:
                        persistent_ctx.original_plan = plan

                    # ==========================================================
                    # PHASE 2: EXECUTION via ReAct Agent
                    # ==========================================================
                    logger.info("Phase 2: ReAct Execution")
                    recorder.push_context({"phase": "execution", "iteration": iteration})
                    execution_step_id = recorder.start_step("execution", "saga_react_execute", {"plan_steps": len(plan)})

                    # Reset executed actions for this iteration
                    self._executed_actions = []

                    try:
                        # Build execution system prompt with the plan
                        formatted_plan = self._format_plan_for_execution(plan)
                        exec_system_prompt = f"""{self.build_system_prompt(policy)}

You are executing a pre-planned action sequence. Follow this plan step by step:

{formatted_plan}

Use the available tools to execute each step. After each tool call, evaluate the result and proceed to the next step.
If a step fails, report the failure clearly.
IMPORTANT: Execute ALL steps in the plan. Do not stop early.
"""
                        # Create ReAct agent
                        agent = create_react_agent(llm, langchain_tools)

                        user_query = get_task_query(task)
                        messages = [
                            SystemMessage(content=exec_system_prompt),
                            HumanMessage(content=user_query),
                        ]

                        result = agent.invoke(
                            {"messages": messages},
                            config=run_config,
                        )

                        all_messages = result.get("messages", [])
                        iter_tool_calls = self._extract_tool_calls(all_messages)
                        all_tool_calls.extend(iter_tool_calls)

                        logger.info(f"ReAct execution completed with {len(iter_tool_calls)} tool calls")
                        for i, tc in enumerate(iter_tool_calls):
                            logger.info(f"  Tool {i+1}: {tc.get('name')} - {tc.get('arguments')}")

                        recorder.end_step(execution_step_id, output={"success": True, "tool_calls": len(iter_tool_calls)})
                        recorder.finish(status="success")
                        recorder.pop_context()

                        return RunnerResult(
                            task_id=task.task_id,
                            framework=self.framework_name,
                            success=True,
                            execution_time=0,
                            tool_calls=all_tool_calls,
                            messages=[self._message_to_dict(m) for m in all_messages],
                            raw_output=result,
                            trace=recorder.trace.to_dict(),
                        )

                    except Exception as e:
                        logger.error(f"ReAct execution failed: {e}")
                        recorder.end_step(execution_step_id, output=None, error=str(e))
                        recorder.pop_context()

                        # Phase 3: Compensation
                        if self._executed_actions:
                            logger.info("Phase 3: Compensation")
                            recorder.push_context({"phase": "compensation", "iteration": iteration})
                            comp_result = self._compensate(tools, recorder)
                            all_compensation_actions.extend(comp_result.get("actions", []))
                            recorder.pop_context()

                        persistent_ctx.add_failure(str(e), "unknown", {}, iteration)
                        persistent_ctx.replan_count += 1

                        if not self.enable_replanning:
                            logger.info("Replanning disabled. Stopping.")
                            break

                except Exception as e:
                    logger.error(f"Iteration {iteration} failed: {e}")

            # End of loop
            recorder.finish(status="failed", metadata={"error": "Max iterations reached"})
            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                tool_calls=all_tool_calls,
                compensation_actions=all_compensation_actions,
                error=f"Max iterations ({self.max_iterations}) reached.",
                trace=recorder.trace.to_dict(),
            )

        except Exception as e:
            logger.error(f"SagaLLM runner failed: {e}")
            if 'recorder' in locals():
                recorder.finish(status="failed", metadata={"error": str(e)})
                trace_dict = recorder.trace.to_dict()
            else:
                trace_dict = None

            return RunnerResult(
                task_id=task.task_id,
                framework=self.framework_name,
                success=False,
                execution_time=0,
                error=str(e),
                trace=trace_dict,
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

META-STRATEGIES (Use these to structure your plan):
1. **DECOMPOSITION**: If the user asks for multiple things (e.g., "flight and hotel", "two different flights"), do NOT try to do them in one step. Create distinct steps for each entitlement.
2. **VERIFICATION FIRST**: Before modifying a reservation (cancel/change), ALWAYS add a step to "get_reservation_details" or "search_..." to verify its current status.
3. **ISOLATION**: If a tool fails, it should not break independent parts of the plan. Keep independent objectives in separate logical groups of steps.
4. **ARGUMENT PRECISION**: Do not guess IDs. Use the output of a search step (e.g., `{{flight_id_from_step_X}}`) as the input for a booking step.
"""
        
        # Add failure context if available
        if failure_context:
            error_history = "\n".join([f"- {err}" for err in failure_context])
            planning_prompt += f"\n\nCRITICAL: Previous attempts failed. Improve your plan based on these errors:\n{error_history}\n"

            # Add full signatures for failed tools to help the LLM fix argument issues
            failed_tool_sigs = self._extract_failed_tool_signatures(failure_context, tools)
            if failed_tool_sigs:
                planning_prompt += f"\nThe following tools had failures. Here are their EXACT required signatures:\n{failed_tool_sigs}\nYou MUST use exactly these argument names.\n"
        
        try:
            # Trace LLM call manually
            step_id = None
            if run_config and "callbacks" in run_config and run_config["callbacks"]:
                 # Try to find recorder in callbacks
                 for cb in run_config["callbacks"]:
                     if hasattr(cb, "recorder"):
                         step_id = cb.recorder.start_step("llm", "plan_generation", planning_prompt)
                         break
            
            response = llm.invoke([
                SystemMessage(content="You are a planning assistant."),
                HumanMessage(content=planning_prompt),
            ])
            
            # Parse plan from response
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
    
    def _convert_tools(self, tools: Dict[str, Any]) -> List[StructuredTool]:
        """Convert τ²-bench tools to LangChain StructuredTool format with disruption checking and action tracking."""
        langchain_tools = []

        for name, func in tools.items():
            def create_wrapper(f, tool_name):
                def wrapped_func(**kwargs):
                    try:
                        # Check for disruptions
                        disruption_engine = get_disruption_engine()
                        disruption_error = disruption_engine.check_disruption(tool_name, kwargs)
                        if disruption_error:
                            return f'{{"status": "failed", "error": "{disruption_error}"}}'

                        result = f(**kwargs)

                        # Track executed action for compensation
                        self._executed_actions.append({
                            "tool": tool_name,
                            "args": kwargs,
                            "result": str(result)[:500],
                        })

                        return self._normalize_tool_output(result, tool_name)
                    except Exception as e:
                        return f'{{"status": "failed", "error": "{str(e)}"}}'
                return wrapped_func

            wrapped = create_wrapper(func, name)
            tool = StructuredTool.from_function(
                func=wrapped,
                name=name,
                description=func.__doc__ or f"Tool: {name}",
            )
            langchain_tools.append(tool)

        return langchain_tools

    def _normalize_tool_output(self, result: Any, tool_name: str) -> str:
        """Normalize tool output to clean JSON string."""
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                result["status"] = "failed"
            return json.dumps(result)

        if isinstance(result, str):
            try:
                cleaned = result.strip()
                if not cleaned.startswith("{") and "error" in cleaned.lower():
                    return json.dumps({"status": "failed", "error": cleaned})
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    if "error" in parsed and parsed["error"]:
                        parsed["status"] = "failed"
                    return json.dumps(parsed)
            except (json.JSONDecodeError, ValueError):
                if "fail" in result.lower() or "error" in result.lower():
                    return json.dumps({"status": "failed", "error": result[:200]})

        return str(result)

    def _format_plan_for_execution(self, plan: List[Dict[str, Any]]) -> str:
        """Format the plan as numbered steps for the ReAct agent's system prompt."""
        lines = []
        for i, step in enumerate(plan, 1):
            tool = step.get("tool", "unknown")
            args = step.get("args", {})
            reasoning = step.get("reasoning", "")
            line = f"Step {i}: Call {tool}({json.dumps(args)})"
            if reasoning:
                line += f" — {reasoning}"
            lines.append(line)
        return "\n".join(lines)

    def _extract_tool_calls(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from ReAct message history."""
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
        result = {}
        if hasattr(msg, "type"):
            result["type"] = msg.type
        if hasattr(msg, "content"):
            result["content"] = str(msg.content)[:500]
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
        if hasattr(msg, "name"):
            result["name"] = msg.name
        return result

    def _compensate(self, tools: Dict[str, Any], recorder: Any = None) -> Dict[str, Any]:
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
                    # Record compensation start
                    step_id = None
                    if recorder:
                         step_id = recorder.start_step("compensation_tool", comp_tool, comp_args)

                    if hasattr(tools[comp_tool], "invoke"):
                         result = tools[comp_tool].invoke(comp_args)
                    else:
                         result = tools[comp_tool](**comp_args)
                    
                    # Record compensation end
                    if recorder and step_id:
                         recorder.end_step(step_id, output=str(result))

                    compensation_actions.append({
                        "original_tool": tool_name,
                        "compensation_tool": comp_tool,
                        "args": comp_args,
                        "success": True,
                    })
                    logger.info(f"Compensated {tool_name} with {comp_tool}")
                    
            except Exception as e:
                logger.error(f"Compensation for {tool_name} failed: {e}")
                if recorder and step_id:
                    recorder.end_step(step_id, output=None, error=str(e))
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
        """Format tool descriptions for the planning prompt with full signatures."""
        import inspect
        descriptions = []

        for name, func in tools.items():
            doc = func.__doc__ or "No description"

            # Extract full function signature with parameter names and types
            try:
                sig = inspect.signature(func)
                params = []
                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue
                    anno = param.annotation
                    if anno is inspect.Parameter.empty:
                        params.append(pname)
                    else:
                        type_name = getattr(anno, "__name__", str(anno))
                        params.append(f"{pname}: {type_name}")
                sig_str = f"{name}({', '.join(params)})"
            except (ValueError, TypeError):
                sig_str = name

            descriptions.append(f"- {sig_str}\n  {doc.strip()}")

        return "\n".join(descriptions)

    def _extract_failed_tool_signatures(
        self, failure_context: List[str], tools: Dict[str, Any]
    ) -> str:
        """Extract full signatures for tools that failed in previous attempts."""
        import inspect
        import re

        failed_tools = set()
        for msg in failure_context:
            match = re.search(r"tool '(\w+)'", msg)
            if match:
                failed_tools.add(match.group(1))

        sigs = []
        for tool_name in failed_tools:
            if tool_name not in tools:
                continue
            func = tools[tool_name]
            doc = func.__doc__ or ""
            try:
                sig = inspect.signature(func)
                params = []
                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue
                    anno = param.annotation
                    if anno is inspect.Parameter.empty:
                        params.append(pname)
                    else:
                        type_name = getattr(anno, "__name__", str(anno))
                        params.append(f"{pname}: {type_name}")
                sigs.append(f"- {tool_name}({', '.join(params)})\n  {doc.strip()}")
            except (ValueError, TypeError):
                sigs.append(f"- {tool_name}: {doc.strip()}")

        return "\n".join(sigs)
