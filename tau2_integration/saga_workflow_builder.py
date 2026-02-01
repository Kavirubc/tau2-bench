"""
SagaLLM Workflow Builder - Corrected Implementation

This module builds LangGraph workflows from SagaLLM plans, using LLM tool calling
at runtime instead of generating executable code. This matches REALM-Bench's
actual implementation.

Key difference from previous approach:
- OLD: Generate code that calls tools → Execute code
- NEW: Generate workflow structure → LLM calls tools at runtime
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Callable

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

logger = logging.getLogger("tau2_integration.saga_workflow_builder")


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SagaState(TypedDict):
    """
    SagaLLM state following the paper's three dimensions.
    
    S_A: Application State (messages, task data)
    S_O: Operation State (current task, status)
    S_D: Dependency State (task outputs, results)
    """
    # S_A: Application State
    messages: List[BaseMessage]
    task_data: Dict[str, Any]
    
    # S_O: Operation State
    current_task: str
    completed_tasks: List[str]
    failed_tasks: List[str]
    
    # S_D: Dependency State
    task_results: Dict[str, Any]
    task_outputs: Dict[str, Any]
    
    # Saga Transaction State
    compensation_stack: List[Dict[str, Any]]
    saga_status: str  # "active", "compensating", "completed", "failed"
    failure_reason: str
    failed_tool: str


# =============================================================================
# WORKFLOW BUILDER
# =============================================================================

class SagaWorkflowBuilder:
    """
    Builds LangGraph workflows from SagaLLM plans.
    
    This creates workflows where LLM decides which tools to call at runtime,
    rather than generating hard-coded tool calls.
    """
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.llm = None
    
    def build_workflow(
        self,
        plan: List[Dict[str, Any]],
        tools: Dict[str, Any],
        llm: ChatGoogleGenerativeAI
    ) -> Any:
        """
        Build LangGraph workflow from plan.
        
        Args:
            plan: List of task steps from Phase 1
            tools: Available tools
            llm: Language model for tool calling
            
        Returns:
            Compiled LangGraph workflow
        """
        self.llm = llm
        workflow = StateGraph(SagaState)
        
        # Create agent nodes for each task
        for i, step in enumerate(plan):
            task_id = step.get('step') or step.get('task_id') or f"task_{i+1}"
            agent_node = self._create_agent_node(step, tools, task_id)
            workflow.add_node(task_id, agent_node)
        
        # Add tool execution node
        tool_node = self._create_tool_node(tools)
        workflow.add_node("tools", tool_node)
        
        # Add compensation node
        comp_node = self._create_compensation_node(tools)
        workflow.add_node("compensate", comp_node)
        
        # Set entry point
        first_task = plan[0].get('step') or plan[0].get('task_id') or "task_1"
        workflow.set_entry_point(first_task)
        
        # Add routing logic
        for i, step in enumerate(plan):
            task_id = step.get('step') or step.get('task_id') or f"task_{i+1}"
            
            # Task → Tools or Compensate
            workflow.add_conditional_edges(
                task_id,
                self._should_continue,
                {
                    "tools": "tools",
                    "compensate": "compensate",
                    "next": plan[i+1].get('step') or plan[i+1].get('task_id') or f"task_{i+2}" if i+1 < len(plan) else END,
                    "end": END
                }
            )
        
        # Tools → Back to agent or Compensate
        workflow.add_conditional_edges(
            "tools",
            self._after_tools,
            {
                "compensate": "compensate",
                "continue": first_task,  # Loop back to continue
                "end": END
            }
        )
        
        # Compensate → End
        workflow.add_edge("compensate", END)
        
        return workflow.compile()
    
    def _create_agent_node(
        self,
        task_spec: Dict[str, Any],
        tools: Dict[str, Any],
        task_id: str
    ) -> Callable:
        """
        Create agent node with LLM tool calling.
        
        This is the KEY difference: instead of generating code that calls tools,
        we create a node that uses LLM to decide which tools to call.
        """
        # Convert tools dict to list for binding
        tool_list = list(tools.values())
        
        # Bind tools to model
        model_with_tools = self.llm.bind_tools(tool_list)
        
        task_name = task_spec.get('tool', 'unknown')
        task_description = task_spec.get('reasoning', '') or task_spec.get('description', '')
        
        def agent_node(state: SagaState) -> Dict:
            """Agent node that uses LLM tool calling."""
            
            # Check if we're in compensation mode
            if state.get("saga_status") == "compensating":
                return {"current_task": task_id}
            
            # Check if task is already completed
            if task_id in state.get("completed_tasks", []):
                logger.info(f"[{task_id}] Already completed, skipping")
                return {"current_task": task_id}

            # Build context-aware prompt
            messages = state.get('messages', [])
            
            # Add task instruction
            task_args = task_spec.get('args', {})
            try:
                task_args_str = json.dumps(task_args, indent=2)
            except:
                task_args_str = str(task_args)

            task_message = HumanMessage(content=f"""Execute this task:

Task: {task_name}
Description: {task_description}
Suggested Arguments:
{task_args_str}

Use the available tools to complete this task using the suggested arguments.
YOU MUST CALL THE TOOL IMMEDIATELY. DO NOT ASK FOR CONFIRMATION.
DO NOT output text, only tool calls.
""")
            
            messages_with_task = messages + [task_message]
            
            logger.info(f"[{task_id}] Invoking LLM with tool calling")
            
            # LLM decides which tools to call
            response = model_with_tools.invoke(messages_with_task)
            
            logger.info(f"[{task_id}] LLM response: {response.content if response.content else 'Tool calls requested'}")
            
            return {
                "messages": [response],
                "current_task": task_id
            }
        
        return agent_node
    
    def _create_tool_node(self, tools: Dict[str, Any]) -> Callable:
        """
        Create tool execution node with saga tracking.
        
        This executes the tool calls requested by the LLM and tracks them
        for compensation.
        """
        # Build tool map - handle both tool objects and raw functions
        tool_map = {}
        for name, tool in tools.items():
            if hasattr(tool, 'name'):
                tool_map[tool.name] = tool
            else:
                tool_map[name] = tool
        
        def tool_node(state: SagaState) -> Dict:
            """Execute tool calls and track for compensation."""
            
            messages = state.get('messages', [])
            if not messages:
                return {}
            
            last_message = messages[-1]
            
            # Check if there are tool calls
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return {}
            
            tool_messages = []
            compensation_stack = state.get('compensation_stack', [])
            failed = False
            failed_tool = ""
            failure_reason = ""
            
            logger.info(f"Executing {len(last_message.tool_calls)} tool calls")
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                
                if tool_name in tool_map:
                    try:
                        tool_func = tool_map[tool_name]
                        if hasattr(tool_func, "invoke"):
                            result = tool_func.invoke(tool_args)
                        else:
                            result = tool_func(**tool_args)
                            
                        logger.info(f"Tool {tool_name} result: {result}")
                        
                        # Track for compensation
                        compensation_stack.append({
                            'tool': tool_name,
                            'args': tool_args,
                            'result': result,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        tool_messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call['id']
                        ))
                        
                    except Exception as e:
                        logger.error(f"Tool {tool_name} failed: {e}")
                        failed = True
                        failed_tool = tool_name
                        failure_reason = str(e)
                        
                        tool_messages.append(ToolMessage(
                            content=f"Error: {e}",
                            tool_call_id=tool_call['id']
                        ))
                else:
                    logger.error(f"Tool {tool_name} not found")
                    failed = True
                    failed_tool = tool_name
                    failure_reason = f"Tool not found: {tool_name}"
                    
                    tool_messages.append(ToolMessage(
                        content=f"Error: Tool {tool_name} not found",
                        tool_call_id=tool_call['id']
                    ))
            
            result = {
                'messages': tool_messages,
                'compensation_stack': compensation_stack
            }
            
            # Mark current task as completed if successful
            current_task = state.get('current_task')
            if current_task and not failed:
                completed = state.get('completed_tasks', [])
                if current_task not in completed:
                    result['completed_tasks'] = completed + [current_task]
            
            if failed:
                result['saga_status'] = 'compensating'
                result['failed_tool'] = failed_tool
                result['failure_reason'] = failure_reason
                logger.warning(f"Tool execution failed, triggering compensation")
            
            return result
        
        return tool_node
    
    def _create_compensation_node(self, tools: Dict[str, Any]) -> Callable:
        """
        Create compensation node that rolls back executed actions.
        
        Executes compensations in LIFO order (reverse of execution).
        """
        # Map of tools to their compensation tools
        COMPENSATION_MAPPING = {
            'book_flight': 'cancel_reservation',
            'reserve_hotel': 'cancel_reservation',
            'book_car': 'cancel_reservation',
            'create_itinerary': 'cancel_reservation',
            # Add more mappings as needed
        }
        
        # Build tool map
        tool_map = {}
        for name, tool in tools.items():
            if hasattr(tool, 'name'):
                tool_map[tool.name] = tool
            else:
                tool_map[name] = tool
        
        def compensation_node(state: SagaState) -> Dict:
            """Execute compensations in reverse order."""
            
            compensation_stack = state.get('compensation_stack', [])
            
            logger.info(f"=== SAGA COMPENSATION ===")
            logger.info(f"Reason: {state.get('failure_reason', 'Unknown')}")
            logger.info(f"Failed tool: {state.get('failed_tool', 'Unknown')}")
            logger.info(f"Actions to compensate: {len(compensation_stack)}")
            
            compensated = []
            
            # Execute compensations in reverse order (LIFO)
            for action in reversed(compensation_stack):
                tool_name = action['tool']
                tool_args = action['args']
                
                # Find compensation tool
                comp_tool_name = COMPENSATION_MAPPING.get(tool_name)
                
                if comp_tool_name and comp_tool_name in tool_map:
                    try:
                        logger.info(f"Compensating {tool_name} with {comp_tool_name}")
                        
                        # Extract ID from result if available
                        result = action.get('result', {})
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except:
                                pass
                        
                        # Call compensation tool
                        comp_tool_func = tool_map[comp_tool_name]
                        if isinstance(result, dict) and 'id' in result:
                            comp_args = {'id': result['id']}
                        else:
                            comp_args = tool_args
                            
                        if hasattr(comp_tool_func, "invoke"):
                            comp_result = comp_tool_func.invoke(comp_args)
                        else:
                            comp_result = comp_tool_func(**comp_args)
                        
                        logger.info(f"Compensation result: {comp_result}")
                        compensated.append(tool_name)
                        
                    except Exception as e:
                        logger.error(f"Compensation failed for {tool_name}: {e}")
                else:
                    logger.warning(f"No compensation tool for {tool_name}")
            
            logger.info(f"Compensation complete. {len(compensated)} actions rolled back.")
            logger.info(f"=========================")
            
            return {
                'saga_status': 'failed',
                'compensation_stack': [],
                'messages': [AIMessage(content=f"Saga compensation complete. Rolled back {len(compensated)} actions.")]
            }
        
        return compensation_node
    
    def _should_continue(self, state: SagaState) -> str:
        """Determine next step after agent node."""
        
        # Check for compensation
        if state.get("saga_status") == "compensating":
            return "compensate"
        
        # Check for tool calls
        messages = state.get('messages', [])
        if messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            return "tools"
        
        # Check if we should continue to next task
        return "next"
    
    def _after_tools(self, state: SagaState) -> str:
        """Determine next step after tool execution."""
        
        # Check for compensation
        if state.get("saga_status") == "compensating":
            return "compensate"
        
        # Check if saga failed
        if state.get("saga_status") == "failed":
            return "end"
        
        # Continue workflow
        return "continue"
