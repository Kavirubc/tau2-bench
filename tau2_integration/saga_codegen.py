"""
Phase 2: Code Generation for SagaLLM (Paper-Compliant)

Implements Algorithm 1, Stage 2 from the SagaLLM paper:
- DefineLogSchema: Generate state tracking schemas
- DefineNodeAgent: Generate task execution code
- DefineCompAgent: Generate compensation code

This module uses LLM to generate Python code for each task, which is then
executed in Phase 3.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tau2_integration.saga_codegen")


@dataclass
class GeneratedSchema:
    """Generated state schema for a task."""
    task_id: str
    schema_name: str
    schema_code: str
    fields: List[str]


@dataclass
class GeneratedAgent:
    """Generated task execution agent."""
    task_id: str
    agent_name: str
    agent_code: str
    required_tools: List[str]


@dataclass
class GeneratedCompensation:
    """Generated compensation agent."""
    task_id: str
    comp_name: str
    comp_code: str


@dataclass
class GeneratedWorkflow:
    """Complete generated workflow."""
    schemas: Dict[str, GeneratedSchema]
    agents: Dict[str, GeneratedAgent]
    compensations: Dict[str, GeneratedCompensation]
    total_tokens: int
    llm_calls: int


class SagaCodeGenerator:
    """
    Phase 2 Code Generator for SagaLLM.
    
    Generates Python code for task execution and compensation using LLM.
    """
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize code generator with LLM."""
        self.model = model
        self.total_tokens = 0
        self.llm_calls = 0
        
        # Initialize LLM
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.2,  # Lower temperature for code generation
                google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            )
        except ImportError:
            raise ImportError("langchain-google-genai required for code generation")
    
    def _call_llm(self, prompt: str) -> str:
        """Make LLM call and track tokens."""
        from langchain_core.messages import HumanMessage
        
        self.llm_calls += 1
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Track tokens (approximate if not available)
        content = response.content
        self.total_tokens += len(prompt.split()) * 1.3 + len(content.split()) * 1.3
        
        return content
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code in markdown blocks
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try without language specifier
        code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Return as-is if no markdown
        return response.strip()
    
    def _validate_generated_code(self, code: str, expected_function: str, tools: Dict[str, Any]) -> tuple[bool, str]:
        """Validate generated code for common errors."""
        issues = []
        
        # Check 1: Function definition exists
        if f"def {expected_function}" not in code:
            issues.append(f"Missing function definition: {expected_function}")
        
        # Check 2: Returns state
        if "return state" not in code:
            issues.append("Missing 'return state' statement")
        
        # Check 3: Tool names are valid
        for tool_name in tools.keys():
            if f"tools['{tool_name}']" in code or f'tools["{tool_name}"]' in code:
                continue  # Valid tool reference found
        
        # Check if any invalid tool names are used
        import re
        tool_pattern = r"tools\['([^']+)'\]|tools\[\"([^\"]+)\"\]"
        found_tools = re.findall(tool_pattern, code)
        for match in found_tools:
            tool_name = match[0] or match[1]
            if tool_name not in tools:
                issues.append(f"Invalid tool name: '{tool_name}' (available: {list(tools.keys())})")
        
        # Check 4: Basic syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "Validation passed"
    
    def _get_tool_signature_details(self, tools: Dict[str, Any]) -> str:
        """Get detailed tool signatures with examples."""
        tool_details = []
        
        for name, tool in tools.items():
            # Get description
            doc = getattr(tool, '__doc__', '') or ''
            desc = doc.split('\n')[0].strip() if doc else f"Tool: {name}"
            
            # Try to get args schema
            try:
                if hasattr(tool, 'args'):
                    args_schema = tool.args
                    if isinstance(args_schema, dict):
                        params = []
                        for arg_name, arg_spec in args_schema.items():
                            arg_type = arg_spec.get('type', 'Any')
                            params.append(f"{arg_name}: {arg_type}")
                        
                        params_str = ", ".join(params)
                        example_args = ", ".join([f"'{k}': value" for k in args_schema.keys()])
                        
                        tool_details.append(
                            f"  • {name}({params_str})\n"
                            f"    Description: {desc}\n"
                            f"    Usage: tools['{name}'].invoke({{{example_args}}})"
                        )
                    else:
                        tool_details.append(f"  • {name}(...)\n    Description: {desc}")
                else:
                    tool_details.append(f"  • {name}(...)\n    Description: {desc}")
            except Exception as e:
                logger.debug(f"Could not extract args for {name}: {e}")
                tool_details.append(f"  • {name}(...)\n    Description: {desc}")
        
        return "\n\n".join(tool_details)
    
    def define_log_schema(
        self,
        task_id: str,
        task_name: str,
        task_description: str,
        tool_names: List[str]
    ) -> GeneratedSchema:
        """
        Algorithm 1, Line 9: DefineLogSchema(ni, Pni)
        
        Generate state tracking schema for a task using LLM.
        """
        logger.info(f"[Phase 2] DefineLogSchema for {task_id}")
        
        prompt = f"""Generate a Python dataclass for tracking the state of this task:

Task ID: {task_id}
Task Name: {task_name}
Description: {task_description}
Available Tools: {', '.join(tool_names)}

Create a dataclass called `{task_id.replace('-', '_').title()}State` that includes:
1. Input parameters (extracted from description)
2. Execution status (pending/running/completed/failed)
3. Result storage
4. Timestamps
5. Error information

Requirements:
- Use @dataclass decorator
- All fields must have default values
- Include typing hints
- Follow this structure:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class {task_id.replace('-', '_').title()}State:
    # Input parameters
    # ... (infer from description)
    
    # Execution state
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
```

Generate ONLY the Python code, no explanations."""

        response = self._call_llm(prompt)
        code = self._extract_code(response)
        
        # Validate syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            logger.warning(f"Generated schema has syntax error: {e}")
        
        # Extract field names
        fields = []
        for line in code.split('\n'):
            if ':' in line and not line.strip().startswith('#'):
                match = re.match(r'\s*(\w+):', line)
                if match:
                    fields.append(match.group(1))
        
        schema_name = f"{task_id.replace('-', '_').title()}State"
        
        return GeneratedSchema(
            task_id=task_id,
            schema_name=schema_name,
            schema_code=code,
            fields=fields
        )
    
    def define_node_agent(
        self,
        task_id: str,
        task_name: str,
        task_description: str,
        schema: GeneratedSchema,
        tools: Dict[str, Any],
        preconditions: List[str] = None,
        postconditions: List[str] = None
    ) -> GeneratedAgent:
        """
        Algorithm 1, Line 10: DefineNodeAgent(ni, Lni)
        
        Generate task execution agent code using LLM.
        """
        logger.info(f"[Phase 2] DefineNodeAgent for {task_id}")
        
        # Get detailed tool signatures
        tools_detailed = self._get_tool_signature_details(tools)
        
        precond_str = ', '.join(preconditions) if preconditions else "None"
        postcond_str = ', '.join(postconditions) if postconditions else "None"
        
        # Extract potential parameters from task description
        param_hints = f"\\nParameter extraction hints for '{task_description}':\\n"
        if 'customer' in task_description.lower() or 'user' in task_description.lower():
            param_hints += "- Look for customer/user IDs (e.g., 'C123', 'user_456')\\n"
        if 'from' in task_description.lower() and 'to' in task_description.lower():
            param_hints += "- Look for origin and destination (e.g., 'from NYC to LAX')\\n"
        if 'flight' in task_description.lower():
            param_hints += "- Look for flight numbers, dates, or booking references\\n"
        if 'reservation' in task_description.lower() or 'booking' in task_description.lower():
            param_hints += "- Look for reservation/booking IDs\\n"
        
        prompt = f"""Generate a Python function that executes this task:

Task ID: {task_id}
Task Name: {task_name}
Description: {task_description}
Preconditions: {precond_str}
Postconditions: {postcond_str}
{param_hints}
State Schema:
{schema.schema_code}

Available Tools (with exact signatures):
{tools_detailed}

Generate a function with this signature:
```python
def {task_id.replace('-', '_')}_agent(state: {schema.schema_name}, tools: Dict[str, Any]) -> {schema.schema_name}:
```

CRITICAL REQUIREMENTS:
1. Use EXACT tool names from the list above
2. Use EXACT parameter names from the tool signatures
3. Extract values from the task description or state fields
4. Handle tool responses (they return strings or dicts)
5. Store results in state.result as a dictionary
6. Set state.status to "completed" on success, "failed" on error
7. Use try/except for error handling (NO finally blocks)
8. Always return the state object

Step-by-step approach:
1. Validate state.status == "pending"
2. Set state.status = "running" and state.started_at
3. Parse task description to extract required parameters
4. Call the appropriate tool with correct parameter names
5. Check if result contains errors
6. Store result in state.result
7. Set state.status = "completed" and state.completed_at
8. Return state

Example (for a booking task):
```python
def book_flight_agent(state: BookFlightState, tools: Dict[str, Any]) -> BookFlightState:
    from datetime import datetime
    
    # Validate
    if state.status != "pending":
        state.error = "Task already executed"
        state.status = "failed"
        return state
    
    # Start execution
    state.status = "running"
    state.started_at = datetime.now().isoformat()
    
    try:
        # Extract parameters from description or state
        # Example: "Book flight for customer C123 from NYC to LAX"
        customer_id = "C123"  # Extract from description
        origin = "NYC"
        destination = "LAX"
        
        # Call tool with EXACT parameter names
        result = tools['book_flight'].invoke({{
            'customer_id': customer_id,
            'origin': origin,
            'destination': destination
        }})
        
        # Store result
        state.result = {{'output': str(result), 'booking_id': 'extracted_id'}}
        state.status = "completed"
        state.completed_at = datetime.now().isoformat()
        
    except Exception as e:
        state.error = str(e)
        state.status = "failed"
        state.completed_at = datetime.now().isoformat()
    
    return state
```

IMPORTANT:
- Match tool parameter names EXACTLY (e.g., if tool uses 'customer_id', don't use 'user_id')
- Extract parameter values from the task description
- Don't invent tool names - use only tools from the list above
- Check result for error indicators before marking as completed
- NO finally blocks, NO return inside finally

Generate ONLY the Python function code, no explanations."""

        # First attempt
        response = self._call_llm(prompt)
        code = self._extract_code(response)
        
        # Validate generated code
        agent_name = f"{task_id.replace('-', '_')}_agent"
        is_valid, validation_msg = self._validate_generated_code(code, agent_name, tools)
        
        # If validation fails, try to fix it
        if not is_valid:
            logger.warning(f"Generated code validation failed: {validation_msg}")
            logger.info("Attempting to regenerate with error feedback...")
            
            # Add error feedback to prompt
            fix_prompt = f"""The previous code generation had issues:
{validation_msg}

Please regenerate the code fixing these issues. Remember:
- Use EXACT tool names from the available tools list
- Include 'return state' at the end
- Ensure proper syntax

Original prompt:
{prompt}

Generate corrected code:"""
            
            response = self._call_llm(fix_prompt)
            code = self._extract_code(response)
            
            # Validate again
            is_valid, validation_msg = self._validate_generated_code(code, agent_name, tools)
            if not is_valid:
                logger.error(f"Code still invalid after retry: {validation_msg}")
        
        # Extract required tools
        required_tools = []
        for tool_name in tools.keys():
            if tool_name in code:
                required_tools.append(tool_name)
        
        return GeneratedAgent(
            task_id=task_id,
            agent_name=agent_name,
            agent_code=code,
            required_tools=required_tools
        )
    
    def define_comp_agent(
        self,
        task_id: str,
        task_name: str,
        schema: GeneratedSchema,
        agent: GeneratedAgent,
        compensation_tool: Optional[str] = None
    ) -> GeneratedCompensation:
        """
        Algorithm 1, Line 11: DefineCompAgent(αni, Lni)
        
        Generate compensation agent code using LLM.
        """
        logger.info(f"[Phase 2] DefineCompAgent for {task_id}")
        
        comp_tool_info = f"Compensation tool: {compensation_tool}" if compensation_tool else "No specific compensation tool"
        
        prompt = f"""Generate a Python function that compensates (undoes) this task:

Task ID: {task_id}
Task Name: {task_name}
{comp_tool_info}

State Schema:
{schema.schema_code}

Original Agent Code:
{agent.agent_code}

Generate a function with this signature:
```python
def {task_id.replace('-', '_')}_compensation(state: {schema.schema_name}, tools: Dict[str, Any]) -> {schema.schema_name}:
```

The compensation function should:
1. Check if compensation is needed (state.status == "completed")
2. If not completed, return state unchanged (idempotent)
3. Extract necessary IDs from state.result
4. Call the compensation tool to undo the action
5. Update state.status to "compensated"
6. Clear state.result
7. Handle errors gracefully (DO NOT use finally blocks)
8. Return the updated state

Important:
- Be idempotent (safe to call multiple times)
- Extract IDs from state.result dictionary
- Use try/except for error handling (NO finally blocks)
- If compensation fails, set state.status to "compensation_failed"
- DO NOT use 'return' inside 'finally' blocks
- Always return state at the end

Example structure:
```python
def task_compensation(state: TaskState, tools: Dict[str, Any]) -> TaskState:
    # Check if compensation needed
    if state.status != "completed":
        return state  # Nothing to compensate
    
    try:
        # Extract ID from result
        if state.result and 'id' in state.result:
            item_id = state.result['id']
            # Call compensation tool
            tools['undo_tool'].invoke({{'id': item_id}})
        
        state.status = "compensated"
        state.result = None
    except Exception as e:
        state.status = "compensation_failed"
        state.error = f"Compensation failed: {{str(e)}}"
    
    return state
```

Generate ONLY the Python function code, no explanations."""

        response = self._call_llm(prompt)
        code = self._extract_code(response)
        
        # Validate syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            logger.warning(f"Generated compensation has syntax error: {e}")
        
        comp_name = f"{task_id.replace('-', '_')}_compensation"
        
        return GeneratedCompensation(
            task_id=task_id,
            comp_name=comp_name,
            comp_code=code
        )
    
    def generate_workflow(
        self,
        plan: List[Dict[str, Any]],
        tools: Dict[str, Any],
        compensation_mapping: Dict[str, str] = None
    ) -> GeneratedWorkflow:
        """
        Generate complete workflow with code for all tasks.
        
        Implements Algorithm 1, Stage 2 (lines 8-15).
        """
        logger.info(f"[Phase 2] Generating code for {len(plan)} tasks")
        
        schemas = {}
        agents = {}
        compensations = {}
        tool_names = list(tools.keys())
        compensation_mapping = compensation_mapping or {}
        
        for i, step in enumerate(plan, 1):
            task_id = step.get('step') or step.get('task_id') or f"task_{i}"
            task_name = step.get('tool', 'unknown')
            task_desc = step.get('reasoning', '') or step.get('description', '')
            
            logger.info(f"[{i}/{len(plan)}] Generating code for {task_id}")
            
            # Step 1: DefineLogSchema
            schema = self.define_log_schema(
                task_id=task_id,
                task_name=task_name,
                task_description=task_desc,
                tool_names=tool_names
            )
            schemas[task_id] = schema
            
            # Step 2: DefineNodeAgent
            agent = self.define_node_agent(
                task_id=task_id,
                task_name=task_name,
                task_description=task_desc,
                schema=schema,
                tools=tools,
                preconditions=step.get('preconditions', []),
                postconditions=step.get('postconditions', [])
            )
            agents[task_id] = agent
            
            # Step 3: DefineCompAgent
            comp_tool = compensation_mapping.get(task_name)
            compensation = self.define_comp_agent(
                task_id=task_id,
                task_name=task_name,
                schema=schema,
                agent=agent,
                compensation_tool=comp_tool
            )
            compensations[task_id] = compensation
        
        logger.info(f"[Phase 2] Code generation complete: {self.llm_calls} LLM calls, ~{int(self.total_tokens)} tokens")
        
        return GeneratedWorkflow(
            schemas=schemas,
            agents=agents,
            compensations=compensations,
            total_tokens=int(self.total_tokens),
            llm_calls=self.llm_calls
        )
    
    def execute_generated_code(
        self,
        workflow: GeneratedWorkflow,
        tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute generated code (Phase 3).
        
        This creates a namespace with all generated code and executes it.
        """
        logger.info("[Phase 3] Executing generated code")
        
        # Create execution namespace
        namespace = {
            'tools': tools,
            'datetime': datetime,
            'Dict': Dict,
            'Any': Any,
            'Optional': Optional,
        }
        
        # Load all schemas
        for schema in workflow.schemas.values():
            try:
                exec(schema.schema_code, namespace)
                logger.debug(f"Loaded schema: {schema.schema_name}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema.schema_name}: {e}")
                raise
        
        # Load all agents
        for agent in workflow.agents.values():
            try:
                exec(agent.agent_code, namespace)
                logger.debug(f"Loaded agent: {agent.agent_name}")
            except Exception as e:
                logger.error(f"Failed to load agent {agent.agent_name}: {e}")
                raise
        
        # Load all compensations
        for comp in workflow.compensations.values():
            try:
                exec(comp.comp_code, namespace)
                logger.debug(f"Loaded compensation: {comp.comp_name}")
            except Exception as e:
                logger.error(f"Failed to load compensation {comp.comp_name}: {e}")
                raise
        
        # Execute tasks in order
        executed_actions = []
        for task_id in workflow.schemas.keys():
            schema = workflow.schemas[task_id]
            agent = workflow.agents[task_id]
            
            try:
                # Create state instance
                state_class = namespace[schema.schema_name]
                state = state_class()
                
                # Execute agent
                agent_func = namespace[agent.agent_name]
                updated_state = agent_func(state, tools)
                
                # Record execution
                executed_actions.append({
                    'task_id': task_id,
                    'agent': agent.agent_name,
                    'state': updated_state,
                    'status': updated_state.status,
                })
                
                # Check for failure
                if updated_state.status == 'failed':
                    logger.warning(f"Task {task_id} failed: {updated_state.error}")
                    return {
                        'success': False,
                        'failed_task': task_id,
                        'error': updated_state.error,
                        'executed_actions': executed_actions,
                        'namespace': namespace
                    }
                
            except Exception as e:
                logger.error(f"Execution failed at {task_id}: {e}")
                return {
                    'success': False,
                    'failed_task': task_id,
                    'error': str(e),
                    'executed_actions': executed_actions,
                    'namespace': namespace
                }
        
        return {
            'success': True,
            'executed_actions': executed_actions,
            'namespace': namespace
        }
    
    def execute_compensation(
        self,
        workflow: GeneratedWorkflow,
        executed_actions: List[Dict[str, Any]],
        namespace: Dict[str, Any],
        tools: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute compensation in reverse order (Phase 4).
        """
        logger.info("[Phase 4] Executing compensation (LIFO)")
        
        compensation_results = []
        
        # Compensate in reverse order (LIFO)
        for action in reversed(executed_actions):
            task_id = action['task_id']
            state = action['state']
            
            if task_id not in workflow.compensations:
                logger.warning(f"No compensation for {task_id}")
                continue
            
            comp = workflow.compensations[task_id]
            
            try:
                # Execute compensation
                comp_func = namespace[comp.comp_name]
                compensated_state = comp_func(state, tools)
                
                compensation_results.append({
                    'task_id': task_id,
                    'compensation': comp.comp_name,
                    'status': compensated_state.status,
                    'success': True
                })
                
                logger.info(f"Compensated {task_id}: {compensated_state.status}")
                
            except Exception as e:
                logger.error(f"Compensation failed for {task_id}: {e}")
                compensation_results.append({
                    'task_id': task_id,
                    'compensation': comp.comp_name,
                    'error': str(e),
                    'success': False
                })
        
        return {
            'success': True,
            'compensation_results': compensation_results
        }
