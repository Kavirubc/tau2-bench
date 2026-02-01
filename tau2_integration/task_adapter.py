"""
Task adapter for converting τ²-bench tasks to framework-compatible format.

Converts τ²-bench task definitions to a format compatible with
SagaLLM, RAC, and LangGraph runners.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tau2_integration.task_adapter")


class TaskCategory(Enum):
    """Categories of τ²-bench tasks."""
    BOOKING = "booking"
    MODIFICATION = "modification"
    CANCELLATION = "cancellation"
    INQUIRY = "inquiry"
    COMPENSATION = "compensation"


@dataclass
class TaskGoal:
    """Definition of a task goal."""
    goal_id: str
    description: str
    weight: float = 1.0
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConstraint:
    """Definition of a task constraint."""
    constraint_id: str
    constraint_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass 
class Tau2TaskDefinition:
    """
    Unified task definition for framework integration.
    
    Combines τ²-bench task format with REALM-Bench style fields.
    """
    task_id: str
    name: str
    category: TaskCategory
    description: str
    user_scenario: Dict[str, Any]
    goals: List[TaskGoal]
    constraints: List[TaskConstraint]
    expected_actions: List[Dict[str, Any]]
    nl_assertions: List[str]
    disruption_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    # Original τ²-bench fields preserved
    original_task: Optional[Dict[str, Any]] = None


def load_tau2_tasks(
    domain: str,
    task_ids: Optional[List[str]] = None
) -> List[Tau2TaskDefinition]:
    """
    Load τ²-bench tasks from a specific domain.

    Args:
        domain: Domain name (airline, retail, telecom, mock).
        task_ids: Optional list of task IDs to load. If None, loads all.

    Returns:
        List of Tau2TaskDefinition objects.
    """
    from .domain_registry import DomainRegistry
    
    # Get domain configuration
    config = DomainRegistry.get_domain(domain)
    tasks_file = config.tasks_path
    
    with open(tasks_file, "r") as f:
        raw_tasks = json.load(f)
    
    tasks = []
    for raw_task in raw_tasks:
        task_id = str(raw_task.get("id", ""))
        
        # Filter by task_ids if specified
        if task_ids is not None and task_id not in task_ids:
            continue
        
        task = convert_tau2_task(raw_task, domain)
        tasks.append(task)
        
    logger.info(f"Loaded {len(tasks)} tasks from {domain} domain ({tasks_file})")
    return tasks


def convert_tau2_task(raw_task: Dict[str, Any], domain: str = "airline") -> Tau2TaskDefinition:
    """
    Convert a raw τ²-bench task to Tau2TaskDefinition.

    Args:
        raw_task: Raw task dictionary from tasks.json.
        domain: Domain name for context.

    Returns:
        Tau2TaskDefinition object.
    """
    task_id = str(raw_task.get("id", "unknown"))
    description_info = raw_task.get("description", {})
    user_scenario = raw_task.get("user_scenario", {})
    evaluation = raw_task.get("evaluation_criteria", {})
    
    # Determine category from task content
    category = _infer_category(raw_task, domain)
    
    # Extract goals from NL assertions
    nl_assertions = evaluation.get("nl_assertions", []) or []
    goals = [
        TaskGoal(
            goal_id=f"goal_{i}",
            description=assertion,
            weight=1.0
        )
        for i, assertion in enumerate(nl_assertions)
    ]
    
    # Extract expected actions
    expected_actions = evaluation.get("actions", []) or []
    
    # Extract constraints from policy references
    constraints = _extract_constraints(raw_task, domain)
    
    # Build name from purpose or reason for call
    purpose = description_info.get("purpose", "")
    reason = user_scenario.get("instructions", {}).get("reason_for_call", "")
    name = purpose[:100] if purpose else reason[:100] if reason else f"Task {task_id}"
    
    return Tau2TaskDefinition(
        task_id=task_id,
        name=name,
        category=category,
        description=purpose or reason,
        user_scenario=user_scenario,
        goals=goals,
        constraints=constraints,
        expected_actions=expected_actions,
        nl_assertions=nl_assertions,
        disruption_scenarios=[],  # Will be injected separately
        original_task=raw_task,
    )


def _infer_category(raw_task: Dict[str, Any], domain: str = "airline") -> TaskCategory:
    """Infer task category from content."""
    user_scenario = raw_task.get("user_scenario", {})
    instructions = user_scenario.get("instructions", {})
    reason = instructions.get("reason_for_call", "").lower()
    task_instructions = instructions.get("task_instructions", "").lower()
    
    actions = raw_task.get("evaluation_criteria", {}).get("actions", [])
    action_names = [a.get("name", "") for a in actions]
    
    # Check for specific patterns
    if "book" in reason or "book_reservation" in action_names:
        return TaskCategory.BOOKING
    elif "cancel" in reason or "cancel_reservation" in action_names:
        return TaskCategory.CANCELLATION
    elif "change" in reason or "modify" in reason or "update" in reason:
        return TaskCategory.MODIFICATION
    elif "compensation" in reason or "send_certificate" in action_names:
        return TaskCategory.COMPENSATION
    else:
        return TaskCategory.INQUIRY


def _extract_constraints(raw_task: Dict[str, Any], domain: str = "airline") -> List[TaskConstraint]:
    """Extract constraints from task definition."""
    constraints = []
    
    # Add constraint about user confirmation for write operations
    constraints.append(TaskConstraint(
        constraint_id="require_confirmation",
        constraint_type="policy",
        description="Agent must obtain explicit user confirmation before write operations"
    ))
    
    # Add constraint about single tool call
    constraints.append(TaskConstraint(
        constraint_id="single_tool_call",
        constraint_type="policy", 
        description="Agent should only make one tool call at a time"
    ))
    
    return constraints


def add_disruption_scenarios(
    task: Tau2TaskDefinition,
    scenarios: List[Dict[str, Any]]
) -> Tau2TaskDefinition:
    """
    Add disruption scenarios to a task definition.

    Args:
        task: Task definition to modify.
        scenarios: List of disruption scenarios to add.

    Returns:
        Modified task definition.
    """
    task.disruption_scenarios = scenarios
    return task


def get_task_query(task: Tau2TaskDefinition) -> str:
    """
    Build the natural language query/prompt for a task.

    Args:
        task: Task definition.

    Returns:
        Query string for the agent.
    """
    instructions = task.user_scenario.get("instructions", {})
    reason = instructions.get("reason_for_call", "")
    known_info = instructions.get("known_info", "")
    task_instructions = instructions.get("task_instructions", "")
    
    query_parts = []
    
    if reason:
        query_parts.append(f"Reason for call: {reason}")
    if known_info:
        query_parts.append(f"Known information: {known_info}")
    if task_instructions:
        query_parts.append(f"Additional instructions: {task_instructions}")
    
    return "\n\n".join(query_parts)


def get_expected_tool_calls(task: Tau2TaskDefinition) -> List[Dict[str, Any]]:
    """
    Get the expected tool calls for a task.

    Args:
        task: Task definition.

    Returns:
        List of expected tool call specifications.
    """
    return task.expected_actions
