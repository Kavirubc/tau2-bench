"""
Evaluation metrics for compensation robustness comparison.

Calculates metrics for comparing framework performance on τ²-bench tasks.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .runners.base import RunnerResult

logger = logging.getLogger("tau2_integration.evaluation")


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single run."""
    # Task completion
    task_success: bool
    goal_completion_rate: float
    
    # Compensation robustness
    disruptions_triggered: int
    compensation_attempted: bool
    compensation_success: bool
    rollback_integrity: float  # 0-1 score
    
    # Efficiency
    tool_call_count: int
    execution_time: float
    
    # Token usage (if available)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def evaluate_run(
    result: RunnerResult,
    expected_actions: List[Dict[str, Any]],
    nl_assertions: List[str],
) -> EvaluationMetrics:
    """
    Evaluate a single run result.

    Args:
        result: RunnerResult from framework execution.
        expected_actions: List of expected tool calls.
        nl_assertions: Natural language assertions to check.

    Returns:
        EvaluationMetrics for this run.
    """
    # Check goal completion
    goals_met = 0
    total_goals = len(expected_actions) + len(nl_assertions)
    
    # Check expected actions
    for expected in expected_actions:
        expected_name = expected.get("name")
        expected_args = expected.get("arguments", {})
        
        for actual in result.tool_calls:
            if actual.get("name") == expected_name:
                # Check if arguments match (flexible matching)
                if _args_match(expected_args, actual.get("arguments", {})):
                    goals_met += 1
                    break
    
    # For NL assertions, we'd need LLM-as-judge or manual review
    # For now, count them as not automatically verifiable
    
    goal_rate = goals_met / total_goals if total_goals > 0 else 0.0
    
    # Evaluate compensation robustness
    disruptions = len(result.disruptions_triggered)
    comp_attempted = len(result.compensation_actions) > 0
    comp_success = result.rollback_success
    
    # Calculate rollback integrity
    # 1.0 if no disruptions or all compensations succeeded
    # Lower score if compensations failed
    if disruptions == 0:
        rollback_integrity = 1.0
    elif comp_attempted and comp_success:
        rollback_integrity = 1.0
    elif comp_attempted and not comp_success:
        rollback_integrity = 0.5  # Partial
    else:
        rollback_integrity = 0.0  # No compensation attempted
    
    # Token usage (if available)
    tokens = result.token_usage or {}
    
    return EvaluationMetrics(
        task_success=result.success,
        goal_completion_rate=goal_rate,
        disruptions_triggered=disruptions,
        compensation_attempted=comp_attempted,
        compensation_success=comp_success,
        rollback_integrity=rollback_integrity,
        tool_call_count=len(result.tool_calls),
        execution_time=result.execution_time,
        input_tokens=tokens.get("input", 0),
        output_tokens=tokens.get("output", 0),
        total_tokens=tokens.get("total", 0),
    )


def _args_match(expected: Dict, actual: Dict) -> bool:
    """Check if actual arguments match expected (flexible matching)."""
    for key, value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        
        # Flexible comparison
        if isinstance(value, str) and isinstance(actual_value, str):
            if value.lower() != actual_value.lower():
                return False
        elif value != actual_value:
            return False
    
    return True


def compare_frameworks(
    results: Dict[str, List[EvaluationMetrics]]
) -> Dict[str, Any]:
    """
    Compare evaluation metrics across frameworks.

    Args:
        results: Dictionary mapping framework name to list of metrics.

    Returns:
        Comparison summary.
    """
    comparison = {}
    
    for framework, metrics_list in results.items():
        if not metrics_list:
            continue
        
        n = len(metrics_list)
        
        # Aggregate metrics
        comparison[framework] = {
            # Task completion
            "success_rate": sum(1 for m in metrics_list if m.task_success) / n,
            "avg_goal_completion": sum(m.goal_completion_rate for m in metrics_list) / n,
            
            # Compensation robustness
            "avg_disruptions": sum(m.disruptions_triggered for m in metrics_list) / n,
            "compensation_rate": sum(1 for m in metrics_list if m.compensation_attempted) / n,
            "rollback_success_rate": sum(
                1 for m in metrics_list 
                if m.compensation_attempted and m.compensation_success
            ) / max(1, sum(1 for m in metrics_list if m.compensation_attempted)),
            "avg_rollback_integrity": sum(m.rollback_integrity for m in metrics_list) / n,
            
            # Efficiency
            "avg_tool_calls": sum(m.tool_call_count for m in metrics_list) / n,
            "avg_execution_time": sum(m.execution_time for m in metrics_list) / n,
            "avg_tokens": sum(m.total_tokens for m in metrics_list) / n,
            
            "total_runs": n,
        }
    
    return comparison


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print a formatted comparison table."""
    if not comparison:
        print("No results to compare.")
        return
    
    print("\n" + "=" * 80)
    print("FRAMEWORK COMPARISON")
    print("=" * 80)
    
    # Header
    metrics = [
        ("Success Rate", "success_rate", "{:.1%}"),
        ("Goal Completion", "avg_goal_completion", "{:.1%}"),
        ("Rollback Integrity", "avg_rollback_integrity", "{:.1%}"),
        ("Avg Tool Calls", "avg_tool_calls", "{:.1f}"),
        ("Avg Time (s)", "avg_execution_time", "{:.2f}"),
    ]
    
    header = f"{'Framework':<15}"
    for label, _, _ in metrics:
        header += f" {label:<18}"
    print(header)
    print("-" * 80)
    
    # Rows
    for framework, stats in comparison.items():
        row = f"{framework:<15}"
        for _, key, fmt in metrics:
            value = stats.get(key, 0)
            row += f" {fmt.format(value):<18}"
        print(row)
    
    print("=" * 80 + "\n")
