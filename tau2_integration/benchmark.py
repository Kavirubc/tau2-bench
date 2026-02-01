"""
Benchmark runner for comparing frameworks on τ²-bench airline domain.

Runs multiple frameworks on the same tasks with disruption injection
and collects comparative metrics.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tau2_integration.disruption_engine import AIRLINE_DISRUPTION_SCENARIOS
from tau2_integration.task_adapter import (
    load_tau2_tasks,
    add_disruption_scenarios,
    Tau2TaskDefinition,
)
from tau2_integration.wrapped_tools import wrap_tool_with_disruption
from tau2_integration.runners.base import RunnerResult
from tau2_integration.runners.langgraph_runner import LangGraphRunner
from tau2_integration.runners.rac_runner import RACRunner
from tau2_integration.runners.saga_runner import SagaLLMRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tau2_integration.benchmark")

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "tau2" / "domains" / "airline"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"

# Default model
DEFAULT_MODEL = "gemini-2.0-flash"

# Write operations that need rollback tracking
WRITE_OPERATIONS = {
    "book_reservation", 
    "cancel_reservation", 
    "update_reservation_flights", 
    "update_reservation_baggages",
    "update_reservation_passengers",
    "send_certificate",
}


def get_runner(framework: str, model: str = DEFAULT_MODEL):
    """Get a runner instance for the specified framework."""
    runners = {
        "langgraph": LangGraphRunner,
        "rac": RACRunner,
        "sagallm": SagaLLMRunner,
    }
    
    if framework not in runners:
        raise ValueError(f"Unknown framework: {framework}. Choose from {list(runners.keys())}")
    
    return runners[framework](model=model)


def load_airline_tools(wrap_with_disruption: bool = True):
    """Load τ²-bench airline tools.
    
    Args:
        wrap_with_disruption: Whether to wrap tools with disruption injection.
    """
    try:
        from tau2.domains.airline.tools import AirlineTools
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.airline.utils import AIRLINE_DB_PATH
        
        db = FlightDB.load(AIRLINE_DB_PATH)
        toolkit = AirlineTools(db)
        
        # Get all tool functions
        tools = {}
        for name in dir(toolkit):
            if name.startswith("_"):
                continue
            attr = getattr(toolkit, name)
            if callable(attr) and hasattr(attr, "__self__"):
                if wrap_with_disruption:
                    # Wrap with disruption injection
                    # Track write operations for potential rollback
                    is_write_op = name in WRITE_OPERATIONS
                    tools[name] = wrap_tool_with_disruption(attr, name, track_for_rollback=is_write_op)
                    logger.debug(f"Wrapped tool with disruption: {name} (track={is_write_op})")
                else:
                    tools[name] = attr
        
        logger.info(f"Loaded {len(tools)} tools (disruption_wrap={wrap_with_disruption})")
        return tools, toolkit
        
    except ImportError as e:
        logger.error(f"Failed to import τ²-bench: {e}")
        logger.info("Make sure τ²-bench is installed: pip install -e .")
        return {}, None


def load_airline_policy() -> str:
    """Load the airline policy document."""
    policy_path = DEFAULT_DATA_DIR / "policy.md"
    if policy_path.exists():
        return policy_path.read_text()
    return "No policy found."


def run_benchmark(
    task_ids: List[str],
    frameworks: List[str],
    disruption_scenario: Optional[str] = None,
    trials: int = 1,
    model: str = DEFAULT_MODEL,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run benchmark comparing frameworks on specified tasks.

    Args:
        task_ids: List of task IDs to run.
        frameworks: List of frameworks to compare.
        disruption_scenario: Optional disruption scenario name.
        trials: Number of trials per task.
        model: LLM model to use.
        output_dir: Directory for output files.
        dry_run: If True, just print what would be done.

    Returns:
        Benchmark results dictionary.
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tasks
    tasks_file = DEFAULT_DATA_DIR / "tasks.json"
    tasks = load_tau2_tasks(tasks_file, task_ids)
    
    if not tasks:
        logger.error(f"No tasks found for IDs: {task_ids}")
        return {"error": "No tasks found"}
    
    logger.info(f"Loaded {len(tasks)} tasks: {[t.task_id for t in tasks]}")
    
    # Load tools with disruption wrapping
    wrap_disruption = disruption_scenario is not None
    tools, toolkit = load_airline_tools(wrap_with_disruption=wrap_disruption)
    if not tools:
        return {"error": "Failed to load airline tools"}
    
    policy = load_airline_policy()
    logger.info(f"Loaded {len(tools)} tools and policy ({len(policy)} chars)")
    
    # Configure disruption scenario
    disruption_config = None
    if disruption_scenario:
        if disruption_scenario in AIRLINE_DISRUPTION_SCENARIOS:
            disruption_config = [AIRLINE_DISRUPTION_SCENARIOS[disruption_scenario]]
            logger.info(f"Using disruption scenario: {disruption_scenario}")
        else:
            logger.warning(f"Unknown disruption scenario: {disruption_scenario}")
    
    if dry_run:
        logger.info("DRY RUN - Would execute:")
        logger.info(f"  Tasks: {[t.task_id for t in tasks]}")
        logger.info(f"  Frameworks: {frameworks}")
        logger.info(f"  Trials: {trials}")
        logger.info(f"  Disruption: {disruption_scenario}")
        return {"dry_run": True}
    
    # Run benchmark
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "task_ids": task_ids,
            "frameworks": frameworks,
            "disruption_scenario": disruption_scenario,
            "trials": trials,
            "model": model,
        },
        "results": [],
    }
    
    for task in tasks:
        # Add disruption scenarios to task
        if disruption_config:
            task = add_disruption_scenarios(task, disruption_config)
        
        for framework in frameworks:
            runner = get_runner(framework, model)
            
            for trial in range(trials):
                logger.info(
                    f"Running: task={task.task_id}, framework={framework}, "
                    f"trial={trial + 1}/{trials}"
                )
                
                try:
                    result = runner.execute(
                        task=task,
                        tools=tools,
                        policy=policy,
                        disruption_scenarios=disruption_config,
                    )
                    
                    results["results"].append({
                        "task_id": task.task_id,
                        "framework": framework,
                        "trial": trial + 1,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "tool_calls": len(result.tool_calls),
                        "disruptions_triggered": len(result.disruptions_triggered),
                        "compensation_actions": len(result.compensation_actions),
                        "rollback_success": result.rollback_success,
                        "error": result.error,
                    })
                    
                    logger.info(
                        f"  Result: success={result.success}, "
                        f"tools={len(result.tool_calls)}, "
                        f"time={result.execution_time:.2f}s"
                    )
                    
                except Exception as e:
                    logger.error(f"  Failed: {e}")
                    results["results"].append({
                        "task_id": task.task_id,
                        "framework": framework,
                        "trial": trial + 1,
                        "success": False,
                        "error": str(e),
                    })
    
    # Save results
    output_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Group by framework
    framework_stats = {}
    for r in results["results"]:
        fw = r["framework"]
        if fw not in framework_stats:
            framework_stats[fw] = {"total": 0, "success": 0, "time": 0}
        
        framework_stats[fw]["total"] += 1
        if r.get("success"):
            framework_stats[fw]["success"] += 1
        framework_stats[fw]["time"] += r.get("execution_time", 0)
    
    print(f"\n{'Framework':<15} {'Success Rate':<15} {'Avg Time':<15}")
    print("-" * 45)
    
    for fw, stats in framework_stats.items():
        success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_time = stats["time"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{fw:<15} {success_rate:.1f}%{'':<10} {avg_time:.2f}s")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run τ²-bench framework comparison benchmark"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task ID(s) to run (comma-separated)",
        default="0",
    )
    parser.add_argument(
        "--frameworks", "-f",
        type=str,
        help="Frameworks to compare (comma-separated)",
        default="langgraph,rac,sagallm",
    )
    parser.add_argument(
        "--inject-disruption", "-d",
        type=str,
        help=f"Disruption scenario to inject. Options: {list(AIRLINE_DISRUPTION_SCENARIOS.keys())}",
        default=None,
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        help="Number of trials per task",
        default=1,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="LLM model to use",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )
    
    args = parser.parse_args()
    
    task_ids = [t.strip() for t in args.task.split(",")]
    frameworks = [f.strip() for f in args.frameworks.split(",")]
    output_dir = Path(args.output) if args.output else None
    
    run_benchmark(
        task_ids=task_ids,
        frameworks=frameworks,
        disruption_scenario=args.inject_disruption,
        trials=args.trials,
        model=args.model,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
