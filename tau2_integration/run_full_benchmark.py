#!/usr/bin/env python3
"""
Comprehensive benchmark runner for SagaLLM, RAC, and vanilla LangGraph.

Runs all three frameworks with enhanced tracing and in-code tracing,
collecting detailed metrics including:
- LLM call counts
- Token usage (input/output/total)
- Execution time
- Tool calls
- Compensation actions
- Trace files

Usage:
    source .venv/bin/activate
    python3 tau2_integration/run_full_benchmark.py --tasks 5,6,8,9 --trials 1
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from tau2_integration.task_adapter import load_tau2_tasks
from tau2_integration.runners.langgraph_runner import LangGraphRunner
from tau2_integration.runners.rac_runner import RACRunner
from tau2_integration.runners.saga_runner import SagaLLMRunner
from tau2_integration.evaluation import evaluate_run, compare_frameworks, print_comparison_table, EvaluationMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_full_benchmark")

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "tau2" / "domains" / "airline"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"
DEFAULT_TRACE_DIR = Path(__file__).parent / "results" / "traces"

# Default model
DEFAULT_MODEL = "gemini-2.0-flash"


def load_airline_tools():
    """Load τ²-bench airline tools without disruption wrapping."""
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
                tools[name] = attr
        
        logger.info(f"Loaded {len(tools)} airline tools")
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


def run_single_benchmark(
    framework_name: str,
    runner,
    task,
    tools,
    policy,
    trial: int,
    trace_dir: Path,
) -> Dict[str, Any]:
    """Run a single benchmark trial."""
    logger.info(f"  [{framework_name}] Trial {trial}: {task.task_id}")
    
    start_time = time.time()
    
    try:
        result = runner.execute(
            task=task,
            tools=tools,
            policy=policy,
            disruption_scenarios=None,
        )
        
        execution_time = time.time() - start_time
        
        # Save trace if available
        trace_file = None
        if result.trace:
            trace_file = trace_dir / f"trace_{framework_name}_{task.task_id}_trial{trial}_{int(start_time)}.json"
            with open(trace_file, "w") as f:
                json.dump(result.trace, f, indent=2)
            logger.info(f"    Saved trace: {trace_file.name}")
        
        # Extract metrics from trace
        llm_calls = 0
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        
        if result.trace and "steps" in result.trace:
            llm_steps = [s for s in result.trace["steps"] if s.get("type") == "llm"]
            llm_calls = len(llm_steps)
            
            for step in result.trace["steps"]:
                tokens = step.get("tokens", {})
                input_tokens += tokens.get("input", 0)
                output_tokens += tokens.get("output", 0)
                total_tokens += tokens.get("total", 0)
        
        return {
            "framework": framework_name,
            "task_id": task.task_id,
            "trial": trial,
            "success": result.success,
            "execution_time": execution_time,
            "tool_calls": len(result.tool_calls),
            "llm_calls": llm_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "compensation_actions": len(result.compensation_actions),
            "rollback_success": result.rollback_success,
            "trace_file": str(trace_file) if trace_file else None,
            "error": result.error,
        }
        
    except Exception as e:
        logger.error(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "framework": framework_name,
            "task_id": task.task_id,
            "trial": trial,
            "success": False,
            "error": str(e),
        }


def run_full_benchmark(
    task_ids: List[str],
    frameworks: List[str],
    trials: int = 1,
    model: str = DEFAULT_MODEL,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark across all frameworks.
    
    Args:
        task_ids: List of task IDs to run
        frameworks: List of frameworks to test
        trials: Number of trials per task
        model: LLM model to use
        output_dir: Output directory for results
    
    Returns:
        Complete benchmark results
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trace_dir = DEFAULT_TRACE_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tasks
    tasks_file = DEFAULT_DATA_DIR / "tasks.json"
    tasks = load_tau2_tasks(tasks_file, task_ids)
    
    if not tasks:
        logger.error(f"No tasks found for IDs: {task_ids}")
        return {"error": "No tasks found"}
    
    logger.info(f"Loaded {len(tasks)} tasks: {[t.task_id for t in tasks]}")
    
    # Load tools and policy
    tools, toolkit = load_airline_tools()
    if not tools:
        return {"error": "Failed to load airline tools"}
    
    policy = load_airline_policy()
    logger.info(f"Loaded policy ({len(policy)} chars)")
    
    # Initialize runners
    runners = {}
    if "langgraph" in frameworks:
        runners["langgraph"] = LangGraphRunner(model=model)
    if "rac" in frameworks:
        runners["rac"] = RACRunner(model=model)
    if "sagallm" in frameworks:
        runners["sagallm"] = SagaLLMRunner(model=model)
    
    logger.info(f"Initialized {len(runners)} runners: {list(runners.keys())}")
    
    # Run benchmarks
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "task_ids": task_ids,
            "frameworks": frameworks,
            "trials": trials,
            "model": model,
            "tracing_enabled": os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        },
        "results": [],
    }
    
    print("\n" + "=" * 80)
    print("🚀 RUNNING COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print(f"Tasks: {task_ids}")
    print(f"Frameworks: {frameworks}")
    print(f"Trials per task: {trials}")
    print(f"Model: {model}")
    print(f"LangSmith Tracing: {results['config']['tracing_enabled']}")
    print("=" * 80 + "\n")
    
    total_runs = len(tasks) * len(frameworks) * trials
    current_run = 0
    
    for task in tasks:
        logger.info(f"\n📋 Task: {task.task_id} - {task.description[:60]}...")
        
        for framework_name, runner in runners.items():
            for trial in range(1, trials + 1):
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running {framework_name} on {task.task_id} (trial {trial}/{trials})")
                
                result = run_single_benchmark(
                    framework_name=framework_name,
                    runner=runner,
                    task=task,
                    tools=tools,
                    policy=policy,
                    trial=trial,
                    trace_dir=trace_dir,
                )
                
                results["results"].append(result)
                
                # Print quick summary
                if result.get("success"):
                    print(f"  ✅ Success | Time: {result.get('execution_time', 0):.2f}s | "
                          f"LLM calls: {result.get('llm_calls', 0)} | "
                          f"Tokens: {result.get('total_tokens', 0)}")
                else:
                    print(f"  ❌ Failed | Error: {result.get('error', 'Unknown')[:50]}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_full_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive summary of benchmark results."""
    print("\n" + "=" * 100)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Group by framework
    framework_stats = {}
    for r in results["results"]:
        fw = r["framework"]
        if fw not in framework_stats:
            framework_stats[fw] = {
                "total": 0,
                "success": 0,
                "time": 0,
                "llm_calls": 0,
                "tokens": 0,
                "tool_calls": 0,
            }
        
        framework_stats[fw]["total"] += 1
        if r.get("success"):
            framework_stats[fw]["success"] += 1
        framework_stats[fw]["time"] += r.get("execution_time", 0)
        framework_stats[fw]["llm_calls"] += r.get("llm_calls", 0)
        framework_stats[fw]["tokens"] += r.get("total_tokens", 0)
        framework_stats[fw]["tool_calls"] += r.get("tool_calls", 0)
    
    # Print table
    print(f"\n{'Framework':<15} {'Success':<12} {'Avg Time':<12} {'Avg LLM':<12} {'Avg Tokens':<15} {'Avg Tools':<12}")
    print("-" * 100)
    
    for fw, stats in framework_stats.items():
        n = stats["total"]
        success_rate = (stats["success"] / n * 100) if n > 0 else 0
        avg_time = stats["time"] / n if n > 0 else 0
        avg_llm = stats["llm_calls"] / n if n > 0 else 0
        avg_tokens = stats["tokens"] / n if n > 0 else 0
        avg_tools = stats["tool_calls"] / n if n > 0 else 0
        
        print(f"{fw:<15} {success_rate:>5.1f}%      {avg_time:>6.2f}s      "
              f"{avg_llm:>6.1f}       {avg_tokens:>9.0f}       {avg_tools:>6.1f}")
    
    print("=" * 100)
    
    # Detailed breakdown by task
    print("\n📋 DETAILED BREAKDOWN BY TASK")
    print("-" * 100)
    
    tasks = sorted(set(r["task_id"] for r in results["results"]))
    for task_id in tasks:
        print(f"\nTask {task_id}:")
        task_results = [r for r in results["results"] if r["task_id"] == task_id]
        
        for fw in sorted(set(r["framework"] for r in task_results)):
            fw_results = [r for r in task_results if r["framework"] == fw]
            successes = sum(1 for r in fw_results if r.get("success"))
            total = len(fw_results)
            avg_tokens = sum(r.get("total_tokens", 0) for r in fw_results) / total if total > 0 else 0
            
            print(f"  {fw:<15} {successes}/{total} success | Avg tokens: {avg_tokens:.0f}")
    
    print("\n" + "=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark with enhanced tracing"
    )
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        help="Task IDs to run (comma-separated)",
        default="5,6,8,9",
    )
    parser.add_argument(
        "--frameworks", "-f",
        type=str,
        help="Frameworks to test (comma-separated)",
        default="langgraph,rac,sagallm",
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
    
    args = parser.parse_args()
    
    task_ids = [t.strip() for t in args.tasks.split(",")]
    frameworks = [f.strip() for f in args.frameworks.split(",")]
    output_dir = Path(args.output) if args.output else None
    
    run_full_benchmark(
        task_ids=task_ids,
        frameworks=frameworks,
        trials=args.trials,
        model=args.model,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
