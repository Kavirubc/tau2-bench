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
from tau2_integration.runners.prompt_engineer_langgraph_runner import PromptEngineerLangGraphRunner
from tau2_integration.evaluation import evaluate_run, compare_frameworks, print_comparison_table, EvaluationMetrics
from tau2_integration.domain_registry import DomainRegistry
from tau2_integration.disruption_engine import AIRLINE_DISRUPTION_SCENARIOS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_full_benchmark")

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"
DEFAULT_TRACE_DIR = Path(__file__).parent / "results" / "traces"

# Default model and domain
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_DOMAIN = "airline"

# Task-specific disruption mapping
# Maps Task ID -> List of disruption scenario names from disruption_engine.py
TASK_DISRUPTION_MAP = {
    "3": ["user_lookup_transient"],                          # Transient: agent must retry get_user_details
    "5": ["user_lookup_transient"],                          # Transient: agent must retry
    "7": ["user_lookup_transient"],                          # Transient: agent must retry
    "8": ["flight_unavailable"],                             # Persistent: book_reservation always fails
    "9": ["reservation_update_conflict"],                    # Persistent: update_reservation_flights always fails
}


def load_domain_tools(domain: str):
    """Load τ²-bench tools for a specific domain."""
    try:
        tools = DomainRegistry.load_domain_tools(domain)
        logger.info(f"Loaded {len(tools)} tools for {domain} domain")
        return tools
    except Exception as e:
        logger.error(f"Failed to load {domain} domain tools: {e}")
        logger.info("Make sure τ²-bench is installed: pip install -e .")
        return {}


def load_domain_policy(domain: str) -> str:
    """Load the policy document for a specific domain."""
    try:
        policy = DomainRegistry.load_domain_policy(domain)
        logger.info(f"Loaded policy for {domain} domain ({len(policy)} chars)")
        return policy
    except Exception as e:
        logger.error(f"Failed to load {domain} domain policy: {e}")
        return f"No policy found for {domain} domain."


def run_single_benchmark(
    framework_name: str,
    runner,
    task,
    tools,
    policy,
    trial: int,
    trace_dir: Path,
    disruption_scenarios: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single benchmark trial."""
    logger.info(f"  [{framework_name}] Trial {trial}: {task.task_id}")
    
    start_time = time.time()
    
    try:
        result = runner.execute(
            task=task,
            tools=tools,
            policy=policy,
            disruption_scenarios=disruption_scenarios,
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
        
        disruptions_fired = len(result.disruptions_triggered) if result.disruptions_triggered else 0
        if disruptions_fired:
            logger.info(f"    Disruptions fired: {disruptions_fired}")

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
            "disruptions_fired": disruptions_fired,
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
    domain: str,
    task_ids: List[str],
    frameworks: List[str],
    trials: int = 1,
    model: str = DEFAULT_MODEL,
    output_dir: Path = None,
    disruptions: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark across all frameworks for a specific domain.
    
    Args:
        domain: Domain name (airline, retail, telecom, mock)
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
    
    trace_dir = DEFAULT_TRACE_DIR / domain
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tasks for the specified domain
    tasks = load_tau2_tasks(domain, task_ids)
    
    if not tasks:
        logger.error(f"No tasks found for domain '{domain}' with IDs: {task_ids}")
        return {"error": "No tasks found"}
    
    logger.info(f"Loaded {len(tasks)} tasks from {domain} domain: {[t.task_id for t in tasks]}")
    
    # Load tools and policy for the domain
    tools = load_domain_tools(domain)
    if not tools:
        return {"error": f"Failed to load {domain} domain tools"}
    
    policy = load_domain_policy(domain)
    
    # Initialize runners
    runners = {}
    if "langgraph" in frameworks:
        runners["langgraph"] = LangGraphRunner(model=model)
    if "rac" in frameworks:
        runners["rac"] = RACRunner(model=model)
    if "sagallm" in frameworks:
        runners["sagallm"] = SagaLLMRunner(model=model)
    if "prompt_engineer_langgraph" in frameworks:
        runners["prompt_engineer_langgraph"] = PromptEngineerLangGraphRunner(model=model)
    
    logger.info(f"Initialized {len(runners)} runners: {list(runners.keys())}")
    
    # Run benchmarks
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "domain": domain,
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
    print(f"Domain: {domain}")
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
                
                # Determine disruption scenarios for this task
                # Determine disruption scenarios for this task
                scenarios = None
                if disruptions and task.task_id in TASK_DISRUPTION_MAP:
                    scenario_names = TASK_DISRUPTION_MAP[task.task_id]
                    scenarios = [AIRLINE_DISRUPTION_SCENARIOS[name] for name in scenario_names if name in AIRLINE_DISRUPTION_SCENARIOS]
                    print(f"  ⚡ Injecting disruptions: {scenario_names} -> {len(scenarios)} scenarios")

                result = run_single_benchmark(
                    framework_name=framework_name,
                    runner=runner,
                    task=task,
                    tools=tools,
                    policy=policy,
                    trial=trial,
                    trace_dir=trace_dir,
                    disruption_scenarios=scenarios
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
    output_file = output_dir / f"benchmark_{domain}_{timestamp}.json"
    
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
        "--domain", "-d",
        type=str,
        help="Domain to benchmark (airline, retail, telecom, mock)",
        default=DEFAULT_DOMAIN,
    )
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        help="Task IDs to run (comma-separated)",
        default="0,1,3,5,8,15,20,28,32,39",
    )
    parser.add_argument(
        "--frameworks", "-f",
        type=str,
        help="Frameworks to test (comma-separated)",
        default="langgraph,rac,sagallm,prompt_engineer_langgraph",
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
        "--disruptions", 
        action="store_true", 
        help="Enable disruption scenarios for compensation testing"
    )
    
    args = parser.parse_args()
    
    
    # Load scenarios if enabled
    disruption_config = None
    if args.disruptions:
        logger.warning("⚠️  DISRUPTION MODE ENABLED - Agents will face failures! ⚠️")
        from tau2_integration.disruption_engine import AIRLINE_DISRUPTION_SCENARIOS
    
    # Validate domain
    valid_domains = DomainRegistry.list_domains()
    if args.domain not in valid_domains:
        logger.error(f"Invalid domain: {args.domain}")
        logger.error(f"Valid domains: {valid_domains}")
        return
    
    task_ids = [t.strip() for t in args.tasks.split(",")]
    frameworks = [f.strip() for f in args.frameworks.split(",")]
    output_dir = Path(args.output) if args.output else None
    
    run_full_benchmark(
        domain=args.domain,
        task_ids=task_ids,
        frameworks=frameworks,
        trials=args.trials,
        model=args.model,
        output_dir=output_dir,
        disruptions=args.disruptions,
    )


if __name__ == "__main__":
    main()
