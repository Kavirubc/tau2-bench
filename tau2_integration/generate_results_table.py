#!/usr/bin/env python3
"""
Generate a comprehensive results table from benchmark JSON files.

Usage:
    python3 tau2_integration/generate_results_table.py <benchmark_file.json>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def generate_table(benchmark_file: Path) -> str:
    """Generate a markdown table from benchmark results."""
    
    with open(benchmark_file) as f:
        data = json.load(f)
    
    results = data.get('results', [])
    config = data.get('config', {})
    
    # Group results by task and framework
    task_framework_data = {}
    
    for r in results:
        task_id = r.get('task_id', 'N/A')
        framework = r.get('framework', 'unknown')
        
        if task_id not in task_framework_data:
            task_framework_data[task_id] = {}
        
        task_framework_data[task_id][framework] = {
            'success': '✓' if r.get('success') else '✗',
            'tokens': r.get('total_tokens', 0),
            'time': r.get('execution_time', 0),
            'llm_calls': r.get('llm_calls', 0),
            'tool_calls': r.get('tool_calls', 0),
        }
    
    # Framework order
    framework_order = ['langgraph', 'prompt_engineer_langgraph', 'rac', 'sagallm']
    framework_names = {
        'langgraph': 'LangGraph',
        'prompt_engineer_langgraph': 'LG (PE)',
        'rac': 'RAC',
        'sagallm': 'SagaLLM',
    }
    
    # Build table
    lines = []
    lines.append("# Benchmark Results Table")
    lines.append(f"\n**File**: `{benchmark_file.name}`")
    lines.append(f"**Model**: {config.get('model', 'N/A')}")
    lines.append(f"**Domain**: {config.get('domain', 'N/A')}")
    lines.append(f"**Timestamp**: {data.get('timestamp', 'N/A')}")
    lines.append("")
    
    # Header
    lines.append("| Task | Framework | Success | Tokens | Time (s) | LLM Calls | Tool Calls |")
    lines.append("|------|-----------|---------|--------|----------|-----------|------------|")
    
    # Sort tasks numerically
    sorted_tasks = sorted(task_framework_data.keys(), key=lambda x: int(x) if x.isdigit() else 999)
    
    for task_id in sorted_tasks:
        frameworks_data = task_framework_data[task_id]
        
        for fw in framework_order:
            if fw in frameworks_data:
                data = frameworks_data[fw]
                fw_name = framework_names.get(fw, fw)
                
                lines.append(
                    f"| {task_id} | {fw_name} | {data['success']} | "
                    f"{data['tokens']:,} | {data['time']:.2f} | "
                    f"{data['llm_calls']} | {data['tool_calls']} |"
                )
    
    # Summary statistics
    lines.append("\n## Summary Statistics\n")
    lines.append("| Framework | Success Rate | Avg Tokens | Avg Time (s) | Avg LLM Calls | Avg Tool Calls |")
    lines.append("|-----------|--------------|------------|--------------|---------------|----------------|")
    
    for fw in framework_order:
        fw_results = [r for r in results if r.get('framework') == fw]
        if not fw_results:
            continue
        
        total = len(fw_results)
        successes = sum(1 for r in fw_results if r.get('success'))
        avg_tokens = sum(r.get('total_tokens', 0) for r in fw_results) / total
        avg_time = sum(r.get('execution_time', 0) for r in fw_results) / total
        avg_llm = sum(r.get('llm_calls', 0) for r in fw_results) / total
        avg_tool = sum(r.get('tool_calls', 0) for r in fw_results) / total
        
        fw_name = framework_names.get(fw, fw)
        lines.append(
            f"| {fw_name} | {successes}/{total} ({100*successes/total:.1f}%) | "
            f"{avg_tokens:,.1f} | {avg_time:.2f} | {avg_llm:.1f} | {avg_tool:.1f} |"
        )
    
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_results_table.py <benchmark_file.json>")
        sys.exit(1)
    
    benchmark_file = Path(sys.argv[1])
    
    if not benchmark_file.exists():
        print(f"Error: File not found: {benchmark_file}")
        sys.exit(1)
    
    table = generate_table(benchmark_file)
    print(table)
    
    # Also save to markdown file
    output_file = benchmark_file.parent / f"{benchmark_file.stem}_table.md"
    with open(output_file, 'w') as f:
        f.write(table)
    
    print(f"\n\n✅ Table saved to: {output_file}")


if __name__ == "__main__":
    main()
