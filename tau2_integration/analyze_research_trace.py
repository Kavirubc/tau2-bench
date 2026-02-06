#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

def analyze_trace(trace_path: str):
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    
    print("=" * 80)
    print(f"🔬 RESEARCH TRACE ANALYSIS: {trace['framework'].upper()} - Task {trace['task_id']}")
    print(f"Status: {trace['status'].upper()}")
    print(f"Total Duration: {trace['duration']:.2f}s")
    print(f"Total Tokens: {trace['total_tokens']['total']} (In: {trace['total_tokens']['input']}, Out: {trace['total_tokens']['output']})")
    print("=" * 80)
    
    # Phase analysis
    phases = {}
    
    for step in trace['steps']:
        metadata = step.get('metadata', {})
        phase = metadata.get('phase', 'unknown')
        iteration = metadata.get('iteration', 1)
        
        phase_key = f"Iter {iteration} - {phase}"
        
        if phase_key not in phases:
            phases[phase_key] = {
                "duration": 0.0,
                "tokens": {"input": 0, "output": 0, "total": 0},
                "steps": 0,
                "events": []
            }
        
        phases[phase_key]["duration"] += step.get('duration', 0.0)
        tokens = step.get('tokens', {})
        phases[phase_key]["tokens"]["input"] += tokens.get("input", 0)
        phases[phase_key]["tokens"]["output"] += tokens.get("output", 0)
        phases[phase_key]["tokens"]["total"] += tokens.get("total", 0)
        phases[phase_key]["steps"] += 1
        
        if step.get('type') == 'event':
            phases[phase_key]["events"].append(f"{step['name']} ({step['input']})")
        elif step.get('error'):
             phases[phase_key]["events"].append(f"ERROR in {step['name']}: {step['error']}")

    print("\nPHASE BREAKDOWN:")
    print(f"{'Phase':<30} | {'Duration':<10} | {'Tokens':<10} | {'Steps':<5}")
    print("-" * 65)
    for name, data in phases.items():
        print(f"{name:<30} | {data['duration']:<10.2f}s | {data['tokens']['total']:<10} | {data['steps']:<5}")
        for event in data['events']:
            print(f"  - [EVENT] {event}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a research trace JSON.")
    parser.add_argument("trace_file", help="Path to the trace JSON file")
    args = parser.parse_args()
    
    analyze_trace(args.trace_file)
