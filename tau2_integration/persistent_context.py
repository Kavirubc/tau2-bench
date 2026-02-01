
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from datetime import datetime

@dataclass
class PersistentExecutionContext:
    """
    Maintains execution context across multiple replan attempts.
    Based on SagaLLM Paper Section 5.3.3 strategies.
    
    Tracks:
    - Cumulative failures (what failed and why)
    - Permanent failures (what can never work)
    - Completed work (what doesn't need re-doing)
    """
    replan_count: int = 0
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list) # Specific parameters that failed
    completed_jobs: Set[str] = field(default_factory=set)
    permanent_failures: Set[str] = field(default_factory=set)
    original_plan: List[Dict[str, Any]] = field(default_factory=list)

    def add_failure(self, reason: str, failed_tool: str, tool_args: Dict[str, Any], attempt: int):
        """Record a failure."""
        self.failure_history.append({
            "attempt": attempt,
            "reason": reason,
            "tool": failed_tool,
            "args": tool_args,
            "timestamp": datetime.now().isoformat()
        })
        
        # Track effective 'permanent' failures (simplified logic)
        # If we failed with specific args, mark them as 'bad'
        failure_signature = f"{failed_tool}:{self._hash_args(tool_args)}"
        self.failed_attempts.append({
            "signature": failure_signature,
            "tool": failed_tool,
            "args": tool_args,
            "reason": reason
        })
        
    def _hash_args(self, args: Dict) -> str:
        """Create a simple hash string for args to identify uniqueness."""
        return str(sorted([(k, str(v)) for k, v in args.items()]))
    
    def get_failure_summary(self) -> str:
        """Generate a summary of failures for the LLM."""
        if not self.failure_history:
            return "No failures recorded."
            
        summary = ["FAILURE HISTORY (Cumulative):"]
        seen_reasons = set()
        
        for fail in self.failure_history:
            if fail['reason'] not in seen_reasons:
                summary.append(f"- Attempt {fail['attempt']}: Tool '{fail['tool']}' failed. Reason: {fail['reason']}")
                # Add specific bad args (e.g. Broken Machine)
                args_str = ", ".join([f"{k}={v}" for k,v in fail['args'].items() if k in ['machine_id', 'vehicle_id', 'job_id']])
                if args_str:
                    summary.append(f"  Failed Parameters: {args_str}")
                seen_reasons.add(fail['reason'])
                
        return "\n".join(summary)
