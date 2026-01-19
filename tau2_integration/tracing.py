import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TraceStep:
    """A single step in the execution trace."""
    step_id: str
    step_type: str  # "tool", "llm", "plan", "compensation"
    name: str
    input: Any
    output: Any
    start_time: float
    end_time: float
    duration: float
    tokens: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class ExecutionTrace:
    """Complete execution trace for a task."""
    trace_id: str
    task_id: str
    framework: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, success, failed
    steps: List[TraceStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def total_tokens(self) -> Dict[str, int]:
        totals = {"input": 0, "output": 0, "total": 0}
        for step in self.steps:
            for k, v in step.tokens.items():
                totals[k] = totals.get(k, 0) + v
                if k not in ["input", "output", "total"]:
                    # Handle unknown keys by adding to total if sensible? 
                    # For now just sum known keys.
                    pass
        return totals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "framework": self.framework,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.total_duration,
            "status": self.status,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
            "steps": [
                {
                    "step_id": s.step_id,
                    "type": s.step_type,
                    "name": s.name,
                    "input": str(s.input)[:1000],  # Truncate for storage
                    "output": str(s.output)[:1000],
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration": s.duration,
                    "tokens": s.tokens,
                    "error": s.error,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ]
        }

class TraceRecorder:
    """Recorder for capturing execution traces."""
    
    def __init__(self, task_id: str, framework: str):
        self.trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            task_id=task_id,
            framework=framework,
            start_time=time.time()
        )
        self.active_steps: Dict[str, TraceStep] = {}
        
    def start_step(self, step_type: str, name: str, input_data: Any, metadata: Dict[str, Any] = None) -> str:
        """Start recording a step. Returns step_id."""
        step_id = str(uuid.uuid4())
        step = TraceStep(
            step_id=step_id,
            step_type=step_type,
            name=name,
            input=input_data,
            output=None,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            metadata=metadata or {}
        )
        self.active_steps[step_id] = step
        return step_id
        
    def end_step(self, step_id: str, output: Any, error: Optional[str] = None, tokens: Dict[str, int] = None):
        """End recording a step."""
        if step_id in self.active_steps:
            step = self.active_steps.pop(step_id)
            step.end_time = time.time()
            step.duration = step.end_time - step.start_time
            step.output = output
            step.error = error
            if tokens:
                step.tokens = tokens
            self.trace.steps.append(step)
            
    def finish(self, status: str = "success", metadata: Dict[str, Any] = None):
        """Finish the trace."""
        self.trace.end_time = time.time()
        self.trace.status = status
        if metadata:
            self.trace.metadata.update(metadata)
            
    def save(self, directory: str):
        """Save trace to JSON file."""
        import os
        os.makedirs(directory, exist_ok=True)
        filename = f"trace_{self.trace.framework}_{self.trace.task_id}_{int(self.trace.start_time)}.json"
        path = os.path.join(directory, filename)
        
        with open(path, "w") as f:
            json.dump(self.trace.to_dict(), f, indent=2)
        return path
