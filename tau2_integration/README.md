# τ²-bench Framework Integration

Integration module for comparing SagaLLM, React-Agent-Compensation (RAC), and vanilla LangGraph frameworks on τ²-bench's airline domain.

## Overview

This module enables compensation robustness testing by:
1. Injecting artificial failures during tool execution (via `DisruptionEngine`)
2. Running multiple frameworks on the same tasks
3. Comparing their compensation/rollback behavior

## Quick Start

```bash
# Install dependencies
cd /path/to/tau2-bench
pip install -e .

# Run benchmark on task 0 with all frameworks
python tau2_integration/benchmark.py --task 0 --frameworks langgraph,rac,sagallm

# Run with disruption injection
python tau2_integration/benchmark.py --task 8 --inject-disruption payment_transient
```

## Components

| File | Purpose |
|------|---------|
| `disruption_engine.py` | Failure injection (transient, persistent) |
| `wrapped_tools.py` | Tool wrappers with disruption hooks |
| `task_adapter.py` | Convert τ²-bench tasks to unified format |
| `runners/` | Framework-specific runners |
| `benchmark.py` | Main benchmark CLI |
| `evaluation.py` | Metrics calculation |

## Frameworks

- **LangGraph** - Vanilla ReAct (baseline, no compensation)
- **RAC** - React-Agent-Compensation with auto-rollback
- **SagaLLM** - 3-phase plan→execute→compensate

## Disruption Scenarios

- `payment_transient` - Payment fails 2x then succeeds (tests retry)
- `flight_unavailable` - Flight becomes unavailable (tests replanning)
- `seat_shortage` - Insufficient seats (tests recovery)
- `system_error` - Transient API error (tests resilience)
