# Compensation Framework Adaptations for τ²-bench

This fork of τ²-bench (`tau2-bench`) adapts the benchmark to evaluate and compare advanced agentic compensation frameworks, specifically **React Agent Compensation (RAC)** and **SagaLLM**, alongside standard LangGraph implementations.

## 🚀 Key Adaptations

We have integrated custom runners and benchmarking tools to test how different agent architectures handle failures and perform compensation (rollback/recovery) in airline booking scenarios.

### 1. Framework Integrations

We implemented three distinct runners in `tau2_integration/runners/`:

*   **RAC Runner (`rac_runner.py`)**: Integrates the [React Agent Compensation](https://github.com/React-Agent-Compensation) library.
    *   **Features**: Automatic tool wrapping, failure detection, and compensation mapping.
    *   **Mechanism**: Uses a `CompensationSchema` to verify tool outputs and triggers reverse actions (e.g., `cancel_hotel`) when downstream steps fail.
*   **SagaLLM Runner (`saga_runner.py`)**: Implements the 3-phase Saga pattern (Planning → Execution → Compensation).
    *   **Features**: Explicit planning step, execution monitoring, and backward recovery.
    *   **Mechanism**: Generates a JSON plan first. If execution fails, it reverses previous successful steps.
*   **LangGraph Runner (`langgraph_runner.py`)**: A vanilla React agent baseline using LangGraph.
    *   **Traceability**: Enhanced with specific Langsmith tags to serve as a control group.

### 2. Disruption Engine

A custom `DisruptionEngine` (`disruption_engine.py`) was ported to inject deterministic failures into tool calls, simulating real-world issues like:
*   Payment gateway timeouts (transient)
*   Database constraints (permanent)
*   Flight unavailability

### 3. Benchmarking & Demos

We added specialized scripts to visualize and verify compensation behavior:

*   **`compensation_demo.py`**: A standalone script that runs a single RAC agent through a "Book → Fail Payment (retry) → Fail Finalize → compensate" flow. Ideal for seeing the mechanism in isolation.
*   **`compensation_benchmark.py`**: A comparative benchmark that runs the same scenario on both RAC and SagaLLM side-by-side, producing timing and success metrics.

## 📊 Langsmith Tracing

All integrations are instrumented with **LangChain Tracing V2** for deep observability.

*   **Tags**: `framework:RAC`, `framework:SagaLLM`, `framework:LangGraph-Vanilla`.
*   **Metadata**: Includes task IDs, model versions, and failure scenarios.
*   **Grouping**: Benchmarks create parent runs (e.g., `🔷 RAC: Full Compensation Flow`) to group all LLM and tool calls.

## 🛠️ How to Run

### Setup
Ensure you have the required environment variables:
```bash
export GOOGLE_API_KEY="your_key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your_langsmith_key"
```

### Run Comparison Benchmark
To see both frameworks handle a rollback scenario:
```bash
python3 tau2_integration/compensation_benchmark.py
```

### Run Full Benchmark
To run the standard τ²-bench tasks with a specific framework:
```bash
# Run Task 8 with RAC
python3 tau2_integration/benchmark.py --task 8 --frameworks rac

# Run Task 8 with all frameworks
python3 tau2_integration/benchmark.py --task 8 --frameworks langgraph,rac,sagallm
```

## 📂 New Directory Structure

*   `tau2_integration/`: Core logic for the adaptations.
    *   `runners/`: Framework-specific runner implementations.
    *   `benchmark.py`: Main entry point for τ²-bench tasks.
    *   `compensation*.py`: Specialized compensation verification scripts.
    *   `disruption_engine.py`: Failure injection logic.
