#!/usr/bin/env python3
"""
Compensation Benchmark for RAC vs SagaLLM.

Runs the same failure scenario on both frameworks to compare
their compensation/rollback behavior.

Features proper Langsmith trace organization:
- Parent run for each framework
- Child runs for phases (Planning, Execution, Compensation)
- Clear naming and metadata

Usage:
    source .venv/bin/activate
    python3 tau2_integration/compensation_benchmark.py
"""

import os
import sys
import time
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("compensation_benchmark")

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable Langsmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    framework: str
    success: bool
    execution_time: float
    tool_calls: List[str]
    compensations_triggered: List[str]
    final_booking_status: str
    error: Optional[str] = None


@contextmanager
def langsmith_trace(name: str, run_type: str = "chain", metadata: dict = None, tags: list = None):
    """Context manager for Langsmith trace grouping."""
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree, traceable as ls_traceable
    
    @ls_traceable(name=name, run_type=run_type, metadata=metadata or {}, tags=tags or [])
    def traced_block():
        yield
    
    # Use a simpler approach - just yield with metadata set
    yield {
        "run_name": name,
        "tags": tags or [],
        "metadata": metadata or {},
    }


def create_test_tools(state: Dict) -> List:
    """Create test tools with failure injection."""
    from langchain_core.tools import StructuredTool
    
    def book_hotel(hotel_name: str, guest_name: str, nights: int) -> dict:
        """Book a hotel room for a guest."""
        state["booking_count"] += 1
        booking_id = f"HOTEL-{state['booking_count']:03d}"
        
        state["tool_calls"].append(f"book_hotel({hotel_name})")
        state["bookings"][booking_id] = {
            "hotel": hotel_name,
            "guest": guest_name,
            "status": "confirmed"
        }
        
        logger.info(f"✅ BOOK_HOTEL: {hotel_name} -> {booking_id}")
        return {
            "booking_id": booking_id,
            "hotel": hotel_name,
            "status": "confirmed",
        }
    
    def cancel_hotel(booking_id: str, reason: str = "rollback") -> dict:
        """Cancel a hotel booking (compensation action)."""
        state["tool_calls"].append(f"cancel_hotel({booking_id})")
        state["compensations"].append(f"cancel_hotel({booking_id})")
        
        if booking_id in state["bookings"]:
            state["bookings"][booking_id]["status"] = "cancelled"
        
        logger.info(f"🔄 CANCEL_HOTEL (COMPENSATION): {booking_id}")
        return {"booking_id": booking_id, "status": "cancelled"}
    
    def process_payment(amount: float, card_number: str) -> dict:
        """Process a payment. Fails first 2 times."""
        state["payment_attempts"] += 1
        state["tool_calls"].append(f"process_payment({amount})")
        
        if state["payment_attempts"] <= 2:
            logger.warning(f"❌ PAYMENT FAILED (attempt {state['payment_attempts']})")
            return {
                "status": "failed",
                "error": f"Payment timeout (attempt {state['payment_attempts']}). Retry.",
            }
        
        logger.info(f"✅ PAYMENT SUCCESS (attempt {state['payment_attempts']})")
        return {"status": "success", "transaction_id": "TXN-001"}
    
    def finalize_reservation(booking_id: str, payment_id: str) -> dict:
        """Finalize reservation. ALWAYS fails to trigger compensation."""
        state["tool_calls"].append(f"finalize_reservation({booking_id})")
        logger.error(f"❌ FINALIZE FAILED - should trigger compensation")
        return {
            "status": "failed", 
            "error": "Database error: Unable to finalize. Rollback required.",
        }
    
    tools = [
        StructuredTool.from_function(book_hotel, name="book_hotel"),
        StructuredTool.from_function(cancel_hotel, name="cancel_hotel"),
        StructuredTool.from_function(process_payment, name="process_payment"),
        StructuredTool.from_function(finalize_reservation, name="finalize_reservation"),
    ]
    
    return tools


def reset_state(state: Dict) -> None:
    """Reset state for next run."""
    state["booking_count"] = 0
    state["bookings"] = {}
    state["payment_attempts"] = 0
    state["tool_calls"] = []
    state["compensations"] = []


def run_rac_benchmark(tools: List, state: Dict) -> BenchmarkResult:
    """Run benchmark using RAC framework with proper Langsmith tracing."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    from langchain_core.tracers import LangChainTracer
    from langchain_core.callbacks import CallbackManager
    from react_agent_compensation.langchain_adaptor import (
        create_compensated_agent,
        get_compensation_middleware,
    )
    from react_agent_compensation.core import CompensationSchema, RetryPolicy
    
    reset_state(state)
    
    # Setup Langsmith tracer
    callbacks = []
    try:
        tracer = LangChainTracer(
            project_name=os.getenv("LANGSMITH_PROJECT", "compensation-benchmark")
        )
        callbacks.append(tracer)
    except Exception as e:
        logger.debug(f"Langsmith not available: {e}")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        callbacks=callbacks,
    )
    
    agent = create_compensated_agent(
        model=llm,
        tools=tools,
        compensation_mapping={
            "book_hotel": "cancel_hotel",
        },
        compensation_schemas={
            "book_hotel": CompensationSchema(
                param_mapping={"booking_id": "result.booking_id"},
                static_params={"reason": "rac_auto_rollback"},
            ),
        },
        retry_policy=RetryPolicy(max_retries=3, base_delay=0.5),
        auto_rollback=True,
        auto_recover=True,
        system_prompt="You are a travel assistant. Book hotel, process payment, then finalize.",
    )
    
    task = "Book Grand Plaza Hotel for John Smith (3 nights), pay $450 with card 4242, then finalize."
    
    start_time = time.time()
    error = None
    
    try:
        # Run with clear Langsmith metadata
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={
                "run_name": "🔷 RAC: Full Compensation Flow",
                "tags": [
                    "framework:RAC",
                    "scenario:compensation-demo",
                    "benchmark",
                ],
                "metadata": {
                    "framework": "RAC (React-Agent-Compensation)",
                    "framework_type": "automatic_compensation",
                    "scenario": "book-pay-finalize-rollback",
                    "expected_outcome": "booking_cancelled",
                    "features": ["auto_retry", "auto_rollback", "compensation_schemas"],
                },
                "callbacks": callbacks,
            },
        )
    except Exception as e:
        error = str(e)
    
    execution_time = time.time() - start_time
    
    final_status = "none"
    for bid, booking in state["bookings"].items():
        final_status = booking["status"]
    
    return BenchmarkResult(
        framework="RAC",
        success=len(state["compensations"]) > 0,
        execution_time=execution_time,
        tool_calls=state["tool_calls"].copy(),
        compensations_triggered=state["compensations"].copy(),
        final_booking_status=final_status,
        error=error,
    )


def run_sagallm_benchmark(tools: List, state: Dict) -> BenchmarkResult:
    """Run benchmark using SagaLLM-style 3-phase approach with proper tracing."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tracers import LangChainTracer
    import json
    
    reset_state(state)
    
    # Setup Langsmith tracer
    callbacks = []
    try:
        tracer = LangChainTracer(
            project_name=os.getenv("LANGSMITH_PROJECT", "compensation-benchmark")
        )
        callbacks.append(tracer)
    except Exception as e:
        logger.debug(f"Langsmith not available: {e}")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        callbacks=callbacks,
    )
    
    tool_map = {t.name: t for t in tools}
    
    start_time = time.time()
    error = None
    
    try:
        # =====================================================================
        # PHASE 1: PLANNING
        # =====================================================================
        logger.info("[SagaLLM] Phase 1: Planning")
        
        plan_prompt = """Generate a JSON plan to: Book Grand Plaza Hotel for John Smith (3 nights), pay $450 with card 4242, then finalize.
        
Available tools with their parameters:

1. book_hotel(hotel_name: str, guest_name: str, nights: int) -> Returns {booking_id, hotel, status}
2. process_payment(amount: float, card_number: str) -> Returns {status, transaction_id} or {status: "failed", error}
3. finalize_reservation(booking_id: str, payment_id: str) -> Returns {status}
4. cancel_hotel(booking_id: str, reason: str) -> Returns {booking_id, status: "cancelled"}

Return ONLY a valid JSON array with exact parameter names:
[
  {"step": 1, "tool": "book_hotel", "args": {"hotel_name": "Grand Plaza Hotel", "guest_name": "John Smith", "nights": 3}},
  {"step": 2, "tool": "process_payment", "args": {"amount": 450.0, "card_number": "4242"}},
  {"step": 3, "tool": "finalize_reservation", "args": {"booking_id": "RESULT_FROM_STEP_1", "payment_id": "RESULT_FROM_STEP_2"}}
]"""
        
        plan_response = llm.invoke(
            [
                SystemMessage(content="You are a planning assistant. Return ONLY valid JSON."),
                HumanMessage(content=plan_prompt),
            ],
            config={
                "run_name": "🟡 SagaLLM Phase 1: PLANNING",
                "tags": ["framework:SagaLLM", "phase:planning", "benchmark"],
                "metadata": {
                    "framework": "SagaLLM (3-Phase Saga)",
                    "phase": "1-Planning",
                    "description": "Generate action plan from user request",
                },
                "callbacks": callbacks,
            },
        )
        
        # Parse plan
        content = plan_response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        plan = json.loads(content)
        
        logger.info(f"[SagaLLM] Plan generated: {len(plan)} steps")
        
        # =====================================================================
        # PHASE 2: EXECUTION
        # =====================================================================
        logger.info(f"[SagaLLM] Phase 2: Execution ({len(plan)} steps)")
        
        executed_actions = []
        step_results = {}
        failed = False
        
        for step in plan:
            step_num = step.get("step", len(executed_actions) + 1)
            tool_name = step.get("tool")
            tool_args = step.get("args", {}).copy()
            
            if tool_name not in tool_map:
                logger.warning(f"[SagaLLM] Unknown tool: {tool_name}")
                continue
            
            # Substitute placeholders with actual results
            for arg_key, arg_val in tool_args.items():
                if isinstance(arg_val, str) and "RESULT_FROM_STEP_" in arg_val:
                    try:
                        ref_step = int(arg_val.replace("RESULT_FROM_STEP_", ""))
                        if ref_step in step_results:
                            prev_result = step_results[ref_step]
                            if arg_key == "booking_id" and "booking_id" in prev_result:
                                tool_args[arg_key] = prev_result["booking_id"]
                            elif arg_key == "payment_id" and "transaction_id" in prev_result:
                                tool_args[arg_key] = prev_result["transaction_id"]
                            elif arg_key == "payment_id":
                                tool_args[arg_key] = "TXN-PENDING"
                    except ValueError:
                        pass
            
            logger.info(f"[SagaLLM] Step {step_num}: {tool_name}")
            
            result = tool_map[tool_name].invoke(tool_args)
            step_results[step_num] = result
            executed_actions.append({"tool": tool_name, "args": tool_args, "result": result})
            
            # Check for failure
            if isinstance(result, dict) and result.get("status") == "failed":
                if "retry" in str(result.get("error", "")).lower():
                    for retry in range(2):
                        logger.info(f"[SagaLLM] Retrying {tool_name} (attempt {retry + 2})")
                        result = tool_map[tool_name].invoke(tool_args)
                        step_results[step_num] = result
                        if result.get("status") != "failed":
                            break
                
                if result.get("status") == "failed":
                    logger.warning(f"[SagaLLM] Step failed: {tool_name}")
                    failed = True
                    break
        
        # =====================================================================
        # PHASE 3: COMPENSATION
        # =====================================================================
        if failed:
            logger.info("[SagaLLM] Phase 3: Compensation")
            
            compensation_mapping = {"book_hotel": "cancel_hotel"}
            
            for action in reversed(executed_actions):
                tool_name = action["tool"]
                comp_tool = compensation_mapping.get(tool_name)
                
                if comp_tool and comp_tool in tool_map:
                    result = action["result"]
                    if isinstance(result, dict) and "booking_id" in result:
                        comp_args = {
                            "booking_id": result["booking_id"], 
                            "reason": "sagallm_rollback"
                        }
                        tool_map[comp_tool].invoke(comp_args)
    
    except Exception as e:
        error = str(e)
        import traceback
        logger.error(f"SagaLLM error: {traceback.format_exc()}")
    
    execution_time = time.time() - start_time
    
    final_status = "none"
    for bid, booking in state["bookings"].items():
        final_status = booking["status"]
    
    return BenchmarkResult(
        framework="SagaLLM",
        success=len(state["compensations"]) > 0,
        execution_time=execution_time,
        tool_calls=state["tool_calls"].copy(),
        compensations_triggered=state["compensations"].copy(),
        final_booking_status=final_status,
        error=error,
    )


def print_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark comparison."""
    print("\n" + "=" * 70)
    print("📊 COMPENSATION BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\n{'Framework':<15} {'Compensation':<15} {'Time':<10} {'Final Status':<15}")
    print("-" * 60)
    
    for r in results:
        comp_status = "✅ Triggered" if r.success else "❌ Not triggered"
        print(f"{r.framework:<15} {comp_status:<15} {r.execution_time:.2f}s     {r.final_booking_status:<15}")
    
    print("\n" + "-" * 70)
    print("DETAILED BREAKDOWN:")
    print("-" * 70)
    
    for r in results:
        print(f"\n🔷 {r.framework}")
        print(f"   Tool calls: {len(r.tool_calls)}")
        for tc in r.tool_calls:
            print(f"      • {tc}")
        print(f"   Compensations: {len(r.compensations_triggered)}")
        for c in r.compensations_triggered:
            print(f"      🔄 {c}")
        if r.error:
            print(f"   ⚠️ Error: {r.error[:80]}")
    
    print("\n" + "=" * 70)
    print("\n📋 LANGSMITH TRACE GUIDE:")
    print("   Filter by tag: 'framework:RAC' or 'framework:SagaLLM'")
    print("   Look for runs named:")
    print("     • '🔷 RAC: Full Compensation Flow'")
    print("     • '🟡 SagaLLM Phase 1: PLANNING'")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("🎬 COMPENSATION BENCHMARK: RAC vs SagaLLM")
    print("=" * 70)
    print("\nScenario: Book hotel → Payment fails 2x → Finalize fails → ROLLBACK")
    print("-" * 70)
    
    # Create shared state
    state = {
        "booking_count": 0,
        "bookings": {},
        "payment_attempts": 0,
        "tool_calls": [],
        "compensations": [],
    }
    
    tools = create_test_tools(state)
    results = []
    
    # Run RAC benchmark
    print("\n🔹 Running RAC benchmark...")
    rac_result = run_rac_benchmark(tools, state)
    results.append(rac_result)
    
    # Run SagaLLM benchmark  
    print("\n🔹 Running SagaLLM benchmark...")
    saga_result = run_sagallm_benchmark(tools, state)
    results.append(saga_result)
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
