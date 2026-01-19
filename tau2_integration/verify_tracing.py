
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Load env vars
load_dotenv()

# Ensure we can import tau2_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tau2_integration.task_adapter import Tau2TaskDefinition
from tau2_integration.runners.saga_runner import SagaLLMRunner
from tau2_integration.runners.rac_runner import RACRunner
from tau2_integration.evaluation import evaluate_run, print_comparison_table
from tau2_integration.tracing import TraceRecorder

from typing import List, Dict, Any

# Mock system to avoid full tool dependencies
class MockSystem:
    def search_flights(self, origin: str, destination: str, date: str) -> str:
        """Search for flights."""
        return str([{"flight_id": "FL123", "price": 100, "seats": 10}])
    def book_flight(self, flight_id: str, passengers: List[Dict[str, Any]]) -> str:
        """Book a flight."""
        return str({"status": "confirmed", "reservation_id": "RES789"})
    def cancel_flight(self, reservation_id: str) -> str:
        """Cancel a flight."""
        return str({"status": "cancelled"})

def run_verification():
    print("=== STARTING TRACE VERIFICATION ===")
    
    # Create simple task
    task = Tau2TaskDefinition(
        task_id="Verify_Tracing",
        name="Verify Tracing",
        category="booking",
        description="Book a flight from JFK to LHR on 2024-01-01 for John Doe.",
        user_scenario={
            "instructions": {
                "task_instructions": "Book a flight from JFK to LHR on 2024-01-01 for John Doe."
            }
        },
        goals=[],
        constraints=[],
        expected_actions=[],
        nl_assertions=[],
        disruption_scenarios=[]
    )
    
    system = MockSystem()
    
    # 1. Run SagaLLM
    print("\n[Running SagaLLM...]")
    from langchain_core.tools import StructuredTool
    saga_tools = {
        "search_flights": StructuredTool.from_function(system.search_flights),
        "book_flight": StructuredTool.from_function(system.book_flight),
        "cancel_flight": StructuredTool.from_function(system.cancel_flight),
    }
    
    saga_runner = SagaLLMRunner(model="gemini-2.0-flash-exp")
    saga_runner.max_iterations = 3
    saga_result = saga_runner.run_task(task, saga_tools, policy="")
    
    print(f"SagaLLM Success: {saga_result.success}")
    if saga_result.trace:
        steps = saga_result.trace.get('steps', [])
        print(f"SagaLLM Trace captured: {len(steps)} steps")
        for i, s in enumerate(steps):
             print(f"  Step {i}: type={s.get('type')}, name={s.get('name')}, tokens={s.get('tokens')}")
    else:
        print("ERROR: No trace captured for SagaLLM!")
        
    # 2. Run RAC
    print("\n[Running RAC...]")
    rac_tools = {
        "search_flights": system.search_flights,
        "book_flight": system.book_flight,
        "cancel_flight": system.cancel_flight,
    }
    
    rac_runner = RACRunner(model="gemini-2.0-flash-exp")
    rac_result = rac_runner.run_task(task, rac_tools, policy="")
    
    print(f"RAC Success: {rac_result.success}")
    if rac_result.trace:
        steps = rac_result.trace.get('steps', [])
        print(f"RAC Trace captured: {len(steps)} steps")
        for i, s in enumerate(steps):
             print(f"  Step {i}: type={s.get('type')}, name={s.get('name')}, tokens={s.get('tokens')}")
    else:
        print("ERROR: No trace captured for RAC!")

    
    # 3. Evaluate and Compare
    print("\n[Evaluating and Printing Table...]")
    # Pass empty lists if task fields are None
    expected = task.expected_actions if task.expected_actions else []
    assertions = task.nl_assertions if task.nl_assertions else []
    
    saga_metrics = evaluate_run(saga_result, expected, assertions)
    rac_metrics = evaluate_run(rac_result, expected, assertions)
    
    print_comparison_table({"SagaLLM": saga_metrics, "RAC": rac_metrics})

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        import traceback
        traceback.print_exc()
