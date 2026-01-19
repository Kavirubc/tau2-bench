#!/usr/bin/env python3
"""
Complex Replanning Demo for SagaLLM vs RAC.

Scenario: Group Booking with Cascading Failures
User wants to book 4 passengers on the same flight.
- Flight A: Only 1 seat left (Fails after 1st booking)
- Flight B: Only 2 seats left (Fails after 2nd booking)
- Flight C: Plenty of seats (Succeeds)

Usage:
    python3 complex_replanning_demo.py --frameworks saga,rac
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("replanning_demo")

# Add paths
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

from tau2_integration.runners.saga_runner import SagaLLMRunner
from tau2_integration.runners.rac_runner import RACRunner
from tau2_integration.task_adapter import Tau2TaskDefinition, TaskCategory
from tau2_integration.wrapped_tools import AIRLINE_COMPENSATION_MAPPING

# =============================================================================
# MOCK TOOLS
# =============================================================================

class FlightSystem:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.flights = {
            "FL-A": {"seats": 1, "price": 100, "name": "Flight A (Cheap)"},
            "FL-B": {"seats": 2, "price": 150, "name": "Flight B (Moderate)"},
            "FL-C": {"seats": 10, "price": 200, "name": "Flight C (Expensive)"},
        }
        self.bookings = {} # booking_id -> flight_id
        
    def search_flights(self, origin: str, destination: str) -> List[Dict]:
        """Search for flights between origin and destination."""
        logger.info(f"🔎 SEARCH: {origin} -> {destination}")
        return [
            {"id": k, "name": v["name"], "price": v["price"], "status": "Available"}
            for k, v in self.flights.items()
        ]

    def book_flight(self, flight_id: str, passenger_name: str) -> Dict:
        """Book a flight for a passenger."""
        if flight_id not in self.flights:
            return {"status": "failed", "error": "Flight not found"}
            
        flight = self.flights[flight_id]
        if flight["seats"] > 0:
            flight["seats"] -= 1
            booking_id = f"BK-{len(self.bookings)+1:03d}"
            self.bookings[booking_id] = flight_id
            logger.info(f"✅ BOOK: {passenger_name} on {flight_id} (ID: {booking_id})")
            return {
                "status": "confirmed",
                "booking_id": booking_id,
                "flight": flight["name"],
                "passenger": passenger_name
            }
        else:
            logger.warning(f"❌ BOOK FAIL: {passenger_name} on {flight_id} - No seats")
            return {
                "status": "failed", 
                "error": f"Flight {flight_id} is fully booked. Only 0 seats remaining."
            }

    def cancel_flight(self, booking_id: str) -> Dict:
        """Cancel a flight booking (Compensation)."""
        if booking_id in self.bookings:
            flight_id = self.bookings[booking_id]
            self.flights[flight_id]["seats"] += 1 # Restore seat
            del self.bookings[booking_id]
            logger.info(f"🔄 CANCEL: {booking_id} (Restored seat on {flight_id})")
            return {"status": "cancelled", "booking_id": booking_id}
        else:
            logger.warning(f"⚠️ CANCEL FAIL: {booking_id} not found")
            return {"status": "failed", "error": "Booking not found"}

# =============================================================================
# RUNNER
# =============================================================================

def run_framework(framework_name: str, system: FlightSystem) -> Dict:
    """Run the benchmark with the specified framework."""
    print("\n" + "-" * 70)
    print(f"🏃 RUNNING FRAMEWORK: {framework_name.upper()}")
    print("-" * 70)
    
    # Reset system state
    system.reset()
    
    
    
    # Initialize Runner
    if framework_name == "saga":
        from langchain_core.tools import StructuredTool
        # Saga runner expects StructuredTools (for now, or handled internally if not)
        tools = {
            "search_flights": StructuredTool.from_function(system.search_flights),
            "book_flight": StructuredTool.from_function(system.book_flight),
            "cancel_flight": StructuredTool.from_function(system.cancel_flight),
        }
        runner = SagaLLMRunner(
            model="gemini-2.0-flash",
            max_iterations=7,
            enable_replanning=True
        )
    elif framework_name == "rac":
        # RAC runner expects a dict of bare functions to convert them itself
        tools = {
            "search_flights": system.search_flights,
            "book_flight": system.book_flight,
            "cancel_flight": system.cancel_flight,
        }
        runner = RACRunner(
            model="gemini-2.0-flash",
            max_iterations=25,
            auto_rollback=True,
            auto_recover=True
        )
    else:
        raise ValueError(f"Unknown framework: {framework_name}")
    
    # Define Task
    task = Tau2TaskDefinition(
        task_id=f"complex-replan-{framework_name}",
        name="Group Booking",
        category=TaskCategory.BOOKING,
        description="Book group flight",
        user_scenario={
            "instructions": {
                "reason_for_call": "Book flights for Alice, Bob, Charlie, and Dave from NYC to LON. ALL passengers must be on the SAME flight."
            }
        },
        goals=[],
        constraints=[],
        expected_actions=[],
        nl_assertions=[]
    )
    
    # Run
    start_time = time.time()
    
    # Monkeypatch compensation mapping for custom tools
    AIRLINE_COMPENSATION_MAPPING["book_flight"] = "cancel_flight"
    
    result = runner.run_task(task, tools, policy="")
    duration = time.time() - start_time
    
    # Report Results
    final_flights = set()
    booking_count = len(system.bookings)
    for bid, fid in system.bookings.items():
        final_flights.add(fid)
    
    success = False
    if result.success and len(final_flights) == 1 and booking_count == 4:
        success = True
    elif result.success:
        # Framework thinks it succeeded, but validation failed (partial/wrong state)
        logger.warning(f"Framework reported success, but state validation failed: {booking_count} bookings on {len(final_flights)} flights.")
        success = False # Strict validation
        
    return {
        "framework": framework_name,
        "reported_success": result.success,
        "validated_success": success,
        "duration": duration,
        "bookings": booking_count,
        "flights_used": len(final_flights),
        "error": result.error if not result.success else None
    }

def main():
    parser = argparse.ArgumentParser(description="Run comparison benchmark")
    parser.add_argument("--frameworks", type=str, default="saga,rac", help="Comma-separated frameworks to run")
    args = parser.parse_args()
    
    frameworks = args.frameworks.split(",")
    
    print("\n" + "=" * 70)
    print("🔄 SAGA vs RAC: COMPLEX REPLANNING BENCHMARK")
    print("=" * 70)
    print("Goal: Book 4 passengers on the same flight.")
    print("Constraints:")
    print("  - Flight A: 1 seat (Will fail after 1 booking)")
    print("  - Flight B: 2 seats (Will fail after 2 bookings)")
    print("  - Flight C: 10 seats (Success)")
    print("=" * 70)
    
    system = FlightSystem()
    results = []
    
    for fw in frameworks:
        try:
            res = run_framework(fw.strip(), system)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to run {fw}: {e}")
            results.append({"framework": fw, "error": str(e), "validated_success": False})

    print("\n" + "=" * 70)
    print("🏁 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'FRAMEWORK':<10} | {'SUCCESS':<10} | {'TIME':<10} | {'BOOKINGS':<10} | {'NOTES'}")
    print("-" * 70)
    
    for res in results:
        status = "✅ PASS" if res.get("validated_success") else "❌ FAIL"
        time_str = f"{res.get('duration', 0):.2f}s"
        bookings = f"{res.get('bookings', 0)}/4"
        note = res.get("error") or ""
        if not note and not res.get("validated_success"):
            note = f"Invalid State ({res.get('flights_used')} flights)"
            
        print(f"{res['framework'].upper():<10} | {status:<10} | {time_str:<10} | {bookings:<10} | {note}")
    print("=" * 70)

if __name__ == "__main__":
    main()
