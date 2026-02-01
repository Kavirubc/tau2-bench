#!/usr/bin/env python3
"""
Compensation Demo Script for τ²-bench.

Demonstrates RAC's compensation capabilities by:
1. Running a booking task that succeeds
2. Injecting a failure after booking
3. Watching RAC automatically roll back the booking

Usage:
    source .venv/bin/activate
    python3 tau2_integration/compensation_demo.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("compensation_demo")

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable Langsmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


def run_compensation_demo():
    """Run a demo that triggers compensation."""
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.tools import StructuredTool
    from langchain_core.messages import HumanMessage
    
    from react_agent_compensation.langchain_adaptor import (
        create_compensated_agent,
        get_compensation_middleware,
    )
    from react_agent_compensation.core import CompensationSchema, RetryPolicy
    
    print("\n" + "=" * 70)
    print("🎬 RAC COMPENSATION DEMONSTRATION")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Create mock tools that simulate failures
    # =========================================================================
    print("\n📦 Setting up tools with failure injection...")
    
    booking_count = [0]
    bookings = {}
    
    def book_hotel(hotel_name: str, guest_name: str, nights: int) -> dict:
        """Book a hotel room for a guest."""
        booking_count[0] += 1
        booking_id = f"HOTEL-{booking_count[0]:03d}"
        
        logger.info(f"✅ BOOK_HOTEL: {hotel_name} for {guest_name}, {nights} nights")
        
        bookings[booking_id] = {
            "hotel": hotel_name,
            "guest": guest_name,
            "nights": nights,
            "status": "confirmed"
        }
        
        return {
            "booking_id": booking_id,
            "hotel": hotel_name,
            "status": "confirmed",
            "total": nights * 150
        }
    
    def cancel_hotel(booking_id: str, reason: str = "auto_rollback") -> dict:
        """Cancel a hotel booking."""
        logger.info(f"🔄 CANCEL_HOTEL (COMPENSATION): {booking_id} - {reason}")
        
        if booking_id in bookings:
            bookings[booking_id]["status"] = "cancelled"
        
        return {
            "booking_id": booking_id,
            "status": "cancelled",
            "refund": "full"
        }
    
    call_count = [0]
    
    def process_payment(amount: float, card_number: str) -> dict:
        """Process a payment. FAILS on first attempt to demonstrate compensation."""
        call_count[0] += 1
        
        if call_count[0] <= 2:
            # First 2 calls fail to trigger retry
            logger.warning(f"❌ PAYMENT FAILED (attempt {call_count[0]})")
            return {
                "status": "failed",
                "error": f"Payment gateway timeout (attempt {call_count[0]}). Please retry.",
            }
        
        logger.info(f"✅ PAYMENT SUCCESS (attempt {call_count[0]})")
        return {
            "status": "success",
            "transaction_id": "TXN-12345",
            "amount": amount
        }
    
    def finalize_reservation(
        booking_id: str, 
        payment_id: str
    ) -> dict:
        """Final confirmation that ALWAYS fails to trigger full rollback."""
        logger.error("❌ FINAL CONFIRMATION FAILED - TRIGGERING COMPENSATION")
        return {
            "status": "failed",
            "error": "System error: Unable to finalize reservation. Database constraint violation.",
        }
    
    # Create LangChain tools
    tools = [
        StructuredTool.from_function(book_hotel, name="book_hotel"),
        StructuredTool.from_function(cancel_hotel, name="cancel_hotel"),
        StructuredTool.from_function(process_payment, name="process_payment"),
        StructuredTool.from_function(
            finalize_reservation, 
            name="finalize_reservation"
        ),
    ]
    
    # =========================================================================
    # Step 2: Create compensated agent with RAC
    # =========================================================================
    print("\n🤖 Creating RAC agent with compensation mapping...")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )
    
    compensation_mapping = {
        "book_hotel": "cancel_hotel",
        "process_payment": None,  # Payments can't be easily rolled back
    }
    
    compensation_schemas = {
        "book_hotel": CompensationSchema(
            param_mapping={"booking_id": "result.booking_id"},
            static_params={"reason": "auto_rollback_due_to_failure"},
        ),
    }
    
    retry_policy = RetryPolicy(
        max_retries=3,
        base_delay=0.5,
    )
    
    agent = create_compensated_agent(
        model=llm,
        tools=tools,
        compensation_mapping=compensation_mapping,
        compensation_schemas=compensation_schemas,
        retry_policy=retry_policy,
        auto_rollback=True,
        auto_recover=True,
        system_prompt="""You are a travel booking assistant.

When the user asks to make a reservation:
1. First book the hotel using book_hotel
2. Then process payment using process_payment  
3. Finally finalize with finalize_reservation

If any step fails, the system will automatically try to recover or rollback previous actions.
""",
    )
    
    middleware = get_compensation_middleware(agent)
    
    # =========================================================================
    # Step 3: Run the agent with a task that will trigger compensation
    # =========================================================================
    print("\n🚀 Running booking task (payment will fail, then succeed)...")
    print("-" * 70)
    
    task = """
    Book a hotel room at the Grand Plaza Hotel for John Smith for 3 nights.
    Process payment of $450 with card ending 4242.
    Then finalize the reservation.
    """
    
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={
                "run_name": "Compensation-Demo",
                "tags": ["compensation-demo", "RAC", "failure-injection"],
                "metadata": {
                    "demo": True,
                    "purpose": "demonstrate_compensation",
                },
            },
        )
        
        print("-" * 70)
        print("\n📋 AGENT RESPONSE:")
        
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                print(last_msg.content[:500])
        
    except Exception as e:
        print(f"\n⚠️ Agent completed with exception (expected): {e}")
    
    # =========================================================================
    # Step 4: Show compensation results
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 COMPENSATION RESULTS")
    print("=" * 70)
    
    if middleware:
        print("\n📜 Transaction Log:")
        try:
            log_snapshot = middleware.transaction_log.snapshot()
            for rid, record in log_snapshot.items():
                status = str(record.status) if hasattr(record, 'status') else 'unknown'
                compensator = record.compensator if hasattr(record, 'compensator') else 'none'
                print(f"   • {record.action}: {status} (compensator: {compensator})")
        except Exception as e:
            print(f"   (Could not read log: {e})")
    
    print("\n📒 Booking State:")
    for bid, booking in bookings.items():
        status_emoji = "✅" if booking["status"] == "confirmed" else "❌"
        print(f"   {status_emoji} {bid}: {booking['hotel']} - {booking['status']}")
    
    print("\n" + "=" * 70)
    print("✨ Demo complete! Check Langsmith for detailed trace.")
    print("   Filter by tag: 'compensation-demo'")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_compensation_demo()
