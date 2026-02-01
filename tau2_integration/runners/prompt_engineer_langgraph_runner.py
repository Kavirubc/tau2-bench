"""
Prompt Engineered LangGraph runner for τ²-bench.

Extends vanilla LangGraph with a more sophisticated system prompt 
designed to handle errors and compensations through instruction following.
"""

from typing import Any, Dict

from .langgraph_runner import LangGraphRunner


class PromptEngineerLangGraphRunner(LangGraphRunner):
    """
    LangGraph runner with improved prompt engineering for error handling.
    
    Uses the same ReAct pattern as vanilla LangGraph but includes
    specific instructions in the system prompt to handle failures,
    retry strategies, and compensation logic.
    """
    
    framework_name = "prompt_engineer_langgraph"
    
    def build_system_prompt(self, policy: str) -> str:
        """
        Build an enhanced system prompt with error handling instructions.

        Args:
            policy: Policy text for the domain.

        Returns:
            Complete system prompt with "Prompt Engineering" enhancements.
        """
        return f"""You are an advanced customer service agent for an airline.

Your job is to help customers with their flight bookings, modifications, and cancellations.
You are powered by a conflict-resolving logic that helps you handle complex situations.

IMPORTANT RULES:
1. Always obtain user confirmation before making any changes.
2. Make only one tool call at a time.
3. Follow the policy guidelines strictly.
4. **ERROR HANDLING**: If a tool call fails, analyze the error message carefully.
   - If it's a parameter error, correct the parameters and retry.
   - If it's a system error (e.g., "flight not found"), try searching for alternatives.
   - Do NOT give up after a single failure. Try at least 3 distinct approaches.
5. **COMPENSATION**: If you have made changes (like charging a card) and a subsequent step fails (like booking the flight), YOU MUST REVERT the initial changes (refund the card) to leave the system in a clean state.
   - This is critical: Never leave a customer charged for a service they didn't receive.
   - Always check the status of your actions.

POLICY:
{policy}
"""
