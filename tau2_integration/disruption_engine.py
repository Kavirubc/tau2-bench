"""
Disruption engine for injecting failures during tool execution.

Ported from REALM-Bench and adapted for τ²-bench airline domain.
Maps task disruption scenarios to tool failures for testing compensation behavior.
"""

import sys
import random
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("tau2_integration.disruption")


class DisruptionTrigger(Enum):
    """When a disruption should trigger."""
    ON_TOOL_CALL = "on_tool_call"
    AFTER_N_ACTIONS = "after_n_actions"
    PROBABILISTIC = "probabilistic"


class DisruptionType(Enum):
    """Types of disruptions for airline domain."""
    # Transient failures - retry may succeed
    PAYMENT_FAILURE = "payment_failure"
    SYSTEM_ERROR = "system_error"
    
    # Persistent failures - requires replanning
    FLIGHT_UNAVAILABLE = "flight_unavailable"
    SEAT_SHORTAGE = "seat_shortage"
    
    # Complex scenarios
    RESERVATION_CONFLICT = "reservation_conflict"
    USER_NOT_FOUND = "user_not_found"


@dataclass
class DisruptionConfig:
    """Configuration for a single disruption."""
    disruption_type: str
    affected_tool: Optional[str] = None
    affected_resource: Optional[str] = None
    trigger: DisruptionTrigger = DisruptionTrigger.AFTER_N_ACTIONS
    trigger_after_n_actions: int = 2
    probability: float = 1.0
    error_message: str = "Disruption occurred"
    triggered: bool = False
    persistent: bool = False  # If True, keeps failing on every retry
    transient_retries_until_success: int = 2  # Fail first N attempts, succeed after


# Mapping of disruption types to affected tools
AIRLINE_DISRUPTION_TOOL_MAPPING = {
    "payment_failure": {
        "tool": "book_reservation",
        "message": "Payment processing failed - please try again"
    },
    "system_error": {
        "tool": None,  # Any tool
        "message": "System temporarily unavailable - please retry"
    },
    "flight_unavailable": {
        "tool": "book_reservation",
        "message": "Flight is no longer available for booking"
    },
    "seat_shortage": {
        "tool": "book_reservation",
        "message": "Not enough seats available on this flight"
    },
    "reservation_conflict": {
        "tool": "update_reservation_flights",
        "message": "Reservation update conflict - concurrent modification detected"
    },
    "user_not_found": {
        "tool": "get_user_details",
        "message": "User not found in system"
    },
}


class DisruptionEngine:
    """
    Manages disruption injection during tool execution.

    Loads disruption scenarios from task definitions and triggers them
    during tool calls to test compensation behavior.
    
    Implements singleton pattern to ensure consistent state across tool calls.
    """

    _instance = None

    def __new__(cls):
        """Ensure true singleton even if imported via different module paths."""
        if not hasattr(sys, "_tau2_disruption_engine"):
            instance = super().__new__(cls)
            instance._disruptions: List[DisruptionConfig] = []
            instance._action_count = 0
            instance._triggered_disruptions: List[Dict[str, Any]] = []
            instance._enabled = True
            instance._executed_actions: List[Dict[str, Any]] = []
            instance._transient_attempt_counts: Dict[str, int] = {}
            sys._tau2_disruption_engine = instance
        return sys._tau2_disruption_engine

    def reset(self) -> None:
        """Reset the disruption engine for a new task."""
        self._disruptions = []
        self._action_count = 0
        self._triggered_disruptions = []
        self._executed_actions = []
        self._transient_attempt_counts = {}
        logger.debug("DisruptionEngine reset")

    def enable(self) -> None:
        """Enable disruption injection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable disruption injection."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if disruption injection is enabled."""
        return self._enabled

    def configure(self, disruption_scenarios: List[Dict[str, Any]]) -> None:
        """
        Configure disruptions from a list of scenario definitions.

        Args:
            disruption_scenarios: List of disruption configurations.
        """
        self._disruptions = []
        
        for scenario in disruption_scenarios:
            config = self._map_scenario_to_config(scenario)
            if config:
                self._disruptions.append(config)
                logger.info(
                    f"Configured disruption: {config.disruption_type} "
                    f"-> {config.affected_tool}"
                )

    def _map_scenario_to_config(
        self,
        scenario: Dict[str, Any]
    ) -> Optional[DisruptionConfig]:
        """Map a task disruption scenario to engine config."""
        dtype = scenario.get("type", "")
        
        # Handle enum types
        if hasattr(dtype, "value"):
            dtype = dtype.value
        
        dtype_lower = str(dtype).lower()
        
        # Look up in mapping
        if dtype_lower in AIRLINE_DISRUPTION_TOOL_MAPPING:
            mapping = AIRLINE_DISRUPTION_TOOL_MAPPING[dtype_lower]
            trigger_after = scenario.get("trigger_after", 2)
            persistent = scenario.get("persistent", False)
            transient_retries = scenario.get("retries_until_success", 2)
            
            error_msg = scenario.get("message", mapping["message"])
            if not persistent and "transient" not in error_msg.lower():
                error_msg = f"{error_msg} (temporary - retry may succeed)"
            
            return DisruptionConfig(
                disruption_type=dtype_lower,
                affected_tool=scenario.get("affected_tool", mapping["tool"]),
                affected_resource=scenario.get("affected_resource"),
                error_message=error_msg,
                trigger=DisruptionTrigger.AFTER_N_ACTIONS,
                trigger_after_n_actions=trigger_after,
                probability=scenario.get("probability", 1.0),
                persistent=persistent,
                transient_retries_until_success=transient_retries,
            )
        
        return None

    def check_disruption(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check if a disruption should occur for this tool call.

        Args:
            tool_name: Name of the tool being called.
            tool_args: Arguments passed to the tool.

        Returns:
            Error message if disruption triggers, None otherwise.
        """
        if not self._enabled:
            return None

        self._action_count += 1

        for disruption in self._disruptions:
            # Skip already triggered non-persistent disruptions
            if disruption.triggered and not disruption.persistent:
                # Check if it's a transient failure that hasn't exhausted retries
                if disruption.disruption_type not in ["payment_failure", "system_error"]:
                    continue
                    
            if self._should_trigger(disruption, tool_name, tool_args):
                if not disruption.persistent:
                    disruption.triggered = True
                    
                self._triggered_disruptions.append({
                    "disruption": disruption.disruption_type,
                    "tool": tool_name,
                    "args": tool_args,
                    "action_count": self._action_count,
                    "error_message": disruption.error_message,
                    "persistent": disruption.persistent
                })
                
                logger.warning(
                    f"DISRUPTION TRIGGERED: {disruption.disruption_type} "
                    f"on {tool_name} at action {self._action_count}"
                )
                
                return disruption.error_message

        return None

    def _should_trigger(
        self,
        disruption: DisruptionConfig,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> bool:
        """Determine if a disruption should trigger."""
        # Check if tool matches (None means any tool)
        if disruption.affected_tool and disruption.affected_tool != tool_name:
            return False

        # Handle transient failures with retry counting
        if disruption.disruption_type in ["payment_failure", "system_error"]:
            transient_key = f"{tool_name}:{disruption.disruption_type}"
            
            if transient_key not in self._transient_attempt_counts:
                self._transient_attempt_counts[transient_key] = 0
            self._transient_attempt_counts[transient_key] += 1
            
            attempt_count = self._transient_attempt_counts[transient_key]
            retries_needed = disruption.transient_retries_until_success
            
            # Must have done enough actions first
            if self._action_count < disruption.trigger_after_n_actions:
                return False
            
            # Fail until we've retried enough times
            if attempt_count <= retries_needed:
                logger.debug(
                    f"Transient failure {transient_key}: "
                    f"attempt {attempt_count}/{retries_needed}"
                )
                return True
            else:
                logger.debug(
                    f"Transient success {transient_key}: "
                    f"attempt {attempt_count} > {retries_needed}"
                )
                return False

        # Check trigger conditions for other disruptions
        if disruption.trigger == DisruptionTrigger.AFTER_N_ACTIONS:
            if self._action_count < disruption.trigger_after_n_actions:
                return False

        # Apply probability
        if random.random() > disruption.probability:
            return False

        return True

    def record_action(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any
    ) -> None:
        """Record an executed action for potential rollback."""
        self._executed_actions.append({
            "tool": tool_name,
            "args": tool_args,
            "result": result,
            "action_id": len(self._executed_actions)
        })
        logger.debug(f"Recorded action: {tool_name}")

    def get_executed_actions(self) -> List[Dict[str, Any]]:
        """Get list of all executed actions."""
        return self._executed_actions.copy()

    def get_triggered_disruptions(self) -> List[Dict[str, Any]]:
        """Get list of all triggered disruptions."""
        return self._triggered_disruptions.copy()

    def get_action_count(self) -> int:
        """Get the current action count."""
        return self._action_count

    def has_disruptions_configured(self) -> bool:
        """Check if any disruptions are configured."""
        return len(self._disruptions) > 0


def get_disruption_engine() -> DisruptionEngine:
    """Get the singleton disruption engine instance."""
    return DisruptionEngine()


# Predefined disruption scenarios for airline domain testing
AIRLINE_DISRUPTION_SCENARIOS = {
    "payment_transient": {
        "type": "payment_failure",
        "trigger_after": 2,
        "persistent": False,
        "retries_until_success": 2,
        "message": "Payment gateway timeout - please retry",
    },
    "flight_unavailable": {
        "type": "flight_unavailable",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "book_reservation",
        "message": "Flight HAT123 is no longer available",
    },
    "seat_shortage": {
        "type": "seat_shortage",
        "trigger_after": 3,
        "persistent": True,
        "affected_tool": "book_reservation",
        "message": "Insufficient seats in requested cabin class",
    },
    "system_error": {
        "type": "system_error",
        "trigger_after": 1,
        "persistent": False,
        "retries_until_success": 1,
        "message": "Internal system error - temporary",
    },
}
