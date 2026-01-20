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

# Retail domain disruption mappings
RETAIL_DISRUPTION_TOOL_MAPPING = {
    "payment_gateway_timeout": {
        "tool": "modify_pending_order_payment",
        "message": "Payment gateway timeout - please retry in a moment"
    },
    "inventory_sync_delay": {
        "tool": "get_product_details",
        "message": "Inventory system temporarily unavailable - retry shortly"
    },
    "database_lock": {
        "tool": None,  # Any write tool
        "message": "Database temporarily locked - please retry"
    },
    "product_out_of_stock": {
        "tool": "modify_pending_order_items",
        "message": "Product variant is currently out of stock"
    },
    "invalid_product_id": {
        "tool": "get_product_details",
        "message": "Product ID not found in catalog"
    },
    "insufficient_balance": {
        "tool": "modify_pending_order_payment",
        "message": "Insufficient gift card balance to complete transaction"
    },
    "order_already_processed": {
        "tool": "cancel_pending_order",
        "message": "Order has already been processed and cannot be modified"
    },
    "warehouse_unavailable": {
        "tool": "exchange_delivered_order_items",
        "message": "Warehouse for your region is temporarily closed"
    },
    "address_validation_failure": {
        "tool": "modify_pending_order_address",
        "message": "Address validation service unavailable - please retry"
    },
}

# Telecom domain disruption mappings
TELECOM_DISRUPTION_TOOL_MAPPING = {
    "billing_system_timeout": {
        "tool": "send_payment_request",
        "message": "Billing system temporarily unavailable - please retry"
    },
    "network_api_delay": {
        "tool": "suspend_line",
        "message": "Network configuration API timeout - retry in a moment"
    },
    "customer_db_lock": {
        "tool": "get_customer_by_phone",
        "message": "Customer database temporarily locked - please retry"
    },
    "service_plan_unavailable": {
        "tool": "get_details_by_id",
        "message": "Service plan is no longer available for new customers"
    },
    "account_suspended": {
        "tool": "refuel_data",
        "message": "Account is suspended due to non-payment - clear outstanding balance first"
    },
    "line_not_found": {
        "tool": "suspend_line",
        "message": "Line not found for customer"
    },
    "line_already_suspended": {
        "tool": "suspend_line",
        "message": "Line is already suspended - cannot suspend again"
    },
    "line_not_suspended": {
        "tool": "resume_line",
        "message": "Line is not suspended - cannot resume"
    },
    "payment_already_pending": {
        "tool": "send_payment_request",
        "message": "A bill is already awaiting payment for this customer"
    },
    "data_refuel_limit_exceeded": {
        "tool": "refuel_data",
        "message": "Data refuel limit exceeded for this billing cycle"
    },
    "roaming_state_conflict": {
        "tool": "enable_roaming",
        "message": "Roaming was already enabled"
    },
}

# Mock domain disruption mappings
MOCK_DISRUPTION_TOOL_MAPPING = {
    "database_lock": {
        "tool": None,  # Any tool
        "message": "Database temporarily locked - please retry"
    },
    "api_timeout": {
        "tool": None,  # Any tool
        "message": "API request timeout - please retry"
    },
    "connection_pool_exhausted": {
        "tool": None,  # Any tool
        "message": "Connection pool exhausted - retry shortly"
    },
    "task_not_found": {
        "tool": "update_task_status",
        "message": "Task not found in system"
    },
    "user_not_found": {
        "tool": "create_task",
        "message": "User not found"
    },
    "user_not_authorized": {
        "tool": "create_task",
        "message": "User not authorized to perform this action"
    },
    "invalid_status_transition": {
        "tool": "update_task_status",
        "message": "Cannot transition task to requested status"
    },
    "task_limit_exceeded": {
        "tool": "create_task",
        "message": "User has reached maximum task limit"
    },
    "duplicate_task_title": {
        "tool": "create_task",
        "message": "Task with this title already exists for this user"
    },
}

# Central mapping of all domains
DISRUPTION_MAPPINGS = {
    "airline": AIRLINE_DISRUPTION_TOOL_MAPPING,
    "retail": RETAIL_DISRUPTION_TOOL_MAPPING,
    "telecom": TELECOM_DISRUPTION_TOOL_MAPPING,
    "mock": MOCK_DISRUPTION_TOOL_MAPPING,
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
            instance._current_domain: Optional[str] = None  # Track current domain
            sys._tau2_disruption_engine = instance
        return sys._tau2_disruption_engine

    def reset(self) -> None:
        """Reset the disruption engine for a new task."""
        self._disruptions = []
        self._action_count = 0
        self._triggered_disruptions = []
        self._executed_actions = []
        self._transient_attempt_counts = {}
        # Don't reset domain - it persists across tasks
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
    
    def set_domain(self, domain: str) -> None:
        """
        Set the active domain for disruption scenarios.
        
        Args:
            domain: Domain name (airline, retail, telecom, mock)
        """
        if domain not in DISRUPTION_MAPPINGS:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Supported domains: {list(DISRUPTION_MAPPINGS.keys())}"
            )
        self._current_domain = domain
        logger.info(f"DisruptionEngine domain set to: {domain}")

    def configure(self, disruption_scenarios: List[Dict[str, Any]], domain: str = None) -> None:
        """
        Configure disruptions from a list of scenario definitions.

        Args:
            disruption_scenarios: List of disruption configurations.
            domain: Optional domain name. If not provided, uses current domain.
        """
        if domain:
            self.set_domain(domain)
        
        if not self._current_domain:
            # Default to airline for backward compatibility
            self._current_domain = "airline"
            logger.warning("No domain set, defaulting to 'airline'")
        
        self._disruptions = []
        
        for scenario in disruption_scenarios:
            config = self._map_scenario_to_config(scenario)
            if config:
                self._disruptions.append(config)
                logger.info(
                    f"Configured disruption: {config.disruption_type} "
                    f"-> {config.affected_tool} (domain: {self._current_domain})"
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
        
        # Get mapping for current domain
        domain_mapping = DISRUPTION_MAPPINGS.get(self._current_domain, {})
        
        # Look up in domain-specific mapping
        if dtype_lower in domain_mapping:
            mapping = domain_mapping[dtype_lower]
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
        
        logger.warning(
            f"Unknown disruption type '{dtype_lower}' for domain '{self._current_domain}'"
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

# Predefined disruption scenarios for retail domain testing
RETAIL_DISRUPTION_SCENARIOS = {
    "payment_gateway_timeout": {
        "type": "payment_gateway_timeout",
        "trigger_after": 2,
        "persistent": False,
        "retries_until_success": 2,
        "message": "Payment gateway timeout - please retry",
    },
    "inventory_sync_delay": {
        "type": "inventory_sync_delay",
        "trigger_after": 3,
        "persistent": False,
        "retries_until_success": 1,
        "message": "Inventory system temporarily unavailable",
    },
    "product_out_of_stock": {
        "type": "product_out_of_stock",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "modify_pending_order_items",
        "message": "Product variant is currently out of stock",
    },
    "order_already_processed": {
        "type": "order_already_processed",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "cancel_pending_order",
        "message": "Order has already been processed",
    },
    "database_lock": {
        "type": "database_lock",
        "trigger_after": 2,
        "persistent": False,
        "retries_until_success": 1,
        "message": "Database temporarily locked",
    },
}

# Predefined disruption scenarios for telecom domain testing
TELECOM_DISRUPTION_SCENARIOS = {
    "billing_system_timeout": {
        "type": "billing_system_timeout",
        "trigger_after": 2,
        "persistent": False,
        "retries_until_success": 2,
        "message": "Billing system temporarily unavailable",
    },
    "network_api_delay": {
        "type": "network_api_delay",
        "trigger_after": 3,
        "persistent": False,
        "retries_until_success": 1,
        "message": "Network configuration API timeout",
    },
    "account_suspended": {
        "type": "account_suspended",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "refuel_data",
        "message": "Account is suspended due to non-payment",
    },
    "line_already_suspended": {
        "type": "line_already_suspended",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "suspend_line",
        "message": "Line is already suspended",
    },
    "payment_already_pending": {
        "type": "payment_already_pending",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "send_payment_request",
        "message": "A bill is already awaiting payment",
    },
}

# Predefined disruption scenarios for mock domain testing
MOCK_DISRUPTION_SCENARIOS = {
    "database_lock": {
        "type": "database_lock",
        "trigger_after": 2,
        "persistent": False,
        "retries_until_success": 1,
        "message": "Database temporarily locked",
    },
    "api_timeout": {
        "type": "api_timeout",
        "trigger_after": 3,
        "persistent": False,
        "retries_until_success": 2,
        "message": "API request timeout",
    },
    "task_not_found": {
        "type": "task_not_found",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "update_task_status",
        "message": "Task not found in system",
    },
    "user_not_found": {
        "type": "user_not_found",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "create_task",
        "message": "User not found",
    },
    "user_not_authorized": {
        "type": "user_not_authorized",
        "trigger_after": 1,
        "persistent": True,
        "affected_tool": "create_task",
        "message": "User not authorized to perform this action",
    },
}

# Central registry of all domain scenarios
DISRUPTION_SCENARIOS = {
    "airline": AIRLINE_DISRUPTION_SCENARIOS,
    "retail": RETAIL_DISRUPTION_SCENARIOS,
    "telecom": TELECOM_DISRUPTION_SCENARIOS,
    "mock": MOCK_DISRUPTION_SCENARIOS,
}
