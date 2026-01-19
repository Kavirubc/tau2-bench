"""
Wrapped tools with disruption injection for τ²-bench.

Wraps τ²-bench airline tools to inject disruptions during execution
and track actions for potential rollback.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional

from .disruption_engine import get_disruption_engine

logger = logging.getLogger("tau2_integration.wrapped_tools")


class ToolExecutionError(Exception):
    """Exception raised when a tool fails due to disruption."""
    
    def __init__(self, message: str, is_transient: bool = False):
        super().__init__(message)
        self.message = message
        self.is_transient = is_transient


def wrap_tool_with_disruption(
    tool_func: Callable,
    tool_name: str,
    track_for_rollback: bool = True
) -> Callable:
    """
    Wrap a τ²-bench tool with disruption injection.

    Args:
        tool_func: The original tool function to wrap.
        tool_name: Name of the tool for disruption matching.
        track_for_rollback: Whether to track this action for potential rollback.

    Returns:
        Wrapped function that checks for disruptions before executing.
    """
    @functools.wraps(tool_func)
    def wrapped(*args, **kwargs) -> Any:
        engine = get_disruption_engine()
        
        # Check for disruption before execution
        error = engine.check_disruption(tool_name, kwargs)
        if error:
            is_transient = "retry" in error.lower() or "temporary" in error.lower()
            logger.warning(f"Tool {tool_name} disrupted: {error}")
            raise ToolExecutionError(error, is_transient=is_transient)
        
        # Execute original tool
        try:
            result = tool_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise
        
        # Track for potential rollback
        if track_for_rollback:
            engine.record_action(tool_name, kwargs, result)
        
        return result
    
    return wrapped


def create_wrapped_toolkit(toolkit_instance: Any) -> Dict[str, Callable]:
    """
    Create wrapped versions of all tools in a τ²-bench toolkit.

    Args:
        toolkit_instance: Instance of a τ²-bench toolkit (e.g., AirlineTools).

    Returns:
        Dictionary mapping tool names to wrapped functions.
    """
    wrapped_tools = {}
    
    # Get all tool methods from the toolkit
    for name in dir(toolkit_instance):
        if name.startswith("_"):
            continue
            
        attr = getattr(toolkit_instance, name)
        if callable(attr) and hasattr(attr, "_is_tool"):
            # Determine if this is a write operation (needs rollback tracking)
            tool_type = getattr(attr, "_tool_type", None)
            track_for_rollback = tool_type == "WRITE" if tool_type else False
            
            wrapped_tools[name] = wrap_tool_with_disruption(
                attr,
                name,
                track_for_rollback=track_for_rollback
            )
            logger.debug(f"Wrapped tool: {name} (track_rollback={track_for_rollback})")
    
    return wrapped_tools


# Compensation mapping for airline tools
AIRLINE_COMPENSATION_MAPPING = {
    "book_reservation": "cancel_reservation",
    "update_reservation_flights": None,  # Complex - needs custom handling
    "update_reservation_baggages": None,  # Complex - needs custom handling
    "update_reservation_passengers": None,  # Cannot revert passenger info
    "cancel_reservation": None,  # Cannot un-cancel
    "send_certificate": None,  # Cannot revoke certificate
}


def get_compensation_action(tool_name: str) -> Optional[str]:
    """
    Get the compensation action for a given tool.

    Args:
        tool_name: Name of the tool to compensate.

    Returns:
        Name of the compensation tool, or None if no compensation exists.
    """
    return AIRLINE_COMPENSATION_MAPPING.get(tool_name)


def build_compensation_args(
    tool_name: str,
    original_args: Dict[str, Any],
    result: Any
) -> Optional[Dict[str, Any]]:
    """
    Build arguments for a compensation action.

    Args:
        tool_name: Name of the original tool.
        original_args: Arguments passed to the original tool.
        result: Result from the original tool execution.

    Returns:
        Arguments for the compensation tool, or None if not applicable.
    """
    if tool_name == "book_reservation":
        # cancel_reservation needs the reservation_id from the result
        if hasattr(result, "reservation_id"):
            return {"reservation_id": result.reservation_id}
        elif isinstance(result, dict) and "reservation_id" in result:
            return {"reservation_id": result["reservation_id"]}
    
    return None
