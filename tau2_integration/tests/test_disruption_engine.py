"""
Tests for the disruption engine.
"""

import pytest
import sys

from tau2_integration.disruption_engine import (
    DisruptionEngine,
    DisruptionConfig,
    DisruptionTrigger,
    AIRLINE_DISRUPTION_SCENARIOS,
    get_disruption_engine,
)


class TestDisruptionEngine:
    """Tests for DisruptionEngine."""
    
    def setup_method(self):
        """Reset disruption engine before each test."""
        # Clear singleton
        if hasattr(sys, "_tau2_disruption_engine"):
            delattr(sys, "_tau2_disruption_engine")
        self.engine = get_disruption_engine()
        self.engine.reset()
    
    def test_singleton_pattern(self):
        """Test that DisruptionEngine is a singleton."""
        engine1 = get_disruption_engine()
        engine2 = get_disruption_engine()
        assert engine1 is engine2
    
    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        self.engine._action_count = 10
        self.engine._triggered_disruptions.append({"test": "data"})
        
        self.engine.reset()
        
        assert self.engine._action_count == 0
        assert len(self.engine._triggered_disruptions) == 0
    
    def test_enable_disable(self):
        """Test enable/disable functionality."""
        assert self.engine.is_enabled()
        
        self.engine.disable()
        assert not self.engine.is_enabled()
        
        self.engine.enable()
        assert self.engine.is_enabled()
    
    def test_no_disruption_when_disabled(self):
        """Test that no disruptions occur when engine is disabled."""
        self.engine.configure([{
            "type": "system_error",
            "trigger_after": 1,
        }])
        self.engine.disable()
        
        result = self.engine.check_disruption("any_tool", {})
        assert result is None
    
    def test_configure_from_scenario(self):
        """Test configuring from predefined scenarios."""
        scenarios = [AIRLINE_DISRUPTION_SCENARIOS["payment_transient"]]
        self.engine.configure(scenarios)
        
        assert self.engine.has_disruptions_configured()
    
    def test_trigger_after_n_actions(self):
        """Test that disruption triggers after N actions."""
        self.engine.configure([{
            "type": "flight_unavailable",
            "trigger_after": 3,
            "persistent": True,
            "affected_tool": "book_reservation",
        }])
        
        # First 2 actions should not trigger
        assert self.engine.check_disruption("book_reservation", {}) is None
        assert self.engine.check_disruption("book_reservation", {}) is None
        
        # Third action should trigger
        result = self.engine.check_disruption("book_reservation", {})
        assert result is not None
        assert "available" in result.lower() or "flight" in result.lower()
    
    def test_transient_failure_retry_success(self):
        """Test that transient failures succeed after retries."""
        self.engine.configure([{
            "type": "payment_failure",
            "trigger_after": 1,
            "persistent": False,
            "retries_until_success": 2,
        }])
        
        # First attempt fails
        result1 = self.engine.check_disruption("book_reservation", {})
        assert result1 is not None
        
        # Second attempt fails
        result2 = self.engine.check_disruption("book_reservation", {})
        assert result2 is not None
        
        # Third attempt succeeds
        result3 = self.engine.check_disruption("book_reservation", {})
        assert result3 is None
    
    def test_persistent_failure(self):
        """Test that persistent failures keep failing."""
        self.engine.configure([{
            "type": "flight_unavailable",
            "trigger_after": 1,
            "persistent": True,
            "affected_tool": "book_reservation",
        }])
        
        # Should fail multiple times
        result1 = self.engine.check_disruption("book_reservation", {})
        assert result1 is not None
        
        result2 = self.engine.check_disruption("book_reservation", {})
        assert result2 is not None
        
        result3 = self.engine.check_disruption("book_reservation", {})
        assert result3 is not None
    
    def test_tool_specific_disruption(self):
        """Test that disruption only affects specific tool."""
        self.engine.configure([{
            "type": "flight_unavailable",
            "trigger_after": 1,
            "persistent": True,
            "affected_tool": "book_reservation",
        }])
        
        # Other tools should not be affected
        assert self.engine.check_disruption("get_user_details", {}) is None
        assert self.engine.check_disruption("search_direct_flight", {}) is None
        
        # Target tool should be affected
        result = self.engine.check_disruption("book_reservation", {})
        assert result is not None
    
    def test_action_recording(self):
        """Test that actions are recorded."""
        self.engine.record_action("test_tool", {"arg": "value"}, {"result": "ok"})
        
        actions = self.engine.get_executed_actions()
        assert len(actions) == 1
        assert actions[0]["tool"] == "test_tool"
    
    def test_get_triggered_disruptions(self):
        """Test getting triggered disruptions."""
        self.engine.configure([{
            "type": "system_error",
            "trigger_after": 1,
            "persistent": False,
            "retries_until_success": 1,
        }])
        
        # Trigger a disruption
        self.engine.check_disruption("any_tool", {})
        
        triggered = self.engine.get_triggered_disruptions()
        assert len(triggered) == 1
        assert triggered[0]["disruption"] == "system_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
