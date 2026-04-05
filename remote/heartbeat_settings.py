"""Shared heartbeat interval state for runtime components."""

_heartbeat_interval_minutes: int = 15


def get_heartbeat_interval() -> int:
    """Get current heartbeat interval in minutes."""
    return _heartbeat_interval_minutes


def set_heartbeat_interval(minutes: int) -> int:
    """Set heartbeat interval (5-15 minutes)."""
    global _heartbeat_interval_minutes
    _heartbeat_interval_minutes = max(5, min(15, minutes))
    return _heartbeat_interval_minutes
