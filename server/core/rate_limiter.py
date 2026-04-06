"""Rate limiting implementation for pairing and API access."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Rate limiting constants
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_BLOCK_SECONDS = 300


@dataclass
class RateLimitEntry:
    """Track rate limit state for a single client."""

    attempts: int = 0
    first_attempt_at: float = 0.0
    locked_until: float = 0.0

    def is_locked(self, now: float) -> bool:
        """Check if this entry is currently blocked."""
        return self.locked_until > now

    def remaining_lockout_seconds(self, now: float) -> int:
        """Return remaining block time in seconds."""
        if not self.is_locked(now):
            return 0
        return int(self.locked_until - now)


class RateLimiter:
    """IP-based rate limiter for pairing code verification.

    Tracks failed attempts per IP and enforces lockouts after too many failures.
    """

    def __init__(
        self,
        max_attempts: int = RATE_LIMIT_MAX_ATTEMPTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
        lockout_seconds: int = RATE_LIMIT_BLOCK_SECONDS,
    ):
        """Initialize rate limiter.

        Args:
            max_attempts: Maximum attempts allowed in the window.
            window_seconds: Time window for counting attempts.
            lockout_seconds: Duration to block after exceeding limit.
        """
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._entries: dict[str, RateLimitEntry] = {}

    def _cleanup_expired(self, now: float) -> None:
        """Remove stale entries that are no longer locked and outside window."""
        cutoff = now - self.window_seconds - self.lockout_seconds
        expired = [
            ip
            for ip, entry in self._entries.items()
            if entry.first_attempt_at < cutoff and not entry.is_locked(now)
        ]
        for ip in expired:
            del self._entries[ip]

    def check_rate_limit(self, client_ip: str) -> tuple[bool, int]:
        """Check if client IP is rate limited.

        Args:
            client_ip: Client IP address.

        Returns:
            Tuple of (is_allowed, remaining_lockout_seconds).
            - is_allowed: True if request should be allowed.
            - remaining_lockout_seconds: Seconds until lockout expires (0 if not locked).
        """
        now = time.time()
        self._cleanup_expired(now)

        entry = self._entries.get(client_ip)
        if entry is None:
            return (True, 0)

        if entry.is_locked(now):
            return (False, entry.remaining_lockout_seconds(now))

        # Reset if window expired
        if now - entry.first_attempt_at > self.window_seconds:
            entry.attempts = 0
            entry.first_attempt_at = 0.0

        return (True, 0)

    def record_attempt(self, client_ip: str, success: bool) -> None:
        """Record an attempt and update rate limit state.

        Args:
            client_ip: Client IP address.
            success: Whether the attempt was successful.
        """
        now = time.time()

        if success:
            # Clear state on successful attempt
            if client_ip in self._entries:
                del self._entries[client_ip]
            return

        # Record failed attempt
        entry = self._entries.get(client_ip)
        if entry is None:
            entry = RateLimitEntry()
            self._entries[client_ip] = entry

        # Reset window if expired
        if entry.first_attempt_at == 0 or now - entry.first_attempt_at > self.window_seconds:
            entry.attempts = 0
            entry.first_attempt_at = now

        entry.attempts += 1

        # Trigger lockout if max attempts exceeded
        if entry.attempts >= self.max_attempts:
            entry.locked_until = now + self.lockout_seconds
            logger.warning(
                "rate_limit: IP %s locked for %ds after %d failed attempts",
                client_ip,
                self.lockout_seconds,
                entry.attempts,
            )

    def get_status(self, client_ip: str) -> dict[str, Any]:
        """Get current rate limit status for an IP.

        Args:
            client_ip: Client IP address.

        Returns:
            Dictionary with attempts, max_attempts, is_locked, remaining_seconds.
        """
        now = time.time()
        entry = self._entries.get(client_ip)

        if entry is None:
            return {
                "attempts": 0,
                "max_attempts": self.max_attempts,
                "is_locked": False,
                "remaining_seconds": 0,
            }

        return {
            "attempts": entry.attempts,
            "max_attempts": self.max_attempts,
            "is_locked": entry.is_locked(now),
            "remaining_seconds": entry.remaining_lockout_seconds(now),
        }

    def reset(self, client_ip: str) -> None:
        """Reset rate limit state for a client.

        Args:
            client_ip: Client identifier.
        """
        if client_ip in self._entries:
            del self._entries[client_ip]

    def cleanup_expired(self) -> int:
        """Remove expired entries to free memory.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        count = len(self._entries)
        self._cleanup_expired(now)
        return count - len(self._entries)
