"""Server-side entitlement checks for Code Bridge.

Code Bridge ships a single paid plan: a $0.99/month subscription that
unlocks every gated API endpoint. The check has three sources, tried in
order:

1. **Local override** (env var ``CODE_BRIDGE_ENTITLEMENT_OVERRIDE=active``):
   used for local dev and for the host user who runs the server on their
   own machine.
2. **RevenueCat REST**: if ``REVENUECAT_SECRET_API_KEY`` is set and the
   caller provides an app user id, look the subscriber up via
   ``GET /v1/subscribers/{app_user_id}`` and check the
   ``code_bridge_pro`` entitlement. Successful lookups are cached for
   5 minutes per app user id.
3. **Stale cache fallback**: if RevenueCat is temporarily unavailable
   (5xx, network errors), cache entries up to 1 hour old are served with
   a ``logger.warning`` so the user is not booted out during an upstream
   blip.

If none of the three give ``active=True``, the service returns
``EntitlementResult(active=False, reason="no_subscription", source="default")``.

Anonymous callers (no ``app_user_id`` header) are denied with
``reason="anonymous"`` unless the local override is set.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Entitlement identifier configured in RevenueCat.
# Must match ``lib/config/revenuecat_config.dart:entitlementId`` — the
# entitlement is registered in the RevenueCat dashboard with this exact
# string (whitespace and capitalization included).
ENTITLEMENT_ID = "Code Bridge Pro"

# Local-dev / host-machine override. Single allowed value: ``active``.
ENV_OVERRIDE = "CODE_BRIDGE_ENTITLEMENT_OVERRIDE"
ENV_OVERRIDE_VALUE = "active"

# RevenueCat REST credentials.
ENV_REVENUECAT_KEY = "REVENUECAT_SECRET_API_KEY"
REVENUECAT_BASE_URL = "https://api.revenuecat.com/v1"

# Cache policy: fresh entries < 5 min, stale entries up to 1 hour.
FRESH_TTL_SECONDS = 5 * 60
STALE_TTL_SECONDS = 60 * 60
REVENUECAT_HTTP_TIMEOUT = 5.0


@dataclass(frozen=True)
class EntitlementResult:
    """Outcome of an entitlement check.

    Attributes:
        active: Whether the subscriber has an active paid entitlement.
        reason: Short machine-readable reason. Examples:
            ``"override"``, ``"revenuecat_active"``, ``"stale_cache"``,
            ``"no_subscription"``, ``"anonymous"``, ``"revenuecat_error"``,
            ``"revenuecat_not_configured"``.
        source: Where the answer came from
            (``"override"``, ``"revenuecat"``, ``"cache"``, ``"default"``).
    """

    active: bool
    reason: str
    source: str

    @property
    def is_terminal_for_cache(self) -> bool:
        """True when the result should be cached as authoritative.

        Stale-cache fallbacks must not be re-cached (otherwise their
        ``source="cache"`` would propagate forever); same for results
        that already represent a fallback layer.
        """
        return self.source not in {"cache", "default"}


@dataclass
class _CacheEntry:
    result: EntitlementResult
    timestamp: float  # monotonic seconds


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    """Parse a RevenueCat ISO-8601 timestamp into UTC datetime."""
    if not value:
        return None
    try:
        # RevenueCat returns either ``...Z`` or ``...+00:00``; accept both.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        logger.warning("entitlement: could not parse expires_date %r", value)
        return None


class EntitlementService:
    """Resolve entitlement state with override/REST/cache layering.

    The service is intentionally process-local: a single in-memory dict
    backs the cache so we do not introduce a Redis dependency for what is
    effectively a per-host server.
    """

    def __init__(
        self,
        *,
        env: Optional[dict] = None,
        http_client_factory=None,
        now_fn=None,
    ) -> None:
        self._env = env if env is not None else os.environ
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_lock = Lock()
        self._http_client_factory = http_client_factory or self._default_client_factory
        self._now = now_fn or time.monotonic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_entitled(self, app_user_id: Optional[str]) -> EntitlementResult:
        """Return an :class:`EntitlementResult` for ``app_user_id``."""

        # 1) Local override always wins.
        if self._env.get(ENV_OVERRIDE) == ENV_OVERRIDE_VALUE:
            return EntitlementResult(active=True, reason="override", source="override")

        api_key = self._env.get(ENV_REVENUECAT_KEY)
        normalized_user_id = (app_user_id or "").strip() or None

        # 2) If we have no way to identify the user, deny.
        if normalized_user_id is None:
            return EntitlementResult(
                active=False, reason="anonymous", source="default"
            )

        # 3) RevenueCat REST path.
        if api_key:
            return self._resolve_via_revenuecat(normalized_user_id, api_key)

        # 4) RevenueCat key missing AND no override.
        return EntitlementResult(
            active=False,
            reason="revenuecat_not_configured",
            source="default",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_via_revenuecat(
        self, app_user_id: str, api_key: str
    ) -> EntitlementResult:
        """Hit RevenueCat REST, with cache + stale fallback."""

        now = self._now()
        cached = self._read_cache(app_user_id)
        if cached is not None and (now - cached.timestamp) < FRESH_TTL_SECONDS:
            return cached.result

        try:
            result = self._fetch_revenuecat(app_user_id, api_key)
        except Exception as exc:  # network/HTTP errors
            logger.warning(
                "entitlement: RevenueCat lookup failed for %s: %s",
                app_user_id,
                exc,
            )
            if cached is not None and (now - cached.timestamp) < STALE_TTL_SECONDS:
                logger.warning(
                    "entitlement: serving stale cache for %s (age=%.0fs)",
                    app_user_id,
                    now - cached.timestamp,
                )
                stale = EntitlementResult(
                    active=cached.result.active,
                    reason="stale_cache",
                    source="cache",
                )
                return stale
            return EntitlementResult(
                active=False,
                reason="revenuecat_error",
                source="default",
            )

        self._write_cache(app_user_id, result)
        return result

    def _fetch_revenuecat(self, app_user_id: str, api_key: str) -> EntitlementResult:
        """Call ``GET /v1/subscribers/{app_user_id}`` and parse the result."""

        url = f"{REVENUECAT_BASE_URL}/subscribers/{app_user_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        with self._http_client_factory() as client:
            response = client.get(url, headers=headers, timeout=REVENUECAT_HTTP_TIMEOUT)

        if response.status_code >= 500:
            # Raise so caller can fall back to stale cache.
            raise RuntimeError(
                f"RevenueCat returned {response.status_code}: {response.text[:200]}"
            )
        if response.status_code == 404:
            return EntitlementResult(
                active=False,
                reason="no_subscription",
                source="revenuecat",
            )
        if response.status_code >= 400:
            # 4xx other than 404: treat as denial but log for diagnosis.
            logger.warning(
                "entitlement: RevenueCat client error %s for %s: %s",
                response.status_code,
                app_user_id,
                response.text[:200],
            )
            return EntitlementResult(
                active=False,
                reason="revenuecat_client_error",
                source="revenuecat",
            )

        payload = response.json()
        subscriber = payload.get("subscriber") or {}
        entitlements = subscriber.get("entitlements") or {}
        entitlement = entitlements.get(ENTITLEMENT_ID)
        if not entitlement:
            return EntitlementResult(
                active=False,
                reason="no_subscription",
                source="revenuecat",
            )

        expires_raw = entitlement.get("expires_date")
        expires_at = _parse_iso8601(expires_raw)
        if expires_at is None:
            # Null/missing expires_date == lifetime entitlement.
            return EntitlementResult(
                active=True,
                reason="revenuecat_active",
                source="revenuecat",
            )

        if expires_at > datetime.now(timezone.utc):
            return EntitlementResult(
                active=True,
                reason="revenuecat_active",
                source="revenuecat",
            )

        return EntitlementResult(
            active=False,
            reason="no_subscription",
            source="revenuecat",
        )

    def _read_cache(self, app_user_id: str) -> Optional[_CacheEntry]:
        with self._cache_lock:
            return self._cache.get(app_user_id)

    def _write_cache(self, app_user_id: str, result: EntitlementResult) -> None:
        with self._cache_lock:
            self._cache[app_user_id] = _CacheEntry(
                result=result, timestamp=self._now()
            )

    def clear_cache(self) -> None:
        """Drop all cached entries (used by tests)."""
        with self._cache_lock:
            self._cache.clear()

    @staticmethod
    def _default_client_factory():
        return httpx.Client()


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------

_service_singleton: Optional[EntitlementService] = None
_singleton_lock = Lock()


def get_entitlement_service() -> EntitlementService:
    """Return the process-wide :class:`EntitlementService` instance."""
    global _service_singleton
    if _service_singleton is None:
        with _singleton_lock:
            if _service_singleton is None:
                _service_singleton = EntitlementService()
    return _service_singleton


def reset_entitlement_service_for_tests() -> None:
    """Reset the singleton; tests only."""
    global _service_singleton
    with _singleton_lock:
        _service_singleton = None
