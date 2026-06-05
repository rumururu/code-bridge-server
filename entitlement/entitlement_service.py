"""Server-side entitlement checks for Code Bridge.

Code Bridge ships a single paid plan: a $0.99/month subscription that
unlocks every gated API endpoint. The check has four sources, tried in
order:

1. **Local override** (env var ``CODE_BRIDGE_ENTITLEMENT_OVERRIDE=active``):
   used for local dev and for the host user who runs the server on their
   own machine.
2. **Supabase fast-path**: if ``SUPABASE_URL`` and
   ``SUPABASE_SERVICE_ROLE_KEY`` are set, look the subscriber up in the
   ``public.entitlement_state`` table that the ``revenuecat-webhook``
   Supabase Edge Function keeps in sync. This is the canonical
   push-based view and is preferred over the RevenueCat REST round-trip
   because cancellations propagate within seconds instead of waiting for
   the REST cache to expire. See
   ``docs/runbooks/revenuecat_webhook_setup.md``.
3. **RevenueCat REST**: if ``REVENUECAT_SECRET_API_KEY`` is set and the
   caller provides an app user id, look the subscriber up via
   ``GET /v1/subscribers/{app_user_id}`` and check the
   ``code_bridge_pro`` entitlement. Used when the Supabase fast-path is
   not configured, returns an empty row (subscriber just installed and
   the webhook has not fired yet), or is temporarily degraded.
   Successful lookups are cached for 5 minutes per app user id.
4. **Stale cache fallback**: if both Supabase and RevenueCat are
   temporarily unavailable (5xx, network errors), cache entries up to
   1 hour old are served with a ``logger.warning`` so the user is not
   booted out during an upstream blip. Supabase responses are cached
   the same way.

If none of the four give ``active=True``, the service returns
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
ENTITLEMENT_ID = "code_bridge_pro"

# Local-dev / host-machine override. Single allowed value: ``active``.
ENV_OVERRIDE = "CODE_BRIDGE_ENTITLEMENT_OVERRIDE"
ENV_OVERRIDE_VALUE = "active"

# RevenueCat REST credentials.
ENV_REVENUECAT_KEY = "REVENUECAT_SECRET_API_KEY"
REVENUECAT_BASE_URL = "https://api.revenuecat.com/v1"

# Supabase fast-path credentials (populated by the
# ``revenuecat-webhook`` Edge Function — see
# ``docs/runbooks/revenuecat_webhook_setup.md``).
ENV_SUPABASE_URL = "SUPABASE_URL"
ENV_SUPABASE_SERVICE_KEY = "SUPABASE_SERVICE_ROLE_KEY"
SUPABASE_ENTITLEMENT_PATH = "/rest/v1/entitlement_state"

# Cache policy: fresh entries < 5 min, stale entries up to 1 hour.
FRESH_TTL_SECONDS = 5 * 60
STALE_TTL_SECONDS = 60 * 60
REVENUECAT_HTTP_TIMEOUT = 5.0
SUPABASE_HTTP_TIMEOUT = 3.0


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
        supabase_url = (self._env.get(ENV_SUPABASE_URL) or "").strip()
        supabase_key = (self._env.get(ENV_SUPABASE_SERVICE_KEY) or "").strip()
        supabase_configured = bool(supabase_url and supabase_key)
        normalized_user_id = (app_user_id or "").strip() or None

        # 2) If we have no way to identify the user, deny.
        if normalized_user_id is None:
            return EntitlementResult(
                active=False, reason="anonymous", source="default"
            )

        # 3) Supabase fast-path (canonical, push-based view kept in
        #    sync by the revenuecat-webhook Edge Function). We try it
        #    before RevenueCat REST so cancellations propagate
        #    immediately.
        if supabase_configured:
            supabase_result = self._resolve_via_supabase(
                normalized_user_id, supabase_url, supabase_key
            )
            if supabase_result is not None:
                return supabase_result
            # ``None`` => fall through (row missing OR upstream error).

        # 4) RevenueCat REST path.
        if api_key:
            return self._resolve_via_revenuecat(normalized_user_id, api_key)

        # 5) Neither Supabase nor RevenueCat is usable AND no override.
        return EntitlementResult(
            active=False,
            reason="revenuecat_not_configured",
            source="default",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_via_supabase(
        self, app_user_id: str, supabase_url: str, service_key: str
    ) -> Optional[EntitlementResult]:
        """Query ``public.entitlement_state`` for ``app_user_id``.

        Returns:
            * an :class:`EntitlementResult` if Supabase returned a row
              (whether active or expired), in which case the caller
              **must not** fall through to RevenueCat REST;
            * ``None`` if the row does not exist yet (newly installed
              user — let the RevenueCat REST path handle it), or if
              Supabase returned a 5xx / timed out (let RevenueCat REST
              answer, then stale cache, then default).
        """

        now = self._now()
        cache_key = self._supabase_cache_key(app_user_id)
        cached = self._read_cache(cache_key)
        if cached is not None and (now - cached.timestamp) < FRESH_TTL_SECONDS:
            return cached.result

        url = f"{supabase_url.rstrip('/')}{SUPABASE_ENTITLEMENT_PATH}"
        params = {
            "app_user_id": f"eq.{app_user_id}",
            "select": "active,expires_at,period_type,last_event,updated_at",
        }
        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json",
        }

        try:
            with self._http_client_factory() as client:
                response = client.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=SUPABASE_HTTP_TIMEOUT,
                )
        except Exception as exc:  # network/timeout
            logger.warning(
                "entitlement: Supabase fast-path failed for %s, falling back: %s",
                app_user_id,
                exc,
            )
            return self._supabase_stale_or_none(cache_key, now)

        if response.status_code >= 500:
            logger.warning(
                "entitlement: Supabase fast-path 5xx (%s) for %s, falling back",
                response.status_code,
                app_user_id,
            )
            return self._supabase_stale_or_none(cache_key, now)
        if response.status_code >= 400:
            # 4xx: misconfiguration. Log loud and fall through so the
            # user is not stuck while we diagnose.
            logger.warning(
                "entitlement: Supabase fast-path %s for %s: %s",
                response.status_code,
                app_user_id,
                response.text[:200],
            )
            return None

        try:
            rows = response.json()
        except ValueError:
            logger.warning(
                "entitlement: Supabase fast-path returned non-JSON for %s", app_user_id
            )
            return None
        if not isinstance(rows, list) or not rows:
            # Row not yet materialised. Webhook may not have fired yet
            # for this just-purchased user — let RevenueCat REST try.
            return None

        row = rows[0]
        active_raw = row.get("active")
        if not isinstance(active_raw, bool):
            logger.warning(
                "entitlement: Supabase row for %s missing 'active' bool: %r",
                app_user_id,
                row,
            )
            return None

        expires_raw = row.get("expires_at")
        expires_at = _parse_iso8601(expires_raw) if expires_raw else None

        if active_raw and expires_at is not None and expires_at <= datetime.now(
            timezone.utc
        ):
            # Webhook said active but expires_at has already passed —
            # treat as inactive. Defensive in case the webhook hasn't
            # processed the EXPIRATION event yet.
            result = EntitlementResult(
                active=False,
                reason="no_subscription",
                source="supabase",
            )
        elif active_raw:
            result = EntitlementResult(
                active=True,
                reason="supabase_active",
                source="supabase",
            )
        else:
            result = EntitlementResult(
                active=False,
                reason="no_subscription",
                source="supabase",
            )

        self._write_cache(cache_key, result)
        return result

    def _supabase_stale_or_none(
        self, cache_key: str, now: float
    ) -> Optional[EntitlementResult]:
        """Return a stale Supabase cache entry, or ``None`` to fall through."""
        cached = self._read_cache(cache_key)
        if cached is not None and (now - cached.timestamp) < STALE_TTL_SECONDS:
            logger.warning(
                "entitlement: serving stale Supabase cache (age=%.0fs)",
                now - cached.timestamp,
            )
            return EntitlementResult(
                active=cached.result.active,
                reason="stale_cache",
                source="cache",
            )
        return None

    @staticmethod
    def _supabase_cache_key(app_user_id: str) -> str:
        """Namespace Supabase cache entries so they don't collide with
        RevenueCat cache entries for the same user."""
        return f"supabase:{app_user_id}"

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
