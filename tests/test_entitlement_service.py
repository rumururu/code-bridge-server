"""Tests for the server-side EntitlementService."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from entitlement.entitlement_service import (  # noqa: E402
    ENV_OVERRIDE,
    ENV_OVERRIDE_VALUE,
    ENV_REVENUECAT_KEY,
    EntitlementService,
)


def _build_service(env: dict, client_factory=None, now_fn=None) -> EntitlementService:
    if client_factory is None:
        client_factory = _client_factory(_mock_response(status_code=404, payload={}))
    return EntitlementService(
        env=env,
        http_client_factory=client_factory,
        now_fn=now_fn,
    )


def _mock_response(*, status_code: int, payload=None, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    response.text = text
    return response


def _client_factory(*responses):
    """Return a factory that yields a mock client whose ``.get`` returns
    successive ``responses``."""

    iter_responses = iter(responses)

    @contextmanager
    def _factory():
        client = MagicMock()

        def _get(*args, **kwargs):
            try:
                return next(iter_responses)
            except StopIteration as exc:  # pragma: no cover - test bug
                raise AssertionError("ran out of mock HTTP responses") from exc

        client.get.side_effect = _get
        try:
            yield client
        finally:
            pass

    return _factory


def _raising_factory(exc: Exception):
    @contextmanager
    def _factory():
        client = MagicMock()
        client.get.side_effect = exc
        try:
            yield client
        finally:
            pass

    return _factory


def test_env_override_returns_active():
    service = _build_service(env={ENV_OVERRIDE: ENV_OVERRIDE_VALUE})

    result = service.is_entitled(app_user_id=None)

    assert result.active is True
    assert result.source == "override"
    assert result.reason == "override"


def test_no_env_and_no_api_key_returns_anonymous():
    service = _build_service(env={})

    result = service.is_entitled(app_user_id=None)

    assert result.active is False
    assert result.reason == "anonymous"


def test_no_user_with_api_key_still_anonymous():
    service = _build_service(env={ENV_REVENUECAT_KEY: "sk_test"})

    result = service.is_entitled(app_user_id=None)

    assert result.active is False
    assert result.reason == "anonymous"


def test_revenuecat_active_entitlement_returns_active():
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    payload = {
        "subscriber": {
            "entitlements": {
                "code_bridge_pro": {"expires_date": future},
            }
        }
    }
    service = _build_service(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        client_factory=_client_factory(_mock_response(status_code=200, payload=payload)),
    )

    result = service.is_entitled(app_user_id="user-123")

    assert result.active is True
    assert result.source == "revenuecat"
    assert result.reason == "revenuecat_active"


def test_revenuecat_expired_entitlement_returns_inactive():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    payload = {
        "subscriber": {
            "entitlements": {
                "code_bridge_pro": {"expires_date": past},
            }
        }
    }
    service = _build_service(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        client_factory=_client_factory(_mock_response(status_code=200, payload=payload)),
    )

    result = service.is_entitled(app_user_id="user-123")

    assert result.active is False
    assert result.reason == "no_subscription"


def test_revenuecat_lifetime_entitlement_returns_active():
    payload = {
        "subscriber": {
            "entitlements": {
                "code_bridge_pro": {"expires_date": None},
            }
        }
    }
    service = _build_service(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        client_factory=_client_factory(_mock_response(status_code=200, payload=payload)),
    )

    result = service.is_entitled(app_user_id="lifetime-user")

    assert result.active is True
    assert result.reason == "revenuecat_active"


def test_revenuecat_no_matching_entitlement_returns_inactive():
    payload = {
        "subscriber": {
            "entitlements": {
                "some_other_entitlement": {"expires_date": None},
            }
        }
    }
    service = _build_service(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        client_factory=_client_factory(_mock_response(status_code=200, payload=payload)),
    )

    result = service.is_entitled(app_user_id="user-other")

    assert result.active is False
    assert result.reason == "no_subscription"


def test_revenuecat_404_returns_no_subscription():
    service = _build_service(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        client_factory=_client_factory(_mock_response(status_code=404, payload={})),
    )

    result = service.is_entitled(app_user_id="unknown-user")

    assert result.active is False
    assert result.reason == "no_subscription"
    assert result.source == "revenuecat"


def test_revenuecat_503_serves_stale_cache(caplog):
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    ok_payload = {
        "subscriber": {
            "entitlements": {
                "code_bridge_pro": {"expires_date": future},
            }
        }
    }

    # Times in order of ``self._now()`` invocations:
    # 1) first call's cache-age check  -> t=0 (no entry yet)
    # 2) first call's cache write       -> t=0 (timestamp stored)
    # 3) second call's cache-age check  -> t=600 (stale, 10 min later)
    now_ticks = iter([0.0, 0.0, 600.0])
    now_fn = lambda: next(now_ticks)  # noqa: E731

    service = EntitlementService(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        http_client_factory=_client_factory(
            _mock_response(status_code=200, payload=ok_payload),
            _mock_response(status_code=503, text="boom"),
        ),
        now_fn=now_fn,
    )

    # 1) Prime the cache with an active result.
    first = service.is_entitled(app_user_id="user-x")
    assert first.active is True

    # 2) RevenueCat goes down → expect stale cache + a warning log.
    with caplog.at_level(logging.WARNING, logger="entitlement.entitlement_service"):
        second = service.is_entitled(app_user_id="user-x")

    assert second.active is True
    assert second.source == "cache"
    assert second.reason == "stale_cache"
    assert any(
        "stale cache" in record.getMessage() or "RevenueCat lookup failed" in record.getMessage()
        for record in caplog.records
    )


def test_revenuecat_503_without_cache_returns_error():
    service = _build_service(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        client_factory=_client_factory(_mock_response(status_code=503, text="boom")),
    )

    result = service.is_entitled(app_user_id="no-cache-user")

    assert result.active is False
    assert result.reason == "revenuecat_error"


def test_revenuecat_network_failure_serves_stale_cache():
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    ok_payload = {
        "subscriber": {
            "entitlements": {
                "code_bridge_pro": {"expires_date": future},
            }
        }
    }

    # See ``test_revenuecat_503_serves_stale_cache`` for tick ordering.
    now_ticks = iter([0.0, 0.0, 1000.0])
    now_fn = lambda: next(now_ticks)  # noqa: E731

    # First call: success. Second call: raise.
    call_state = {"n": 0}

    @contextmanager
    def factory():
        client = MagicMock()

        def _get(*args, **kwargs):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return _mock_response(status_code=200, payload=ok_payload)
            raise RuntimeError("network down")

        client.get.side_effect = _get
        yield client

    service = EntitlementService(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        http_client_factory=factory,
        now_fn=now_fn,
    )

    service.is_entitled(app_user_id="user-net")
    second = service.is_entitled(app_user_id="user-net")

    assert second.active is True
    assert second.source == "cache"


def test_fresh_cache_is_reused_without_http_call():
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    ok_payload = {
        "subscriber": {
            "entitlements": {
                "code_bridge_pro": {"expires_date": future},
            }
        }
    }

    # First call: cache-age check (0.0), cache write (0.0).
    # Second call: cache-age check (10.0) — within fresh TTL (5 min).
    now_ticks = iter([0.0, 0.0, 10.0])
    now_fn = lambda: next(now_ticks)  # noqa: E731

    service = EntitlementService(
        env={ENV_REVENUECAT_KEY: "sk_test"},
        http_client_factory=_client_factory(
            _mock_response(status_code=200, payload=ok_payload),
        ),
        now_fn=now_fn,
    )

    first = service.is_entitled(app_user_id="user-cache")
    second = service.is_entitled(app_user_id="user-cache")

    assert first.active is True
    assert second.active is True
    # Only one HTTP response should have been consumed; second hit was cached.
