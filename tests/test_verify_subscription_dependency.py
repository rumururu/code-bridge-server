"""Tests for the ``verify_subscription`` FastAPI dependency."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from entitlement.entitlement_service import EntitlementResult  # noqa: E402
from routes.deps import verify_subscription  # noqa: E402


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/gated", dependencies=[Depends(verify_subscription)])
    def gated_endpoint():
        return {"ok": True}

    return app


def test_non_local_request_without_headers_returns_402():
    app = _build_app()
    client = TestClient(app)

    # ``TestClient`` defaults to ``testclient``/``testserver`` host which is
    # not 127.0.0.1, so we explicitly mark it as not loopback by passing a
    # ``CF-Ray`` header (tunnel marker) so ``is_request_from_tunnel`` is True.
    response = client.get("/gated", headers={"CF-Ray": "test-ray"})

    assert response.status_code == 402
    body = response.json()
    assert body["detail"]["error"] == "subscription_required"
    assert "reason" in body["detail"]


def test_local_request_without_headers_returns_200():
    app = _build_app()
    # Default TestClient client host is ``testclient`` (not localhost).
    # We patch the localhost check so the request is treated as loopback.
    with patch("routes.deps.is_localhost_request", return_value=True):
        client = TestClient(app)
        response = client.get("/gated")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_non_local_request_with_active_entitlement_returns_200():
    app = _build_app()

    fake_result = EntitlementResult(
        active=True, reason="revenuecat_active", source="revenuecat"
    )

    class FakeService:
        def is_entitled(self, app_user_id):  # noqa: D401, ARG002
            return fake_result

    with patch("routes.deps.get_entitlement_service", return_value=FakeService()):
        client = TestClient(app)
        response = client.get(
            "/gated",
            headers={
                "X-App-User-Id": "user-abc",
                "X-Entitlement-Token": "tok-xyz",
                "CF-Ray": "test-ray",  # mark as tunnel so loopback bypass doesn't fire
            },
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_non_local_request_with_inactive_entitlement_returns_402():
    app = _build_app()

    fake_result = EntitlementResult(
        active=False, reason="no_subscription", source="revenuecat"
    )

    class FakeService:
        def is_entitled(self, app_user_id):  # noqa: ARG002
            return fake_result

    with patch("routes.deps.get_entitlement_service", return_value=FakeService()):
        client = TestClient(app)
        response = client.get(
            "/gated",
            headers={
                "X-App-User-Id": "user-abc",
                "CF-Ray": "test-ray",
            },
        )

    assert response.status_code == 402
    assert response.json()["detail"]["reason"] == "no_subscription"


def test_entitlement_token_is_accepted_but_not_validated():
    """The ``X-Entitlement-Token`` header is reserved for future use; we
    must accept any value (even garbage) without failing the request when
    the underlying entitlement check passes."""

    app = _build_app()
    fake_result = EntitlementResult(
        active=True, reason="revenuecat_active", source="revenuecat"
    )

    class FakeService:
        def is_entitled(self, app_user_id):  # noqa: ARG002
            return fake_result

    with patch("routes.deps.get_entitlement_service", return_value=FakeService()):
        client = TestClient(app)
        response = client.get(
            "/gated",
            headers={
                "X-App-User-Id": "user-abc",
                "X-Entitlement-Token": "literally-anything",
                "CF-Ray": "test-ray",
            },
        )

    assert response.status_code == 200
