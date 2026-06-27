"""Integration tests for ``/api/secrets`` routes.

Covers the contract demanded by ``TASK_001_CLAUDE.md`` § 6:
  1. ``GET`` on an empty store returns ``{"items": []}``.
  2. ``POST`` accepts ``{"name", "value"}`` and returns ``has_value: true``.
  3. After ``POST``, ``GET`` shows the name but **no** value field.
  4. ``DELETE`` returns 200 when key exists, 404 when missing.
  5. ``POST`` with an invalid (lowercase) name returns 422 from pydantic.
  6. (extra) ``POST`` with a newline in value returns 400 from store layer.

Test isolation: ``secret_store.ENV_FILE`` is redirected to a temp file
per test so the developer's real ``~/.code-bridge/.env`` is never read
or written. ``verify_api_key`` is overridden so no Firebase / pairing
state is required.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes import secrets as secrets_routes  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402
from secret_store_pkg import secret_store  # noqa: E402


class SecretsRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._tmp_env = Path(self._tmp.name) / "code-bridge" / ".env"
        self._original_env_file = secret_store.ENV_FILE
        secret_store.ENV_FILE = self._tmp_env

        # Snapshot env so leaked keys from one test don't bleed into
        # another. (Store does ``os.environ[key] = value`` on upsert.)
        self._env_snapshot = dict(os.environ)

        app = FastAPI()
        app.include_router(secrets_routes.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        secret_store.ENV_FILE = self._original_env_file
        for key in ("NAVER_ID", "NAVER_PW", "FOO_KEY"):
            if key in os.environ and key not in self._env_snapshot:
                del os.environ[key]
            elif key in self._env_snapshot:
                os.environ[key] = self._env_snapshot[key]
        self._tmp.cleanup()

    # ------------------------------------------------------------------

    def test_get_empty_returns_no_items(self) -> None:
        resp = self.client.get("/api/secrets")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"items": []})

    def test_post_upserts_and_returns_has_value_true(self) -> None:
        resp = self.client.post(
            "/api/secrets",
            json={"name": "NAVER_ID", "value": "shh-not-a-secret"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body, {"name": "NAVER_ID", "has_value": True})

        # File state side effect:
        self.assertTrue(self._tmp_env.is_file())
        self.assertIn("NAVER_ID=shh-not-a-secret", self._tmp_env.read_text("utf-8"))
        # Environ propagation:
        self.assertEqual(os.environ["NAVER_ID"], "shh-not-a-secret")

    def test_get_does_not_leak_values(self) -> None:
        # Seed two secrets with distinct values so we can check both.
        self.client.post(
            "/api/secrets", json={"name": "NAVER_ID", "value": "id-secret"}
        )
        self.client.post(
            "/api/secrets", json={"name": "NAVER_PW", "value": "pw-secret"}
        )

        resp = self.client.get("/api/secrets")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(
            body,
            {
                "items": [
                    {"name": "NAVER_ID", "has_value": True},
                    {"name": "NAVER_PW", "has_value": True},
                ]
            },
        )
        # Crucial: no "value" field anywhere in the response payload, and
        # neither secret string appears in the serialised body. We check
        # the raw text so we catch both ``value`` key and accidental
        # logging of the literal string in some other field.
        raw = resp.text
        self.assertNotIn("id-secret", raw)
        self.assertNotIn("pw-secret", raw)
        self.assertNotIn("\"value\"", raw)

    def test_delete_existing_returns_200_missing_returns_404(self) -> None:
        # Seed and delete.
        self.client.post(
            "/api/secrets", json={"name": "NAVER_ID", "value": "v"}
        )

        ok = self.client.delete("/api/secrets/NAVER_ID")
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(ok.json(), {"name": "NAVER_ID", "deleted": True})

        # Second delete should 404 — store reported existed=False.
        missing = self.client.delete("/api/secrets/NAVER_ID")
        self.assertEqual(missing.status_code, 404, missing.text)
        # And environ is cleaned up.
        self.assertNotIn("NAVER_ID", os.environ)

    def test_post_with_lowercase_name_rejected_at_validation_layer(self) -> None:
        resp = self.client.post(
            "/api/secrets",
            json={"name": "naver_id", "value": "v"},
        )
        # Pydantic field pattern -> 422 before reaching the store.
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_post_with_newline_in_value_rejected_by_store(self) -> None:
        resp = self.client.post(
            "/api/secrets",
            json={"name": "NAVER_ID", "value": "line1\nline2"},
        )
        # Pydantic itself does not reject the newline (value is just a
        # bounded string), so this becomes the store-layer 400 path.
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("newline", resp.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()
