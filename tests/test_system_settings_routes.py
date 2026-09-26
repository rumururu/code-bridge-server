import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes.deps import verify_api_key, verify_api_key_or_localhost
from routes.system_settings import router as system_settings_router
from system.system_settings_service import SystemSettingsResult


class SystemSettingsRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(system_settings_router)
        app.dependency_overrides[verify_api_key] = lambda: True
        # `/llm/selection` guards with verify_api_key_or_localhost, which calls
        # verify_api_key directly rather than through DI — so the override
        # above does not reach it and this second one is what keeps these
        # payload-shape tests about payloads. The auth behaviour of that
        # dependency is covered on its own in LlmSelectionAccessTest below.
        app.dependency_overrides[verify_api_key_or_localhost] = lambda: True
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_get_heartbeat_settings_success(self):
        with patch(
            "routes.system_settings.get_heartbeat_settings_for_current_server",
            return_value=SystemSettingsResult(
                success=True,
                status_code=200,
                payload={"interval_minutes": 10, "min": 5, "max": 15},
            ),
        ):
            response = self.client.get("/api/system/heartbeat")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("interval_minutes"), 10)

    def test_update_llm_selection_validation_error_returns_400(self):
        with patch(
            "routes.system_settings.update_llm_selection_for_current_server",
            return_value=SystemSettingsResult(
                success=False,
                status_code=400,
                payload={"error": "Unknown LLM provider"},
            ),
        ):
            response = self.client.put(
                "/api/system/llm/selection",
                json={"company_id": "unknown", "model": "x"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error"), "Unknown LLM provider")

    def test_get_llm_commands_success(self):
        with patch(
            "routes.system_settings.get_llm_command_snapshot",
            return_value={
                "provider_id": "openai",
                "model": "o3",
                "scope": "global",
                "commands": [{"name": "/project"}],
                "capabilities": {"slash_commands_executable": False},
            },
        ) as command_snapshot:
            response = self.client.get(
                "/api/system/llm/commands?provider_id=openai&model=o3&scope=global"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("commands")[0].get("name"), "/project")
        command_snapshot.assert_called_once_with(
            provider_id="openai",
            model="o3",
            scope="global",
            refresh=False,
        )

    def test_execute_llm_command_success(self):
        with patch(
            "routes.system_settings.execute_llm_command",
            return_value={
                "success": True,
                "command": "/status",
                "execution": "server_action",
                "server_action": "status",
                "payload": {"scope": "project"},
            },
        ) as execute_command:
            response = self.client.post(
                "/api/system/llm/commands/execute",
                json={
                    "name": "/status",
                    "provider_id": "openai",
                    "model": "o3",
                    "scope": "project",
                    "project_name": "demo",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("server_action"), "status")
        execute_command.assert_called_once_with(
            name="/status",
            provider_id="openai",
            model="o3",
            scope="project",
            project_name="demo",
        )

    def test_update_codex_settings_success(self):
        with patch(
            "routes.system_settings.update_codex_settings_for_current_server",
            return_value=SystemSettingsResult(
                success=True,
                status_code=200,
                payload={"sandbox_mode": "workspace-write", "sandbox_modes": []},
            ),
        ):
            response = self.client.put(
                "/api/system/llm/codex/settings",
                json={"sandbox_mode": "workspace-write"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("sandbox_mode"), "workspace-write")

    def test_install_llm_provider_returns_job_contract(self):
        payload = {
            "job_id": "abc123",
            "provider_id": "openai",
            "method": "npm",
            "status": "queued",
            "finished": False,
        }
        with patch(
            "routes.system_settings.install_llm_provider_for_current_server",
            new=AsyncMock(return_value=SystemSettingsResult.ok(payload, status_code=202)),
        ):
            response = self.client.post(
                "/api/system/llm/providers/openai/install",
                json={"method": "npm"},
            )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body.get("job_id"), "abc123")
        self.assertEqual(body.get("status"), "queued")

    def test_install_llm_provider_invalid_method_returns_error_code(self):
        with patch(
            "routes.system_settings.install_llm_provider_for_current_server",
            new=AsyncMock(
                return_value=SystemSettingsResult.error(
                    400,
                    "Install method 'brew' not available",
                    error_code="invalid_install_method",
                )
            ),
        ):
            response = self.client.post(
                "/api/system/llm/providers/openai/install",
                json={"method": "brew"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error_code"), "invalid_install_method")

    def test_get_llm_provider_install_job_status(self):
        with patch(
            "routes.system_settings.get_llm_provider_install_job_for_current_server",
            return_value=SystemSettingsResult.ok(
                {
                    "job_id": "abc123",
                    "provider_id": "openai",
                    "method": "npm",
                    "status": "completed",
                    "finished": True,
                    "installed": True,
                    "options": {"companies": []},
                }
            ),
        ):
            response = self.client.get("/api/system/llm/providers/install/jobs/abc123")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("finished"))
        self.assertIn("options", response.json())


class LlmSelectionAccessTest(unittest.TestCase):
    """Who may change the app-wide LLM selection.

    The agents page tells a user their selected provider is out of quota and
    offers to switch. That offer is only real if the page can act on it, and
    the page holds no API key — it talks from localhost. So this route carries
    `verify_api_key_or_localhost`, the same dependency its `/llm/access`
    neighbour has carried all along.

    Widening an auth dependency is exactly the change that deserves a test
    saying what it did *not* widen: a tunnelled request is still refused, so
    "reachable from the dashboard" never quietly became "reachable from the
    internet".
    """

    def setUp(self):
        app = FastAPI()
        app.include_router(system_settings_router)
        self.app = app

    def _client(self, host: str) -> TestClient:
        # TestClient reports a client host of "testclient", which
        # `is_localhost_request` rightly does not consider local. Rewriting the
        # ASGI scope is how these tests reach the real dependency instead of a
        # stand-in for it — patching `is_localhost_request` would prove only
        # that the patch works.
        app = self.app

        async def with_client_host(scope, receive, send):
            if scope["type"] == "http":
                scope = dict(scope)
                scope["client"] = (host, 12345)
            await app(scope, receive, send)

        return TestClient(with_client_host)

    def test_localhost_may_switch_provider_without_an_api_key(self):
        with patch(
            "routes.system_settings.update_llm_selection_for_current_server",
            return_value=SystemSettingsResult(
                success=True,
                status_code=200,
                payload={"selected": {"company_id": "anthropic", "model": "sonnet"}},
            ),
        ) as update:
            with self._client("127.0.0.1") as client:
                response = client.put(
                    "/api/system/llm/selection",
                    json={"company_id": "anthropic", "model": "sonnet"},
                )

        self.assertEqual(response.status_code, 200)
        update.assert_called_once_with("anthropic", "sonnet")

    def test_tunnelled_request_is_still_refused(self):
        with patch(
            "routes.system_settings.update_llm_selection_for_current_server",
        ) as update:
            with self._client("127.0.0.1") as client:
                response = client.put(
                    "/api/system/llm/selection",
                    json={"company_id": "anthropic", "model": "sonnet"},
                    # Cloudflare stamps these; routes.deps reads them as proof
                    # the request arrived through the tunnel even though the
                    # socket looks local.
                    headers={"CF-Ray": "abc123", "CF-Connecting-IP": "203.0.113.7"},
                )

        self.assertIn(response.status_code, (401, 403))
        update.assert_not_called()

    def test_non_local_request_without_a_key_is_refused(self):
        with patch(
            "routes.system_settings.update_llm_selection_for_current_server",
        ) as update:
            with self._client("192.168.1.50") as client:
                response = client.put(
                    "/api/system/llm/selection",
                    json={"company_id": "anthropic", "model": "sonnet"},
                )

        self.assertIn(response.status_code, (401, 403))
        update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
