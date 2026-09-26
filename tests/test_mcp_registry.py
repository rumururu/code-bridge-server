"""Code Bridge must be able to register an MCP server by itself.

Detection read exactly two files — `~/.claude.json`, which belongs to the
Claude Code CLI, and `<cwd>/.mcp.json`. So the only way to give a Code Bridge
agent an MCP server was to install a different product and hand-edit its JSON.
For a phone app that builds agents, that is not a path: the tool picker offered
nothing, and every `mcp_tool` step parked with "not configured on this machine".

The acceptance test for the whole feature is the last class here — a server
registered through Code Bridge and nowhere else must let an `mcp_tool` step
*run*. A registration that shows up in a list but still parks the step has
changed nothing that matters.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import capability_registry  # noqa: E402
from core import database  # noqa: E402
from system import mcp_registry  # noqa: E402

STDIO = {"command": "npx", "args": ["-y", "@acme/mcp@latest"]}
HTTP = {"type": "http", "url": "https://mcp.example.com/sse"}


class _RegistryCase(unittest.TestCase):
    """Each test gets its own settings database and its own fake home."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._original_db_path = database.DB_PATH
        database.DB_PATH = root / "mcp_registry.db"
        database._settings_db = None

        self.home = root / "home"
        self.home.mkdir()
        self.project = root / "project"
        self.project.mkdir()

    def tearDown(self) -> None:
        database._settings_db = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def write_claude_config(self, servers: dict) -> None:
        (self.home / ".claude.json").write_text(
            json.dumps({"mcpServers": servers}), encoding="utf-8"
        )

    def write_project_config(self, servers: dict) -> None:
        (self.project / ".mcp.json").write_text(
            json.dumps({"mcpServers": servers}), encoding="utf-8"
        )

    def isolated_paths(self):
        """Point the file half of detection at this test's temp directories."""
        return (
            patch.object(capability_registry.Path, "home", staticmethod(lambda: self.home)),
            patch.object(capability_registry.Path, "cwd", staticmethod(lambda: self.project)),
        )

    def detect(self):
        home_patch, cwd_patch = self.isolated_paths()
        with home_patch, cwd_patch:
            return capability_registry._detect_mcp_servers()

    def launch_configs(self):
        home_patch, cwd_patch = self.isolated_paths()
        with home_patch, cwd_patch:
            return capability_registry.detected_mcp_server_configs()


class StoringAServerTest(_RegistryCase):
    def test_a_registered_server_comes_back(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)

        self.assertEqual(mcp_registry.list_registered_servers(), {"acme": STDIO})

    def test_registering_the_same_name_replaces_it(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)
        mcp_registry.upsert_server("acme", HTTP)

        self.assertEqual(mcp_registry.list_registered_servers()["acme"], HTTP)

    def test_removing_reports_whether_there_was_anything_to_remove(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)

        self.assertTrue(mcp_registry.remove_server("acme"))
        self.assertFalse(mcp_registry.remove_server("acme"))
        self.assertEqual(mcp_registry.list_registered_servers(), {})

    def test_an_unreadable_store_reads_as_empty_rather_than_raising(self) -> None:
        """Detection runs through this on every agent run; it must not be the
        thing that takes the server down."""
        with patch.object(
            database, "get_settings_db", side_effect=RuntimeError("db is gone")
        ):
            self.assertEqual(mcp_registry.list_registered_servers(), {})

    def test_a_row_of_the_wrong_shape_reads_as_empty(self) -> None:
        database.get_settings_db().set_json(mcp_registry.SETTING_KEY, ["not", "a", "map"])

        self.assertEqual(mcp_registry.list_registered_servers(), {})


class RefusingWhatCannotBeLaunchedTest(_RegistryCase):
    """The store is validated by the real translator, not a lookalike of it.

    `detected_mcp_server_configs` omits any entry `_sdk_mcp_server_config`
    cannot translate. An entry accepted here but dropped there is a
    registration the user watched succeed, that lists fine, and that surfaces
    days later as a parked step reporting the server as missing.
    """

    def _rejects(self, config) -> str:
        with self.assertRaises(mcp_registry.McpRegistryError) as caught:
            mcp_registry.upsert_server("bad", config)
        self.assertEqual(mcp_registry.list_registered_servers(), {})
        return str(caught.exception)

    def test_a_stdio_entry_with_no_command_is_refused(self) -> None:
        self.assertIn("command", self._rejects({"type": "stdio", "args": ["x"]}))

    def test_an_http_entry_with_no_url_is_refused(self) -> None:
        self.assertIn("url", self._rejects({"type": "http", "headers": {"A": "b"}}))

    def test_an_empty_entry_is_refused(self) -> None:
        self._rejects({})

    def test_an_sdk_entry_is_refused(self) -> None:
        """An in-process server object cannot be described by stored JSON."""
        self._rejects({"type": "sdk", "name": "in-process"})

    def test_a_non_object_entry_is_refused(self) -> None:
        self._rejects("npx @acme/mcp")

    def test_a_nameless_registration_is_refused(self) -> None:
        with self.assertRaises(mcp_registry.McpRegistryError):
            mcp_registry.upsert_server("   ", STDIO)

    def test_the_builtin_browser_name_is_refused(self) -> None:
        """`_merged_mcp_servers` drops it, so storing it would be a silent
        no-op — exactly what write-time validation exists to prevent."""
        with self.assertRaises(mcp_registry.McpRegistryError) as caught:
            mcp_registry.upsert_server(
                capability_registry.BROWSER_RUNTIME_CAPABILITY_NAME, STDIO
            )
        self.assertIn("reserved", str(caught.exception))

    def test_everything_stored_can_be_launched(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)
        mcp_registry.upsert_server("remote", HTTP)

        configs = self.launch_configs()

        self.assertEqual(configs["acme"]["type"], "stdio")
        self.assertEqual(configs["remote"]["type"], "http")


class MaskingTest(_RegistryCase):
    """Stored verbatim because the SDK needs it; never rendered back."""

    def test_env_and_header_values_never_appear_in_the_public_view(self) -> None:
        mcp_registry.upsert_server(
            "acme",
            {"command": "npx", "env": {"ACME_TOKEN": "sk-live-secret-value"}},
        )
        mcp_registry.upsert_server(
            "remote",
            {"type": "http", "url": "https://x.example", "headers": {"Authorization": "Bearer abc123"}},
        )

        rendered = json.dumps(mcp_registry.public_registry())

        self.assertNotIn("sk-live-secret-value", rendered)
        self.assertNotIn("abc123", rendered)

    def test_the_public_view_still_says_which_variables_are_set(self) -> None:
        """Names, not masked values: "this variable is set" is the fact a UI
        needs, and a partially masked token still leaks its shape."""
        mcp_registry.upsert_server(
            "acme", {"command": "npx", "env": {"ACME_TOKEN": "x", "ACME_URL": "y"}}
        )

        view = mcp_registry.public_registry()[0]

        self.assertEqual(view["env_keys"], ["ACME_TOKEN", "ACME_URL"])
        self.assertEqual(view["command"], "npx")
        self.assertEqual(view["transport"], "stdio")
        self.assertIs(view["launchable"], True)

    def test_a_credential_passed_as_an_argument_is_masked_both_ways(self) -> None:
        mcp_registry.upsert_server(
            "acme",
            {
                "command": "npx",
                "args": ["-y", "@acme/mcp", "--api-key=sk-inline", "--token", "sk-spaced"],
            },
        )

        args = mcp_registry.public_registry()[0]["args"]

        self.assertIn("@acme/mcp", args, "the recognisable part is kept")
        self.assertNotIn("sk-inline", json.dumps(args))
        self.assertNotIn("sk-spaced", json.dumps(args))

    def test_a_credential_in_a_url_is_masked(self) -> None:
        mcp_registry.upsert_server(
            "remote",
            {"type": "http", "url": "https://user:pw123@mcp.example.com/x?access_token=tok987&page=2"},
        )

        url = mcp_registry.public_registry()[0]["url"]

        self.assertNotIn("pw123", url)
        self.assertNotIn("tok987", url)
        self.assertIn("mcp.example.com", url, "the host is what identifies the row")
        self.assertIn("page=2", url)


class MergingWithTheConfigFilesTest(_RegistryCase):
    def test_a_registered_server_is_detected(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)

        detected = {entry["name"]: entry for entry in self.detect()}

        self.assertIn("acme", detected)
        self.assertEqual(detected["acme"]["status"], "available")
        self.assertEqual(
            detected["acme"]["metadata"]["config_origin"], mcp_registry.REGISTRY_ORIGIN
        )
        self.assertIsNone(detected["acme"]["metadata"]["config_path"])
        self.assertIn("Code Bridge", detected["acme"]["description"])

    def test_file_servers_and_registered_servers_both_appear(self) -> None:
        self.write_claude_config({"from_cli": {"command": "cli-command"}})
        mcp_registry.upsert_server("from_app", STDIO)

        self.assertEqual(
            {entry["name"] for entry in self.detect()}, {"from_cli", "from_app"}
        )

    def test_a_registration_overrides_a_file_of_the_same_name(self) -> None:
        """The user of this product typed this into this product, naming that
        server. If the file won, the registration would be a silent no-op."""
        self.write_claude_config({"shared": {"command": "cli-command"}})
        self.write_project_config({"shared": {"command": "project-command"}})
        mcp_registry.upsert_server("shared", {"command": "code-bridge-command"})

        detected = {entry["name"]: entry for entry in self.detect()}

        self.assertEqual(detected["shared"]["metadata"]["command"], "code-bridge-command")
        self.assertEqual(
            detected["shared"]["metadata"]["config_origin"], mcp_registry.REGISTRY_ORIGIN
        )
        self.assertEqual(self.launch_configs()["shared"]["command"], "code-bridge-command")

    def test_a_registration_is_reported_as_verified_and_says_where(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)

        home_patch, cwd_patch = self.isolated_paths()
        with home_patch, cwd_patch:
            verdict = capability_registry.verify_declared_mcp_ids(["acme"])[0]

        self.assertTrue(verdict.verified)
        self.assertIn("Code Bridge", verdict.detail)
        self.assertNotIn("None", verdict.detail)

    def test_an_unknown_server_says_the_registry_was_looked_in_too(self) -> None:
        home_patch, cwd_patch = self.isolated_paths()
        with home_patch, cwd_patch:
            verdict = capability_registry.verify_declared_mcp_ids(["nowhere"])[0]

        self.assertFalse(verdict.verified)
        self.assertIn("Code Bridge", verdict.detail)


class NothingChangesWhenNothingIsRegisteredTest(_RegistryCase):
    """The regression guard. An install that registered nothing has no row, and
    must behave exactly as it did before this module existed."""

    def test_detection_of_the_files_is_untouched(self) -> None:
        self.write_claude_config(
            {"marionette": {"type": "stdio", "command": "/bin/marionette_mcp", "args": []}}
        )
        self.write_project_config({"extra": {"url": "https://x"}})

        detected = {entry["name"]: entry for entry in self.detect()}

        self.assertEqual(set(detected), {"marionette", "extra"})
        self.assertEqual(
            detected["marionette"]["metadata"]["config_path"],
            str(self.home / ".claude.json"),
        )
        self.assertEqual(detected["marionette"]["metadata"]["config_origin"], "claude_cli_config")
        self.assertIn(".claude.json", detected["marionette"]["description"])
        self.assertEqual(detected["extra"]["metadata"]["transport"], "http")

    def test_no_config_and_no_registration_still_detects_nothing(self) -> None:
        self.assertEqual(self.detect(), [])
        self.assertEqual(self.launch_configs(), {})

    def test_a_broken_registry_does_not_hide_the_file_servers(self) -> None:
        """Adding a second source must not give detection a second way to fail."""
        self.write_claude_config({"marionette": {"command": "/bin/marionette_mcp"}})

        with patch(
            "system.mcp_registry.list_registered_servers",
            side_effect=RuntimeError("store exploded"),
        ):
            detected = {entry["name"] for entry in self.detect()}
            configs = self.launch_configs()

        self.assertEqual(detected, {"marionette"})
        self.assertEqual(set(configs), {"marionette"})


class AnMcpStepRunsOnARegisteredServerTest(_RegistryCase):
    """The acceptance criterion for the whole feature.

    Registering a server has to make an `mcp_tool` step *run*. Everything above
    is plumbing; if `_mcp_step_blocker` still parks the step, a user who
    registered a server in the app gained nothing — the agent still stops every
    firing and waits for a person, which for a scheduled agent means it never
    runs unattended at all.
    """

    def _blocker(self, server_id: str, declared: str):
        import agent.task_orchestrator as orchestrator

        step = {
            "id": "step_1",
            "title": "Trigger the workflow",
            "input": {"workflow_type": "mcp_tool", "tool_hint": server_id},
        }
        agent = {"id": "agent_1", "tools_json": [{"mcp_id": declared}]}
        home_patch, cwd_patch = self.isolated_paths()
        with home_patch, cwd_patch:
            return orchestrator._mcp_step_blocker(step, agent)

    def test_the_step_parks_before_the_server_is_registered(self) -> None:
        blocker = self._blocker("acme", "acme")

        self.assertIsNotNone(blocker, "nothing declares this server yet")
        self.assertIn("acme", blocker)

    def test_the_step_runs_once_the_server_is_registered_in_code_bridge(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)

        self.assertIsNone(self._blocker("acme", "acme"))

    def test_it_runs_for_an_http_registration_too(self) -> None:
        mcp_registry.upsert_server("remote", HTTP)

        self.assertIsNone(self._blocker("remote", "remote"))

    def test_removing_the_registration_parks_the_step_again(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)
        mcp_registry.remove_server("acme")

        self.assertIsNotNone(self._blocker("acme", "acme"))


class TheApiNeverReturnsASecretTest(_RegistryCase):
    def setUp(self) -> None:
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from routes import deps, system_settings

        app = FastAPI()
        app.include_router(system_settings.router)
        # The endpoints under test are about the registry, not about auth; the
        # dependency is pinned by `test_the_endpoints_require_the_api_key`.
        app.dependency_overrides[deps.verify_api_key] = lambda: "test-key"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_the_endpoints_require_the_api_key(self) -> None:
        """Pinned by inspection rather than a live call, because the fixture
        above overrides the dependency for every other test in this class."""
        from routes import deps, system_settings

        found = set()
        for route in system_settings.router.routes:
            if "mcp-servers" not in route.path:
                continue
            found.add((route.path, tuple(sorted(route.methods))))
            self.assertIn(
                deps.verify_api_key,
                [dependency.call for dependency in route.dependant.dependencies],
                f"{route.path} {route.methods} is unauthenticated",
            )
        self.assertEqual(
            found,
            {
                ("/api/system/mcp-servers", ("GET",)),
                ("/api/system/mcp-servers", ("POST",)),
                # Read-only names catalog behind the step schema's
                # `mcp-servers` option source (T-I2-06). Key-gated like the
                # rest; its keyless mirror lives on the localhost-only
                # dashboard router, not here.
                ("/api/system/mcp-servers/detected", ("GET",)),
                ("/api/system/mcp-servers/{name}", ("DELETE",)),
            },
        )

    def test_registering_then_listing_never_echoes_the_token(self) -> None:
        created = self.client.post(
            "/api/system/mcp-servers",
            json={
                "name": "acme",
                "config": {"command": "npx", "env": {"ACME_TOKEN": "sk-live-do-not-leak"}},
            },
        )

        self.assertEqual(created.status_code, 200)
        self.assertNotIn("sk-live-do-not-leak", created.text)
        self.assertEqual(created.json()["env_keys"], ["ACME_TOKEN"])

        listed = self.client.get("/api/system/mcp-servers")

        self.assertEqual(listed.status_code, 200)
        self.assertNotIn("sk-live-do-not-leak", listed.text)
        self.assertEqual([item["name"] for item in listed.json()["items"]], ["acme"])
        # …and the value really is stored, not dropped on the way in.
        self.assertEqual(
            mcp_registry.list_registered_servers()["acme"]["env"]["ACME_TOKEN"],
            "sk-live-do-not-leak",
        )

    def test_an_unlaunchable_entry_is_rejected_with_a_reason(self) -> None:
        response = self.client.post(
            "/api/system/mcp-servers", json={"name": "bad", "config": {"type": "stdio"}}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("command", response.json()["detail"])
        self.assertEqual(mcp_registry.list_registered_servers(), {})

    def test_deleting_something_that_was_never_registered_is_a_404(self) -> None:
        self.assertEqual(
            self.client.delete("/api/system/mcp-servers/ghost").status_code, 404
        )

    def test_delete_removes_it(self) -> None:
        mcp_registry.upsert_server("acme", STDIO)

        response = self.client.delete("/api/system/mcp-servers/acme")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mcp_registry.list_registered_servers(), {})


if __name__ == "__main__":
    unittest.main()
