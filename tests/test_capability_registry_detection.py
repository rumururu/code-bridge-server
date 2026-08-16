"""The capability catalog must only claim what this machine actually has.

The catalog used to ship a hardcoded MCP list — github, gmail, browser — and
upsert all three as ``status="available"`` on every refresh, on every install,
regardless of what was configured. On the machine this was written on, the
Claude CLI config declares exactly two servers (marionette, playwright) and
neither github nor gmail; the catalog was advertising two connectors that
could never run, and a builder that reads the catalog would happily hand the
user an agent built on them.

These tests pin the replacement: entries come from config files that exist,
nothing is invented when a config is missing or malformed, and the built-in
browser runtime's status comes from a readiness probe — with "not probed" as
its own answer, never folded into "available".
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

from agent import agent_store  # noqa: E402
from agent import capability_registry  # noqa: E402
from agent.capability_registry import (  # noqa: E402
    MCP_CATALOG_SOURCE,
    SKILL_PLACEHOLDERS,
    _browser_runtime_entry,
    _detect_mcp_servers,
    refresh_capability_registry,
)
from core import database  # noqa: E402


READY = {
    "ready": True,
    "install_command": "python -m playwright install chromium",
    "message": "",
}
NOT_READY = {
    "ready": False,
    "install_command": "python -m playwright install chromium",
    "message": "Python Playwright package is not available.",
}


class McpDetectionTest(unittest.TestCase):
    """`_detect_mcp_servers` reports config contents and nothing else."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir()
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _detect(self):
        with patch.object(capability_registry.Path, "home", staticmethod(lambda: self.home)), \
             patch.object(capability_registry.Path, "cwd", staticmethod(lambda: self.project)):
            return _detect_mcp_servers()

    def _write_claude_config(self, payload):
        (self.home / ".claude.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_reports_exactly_the_servers_in_the_config(self):
        self._write_claude_config(
            {
                "numStartups": 12,
                "mcpServers": {
                    "marionette": {"type": "stdio", "command": "/bin/marionette_mcp", "args": []},
                    "playwright": {"type": "stdio", "command": "npx", "args": ["-y", "@playwright/mcp@latest"]},
                },
            }
        )

        detected = self._detect()

        self.assertEqual([entry["name"] for entry in detected], ["marionette", "playwright"])
        for entry in detected:
            self.assertEqual(entry["status"], "available")
            self.assertEqual(entry["type"], "mcp_server")
        self.assertEqual(detected[1]["metadata"]["command"], "npx")
        self.assertEqual(detected[1]["metadata"]["args"], ["-y", "@playwright/mcp@latest"])
        self.assertEqual(detected[1]["metadata"]["transport"], "stdio")
        self.assertEqual(
            detected[1]["metadata"]["config_path"],
            str(self.home / ".claude.json"),
        )

    def test_invents_no_servers_the_config_does_not_declare(self):
        """The exact regression: github/gmail were advertised by the code."""
        self._write_claude_config({"mcpServers": {"marionette": {"command": "/bin/marionette_mcp"}}})

        names = {entry["name"] for entry in self._detect()}

        self.assertEqual(names, {"marionette"})
        self.assertNotIn("github", names)
        self.assertNotIn("gmail", names)

    def test_missing_config_yields_nothing(self):
        self.assertEqual(self._detect(), [])

    def test_malformed_config_yields_nothing_rather_than_defaults(self):
        (self.home / ".claude.json").write_text("{not json at all", encoding="utf-8")

        self.assertEqual(self._detect(), [])

    def test_config_without_mcp_servers_key_yields_nothing(self):
        self._write_claude_config({"numStartups": 3, "projects": {}})

        self.assertEqual(self._detect(), [])

    def test_project_mcp_config_is_merged_and_wins_on_name_clash(self):
        self._write_claude_config({"mcpServers": {"shared": {"command": "home-command"}}})
        (self.project / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"shared": {"command": "project-command"}, "extra": {"url": "https://x"}}}),
            encoding="utf-8",
        )

        detected = {entry["name"]: entry for entry in self._detect()}

        self.assertEqual(set(detected), {"shared", "extra"})
        self.assertEqual(detected["shared"]["metadata"]["command"], "project-command")
        self.assertEqual(detected["extra"]["metadata"]["transport"], "http")

    def test_configured_server_cannot_take_the_builtin_browser_name(self):
        """`browser` means the server's own runtime to every browser_action step."""
        self._write_claude_config({"mcpServers": {"browser": {"command": "something-else"}}})

        self.assertEqual(self._detect(), [])


class BrowserRuntimeEntryTest(unittest.TestCase):
    """The browser row follows the readiness probe, including "unknown"."""

    def test_ready_probe_makes_it_available(self):
        with patch.object(capability_registry, "get_cached_browser_readiness_sync", return_value=READY):
            entry = _browser_runtime_entry()

        self.assertEqual(entry["status"], "available")
        self.assertIs(entry["metadata"]["ready"], True)
        self.assertIs(entry["metadata"]["readiness_probed"], True)

    def test_unready_probe_makes_it_unavailable_with_install_command(self):
        with patch.object(capability_registry, "get_cached_browser_readiness_sync", return_value=NOT_READY):
            entry = _browser_runtime_entry()

        self.assertEqual(entry["status"], "unavailable")
        self.assertIs(entry["metadata"]["ready"], False)
        self.assertEqual(
            entry["metadata"]["install_command"],
            "python -m playwright install chromium",
        )

    def test_unprobed_cache_is_unknown_not_available(self):
        """`None` from the sync accessor means nobody asked yet."""
        with patch.object(capability_registry, "get_cached_browser_readiness_sync", return_value=None):
            entry = _browser_runtime_entry()

        self.assertEqual(entry["status"], "unknown")
        self.assertIsNone(entry["metadata"]["ready"])
        self.assertIs(entry["metadata"]["readiness_probed"], False)
        self.assertTrue(entry["metadata"]["install_command"])


class CatalogRefreshTest(unittest.TestCase):
    """End to end through the store: what a refresh leaves in the catalog."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "capability_registry.db"
        agent_store._agent_store = None

        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir()
        self.project = Path(self._tmp.name) / "project"
        self.project.mkdir()
        (self.home / ".claude.json").write_text(
            json.dumps({"mcpServers": {"marionette": {"command": "/bin/marionette_mcp"}}}),
            encoding="utf-8",
        )

    def tearDown(self):
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _refresh(self, readiness=NOT_READY):
        with patch.object(capability_registry.Path, "home", staticmethod(lambda: self.home)), \
             patch.object(capability_registry.Path, "cwd", staticmethod(lambda: self.project)), \
             patch.object(capability_registry, "get_cached_browser_readiness_sync", return_value=readiness), \
             patch.object(capability_registry, "get_llm_options_snapshot", return_value={"companies": []}):
            return refresh_capability_registry()

    def test_refresh_stores_only_detected_mcp_rows(self):
        catalog = self._refresh()

        mcp_rows = {row["name"]: row for row in catalog if row["type"] == "mcp_server"}
        self.assertEqual(set(mcp_rows), {"marionette", "browser"})
        self.assertEqual(mcp_rows["marionette"]["status"], "available")
        self.assertEqual(mcp_rows["browser"]["status"], "unavailable")

    def test_skills_are_unverified_not_available(self):
        catalog = self._refresh()

        skills = {row["name"]: row for row in catalog if row["type"] == "skill"}
        self.assertEqual(set(skills), {item["name"] for item in SKILL_PLACEHOLDERS})
        for name, row in skills.items():
            self.assertEqual(row["status"], "unverified", f"skill {name} still claims availability")

    def test_stale_available_rows_are_demoted_on_refresh(self):
        """A row an older build wrote must stop advertising availability."""
        store = agent_store.get_agent_store()
        store.upsert_capability(
            capability_type="mcp_server",
            name="github",
            status="available",
            source=MCP_CATALOG_SOURCE,
            description="GitHub repository, issue, and pull request access.",
        )

        self._refresh()

        rows = {
            row["name"]: row
            for row in store.list_capabilities(capability_type="mcp_server")
        }
        self.assertIn("github", rows, "the row is downgraded, not deleted")
        self.assertEqual(rows["github"]["status"], "unverified")
        self.assertIs(rows["github"]["metadata"]["detected"], False)
        self.assertEqual(rows["marionette"]["status"], "available")

    def test_refresh_is_idempotent(self):
        first = self._refresh()
        second = self._refresh()

        def shape(catalog):
            return sorted((row["type"], row["name"], row["status"]) for row in catalog)

        self.assertEqual(shape(first), shape(second))

    def test_browser_row_tracks_a_later_successful_probe(self):
        self._refresh(readiness=NOT_READY)
        catalog = self._refresh(readiness=READY)

        browser = next(
            row for row in catalog if row["type"] == "mcp_server" and row["name"] == "browser"
        )
        self.assertEqual(browser["status"], "available")


if __name__ == "__main__":
    unittest.main()
