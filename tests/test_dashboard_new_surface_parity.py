"""The PC has to show the three things the phone learned to show (G7).

Waves 1-3 added three server-side surfaces and wired the phone to them. The
dashboard was never checked, and a real user builds the agent on the PC and
approves on the phone — so anything that exists only on the phone makes the PC
half of the product lie by omission:

- **C3, the commit gate.** `POST /agents`, `PATCH /agents/{id}` and
  `/builder/commit` refuse a workflow that cannot run as written with a 400
  carrying `unresolved[]` / `unknown_step_references[]` (each naming a
  `step_id` and a 0-based `action_index`), a human `ask` per finding, and
  `can_save_incomplete`. On success `commit_result.readiness` says what the
  saved workflow still cannot do. The page used to keep only the sentence and
  had no way to accept the escape hatch, and it rendered a
  `readiness.ok: false` commit as a clean success.
- **E-S1, the run-status fields.** `GET /agents` carries `last_run_status` and
  `waiting_run_count` per agent. The roster derived its status from a page-wide
  `/runs?limit=60` window instead, which drops an agent whose last run fell out
  of the newest sixty, and never distinguished "parked waiting for you" from
  "nothing happening".
- **B6, `details.display`.** An approval carries `{action, target}` so a client
  can say what is being asked instead of printing an operation id. The
  approvals table showed the id and a guessed preview.

The route assertions below prove the dashboard's no-key mirrors hand the page
the same payloads the phone gets; the template assertions prove the page reads
them. Neither is a live-server check — that happens after deploy.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import app_factory  # noqa: E402
from agent import agent_store, schedule_store  # noqa: E402
from agent.browser_action_adapter import reset_browser_readiness_cache  # noqa: E402
from core import database  # noqa: E402
from routes import agents as agents_routes  # noqa: E402
from routes.deps import require_local_access  # noqa: E402

MARKUP = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
    encoding="utf-8"
)

UNREADY_RUNTIME = {
    "ready": False,
    "playwright_python": False,
    "chromium_executable": False,
    "install_command": "/opt/venv/bin/python -m playwright install chromium",
    "message": "Playwright is not installed in this server's environment.",
}

PLACEHOLDER_FLOW = [
    {
        "id": "open_cafe",
        "name": "카페 열기",
        "type": "browser_action",
        "actions": [
            {"type": "navigate", "url": "configured_cafe_url"},
            {"type": "click", "selector": "#write"},
        ],
    }
]


class DashboardCommitGateTest(unittest.TestCase):
    """The refusal and the readiness fact must survive the mirror."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "dashboard_parity.db"
        agent_store._agent_store = None
        schedule_store._store = None
        reset_browser_readiness_cache()

        self.app = app_factory.create_dashboard_app()
        self.app.dependency_overrides[require_local_access] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        reset_browser_readiness_cache()
        self._tmp.cleanup()

    def _readiness(self, snapshot):
        return mock.patch.multiple(
            agents_routes,
            get_cached_browser_readiness_sync=mock.Mock(return_value=snapshot),
            get_browser_runtime_readiness=mock.AsyncMock(return_value=snapshot),
        )

    def _create(self, **extra):
        body = {"name": "카페 글쓰기", "flow_json": PLACEHOLDER_FLOW}
        body.update(extra)
        return self.client.post("/api/dashboard/agent/agents", json=body)

    def test_the_dashboard_gets_the_whole_refusal_not_just_the_sentence(self):
        with self._readiness(UNREADY_RUNTIME):
            response = self._create()

        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()
        self.assertTrue(payload["can_save_incomplete"])
        self.assertTrue(payload["blocking"])
        finding = payload["unresolved"][0]
        self.assertEqual(finding["step_id"], "open_cafe")
        self.assertTrue(finding["ask"].strip())
        # 0-based on the wire; the page renders `action_index + 1`.
        self.assertEqual(finding["detail"]["action_index"], 0)
        self.assertEqual(finding["detail"]["value"], "configured_cafe_url")

    def test_the_escape_hatch_works_from_the_dashboard_and_says_it_did(self):
        with self._readiness(UNREADY_RUNTIME):
            response = self._create(commit_incomplete=True)

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        readiness = payload["readiness"]
        self.assertFalse(readiness["ok"])
        self.assertTrue(readiness["saved_incomplete"])
        self.assertTrue(readiness["unresolved_targets"])
        self.assertFalse(readiness["browser_runtime"]["ready"])

    def test_a_run_activity_field_the_roster_reads_is_actually_sent(self):
        with self._readiness(UNREADY_RUNTIME):
            self._create(commit_incomplete=True)
        response = self.client.get("/api/dashboard/agent/agents")

        self.assertEqual(response.status_code, 200, response.text)
        agents = response.json()["agents"]
        self.assertTrue(agents)
        for key in ("last_run_status", "waiting_run_count", "active_run_count"):
            with self.subTest(field=key):
                self.assertIn(key, agents[0])


class CommitGateTemplateTest(unittest.TestCase):
    def test_the_structured_refusal_body_is_not_thrown_away(self):
        """`api()` used to collapse an error to one string, which is what left
        the page unable to list the findings or accept the offer."""
        self.assertIn("error_body:", MARKUP)
        self.assertIn("contractRefusalFrom", MARKUP)

    def test_a_bare_400_is_not_mistaken_for_a_contract_refusal(self):
        self.assertIn(
            "if (!blocking.length && !unresolved.length && !unknown.length) return null;",
            MARKUP,
        )

    def test_each_finding_names_its_step_action_ask_and_value(self):
        self.assertIn("detail.action_index", MARKUP)
        # Shown +1: a person counts actions from one, and the app already does.
        self.assertIn("String(index + 1)", MARKUP)
        self.assertIn("finding.ask", MARKUP)
        self.assertIn("detail.value", MARKUP)

    def test_the_save_it_unfinished_offer_is_wired_to_the_server_flag(self):
        self.assertIn("refusal.canSaveIncomplete", MARKUP)
        self.assertIn("body.can_save_incomplete === true", MARKUP)
        self.assertIn("commit_incomplete: true", MARKUP)

    def test_dismissing_the_dialog_answers_it_rather_than_hanging_the_save(self):
        self.assertIn("contractRefusalResolve", MARKUP)
        self.assertIn("resolveContractRefusal", MARKUP)

    def test_a_successful_commit_with_gaps_does_not_render_as_a_clean_save(self):
        self.assertIn("readinessWorthShowing", MARKUP)
        self.assertIn("renderReadinessBlock", MARKUP)
        # `ok` false is the blocking styling, not a neutral note.
        self.assertIn("const blocked = readiness.ok !== true;", MARKUP)
        # POST/PATCH attach `readiness` only when it is not ok, so its mere
        # presence there has to be picked up too.
        self.assertIn("result.readiness", MARKUP)

    def test_absent_readiness_says_nothing_rather_than_saying_fine(self):
        self.assertIn(
            "if (!readiness || typeof readiness !== 'object') return false;", MARKUP
        )

    def test_the_browser_install_command_is_offered_with_a_copy_button(self):
        self.assertIn("runtime.install_command", MARKUP)
        self.assertIn("copyReadinessCommand", MARKUP)

    def test_the_gate_wording_exists_in_both_locale_tables(self):
        for key in (
            "refusal_title:",
            "refusal_step_line:",
            "refusal_save_incomplete:",
            "refusal_fix_first:",
            "readiness_gaps_title:",
            "readiness_warnings_title:",
            "readiness_saved_incomplete:",
            "readiness_browser_missing:",
        ):
            with self.subTest(key=key):
                self.assertEqual(MARKUP.count(key), 2, key)


class RunStatusChipTemplateTest(unittest.TestCase):
    def test_the_chip_reads_the_agents_own_fields_not_the_global_run_window(self):
        self.assertIn("agent.waiting_run_count", MARKUP)
        self.assertIn("agent.last_run_status === 'failed'", MARKUP)
        self.assertIn("agent.active_run_count", MARKUP)

    def test_needs_response_outranks_a_failure_and_both_outrank_the_count(self):
        chip = MARKUP[MARKUP.index("function runStatusChip") :]
        chip = chip[: chip.index("function renderAgentList")]
        self.assertLess(
            chip.index("chip_needs_response"),
            chip.index("chip_last_run_failed"),
        )
        self.assertLess(
            chip.index("chip_last_run_failed"),
            chip.index("chip_active_runs"),
        )

    def test_a_server_that_sends_neither_field_renders_no_chip(self):
        """Silence is not "0 active" — an older server simply does not know."""
        self.assertIn("if (active === null && waiting === null) return '';", MARKUP)

    def test_the_chip_is_actually_placed_in_the_roster_row(self):
        self.assertIn("runStatusChip(agent)", MARKUP)
        self.assertIn('class="roster-head"', MARKUP)

    def test_the_chip_wording_exists_in_both_locale_tables(self):
        for key in ("chip_needs_response:", "chip_last_run_failed:", "chip_active_runs:"):
            with self.subTest(key=key):
                self.assertEqual(MARKUP.count(key), 2, key)


class ApprovalDisplayTemplateTest(unittest.TestCase):
    def test_the_approval_row_reads_the_normalized_display(self):
        self.assertIn("details.display", MARKUP)
        self.assertIn("approvalActionSentence", MARKUP)

    def test_every_action_the_server_can_send_has_a_sentence(self):
        for action in (
            "read_file",
            "write_file",
            "edit_file",
            "delete_file",
            "list_files",
            "search_files",
            "run_command",
            "fetch_url",
        ):
            with self.subTest(action=action):
                self.assertIn(f"{action}: 'appr_{action}'", MARKUP)

    def test_an_unknown_action_keeps_the_raw_operation_instead_of_a_guess(self):
        self.assertIn("return key ? t(key) : '';", MARKUP)

    def test_a_null_target_stays_absent_rather_than_being_filled_in(self):
        self.assertIn("const command = sentence ? target : (target || legacy);", MARKUP)

    def test_the_target_is_rendered_monospace(self):
        self.assertIn("target-line", MARKUP)

    def test_the_action_sentences_exist_in_both_locale_tables(self):
        for action in (
            "read_file",
            "write_file",
            "edit_file",
            "delete_file",
            "list_files",
            "search_files",
            "run_command",
            "fetch_url",
        ):
            with self.subTest(action=action):
                self.assertEqual(MARKUP.count(f"appr_{action}:"), 2, action)


if __name__ == "__main__":
    unittest.main()
