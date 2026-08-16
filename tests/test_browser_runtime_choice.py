"""The browser is the operator's, and the product asks rather than decides.

Four things are pinned here, each of which was previously a constant in code:

*   the adapter launches what the stored plan says — installed Chrome by
    channel, headless per the setting, in the profile that was chosen,
*   readiness tells the truth about *which* browser will run and whether a
    download is still owed, on a machine with Chrome and on one without,
*   a choice this machine cannot honour stops the step and names the reason
    instead of silently running with a different one,
*   and a storage state written by one run is the input to the next, which is
    what makes a scheduled agent stay logged in.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import agent.browser_action_adapter as adapter  # noqa: E402
from agent import browser_session_store  # noqa: E402
from agent.browser_action_adapter import PlaywrightBrowserActionAdapter  # noqa: E402
from core import database  # noqa: E402
from system import browser_preferences as prefs  # noqa: E402
from system.browser_runtime_setup import InstalledBrowser  # noqa: E402

CHROME = InstalledBrowser(
    channel="chrome",
    name="Google Chrome",
    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    user_data_dir="/Users/someone/Library/Application Support/Google/Chrome",
)


# --- a Playwright that records instead of launching --------------------------


class _FakePage:
    def __init__(self):
        self.url = "https://example.test/"
        self.goto_calls: list[str] = []

    async def goto(self, url, **_kwargs):
        self.goto_calls.append(url)
        self.url = url

    async def title(self):
        return "fake"

    def locator(self, _selector):
        return self

    async def inner_text(self, **_kwargs):
        return "body text"


class _FakeContext:
    def __init__(self):
        self.pages: list[_FakePage] = [_FakePage()]
        self.storage_state_calls: list[str] = []
        self.storage_state_input: str | None = None
        self.closed = False

    async def new_page(self):
        page = _FakePage()
        self.pages.append(page)
        return page

    async def storage_state(self, path=None):
        self.storage_state_calls.append(str(path))
        Path(path).write_text(json.dumps({"cookies": []}), encoding="utf-8")
        return {"cookies": []}

    async def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self, context):
        self._context = context
        self.closed = False

    async def new_context(self, **options):
        self._context.storage_state_input = options.get("storage_state")
        return self._context

    async def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self):
        self.launch_calls: list[dict] = []
        self.persistent_calls: list[tuple[str, dict]] = []
        self.context = _FakeContext()

    async def launch(self, **options):
        self.launch_calls.append(options)
        return _FakeBrowser(self.context)

    async def launch_persistent_context(self, user_data_dir, **options):
        self.persistent_calls.append((str(user_data_dir), options))
        return self.context


class _FakePlaywright:
    def __init__(self):
        self.chromium = _FakeChromium()
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _FakeAsyncPlaywright:
    def __init__(self, playwright):
        self._playwright = playwright

    async def start(self):
        return self._playwright


def _install_fake_playwright(playwright):
    """Make `from playwright.async_api import async_playwright` return the fake."""
    module = mock.MagicMock()
    module.async_playwright = lambda: _FakeAsyncPlaywright(playwright)
    return mock.patch.dict(sys.modules, {"playwright.async_api": module})


class _AdapterTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._original_db = database.DB_PATH
        database.DB_PATH = self.tmp / "browser_choice.db"
        database._settings_db = None
        database.init_db()
        browser_session_store._browser_session_store = None
        self.playwright = _FakePlaywright()

    def tearDown(self):
        browser_session_store._browser_session_store = None
        database._settings_db = None
        database.DB_PATH = self._original_db
        self._tmp.cleanup()

    def run_actions(self, actions=None, *, context=None):
        with _install_fake_playwright(self.playwright):
            return asyncio.run(
                PlaywrightBrowserActionAdapter().run_actions(
                    actions or [{"type": "navigate", "url": "https://example.test/"}],
                    context=context or {"run_id": "run_1", "step_id": "step_1"},
                )
            )


class LaunchFollowsTheChoiceTest(_AdapterTestCase):
    def test_installed_chrome_is_launched_by_channel(self):
        with mock.patch.object(adapter, "resolve_browser_launch_plan") as resolve:
            resolve.return_value = prefs.resolve_browser_launch_plan(
                prefs.BrowserPreferences(profile=prefs.PROFILE_EPHEMERAL),
                chrome=CHROME,
                detect_chrome=False,
            )
            result = self.run_actions()

        self.assertEqual(result.status, "completed", result.to_output())
        self.assertEqual(len(self.playwright.chromium.launch_calls), 1)
        self.assertEqual(self.playwright.chromium.launch_calls[0]["channel"], "chrome")

    def test_bundled_chromium_passes_no_channel(self):
        with mock.patch.object(adapter, "resolve_browser_launch_plan") as resolve:
            resolve.return_value = prefs.resolve_browser_launch_plan(
                prefs.BrowserPreferences(profile=prefs.PROFILE_EPHEMERAL),
                chrome=None,
                detect_chrome=False,
            )
            self.run_actions()

        self.assertNotIn("channel", self.playwright.chromium.launch_calls[0])

    def test_headed_setting_reaches_the_launch(self):
        with mock.patch.object(adapter, "resolve_browser_launch_plan") as resolve:
            resolve.return_value = prefs.resolve_browser_launch_plan(
                prefs.BrowserPreferences(
                    headless_mode=prefs.HEADLESS_OFF, profile=prefs.PROFILE_EPHEMERAL
                ),
                chrome=CHROME,
                detect_chrome=False,
            )
            self.run_actions()

        options = self.playwright.chromium.launch_calls[0]
        self.assertFalse(options["headless"])
        # Offscreen, not in the operator's face: the same args the live
        # handoff uses.
        self.assertTrue(any("--window-position" in arg for arg in options["args"]))

    def test_headless_setting_reaches_the_launch(self):
        with mock.patch.object(adapter, "resolve_browser_launch_plan") as resolve:
            resolve.return_value = prefs.resolve_browser_launch_plan(
                prefs.BrowserPreferences(
                    headless_mode=prefs.HEADLESS_ON, profile=prefs.PROFILE_EPHEMERAL
                ),
                chrome=CHROME,
                detect_chrome=False,
            )
            self.run_actions()

        self.assertTrue(self.playwright.chromium.launch_calls[0]["headless"])

    def test_the_dedicated_profile_is_a_persistent_context(self):
        plan = prefs.resolve_browser_launch_plan(
            prefs.BrowserPreferences(), chrome=CHROME, detect_chrome=False
        )
        with mock.patch.object(adapter, "resolve_browser_launch_plan", return_value=plan):
            self.run_actions()

        self.assertEqual(self.playwright.chromium.launch_calls, [], "must not use a throwaway context")
        self.assertEqual(len(self.playwright.chromium.persistent_calls), 1)
        user_data_dir, options = self.playwright.chromium.persistent_calls[0]
        self.assertEqual(user_data_dir, str(prefs.dedicated_profile_dir()))
        self.assertEqual(options["channel"], "chrome")

    def test_a_choice_this_machine_cannot_honour_stops_the_step(self):
        plan = prefs.resolve_browser_launch_plan(
            prefs.BrowserPreferences(browser=prefs.BROWSER_CHROME),
            chrome=None,
            detect_chrome=False,
        )
        with mock.patch.object(adapter, "resolve_browser_launch_plan", return_value=plan):
            result = self.run_actions()

        self.assertEqual(result.status, "waiting_for_user")
        self.assertEqual(result.wait_reason, prefs.BLOCKED_CHROME_NOT_INSTALLED)
        self.assertIn("Chrome", result.prompt or "")
        self.assertEqual(self.playwright.chromium.launch_calls, [])
        self.assertEqual(self.playwright.chromium.persistent_calls, [])


class LoginSurvivesBetweenRunsTest(_AdapterTestCase):
    """A storage state written by one run is the input to the next one."""

    def _ephemeral_plan(self):
        return prefs.resolve_browser_launch_plan(
            prefs.BrowserPreferences(profile=prefs.PROFILE_EPHEMERAL),
            chrome=CHROME,
            detect_chrome=False,
        )

    def test_the_next_run_of_the_same_task_reads_the_previous_state(self):
        store = browser_session_store.get_browser_session_store()
        monday = store.create(run_id="run_monday", task_id="task_daily", step_id="step_1")
        monday_state = Path(monday["context_dir"]) / "storage_state.json"
        monday_state.parent.mkdir(parents=True, exist_ok=True)
        monday_state.write_text(json.dumps({"cookies": ["session"]}), encoding="utf-8")
        store.update(monday["id"], {"storage_state_path": str(monday_state)})
        store.close(monday["id"])

        with mock.patch.object(
            adapter, "resolve_browser_launch_plan", return_value=self._ephemeral_plan()
        ):
            result = self.run_actions(
                context={
                    "run_id": "run_tuesday",
                    "step_id": "step_1",
                    "task_id": "task_daily",
                }
            )

        self.assertEqual(result.status, "completed", result.to_output())
        self.assertEqual(
            self.playwright.chromium.context.storage_state_input,
            str(monday_state),
            "Tuesday's run started signed out",
        )

    def test_a_different_task_does_not_inherit_someone_elses_login(self):
        store = browser_session_store.get_browser_session_store()
        other = store.create(run_id="run_other", task_id="task_other", step_id="step_1")
        other_state = Path(other["context_dir"]) / "storage_state.json"
        other_state.parent.mkdir(parents=True, exist_ok=True)
        other_state.write_text("{}", encoding="utf-8")
        store.update(other["id"], {"storage_state_path": str(other_state)})

        with mock.patch.object(
            adapter, "resolve_browser_launch_plan", return_value=self._ephemeral_plan()
        ):
            self.run_actions(
                context={"run_id": "run_mine", "step_id": "step_1", "task_id": "task_mine"}
            )

        self.assertIsNone(self.playwright.chromium.context.storage_state_input)

    def test_the_current_run_is_excluded_so_it_does_not_read_its_own_half_state(self):
        store = browser_session_store.get_browser_session_store()
        session = store.create(run_id="run_now", task_id="task_daily", step_id="step_1")
        state = Path(session["context_dir"]) / "storage_state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text("{}", encoding="utf-8")
        store.update(session["id"], {"storage_state_path": str(state)})

        self.assertIsNone(
            store.latest_storage_state_for_task("task_daily", exclude_run_id="run_now")
        )
        self.assertEqual(
            store.latest_storage_state_for_task("task_daily"), str(state)
        )


class ReadinessTellsTheTruthTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "readiness.db"
        database._settings_db = None
        database.init_db()
        adapter.reset_browser_readiness_cache()

    def tearDown(self):
        adapter.reset_browser_readiness_cache()
        database._settings_db = None
        database.DB_PATH = self._original_db
        self._tmp.cleanup()

    def _probe(self, *, chrome, chromium_path):
        module = mock.MagicMock()

        class _Ctx:
            async def __aenter__(self_inner):
                playwright = mock.MagicMock()
                playwright.chromium.executable_path = chromium_path
                return playwright

            async def __aexit__(self_inner, *_args):
                return False

        module.async_playwright = _Ctx
        with mock.patch.dict(sys.modules, {"playwright.async_api": module}):
            with mock.patch.object(adapter, "detect_installed_chrome", return_value=chrome):
                return asyncio.run(adapter.get_browser_runtime_readiness(force_refresh=True))

    def test_chrome_present_means_ready_and_no_download(self):
        readiness = self._probe(chrome=CHROME, chromium_path="/nowhere/chromium")
        self.assertTrue(readiness["ready"])
        self.assertFalse(readiness["install_required"])
        self.assertFalse(readiness["chromium_executable"], "the bundled build really is absent")
        self.assertEqual(readiness["plan"]["browser"], "chrome")
        self.assertEqual(readiness["installed_chrome"]["executable_path"], CHROME.executable_path)
        self.assertIn("no download", readiness["message"].lower())

    def test_no_chrome_and_no_chromium_says_what_is_missing(self):
        readiness = self._probe(chrome=None, chromium_path="/nowhere/chromium")
        self.assertFalse(readiness["ready"])
        self.assertTrue(readiness["install_required"])
        self.assertEqual(readiness["plan"]["browser"], "chromium")
        codes = {item["code"] for item in readiness["diagnostics"]}
        self.assertIn("chromium_executable_missing", codes)

    def test_no_chrome_but_chromium_present_is_ready(self):
        readiness = self._probe(chrome=None, chromium_path=str(Path(sys.executable)))
        self.assertTrue(readiness["ready"])
        self.assertFalse(readiness["install_required"])
        self.assertEqual(readiness["plan"]["browser"], "chromium")

    def test_a_blocked_choice_is_not_reported_as_ready(self):
        prefs.set_browser_preferences(browser=prefs.BROWSER_CHROME)
        readiness = self._probe(chrome=None, chromium_path=str(Path(sys.executable)))
        self.assertFalse(
            readiness["ready"],
            "bundled Chromium being present must not satisfy a Chrome-only setting",
        )
        self.assertEqual(
            readiness["plan"]["blocked_reason"], prefs.BLOCKED_CHROME_NOT_INSTALLED
        )

    def test_readiness_reports_the_profile_the_run_will_use(self):
        prefs.set_browser_preferences(profile=prefs.PROFILE_SHARED_CHROME)
        readiness = self._probe(chrome=CHROME, chromium_path="/nowhere/chromium")
        self.assertEqual(readiness["plan"]["user_data_dir"], CHROME.user_data_dir)
        self.assertIn(
            prefs.WARN_SHARED_PROFILE_SCOPE,
            {warning["code"] for warning in readiness["plan"]["warnings"]},
        )


class OrchestratorHandsOverTheLoginTest(unittest.TestCase):
    """The plumbing, not the helper.

    A unit test of `latest_storage_state_for_task` passed while the product did
    not: the orchestrator resolved the previous session with a *run*-scoped
    lookup, so the right answer was computed and nothing consumed it. Every
    assertion here goes through the two functions the orchestrator actually
    calls — `_prepare_browser_session_for_execution` and
    `_browser_session_context` — and asserts on the context dict handed to the
    adapter, which is the thing that was empty.
    """

    TASK = "task_daily_board_check"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._original_db = database.DB_PATH
        database.DB_PATH = self.tmp / "handoff.db"
        database._settings_db = None
        database.init_db()
        browser_session_store._browser_session_store = None
        self._original_root = browser_session_store.BROWSER_SESSION_ROOT
        browser_session_store.BROWSER_SESSION_ROOT = self.tmp / "browser_sessions"
        self.store = browser_session_store.get_browser_session_store()

    def tearDown(self):
        browser_session_store.BROWSER_SESSION_ROOT = self._original_root
        browser_session_store._browser_session_store = None
        database._settings_db = None
        database.DB_PATH = self._original_db
        self._tmp.cleanup()

    def _finished_run_with_a_login(self, *, run_id, task_id, step_id="step_1"):
        """A run that logged in and left its storage state behind, then closed."""
        session = self.store.create(run_id=run_id, task_id=task_id, step_id=step_id)
        state = Path(session["context_dir"]) / "storage_state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"cookies": [{"name": "session"}]}), encoding="utf-8")
        self.store.update(session["id"], {"storage_state_path": str(state)})
        self.store.close(session["id"])
        return session, state

    def _context_for(self, *, run_id, task_id, step_id="step_1"):
        """Exactly what the orchestrator does before calling the adapter."""
        from agent import task_orchestrator as orchestrator

        session = orchestrator._prepare_browser_session_for_execution(
            task_id=task_id,
            run_id=run_id,
            step_id=step_id,
            workflow_step_id=None,
            enabled=True,
        )
        previous = orchestrator._previous_browser_session_for_execution(
            run_id=run_id,
            task_id=task_id,
            current_session_id=session["id"] if session else None,
        )
        context = orchestrator._browser_session_context(
            browser_session=session, previous_session=previous
        )
        return session, context

    def test_tonights_run_is_handed_last_nights_login(self):
        _, yesterday_state = self._finished_run_with_a_login(
            run_id="run_monday", task_id=self.TASK
        )

        _, context = self._context_for(run_id="run_tuesday", task_id=self.TASK)

        self.assertEqual(
            context.get("browser_input_storage_state_path"),
            str(yesterday_state),
            "tonight's run started signed out",
        )

    def test_a_different_task_is_not_handed_someone_elses_login(self):
        self._finished_run_with_a_login(run_id="run_monday", task_id=self.TASK)

        _, context = self._context_for(run_id="run_other", task_id="task_someone_else")

        self.assertIsNone(context.get("browser_input_storage_state_path"))
        self.assertIsNone(context.get("previous_browser_session_id"))

    def test_the_current_runs_own_state_does_not_shadow_a_real_previous_one(self):
        _, yesterday_state = self._finished_run_with_a_login(
            run_id="run_monday", task_id=self.TASK
        )
        # Tonight's own session, already carrying a recorded storage state —
        # the ordering that made the earlier fallback unreachable.
        tonight = self.store.create(
            run_id="run_tuesday", task_id=self.TASK, step_id="step_1"
        )
        tonight_state = Path(tonight["context_dir"]) / "storage_state.json"
        tonight_state.parent.mkdir(parents=True, exist_ok=True)
        tonight_state.write_text("{}", encoding="utf-8")
        self.store.update(tonight["id"], {"storage_state_path": str(tonight_state)})

        session, context = self._context_for(run_id="run_tuesday", task_id=self.TASK)

        self.assertEqual(session["id"], tonight["id"], "should resume its own session")
        self.assertEqual(
            context.get("browser_input_storage_state_path"), str(yesterday_state)
        )

    def test_an_earlier_step_in_the_same_run_still_wins(self):
        # The pre-existing behaviour the cross-run lookup must not displace:
        # two browser steps in one run are one execution.
        _, yesterday_state = self._finished_run_with_a_login(
            run_id="run_monday", task_id=self.TASK
        )
        _, step_one_state = self._finished_run_with_a_login(
            run_id="run_tuesday", task_id=self.TASK, step_id="step_1"
        )

        _, context = self._context_for(
            run_id="run_tuesday", task_id=self.TASK, step_id="step_2"
        )

        self.assertNotEqual(step_one_state, yesterday_state)
        self.assertEqual(
            context.get("browser_input_storage_state_path"), str(step_one_state)
        )

    def test_the_sessions_recorded_provenance_matches_what_was_handed_over(self):
        yesterday, yesterday_state = self._finished_run_with_a_login(
            run_id="run_monday", task_id=self.TASK
        )

        session, context = self._context_for(run_id="run_tuesday", task_id=self.TASK)

        self.assertEqual(context["previous_browser_session_id"], yesterday["id"])
        self.assertEqual(
            session["metadata"].get("previous_browser_session_id"), yesterday["id"]
        )
        self.assertEqual(
            session["metadata"].get("input_storage_state_path"), str(yesterday_state)
        )

    def test_a_login_that_has_been_deleted_off_disk_is_not_claimed(self):
        _, state = self._finished_run_with_a_login(
            run_id="run_monday", task_id=self.TASK
        )
        state.unlink()

        _, context = self._context_for(run_id="run_tuesday", task_id=self.TASK)

        # The context still names it — the store records paths, not files — but
        # the adapter is what decides, and it checks the file exists.
        self.assertIsNone(
            adapter._first_existing_file(
                context.get("browser_input_storage_state_path")
            )
        )


class HttpStatusIsOnTheRecordTest(_AdapterTestCase):
    """A 503 is a 503, not "whatever failed next"."""

    def _run_against_status(self, status):
        page = self.playwright.chromium.context.pages[0]

        class _Response:
            def __init__(self, code):
                self.status = code

        async def goto(url, **_kwargs):
            page.url = url
            return _Response(status) if status is not None else None

        page.goto = goto
        plan = prefs.resolve_browser_launch_plan(
            prefs.BrowserPreferences(profile=prefs.PROFILE_EPHEMERAL),
            chrome=CHROME,
            detect_chrome=False,
        )
        with mock.patch.object(adapter, "resolve_browser_launch_plan", return_value=plan):
            return self.run_actions(
                [{"type": "navigate", "url": "https://example.test/thing"}]
            )

    def test_a_server_error_is_recorded_rather_than_swallowed(self):
        result = self._run_against_status(503)
        self.assertEqual(result.status, "completed", result.to_output())
        self.assertEqual(result.observations[0]["http_status"], 503)
        self.assertFalse(result.observations[0]["http_ok"])

    def test_a_healthy_page_says_so(self):
        result = self._run_against_status(200)
        self.assertEqual(result.observations[0]["http_status"], 200)
        self.assertTrue(result.observations[0]["http_ok"])

    def test_an_error_page_does_not_park_the_run(self):
        # "Is the site up?" is one of the things these agents are for. Parking
        # on a 503 would break the check that wants to observe it.
        result = self._run_against_status(503)
        self.assertNotEqual(result.status, "waiting_for_user")
        self.assertIsNone(result.wait_reason)

    def test_an_unknown_status_is_omitted_not_guessed(self):
        result = self._run_against_status(None)
        self.assertNotIn("http_status", result.observations[0])


class CommitGateStaysHonestTest(unittest.TestCase):
    """"Not ready" no longer always means "one download away"."""

    FLOW = [{"id": "s1", "type": "browser_action"}]

    def _findings(self, readiness):
        from agent.workflow_contract import _check_browser_runtime

        return _check_browser_runtime(self.FLOW, readiness)

    def test_a_missing_download_still_hands_over_the_install_command(self):
        findings = self._findings(
            {
                "ready": False,
                "install_required": True,
                "install_command": "/opt/venv/bin/python -m playwright install chromium",
                "message": "Playwright Chromium executable is missing.",
            }
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("playwright install chromium", findings[0].ask)

    def test_a_setting_this_machine_cannot_honour_is_not_answered_with_a_download(self):
        findings = self._findings(
            {
                "ready": False,
                "install_required": False,
                "install_command": "/opt/venv/bin/python -m playwright install chromium",
                "message": "This server is set to use installed Google Chrome, but no Chrome was found on it.",
            }
        )
        self.assertEqual(len(findings), 1)
        self.assertNotIn("playwright install chromium", findings[0].ask)
        self.assertIn("Chrome", findings[0].ask)

    def test_a_ready_runtime_produces_no_finding(self):
        self.assertEqual(self._findings({"ready": True, "install_required": False}), [])

    def test_the_phone_is_not_handed_a_command_that_would_change_nothing(self):
        # `commit_readiness_views.dart` renders a copyable install command
        # whenever one is present, so carrying it for a setting problem would
        # put a wrong instruction on the phone.
        from agent.workflow_contract import ContractReport
        from routes.agents import _contract_readiness_fact

        findings = self._findings(
            {
                "ready": False,
                "install_required": False,
                "install_command": "/opt/venv/bin/python -m playwright install chromium",
                "message": "No Chrome was found on this server.",
            }
        )
        payload = _contract_readiness_fact(
            ContractReport(findings=findings), saved_incomplete=False
        )
        self.assertIsNotNone(payload["browser_runtime"])
        self.assertIsNone(payload["browser_runtime"]["install_command"])
        self.assertFalse(payload["browser_runtime"]["install_required"])
        self.assertIn("Chrome", payload["browser_runtime"]["message"])


if __name__ == "__main__":
    unittest.main()
