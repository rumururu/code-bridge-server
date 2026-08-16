"""The dashboard's one-click browser runtime install.

The product already said the browser runtime was missing — a red chip, a
commit-gate warning, a copy-paste command. What it could not do was fix it.
These tests cover the job that fixes it, and specifically the ways such a job
lies if you let it:

*   reporting success because the install command exited 0,
*   turning readiness green because an install was *started*,
*   starting a second 450MB download on a second click,
*   and ignoring an operator who said this machine must not have Chromium.

Also asserted: creating the apps starts nothing. A missing browser runtime must
never block a server start or a run.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import app_factory  # noqa: E402
from routes.deps import require_local_access  # noqa: E402
from system import browser_runtime_install_jobs as jobs  # noqa: E402

READY = {"ready": True, "message": "Playwright Chromium is ready."}
NOT_READY = {"ready": False, "message": "Playwright Chromium executable is missing."}


class _FakeProcess:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr

    def terminate(self):  # pragma: no cover - only used by cancel paths
        self.returncode = -15

    async def wait(self):  # pragma: no cover
        return self.returncode


def _no_real_download():
    """Belt and braces: no test in this file may ever spawn a real download.

    The route tests patch the manager, but a patch aimed at the wrong module
    once let a real `playwright install chromium` start from a unit test. This
    makes that impossible rather than unlikely.
    """

    async def refuse(*argv, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError(f"test tried to spawn a real subprocess: {argv}")

    return mock.patch("asyncio.create_subprocess_exec", refuse)


def _spawn(process: _FakeProcess, recorder: list[list[str]] | None = None):
    async def create(*argv, **_kwargs):
        if recorder is not None:
            recorder.append(list(argv))
        return process

    return create


def _run(manager: jobs.BrowserRuntimeInstallJobManager, readiness: dict) -> jobs.BrowserRuntimeInstallJob:
    """Start one job and wait for it to finish."""

    async def go():
        job = await manager.start_install()
        task = manager._tasks[job.job_id]  # noqa: SLF001 - deterministic wait in tests
        await task
        return job

    with mock.patch.object(
        jobs, "get_browser_runtime_readiness", mock.AsyncMock(return_value=readiness)
    ), mock.patch.object(jobs, "reset_browser_readiness_cache", mock.Mock()):
        return asyncio.run(go())


class InstallJobHonestyTest(unittest.TestCase):
    def setUp(self):
        self.manager = jobs.BrowserRuntimeInstallJobManager()

    def test_a_failing_command_fails_the_job_and_says_why(self):
        with mock.patch(
            "asyncio.create_subprocess_exec",
            _spawn(_FakeProcess(1, stderr=b"network unreachable")),
        ):
            job = _run(self.manager, NOT_READY)

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error_code, jobs.BROWSER_INSTALL_FAILED)
        self.assertIs(job.installed, False)
        self.assertIn("network unreachable", job.error_message or "")

    def test_exit_zero_without_a_usable_runtime_is_still_a_failure(self):
        """`installed` follows the re-probe, never the exit code."""
        with mock.patch(
            "asyncio.create_subprocess_exec", _spawn(_FakeProcess(0, stdout=b"done"))
        ):
            job = _run(self.manager, NOT_READY)

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error_code, jobs.BROWSER_RUNTIME_UNAVAILABLE)
        self.assertIs(job.installed, False)
        self.assertIs(self.manager.serialize(job)["readiness"]["ready"], False)

    def test_success_requires_the_probe_to_say_ready(self):
        with mock.patch(
            "asyncio.create_subprocess_exec", _spawn(_FakeProcess(0, stdout=b"done"))
        ):
            job = _run(self.manager, READY)

        self.assertEqual(job.status, "completed")
        self.assertIs(job.installed, True)
        self.assertIsNone(job.error_code)
        payload = self.manager.serialize(job)
        self.assertTrue(payload["finished"])
        self.assertIs(payload["readiness"]["ready"], True)

    def test_the_command_is_the_one_the_diagnostics_hand_out(self):
        from agent.browser_action_adapter import PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND

        recorder: list[list[str]] = []
        with mock.patch(
            "asyncio.create_subprocess_exec", _spawn(_FakeProcess(0), recorder)
        ):
            job = _run(self.manager, READY)

        self.assertEqual(len(recorder), 1)
        self.assertEqual(recorder[0][1:], ["-m", "playwright", "install", "chromium"])
        self.assertEqual(job.command[0], sys.executable)
        # The chip's copy-paste command and the button must not drift apart.
        self.assertIn("-m playwright install chromium", PLAYWRIGHT_CHROMIUM_INSTALL_COMMAND)

    def test_the_stated_cost_travels_with_the_job(self):
        with mock.patch("asyncio.create_subprocess_exec", _spawn(_FakeProcess(0))):
            job = _run(self.manager, READY)
        payload = self.manager.serialize(job)
        self.assertEqual(payload["download_mb"], 200)
        self.assertEqual(payload["disk_mb"], 450)


class OptOutTest(unittest.TestCase):
    def test_an_opted_out_machine_refuses_to_start_a_job(self):
        manager = jobs.BrowserRuntimeInstallJobManager()
        with mock.patch.object(jobs, "browser_runtime_opt_out", return_value=True):
            with self.assertRaises(jobs.BrowserRuntimeInstallDisabled):
                asyncio.run(manager.start_install())
        self.assertIsNone(manager.active_job())


class SingleFlightTest(unittest.TestCase):
    def test_a_second_request_joins_the_running_job(self):
        """Two clicks must not become two 450MB downloads."""
        manager = jobs.BrowserRuntimeInstallJobManager()
        started = asyncio.Event()
        release = asyncio.Event()
        spawned: list[list[str]] = []

        async def create(*argv, **_kwargs):
            spawned.append(list(argv))
            started.set()
            await release.wait()
            return _FakeProcess(0)

        async def go():
            first = await manager.start_install()
            await started.wait()
            second = await manager.start_install()
            release.set()
            await manager._tasks[first.job_id]  # noqa: SLF001
            return first, second

        with mock.patch("asyncio.create_subprocess_exec", create), mock.patch.object(
            jobs, "get_browser_runtime_readiness", mock.AsyncMock(return_value=READY)
        ), mock.patch.object(jobs, "reset_browser_readiness_cache", mock.Mock()):
            first, second = asyncio.run(go())

        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(len(spawned), 1)


class InstallRouteTest(unittest.TestCase):
    def setUp(self):
        self.app = app_factory.create_dashboard_app()
        self.app.dependency_overrides[require_local_access] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_the_trigger_is_localhost_only_and_never_on_the_tunnel(self):
        """A leaked pairing key must not be able to make a host fetch 450MB."""
        api_paths = {
            getattr(route, "path", "") for route in app_factory.create_api_app().routes
        }
        self.assertNotIn("/api/agent/browser-runtime/install", api_paths)
        dashboard_paths = {getattr(route, "path", "") for route in self.app.routes}
        self.assertIn("/api/dashboard/agent/browser-runtime/install", dashboard_paths)

    def test_starting_an_install_answers_202_with_a_job_to_poll(self):
        manager = jobs.BrowserRuntimeInstallJobManager()
        job = jobs.BrowserRuntimeInstallJob(job_id="abc123", command=["python"])
        with _no_real_download(), mock.patch(
            "routes.dashboard_agents.get_browser_runtime_install_job_manager",
            return_value=manager,
        ), mock.patch.object(
            manager, "start_install", mock.AsyncMock(return_value=job)
        ):
            response = self.client.post("/api/dashboard/agent/browser-runtime/install")

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["job_id"], "abc123")
        self.assertEqual(body["status"], "queued")
        self.assertIsNone(body["installed"])
        self.assertEqual(body["download_mb"], 200)

    def test_an_opted_out_machine_answers_409_with_a_named_reason(self):
        manager = jobs.BrowserRuntimeInstallJobManager()
        with _no_real_download(), mock.patch(
            "routes.dashboard_agents.get_browser_runtime_install_job_manager",
            return_value=manager,
        ), mock.patch.object(
            manager,
            "start_install",
            mock.AsyncMock(side_effect=jobs.BrowserRuntimeInstallDisabled("switched off")),
        ):
            response = self.client.post("/api/dashboard/agent/browser-runtime/install")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error_code"], jobs.BROWSER_INSTALL_DISABLED)

    def test_an_unknown_job_is_404_not_an_invented_status(self):
        response = self.client.get(
            "/api/dashboard/agent/browser-runtime/install/jobs/nope"
        )
        self.assertEqual(response.status_code, 404)
        response = self.client.post(
            "/api/dashboard/agent/browser-runtime/install/jobs/nope/cancel"
        )
        self.assertEqual(response.status_code, 404)

    def test_building_the_apps_starts_no_install(self):
        """A missing browser runtime must never block a server start."""
        app_factory.create_dashboard_app()
        app_factory.create_api_app()
        self.assertIsNone(jobs.get_browser_runtime_install_job_manager().active_job())


class CancelTest(unittest.TestCase):
    def test_a_cancelled_job_never_claims_it_installed_anything(self):
        manager = jobs.BrowserRuntimeInstallJobManager()
        job = jobs.BrowserRuntimeInstallJob(job_id="c1", command=["python"], status="running")
        manager._jobs[job.job_id] = job  # noqa: SLF001

        cancelled = asyncio.run(manager.cancel_job("c1"))
        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertIs(cancelled.installed, False)


class DashboardSurfaceTest(unittest.TestCase):
    """The button has to exist on the page, and only where it is honest."""

    @classmethod
    def setUpClass(cls):
        cls.markup = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
            encoding="utf-8"
        )

    def test_the_modal_offers_an_install_button(self):
        self.assertIn("startBrowserRuntimeInstall", self.markup)
        self.assertIn("'/browser-runtime/install'", self.markup)

    def test_the_cost_is_stated_before_the_button(self):
        cost_at = self.markup.index("modal_browser_cost')")
        button_at = self.markup.index("modal_browser_install')")
        self.assertLess(cost_at, button_at)

    def test_success_is_taken_from_the_servers_reprobe_not_the_exit_code(self):
        self.assertIn("job.installed === true", self.markup)

    def test_the_button_is_only_offered_when_the_probe_answered_not_ready(self):
        self.assertIn("known && browserRuntime.ready !== true", self.markup)

    def test_every_new_string_exists_in_both_locales(self):
        for key in (
            "modal_browser_install:",
            "modal_browser_cost:",
            "modal_browser_installing:",
            "modal_browser_install_done:",
            "modal_browser_install_failed:",
            "modal_browser_install_disabled:",
        ):
            self.assertEqual(
                self.markup.count(key), 2, f"{key} must be defined in both en and ko"
            )


if __name__ == "__main__":
    unittest.main()
