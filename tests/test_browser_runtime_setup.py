"""The one place that decides whether this machine needs a 450MB download.

`install/install.sh`, `install/install.ps1`, `install/sync-local-install.sh`
and the dashboard install job all route through
`system/browser_runtime_setup.py`. What is asserted here is what those four
callers rely on and what nobody could verify while the logic lived in three
separate shell scripts:

*   a machine that already has Chromium downloads nothing,
*   `CODE_BRIDGE_BROWSER=0` is honoured,
*   a dry run downloads nothing,
*   and an install command that exits 0 without leaving an executable behind
    is reported as a failure, not as success.
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from system import browser_runtime_setup as setup  # noqa: E402


READY = setup.ProbeResult(
    status=setup.STATUS_READY,
    ready=True,
    message="Playwright Chromium is ready.",
    executable_path="/tmp/chrome",
)
MISSING = setup.ProbeResult(
    status=setup.STATUS_CHROMIUM_MISSING,
    ready=False,
    message="Playwright Chromium executable is missing.",
    executable_path="/tmp/chrome",
)
NO_PACKAGE = setup.ProbeResult(
    status=setup.STATUS_PLAYWRIGHT_MISSING,
    ready=False,
    message="Python Playwright package is not available.",
)


class _Recorder:
    """Stands in for the subprocess that would download 450MB."""

    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.returncode


def _probes(*results: setup.ProbeResult):
    """Return probe results in order, repeating the last one."""
    remaining = list(results)

    def probe() -> setup.ProbeResult:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return probe


class InstallCommandTest(unittest.TestCase):
    def test_command_is_derived_from_the_running_interpreter(self):
        argv = setup.chromium_install_argv()
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:], ["-m", "playwright", "install", "chromium"])

    def test_a_path_with_spaces_stays_runnable_when_quoted(self):
        command = setup.chromium_install_command("/Applications/My Python/bin/python")
        self.assertIn("'/Applications/My Python/bin/python'", command)

    def test_the_cost_is_stated_in_the_notice(self):
        notice = setup.cost_notice()
        self.assertIn("200MB", notice)
        self.assertIn("450MB", notice)


class OptOutTest(unittest.TestCase):
    def test_zero_means_opted_out(self):
        self.assertTrue(setup.browser_runtime_opt_out({"CODE_BRIDGE_BROWSER": "0"}))

    def test_unset_means_enabled(self):
        self.assertFalse(setup.browser_runtime_opt_out({}))

    def test_any_other_value_means_enabled(self):
        self.assertFalse(setup.browser_runtime_opt_out({"CODE_BRIDGE_BROWSER": "1"}))

    def test_ensure_downloads_nothing_when_opted_out(self):
        runner = _Recorder()
        code, result = ensure(
            env={"CODE_BRIDGE_BROWSER": "0"},
            probe=_probes(MISSING),
            runner=runner,
        )
        self.assertEqual(code, setup.EXIT_OK)
        self.assertEqual(runner.calls, [])
        self.assertFalse(result.ready)


class AlreadyPresentTest(unittest.TestCase):
    def test_a_machine_that_has_chromium_downloads_nothing(self):
        """The 'no repeated 450MB' guarantee every installer leans on."""
        runner = _Recorder()
        code, result = ensure(probe=_probes(READY), runner=runner)
        self.assertEqual(code, setup.EXIT_OK)
        self.assertEqual(runner.calls, [], "must not shell out when Chromium is present")
        self.assertTrue(result.ready)

    def test_dry_run_reports_the_gap_without_downloading(self):
        runner = _Recorder()
        lines: list[str] = []
        code, result = ensure(
            dry_run=True,
            probe=_probes(MISSING),
            runner=runner,
            log=lines.append,
        )
        self.assertEqual(code, setup.EXIT_CHROMIUM_MISSING)
        self.assertEqual(runner.calls, [])
        self.assertFalse(result.ready)
        self.assertTrue(any("would run" in line for line in lines))
        self.assertTrue(any("450MB" in line for line in lines))


class InstallHonestyTest(unittest.TestCase):
    def test_a_successful_install_is_confirmed_by_a_second_probe(self):
        runner = _Recorder(returncode=0)
        code, result = ensure(probe=_probes(MISSING, READY), runner=runner)
        self.assertEqual(code, setup.EXIT_OK)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0][1:], ["-m", "playwright", "install", "chromium"])
        self.assertTrue(result.ready)

    def test_exit_zero_without_an_executable_is_a_failure(self):
        """The whole reason readiness is re-probed rather than inferred."""
        runner = _Recorder(returncode=0)
        code, result = ensure(probe=_probes(MISSING, MISSING), runner=runner)
        self.assertEqual(code, setup.EXIT_INSTALL_FAILED)
        self.assertFalse(result.ready)

    def test_a_failing_command_is_reported_as_a_failure(self):
        runner = _Recorder(returncode=1)
        lines: list[str] = []
        code, result = ensure(probe=_probes(MISSING, MISSING), runner=runner, log=lines.append)
        self.assertEqual(code, setup.EXIT_INSTALL_FAILED)
        self.assertFalse(result.ready)
        self.assertTrue(any("exit 1" in line for line in lines))

    def test_a_missing_playwright_package_is_pips_problem_not_ours(self):
        runner = _Recorder()
        code, _ = ensure(probe=_probes(NO_PACKAGE), runner=runner)
        self.assertEqual(code, setup.EXIT_PLAYWRIGHT_MISSING)
        self.assertEqual(runner.calls, [], "running `playwright install` cannot fix a missing import")


CHROME = setup.InstalledBrowser(
    channel="chrome",
    name="Google Chrome",
    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    user_data_dir="/Users/someone/Library/Application Support/Google/Chrome",
)


class InstalledChromeTest(unittest.TestCase):
    """A machine with Chrome must not be made to download a second browser."""

    def test_chrome_short_circuits_the_download(self):
        runner = _Recorder()
        lines: list[str] = []
        code, result = ensure(
            probe=_probes(MISSING),
            runner=runner,
            chrome_detector=lambda: CHROME,
            log=lines.append,
        )
        self.assertEqual(code, setup.EXIT_OK)
        self.assertEqual(runner.calls, [], "installed Chrome means nothing to download")
        self.assertTrue(result.ready)
        self.assertEqual(result.channel, "chrome")
        self.assertTrue(any("no download needed" in line.lower() for line in lines))

    def test_force_chromium_downloads_even_with_chrome_present(self):
        runner = _Recorder(returncode=0)
        code, result = ensure(
            probe=_probes(MISSING, READY),
            runner=runner,
            chrome_detector=lambda: CHROME,
            allow_installed_chrome=False,
        )
        self.assertEqual(code, setup.EXIT_OK)
        self.assertEqual(len(runner.calls), 1)
        self.assertTrue(result.ready)

    def test_opt_out_still_wins_over_installed_chrome(self):
        runner = _Recorder()
        code, _ = ensure(
            env={"CODE_BRIDGE_BROWSER": "0"},
            probe=_probes(MISSING),
            runner=runner,
            chrome_detector=lambda: CHROME,
        )
        self.assertEqual(code, setup.EXIT_OK)
        self.assertEqual(runner.calls, [])

    def test_detection_looks_at_the_filesystem_not_the_platform(self):
        # A Mac whose Chrome is not where Chrome lives must answer "none".
        # Assuming Chrome from the platform is what would make readiness claim
        # no download is needed and then park every browser step at run time.
        with unittest.mock.patch.dict(
            setup._CHROME_EXECUTABLES,
            {"darwin": ("/nowhere/Google Chrome",)},
            clear=False,
        ):
            self.assertIsNone(
                setup.detect_installed_chrome(platform="darwin", env={})
            )

    def test_an_explicit_path_that_exists_is_accepted(self):
        found = setup.detect_installed_chrome(
            platform="darwin",
            env={"CODE_BRIDGE_CHROME_PATH": sys.executable},
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.executable_path, sys.executable)
        self.assertEqual(found.channel, setup.CHROME_CHANNEL)

    def test_probe_reports_chrome_without_needing_the_bundled_build(self):
        result = setup.probe_browser_runtime(chrome_detector=lambda: CHROME)
        self.assertTrue(result.ready)
        self.assertEqual(result.status, setup.STATUS_READY)
        self.assertEqual(result.browser_name, "Google Chrome")


def ensure(**kwargs):
    kwargs.setdefault("env", {})
    kwargs.setdefault("log", lambda _message: None)
    # Default to "no Chrome on this machine" so the download-path tests below
    # assert the download path, on a host with Chrome or without one. Before
    # this the same tests would have passed or failed depending on what
    # happened to be installed on the machine running them.
    kwargs.setdefault("chrome_detector", lambda: None)
    return setup.ensure_chromium(**kwargs)


if __name__ == "__main__":
    unittest.main()
