"""PATH bootstrap must repair a bare launchd PATH without ever blocking startup.

launchd / a macOS login item starts the server with PATH=/usr/bin:/bin:...,
which hides every Homebrew/npm/Android SDK install from shutil.which(). These
tests pin down the contract of core.env_bootstrap.bootstrap_path(): static
candidates are only added when they exist on disk, everything is
de-duplicated, the caller's existing PATH entries are preserved (never
dropped), the login-shell probe is merged in as a supplement (not a
replacement) for the static list, and any failure in the shell probe -
non-zero exit, timeout, missing $SHELL - degrades harmlessly to "static list
only" rather than raising or blocking startup.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import core.env_bootstrap as env_bootstrap  # noqa: E402


class StaticCandidatesTest(unittest.TestCase):
    """_static_candidates() only returns directories that exist."""

    def test_only_existing_directories_are_returned(self):
        existing = {os.path.expanduser("/opt/homebrew/bin"), os.path.expanduser("~/.local/bin")}

        with patch("os.path.isdir", side_effect=lambda p: p in existing):
            result = env_bootstrap._static_candidates()

        self.assertEqual(set(result), existing)

    def test_no_directories_exist_yields_empty_list(self):
        with patch("os.path.isdir", return_value=False):
            result = env_bootstrap._static_candidates()

        self.assertEqual(result, [])

    def test_all_static_candidates_checked(self):
        with patch("os.path.isdir", return_value=True) as mock_isdir:
            result = env_bootstrap._static_candidates()

        self.assertEqual(len(result), len(env_bootstrap._STATIC_PATH_CANDIDATES))
        self.assertEqual(mock_isdir.call_count, len(env_bootstrap._STATIC_PATH_CANDIDATES))


class LoginShellPathTest(unittest.TestCase):
    """_login_shell_path() is a best-effort probe that never raises."""

    def test_missing_shell_env_returns_empty(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SHELL", None)
            result = env_bootstrap._login_shell_path()

        self.assertEqual(result, [])

    def test_successful_probe_splits_path_entries(self):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="/usr/bin:/bin:/opt/homebrew/bin")

        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}), patch("subprocess.run", return_value=fake):
            result = env_bootstrap._login_shell_path()

        self.assertEqual(result, ["/usr/bin", "/bin", "/opt/homebrew/bin"])

    def test_multiline_output_takes_only_last_non_empty_line(self):
        # rc files can print banners/warnings to stdout ahead of the PATH
        # line the printf command actually produces.
        noisy = "Warning: some rc file banner\n\n/usr/bin:/bin:/opt/homebrew/bin\n"
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=noisy)

        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}), patch("subprocess.run", return_value=fake):
            result = env_bootstrap._login_shell_path()

        self.assertEqual(result, ["/usr/bin", "/bin", "/opt/homebrew/bin"])

    def test_nonzero_exit_returns_empty(self):
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="/usr/bin:/bin")

        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}), patch("subprocess.run", return_value=fake):
            result = env_bootstrap._login_shell_path()

        self.assertEqual(result, [])

    def test_timeout_returns_empty(self):
        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}), patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="zsh", timeout=5),
        ):
            result = env_bootstrap._login_shell_path()

        self.assertEqual(result, [])

    def test_oserror_returns_empty(self):
        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}), patch(
            "subprocess.run", side_effect=OSError("no such shell")
        ):
            result = env_bootstrap._login_shell_path()

        self.assertEqual(result, [])

    def test_empty_output_returns_empty(self):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="   \n  \n")

        with patch.dict(os.environ, {"SHELL": "/bin/zsh"}), patch("subprocess.run", return_value=fake):
            result = env_bootstrap._login_shell_path()

        self.assertEqual(result, [])


class BootstrapPathTest(unittest.TestCase):
    """bootstrap_path() merges static + shell candidates into env['PATH']."""

    def test_static_and_shell_candidates_prepended_in_order(self):
        env = {"PATH": "/usr/bin:/bin"}

        with patch("core.env_bootstrap._static_candidates", return_value=["/opt/homebrew/bin"]), patch(
            "core.env_bootstrap._login_shell_path", return_value=["/usr/local/bin"]
        ):
            env_bootstrap.bootstrap_path(env)

        self.assertEqual(env["PATH"], "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")

    def test_existing_path_entries_are_preserved(self):
        env = {"PATH": "/usr/bin:/bin:/custom/tool"}

        with patch("core.env_bootstrap._static_candidates", return_value=[]), patch(
            "core.env_bootstrap._login_shell_path", return_value=[]
        ):
            env_bootstrap.bootstrap_path(env)

        for entry in ("/usr/bin", "/bin", "/custom/tool"):
            self.assertIn(entry, env["PATH"].split(os.pathsep))

    def test_deduplicates_entries_already_on_path(self):
        env = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin"}

        with patch(
            "core.env_bootstrap._static_candidates",
            return_value=["/opt/homebrew/bin", "/usr/local/bin"],
        ), patch("core.env_bootstrap._login_shell_path", return_value=["/usr/local/bin"]):
            env_bootstrap.bootstrap_path(env)

        entries = env["PATH"].split(os.pathsep)
        self.assertEqual(entries.count("/opt/homebrew/bin"), 1)
        self.assertEqual(entries.count("/usr/local/bin"), 1)

    def test_deduplicates_between_static_and_shell_sources(self):
        env = {"PATH": "/usr/bin"}

        with patch(
            "core.env_bootstrap._static_candidates",
            return_value=["/opt/homebrew/bin"],
        ), patch(
            "core.env_bootstrap._login_shell_path",
            return_value=["/opt/homebrew/bin", "/usr/local/bin"],
        ):
            env_bootstrap.bootstrap_path(env)

        entries = env["PATH"].split(os.pathsep)
        self.assertEqual(entries.count("/opt/homebrew/bin"), 1)
        self.assertEqual(entries, ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"])

    def test_no_candidates_leaves_path_unchanged(self):
        env = {"PATH": "/usr/bin:/bin"}

        with patch("core.env_bootstrap._static_candidates", return_value=[]), patch(
            "core.env_bootstrap._login_shell_path", return_value=[]
        ):
            env_bootstrap.bootstrap_path(env)

        self.assertEqual(env["PATH"], "/usr/bin:/bin")

    def test_shell_probe_failure_falls_back_to_static_list_only(self):
        """A real failing subprocess.run() must still leave a healthy PATH."""
        env = {"PATH": "/usr/bin:/bin"}

        with patch(
            "core.env_bootstrap._static_candidates",
            return_value=["/opt/homebrew/bin"],
        ), patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="zsh", timeout=5)), patch.dict(
            os.environ, {"SHELL": "/bin/zsh"}
        ):
            env_bootstrap.bootstrap_path(env)

        self.assertEqual(env["PATH"], "/opt/homebrew/bin:/usr/bin:/bin")

    def test_missing_path_key_is_handled(self):
        env: dict[str, str] = {}

        with patch("core.env_bootstrap._static_candidates", return_value=["/opt/homebrew/bin"]), patch(
            "core.env_bootstrap._login_shell_path", return_value=[]
        ):
            env_bootstrap.bootstrap_path(env)

        self.assertEqual(env["PATH"], "/opt/homebrew/bin")

    def test_defaults_to_os_environ_when_no_env_given(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=False), patch(
            "core.env_bootstrap._static_candidates", return_value=["/opt/homebrew/bin"]
        ), patch("core.env_bootstrap._login_shell_path", return_value=[]):
            env_bootstrap.bootstrap_path()

            self.assertEqual(os.environ["PATH"], "/opt/homebrew/bin:/usr/bin:/bin")

    def test_never_raises_even_if_internals_blow_up(self):
        env = {"PATH": "/usr/bin:/bin"}

        with patch("core.env_bootstrap._static_candidates", side_effect=RuntimeError("boom")):
            try:
                env_bootstrap.bootstrap_path(env)
            except Exception as exc:  # pragma: no cover - failure path under test
                self.fail(f"bootstrap_path() raised {exc!r}; it must never block startup")

        # Untouched, since the failure happened before any mutation.
        self.assertEqual(env["PATH"], "/usr/bin:/bin")


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class BootstrapLogReplayTest(unittest.TestCase):
    """bootstrap_path() is silent; its summary replays after logging exists.

    The bootstrap runs at server_cli.main() top, before
    configure_server_logging() attaches any handler, so an immediate log line
    used to vanish. These tests pin down the replacement contract: bootstrap
    records, emit_recorded_bootstrap_log() emits exactly once through
    handlers attached *after* the bootstrap ran.
    """

    def setUp(self) -> None:
        import core.env_bootstrap as env_bootstrap

        self.env_bootstrap = env_bootstrap
        self._saved = (env_bootstrap._recorded_summary, env_bootstrap._summary_emitted)
        env_bootstrap._recorded_summary = None
        env_bootstrap._summary_emitted = False
        self.logger = logging.getLogger("core.env_bootstrap")
        self._saved_level = self.logger.level
        self.handler = _ListHandler()

    def tearDown(self) -> None:
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._saved_level)
        (
            self.env_bootstrap._recorded_summary,
            self.env_bootstrap._summary_emitted,
        ) = self._saved

    def _bootstrap(self) -> None:
        with patch(
            "core.env_bootstrap._static_candidates", return_value=["/opt/homebrew/bin"]
        ), patch("core.env_bootstrap._login_shell_path", return_value=[]):
            self.env_bootstrap.bootstrap_path({"PATH": "/usr/bin:/bin"})

    def test_bootstrap_emits_nothing_itself(self):
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

        self._bootstrap()

        self.assertEqual(self.handler.records, [])

    def test_summary_reaches_handler_attached_after_bootstrap(self):
        self._bootstrap()

        # Handler attached only now — the situation configure_server_logging
        # creates on a normal start.
        self.logger.addHandler(self.handler)

        self.env_bootstrap.emit_recorded_bootstrap_log()

        self.assertEqual(len(self.handler.records), 1)
        message = self.handler.records[0].getMessage()
        self.assertIn("PATH bootstrapped", message)
        self.assertIn("/opt/homebrew/bin", message)

    def test_summary_emitted_at_most_once(self):
        self._bootstrap()
        self.logger.addHandler(self.handler)

        self.env_bootstrap.emit_recorded_bootstrap_log()
        self.env_bootstrap.emit_recorded_bootstrap_log()  # dual-server: called twice

        self.assertEqual(len(self.handler.records), 1)

    def test_emit_without_bootstrap_is_a_noop(self):
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

        self.env_bootstrap.emit_recorded_bootstrap_log()

        self.assertEqual(self.handler.records, [])

    def test_configure_server_logging_replays_summary_into_server_log(self):
        """End to end: the line lands in the actual server.log file."""
        from core.server_logging import configure_server_logging

        self._bootstrap()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            root_logger = logging.getLogger("")
            with patch.dict(
                os.environ, {"CODE_BRIDGE_SERVER_LOG_PATH": str(log_path)}
            ):
                configure_server_logging("info")
            try:
                content = log_path.read_text(encoding="utf-8")
            finally:
                for handler in list(root_logger.handlers):
                    if getattr(handler, "baseFilename", None) == str(log_path):
                        root_logger.removeHandler(handler)
                        handler.close()

        self.assertIn("PATH bootstrapped", content)
        self.assertIn("/opt/homebrew/bin", content)


if __name__ == "__main__":
    unittest.main()
