"""The deploy path has to carry the browser runtime, not just the installer.

`install/install.sh` grew a Chromium step; `install/sync-local-install.sh` —
the path actually used to deploy every day — did not. The result was a machine
with the `playwright` Python package importable and no browser behind it, so
every `browser_action` step failed within seconds. Nothing caught it because no
test ever looked at what the sync script does.

These tests run `sync-local-install.sh --browser-only` against a throwaway
install directory whose `venv/bin/python` is a stub that records its arguments
and exits with a chosen code. Nothing here can reach the network, touch
~/.code-bridge, or download anything: `--browser-only` performs no rsync, no
pip and no integrity report, and the stub is a shell script.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = REPO_ROOT / "install" / "sync-local-install.sh"
INSTALL_SH = REPO_ROOT / "install" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install" / "install.ps1"

# The stub stands in for $INSTALL_DIR/venv/bin/python. It appends its argv to a
# log and exits with $STUB_EXIT, so a test can play "already present" (0) or
# "missing, install failed" (5) without any of it being real.
_STUB = """#!/bin/bash
printf '%s\\n' "$*" >> "$STUB_LOG"
exit "${STUB_EXIT:-0}"
"""


class SyncScriptBrowserStepTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.install_dir = Path(self._tmp.name)
        (self.install_dir / "venv" / "bin").mkdir(parents=True)
        (self.install_dir / "system").mkdir()
        # Only its presence matters: the stub python never reads it.
        (self.install_dir / "system" / "browser_runtime_setup.py").write_text("", encoding="utf-8")
        self.python = self.install_dir / "venv" / "bin" / "python"
        self.python.write_text(_STUB, encoding="utf-8")
        self.python.chmod(0o755)
        self.log = self.install_dir / "stub.log"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *args: str, stub_exit: int = 0) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "STUB_LOG": str(self.log),
            "STUB_EXIT": str(stub_exit),
            # Belt and braces: even if an argument were mis-parsed, the target
            # is still the throwaway directory, never ~/.code-bridge.
            "CODE_BRIDGE_INSTALL_DIR": str(self.install_dir),
        }
        return subprocess.run(
            ["bash", str(SYNC_SCRIPT), "--browser-only", "--install-dir", str(self.install_dir), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

    def _stub_calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return [line for line in self.log.read_text(encoding="utf-8").splitlines() if line]

    def test_browser_only_delegates_to_the_shared_helper(self):
        result = self._run("--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self._stub_calls()
        self.assertEqual(len(calls), 1, f"expected exactly one helper call, got {calls}")
        self.assertIn("browser_runtime_setup.py", calls[0])
        self.assertIn("--ensure", calls[0])
        self.assertNotIn("--dry-run", calls[0])

    def test_a_dry_run_may_not_download_anything(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self._stub_calls()
        self.assertEqual(len(calls), 1)
        self.assertIn("--dry-run", calls[0], "without --apply the step must only report")

    def test_browser_only_does_no_rsync_and_no_pip(self):
        """It exists for the already-deployed machine that just lacks Chromium."""
        result = self._run("--apply")
        self.assertNotIn("dependencies", result.stdout)
        self.assertNotIn("integrity report", result.stdout)
        self.assertNotIn("desktop_server_app", result.stdout)

    def test_a_failing_helper_never_fails_the_deploy(self):
        """A server with no browser starts fine and says so; it is not an error."""
        result = self._run("--apply", stub_exit=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self._stub_calls()), 1)

    def test_no_venv_python_is_reported_not_guessed(self):
        self.python.unlink()
        result = self._run("--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("venv python not found", result.stdout)
        self.assertEqual(self._stub_calls(), [])

    def test_an_undeployed_helper_is_reported_not_guessed(self):
        (self.install_dir / "system" / "browser_runtime_setup.py").unlink()
        result = self._run("--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not deployed yet", result.stdout)
        self.assertEqual(self._stub_calls(), [])

    def test_no_browser_skips_the_step_in_a_full_run(self):
        """The opt-out has to exist on the flag as well as in the environment."""
        source = SYNC_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--no-browser)   SKIP_BROWSER=1", source)
        self.assertIn('skipped (--no-browser)', source)

    def test_the_full_run_reaches_the_browser_step(self):
        """The gap that started all of this: the sync path never called it."""
        source = SYNC_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--- browser runtime", source)
        # Called once for --browser-only and once at the end of a full run.
        self.assertEqual(source.count("sync_browser_runtime\n"), 2)


class InstallerDelegationTest(unittest.TestCase):
    """All four surfaces must ask the same module the same question."""

    def test_install_sh_calls_the_shared_helper(self):
        source = INSTALL_SH.read_text(encoding="utf-8")
        self.assertIn('helper="$INSTALL_DIR/system/browser_runtime_setup.py"', source)
        self.assertIn('"$python" "$helper" --ensure', source)

    def test_install_sh_still_works_against_an_older_pinned_ref(self):
        """A deployment SHA without the helper must not silently lose Chromium."""
        source = INSTALL_SH.read_text(encoding="utf-8")
        self.assertIn('if [ ! -f "$helper" ]; then', source)
        self.assertIn("-m playwright install chromium", source)

    def test_install_ps1_calls_the_shared_helper(self):
        source = INSTALL_PS1.read_text(encoding="utf-8")
        self.assertIn("$helper = \"$INSTALL_DIR\\system\\browser_runtime_setup.py\"", source)
        self.assertIn("& $python $helper --ensure", source)

    def test_the_helper_ships_where_every_installer_looks_for_it(self):
        """install.sh clones server/ flat into $INSTALL_DIR; the sync rsyncs it."""
        self.assertTrue(
            (REPO_ROOT / "server" / "system" / "browser_runtime_setup.py").is_file()
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(unittest.main())
