"""The deploy script installs the agent-flow-core kernel — guarded, as a snapshot.

The flow kernel lives in a separate local git repository, not on any package
index, so `server/requirements.txt` cannot name it: a local path there would
break every install on a machine without that checkout, and the public mirror
installs from the same file. `install/sync-local-install.sh` is therefore the
one path that puts the kernel into the deployed venv, and it must keep two
properties:

* **Guarded** — when the checkout is absent (another developer's machine, a
  public install) the step logs one line and skips. If the guard is lost, the
  deploy fails on every machine that is not this one.
* **Non-editable** — the install is a snapshot taken at deploy time. An
  editable install (`pip install -e`) would point the live server at the
  kernel repo's *working tree*, so every half-finished edit there would reach
  the running server immediately — the exact partial-deployment accident the
  sync script exists to prevent for `server/` code.

These tests pin both properties in the script text itself (the same style as
`test_deploy_protects_runtime_state.py`): they do not execute the script —
execution against a real install is the orchestrator's job — they make sure a
future edit cannot silently drop the guard or flip the install to editable.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = REPO_ROOT / "install" / "sync-local-install.sh"
REQUIREMENTS = REPO_ROOT / "server" / "requirements.txt"


class FlowKernelInstallStepTest(unittest.TestCase):
    """The kernel step exists, is guarded, and installs a snapshot."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SYNC_SCRIPT.read_text(encoding="utf-8")

    def _function_body(self) -> str:
        """The text of sync_flow_core(), so assertions cannot match a comment."""
        match = re.search(
            r"^sync_flow_core\(\)\s*\{\n(.*?)^\}", self.script, re.M | re.S
        )
        if match is None:
            self.fail(
                "sync-local-install.sh no longer defines sync_flow_core(); the "
                "kernel install step is gone (or renamed — point this test at "
                "whatever replaced it)"
            )
        return match.group(1)

    def test_the_kernel_install_step_exists_and_is_invoked(self) -> None:
        body = self._function_body()  # fails with the message above if absent
        self.assertTrue(body.strip(), "sync_flow_core() is defined but empty")
        # Defining the function is not enough — it must actually run.
        after_def = self.script.split("sync_flow_core() {", 1)[1]
        self.assertRegex(
            after_def,
            re.compile(r"^\s*sync_flow_core\s*$", re.M),
            "sync_flow_core is defined but never called; the kernel would "
            "silently stop being deployed",
        )

    def test_the_kernel_path_is_overridable(self) -> None:
        self.assertIn(
            "CODE_BRIDGE_FLOW_CORE_DIR",
            self._function_body(),
            "the kernel path must be overridable via CODE_BRIDGE_FLOW_CORE_DIR "
            "so a machine with a non-standard checkout location can deploy",
        )

    def test_a_missing_checkout_skips_instead_of_failing(self) -> None:
        """Property (a): the guard. Public/other-machine installs must survive."""
        body = self._function_body()
        guard = re.search(
            r"if \[ ! -d \"\$kernel_dir\" \];\s*then\n(.*?)\bfi\b",
            body,
            re.S,
        )
        self.assertIsNotNone(
            guard,
            "sync_flow_core() no longer checks whether the kernel checkout "
            "exists; on any machine without ~/VSCodeProject/agent-flow-core "
            "the deploy would now fail instead of skipping",
        )
        self.assertIn(
            "return 0",
            guard.group(1),
            "the missing-checkout branch must return success (skip), not fall "
            "through to pip or fail the deploy",
        )
        # The guard must come before any pip invocation in the function.
        self.assertLess(
            body.index('! -d "$kernel_dir"'),
            body.index("$pip"),
            "the existence guard must run before pip is touched",
        )

    def test_the_install_is_a_snapshot_not_editable(self) -> None:
        """Property (b): no -e/--editable, ever."""
        body = self._function_body()
        install_lines = [
            line
            for line in body.splitlines()
            if "install" in line and "$kernel_dir" in line
        ]
        self.assertTrue(
            install_lines,
            "sync_flow_core() has no `pip install ... $kernel_dir` line; the "
            "kernel is no longer installed",
        )
        for line in install_lines:
            with self.subTest(line=line.strip()):
                self.assertIsNone(
                    re.search(r"(?:^|\s)(-e|--editable)(?:\s|$)", line),
                    "the kernel must be installed as a non-editable snapshot: "
                    "an editable install points the live server at the kernel "
                    "repo's working tree, so unfinished edits there would "
                    "reach the running server immediately",
                )

    def test_dry_run_does_not_install(self) -> None:
        """The script's convention: a dry run announces, only --apply acts."""
        body = self._function_body()
        dry = re.search(r"if \[ \"\$APPLY\" -ne 1 \];\s*then\n(.*?)\bfi\b", body, re.S)
        self.assertIsNotNone(
            dry,
            "sync_flow_core() no longer distinguishes dry run from --apply; "
            "a default (dry) run of the sync script would install packages",
        )
        self.assertIn("return 0", dry.group(1))

    def test_requirements_txt_never_names_the_kernel(self) -> None:
        """The decision this step exists to uphold: no local path in the
        shared requirements file, because the public mirror installs from it
        on machines where that path does not exist."""
        requirements = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertNotIn(
            "agent-flow-core",
            requirements,
            "server/requirements.txt must not reference the local kernel "
            "checkout — it would break every install on a machine without "
            "that path (public mirror included); the sync script installs "
            "the kernel instead",
        )


if __name__ == "__main__":
    unittest.main()
