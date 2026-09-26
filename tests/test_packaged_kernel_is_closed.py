"""The desktop build must not package a readable flow kernel.

PyInstaller collects site-packages from the interpreter running the build
script, so whichever `agent_flow_core` is installed there is the one that ends
up inside the `.app`. A developer environment installs it from the local
checkout as plain `.py` — right for development, wrong for distribution, and
the difference is invisible in the build log.

That is not hypothetical. The `.dmg` built on 2026-06-28 was opened and
inspected: 216 plain `.py` files, the whole server, comments intact. A `.app`
is a folder and Show Package Contents is one click; a Developer ID signature
proves who built it, not that it cannot be read.

So the build asks the artifact rather than the build steps — can this module's
source be recovered at runtime? — and this file holds that question in place.
The three cases below are the three states a build machine can be in, and each
has to end differently: no kernel at all, a source kernel, and a compiled one.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_desktop_server_app.py"
SPEC = importlib.util.spec_from_file_location(
    "build_desktop_server_app_kernel_guard", SCRIPT_PATH
)
packaging = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = packaging
SPEC.loader.exec_module(packaging)


#: The two closed packages and the symbol the guard probes in each. Separate
#: packages for the reason ADR-002 gives — the kernel is shared with
#: Infergraph and its G5 invariant forbids Code Bridge's step vocabulary —
#: and either one shipping as source defeats the same purpose.
CLOSED_PACKAGES = (
    ("agent_flow_core", "agent_flow_core.model", "Flow"),
    ("code_bridge_core", "code_bridge_core.step_cursor", "StepCursor"),
)


def _fake_modules():
    """Stand-ins for both closed packages, with a probe symbol in each.

    Built rather than imported so the test says what it means on any machine:
    this repo's own environment installs both from source, so importing the
    real ones would only ever exercise one branch of the guard.
    """
    modules = {}
    for package, dotted, symbol in CLOSED_PACKAGES:
        leaf = types.ModuleType(dotted)
        setattr(leaf, symbol, type(symbol, (), {}))
        root = types.ModuleType(package)
        setattr(root, dotted.split(".", 1)[1], leaf)
        modules[package] = root
        modules[dotted] = leaf
    return modules


class TheGuardReadsTheInstalledKernelTest(unittest.TestCase):
    def _run_with(self, *, getsource):
        with patch.dict(sys.modules, _fake_modules()), patch(
            "inspect.getsource", getsource
        ), patch(
            "inspect.getfile", lambda _obj: "/somewhere/model.py"
        ):
            packaging.verify_kernel_is_compiled()

    def test_a_compiled_kernel_passes(self):
        # `inspect.getsource` raising OSError is exactly what a Nuitka module
        # does — there is no file behind it to read.
        def raises(_obj):
            raise OSError("source not available")

        self._run_with(getsource=raises)  # no SystemExit

    def test_a_source_kernel_stops_the_build(self):
        with self.assertRaises(SystemExit) as caught:
            self._run_with(getsource=lambda _obj: "class Flow: ...")
        message = str(caught.exception)
        # The refusal has to be actionable: what is wrong, where, and the two
        # commands that fix it. A bare "refused" sends someone reading the
        # PyInstaller docs.
        self.assertIn("readable source", message)
        self.assertIn("build_closed_kernel.py", message)
        self.assertIn("pip install --force-reinstall", message)

    def test_both_closed_packages_are_probed(self):
        # A guard that checked only the kernel would wave through an app whose
        # whole decision layer — vocabulary, gates, Configurator — shipped as
        # source. That is the larger half of what ADR-002 closes.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        guard = source[source.index("def verify_kernel_is_compiled("):]
        guard = guard[: guard.index("\ndef ")]
        for package, _dotted, symbol in CLOSED_PACKAGES:
            self.assertIn(package, guard)
            self.assertIn(symbol, guard)

    def test_a_source_second_package_is_named_in_its_own_refusal(self):
        # The refusal has to name which package is at fault; "something is
        # source" sends the reader to rebuild the wrong one.
        modules = _fake_modules()
        kernel_symbol = modules["agent_flow_core.model"].Flow

        def compiled_kernel_only(obj):
            if obj is kernel_symbol:
                raise OSError("compiled")
            return "class StepCursor: ..."

        with patch.dict(sys.modules, modules), patch(
            "inspect.getsource", compiled_kernel_only
        ), patch("inspect.getfile", lambda _obj: "/somewhere/step_cursor.py"):
            with self.assertRaises(SystemExit) as caught:
                packaging.verify_kernel_is_compiled()
        self.assertIn("code_bridge_core", str(caught.exception))
        self.assertNotIn("agent_flow_core installed here", str(caught.exception))

    def test_a_missing_kernel_stops_the_build_too(self):
        # Not the same failure and not the same fix: this app would ship with
        # no kernel at all, and the message must not send someone off to
        # compile one they have not installed.
        real_import = __import__

        def missing(name, *args, **kwargs):
            if name.startswith("agent_flow_core"):
                raise ImportError("No module named 'agent_flow_core'")
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {}, clear=False):
            for name in list(sys.modules):
                if name.startswith("agent_flow_core"):
                    sys.modules.pop(name, None)
            with patch("builtins.__import__", missing):
                with self.assertRaises(SystemExit) as caught:
                    packaging.verify_kernel_is_compiled()
        self.assertIn("not installed", str(caught.exception))


class TheSourceDoesNotTravelBesideTheBinaryTest(unittest.TestCase):
    """A staged source copy beats the installed extension, silently.

    `server/` goes on `sys.path` ahead of site-packages, so if
    `server/code_bridge_core/` is copied into the app next to the compiled
    wheel, Python imports the `.py` and the whole decision layer ships
    readable — with the guard above satisfied, because the *installed*
    package really is compiled.

    Measured, not reasoned: with both present, `__loader__` came back
    `SourceFileLoader` and `inspect.getsource` returned the file.
    """

    def test_the_package_source_is_excluded_from_the_staged_tree(self):
        self.assertIn("code_bridge_core", packaging.SERVER_EXCLUDE_DIRS)

    def test_the_staging_filter_actually_drops_it(self):
        # The set is one thing; what the copier does with it is another.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / "server"
            (server / "code_bridge_core").mkdir(parents=True)
            (server / "code_bridge_core" / "step_cursor.py").write_text("x = 1\n")
            (server / "agent").mkdir()
            (server / "agent" / "task_orchestrator.py").write_text("y = 2\n")

            with patch.object(packaging, "SERVER_DIR", server):
                staged = packaging.stage_server_tree(root / "build")

            survivors = {
                path.relative_to(staged).as_posix()
                for path in staged.rglob("*.py")
            }
        # The effect layer ships; the decision layer does not.
        self.assertIn("agent/task_orchestrator.py", survivors)
        self.assertNotIn("code_bridge_core/step_cursor.py", survivors)


class TheGuardRunsBeforeAnythingIsPackagedTest(unittest.TestCase):
    def test_main_checks_the_kernel_before_invoking_pyinstaller(self):
        # Order is the point: a build that discovers this at the end has
        # already spent minutes producing an artifact nobody should ship.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        guard = source.index("verify_kernel_is_compiled()\n", source.index("def main("))
        run_call = source.index("run(command, cwd=REPO_ROOT)")
        self.assertLess(guard, run_call)

    def test_a_dry_run_does_not_require_a_compiled_kernel(self):
        # Printing the command is not producing an app, and a developer
        # checking the invocation should not be made to compile first.
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        main = source[source.index("def main("):]
        dry_run_return = main.index("if args.dry_run:")
        guard = main.index("verify_kernel_is_compiled()")
        self.assertLess(guard, dry_run_return)
        # ...and it sits inside the `not args.dry_run` block that precedes it.
        self.assertIn("if not args.dry_run:", main[:guard])


if __name__ == "__main__":
    unittest.main()
