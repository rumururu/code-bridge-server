"""The committed canvas bundle matches the kernel source it was built from.

`server/webui/canvas/` is not source. It is a build artifact produced in the
flow kernel repository and committed here, because the machines this server
installs onto have no node (`routes/canvas_static.py:55-59`). Committed
artifacts rot quietly: `install/verify_install.py` compares repository bytes to
install-directory bytes, so a bundle six weeks behind its source is
byte-identical on both sides and the deploy reports success while the screen
shows the old canvas.

This is the red light for that, placed in the ordinary test suite on purpose.
The deploy report (`verify_install.py` section [4]) asks the same question, but
it is asked at deploy time by whoever is deploying — which is too late and
often somebody else. `pytest tests/` is what the person who just changed the
canvas runs, so this is where they should find out.

The verdict logic is `install/canvas_bundle_freshness.py`, shared with the
deploy report so the two cannot drift. What this module adds is the two things
a test must not get wrong:

* **A missing kernel checkout skips, loudly.** Most machines that run this
  suite have no flow kernel, and a gate that passes on them is
  indistinguishable from a gate that never ran.
* **A `-dirty` or `unknown` stamp fails even there.** Neither is reproducible
  from any commit, so no comparison could ever make them right, and no kernel
  is needed to say so.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRESHNESS_MODULE = REPO_ROOT / "install" / "canvas_bundle_freshness.py"


def _load_freshness():
    """Load the checker by path — `install/` is not an importable package.

    Same approach as `test_deploy_protects_runtime_state.py`, and for the same
    reason: the deploy tooling is a directory of scripts, deliberately not
    shipped into the install directory, so there is nothing to `import`.
    """
    spec = importlib.util.spec_from_file_location(
        "canvas_bundle_freshness", FRESHNESS_MODULE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["canvas_bundle_freshness"] = module
    spec.loader.exec_module(module)
    return module


class CanvasBundleFreshnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FRESHNESS_MODULE.is_file():
            raise unittest.SkipTest(
                f"{FRESHNESS_MODULE} not present — install/ is excluded from the "
                "deployed install directory, so this suite can only run this "
                "check from a repository checkout"
            )
        cls.freshness = _load_freshness()

    def test_committed_bundle_is_built_from_the_kernel_source(self) -> None:
        result = self.freshness.check(REPO_ROOT)

        if result.status == self.freshness.SKIPPED:
            # Named, not silent. The reason is the point: "no kernel checkout"
            # is a fact about this machine, "stale bundle" is a fact about the
            # repository, and only one of them is anybody's fault.
            self.skipTest(result.report())

        self.assertEqual(
            self.freshness.OK,
            result.status,
            "\n" + result.report(),
        )

    def test_an_unreproducible_stamp_fails_without_a_kernel_checkout(self) -> None:
        """`-dirty` and `unknown` are rejected on machines with no kernel.

        This is the half of the gate that must work everywhere. Both stamps say
        "no commit produced this artifact", which is decidable from the bundle
        alone — deferring it to a machine that happens to have the kernel is
        how an unreproducible bundle reaches a release.
        """
        absent_kernel = REPO_ROOT / "does-not-exist-flow-kernel"
        self.assertFalse(absent_kernel.exists())

        for stamp in ("a" * 40 + "-dirty", "unknown"):
            with self.subTest(stamp=stamp):
                verdict = self._verdict_for_stamp(stamp, kernel=absent_kernel)
                self.assertEqual(self.freshness.STALE, verdict.status)
                self.assertIn(verdict.reason, {"dirty-stamp", "unknown-stamp"})

    def test_a_clean_stamp_skips_rather_than_passes_without_a_kernel(self) -> None:
        """The comparison half degrades to `skipped`, never to `ok`.

        A clean 40-hex stamp is only correct relative to a kernel tree. With no
        kernel to read, "looks fine" is not an answer this may give.
        """
        absent_kernel = REPO_ROOT / "does-not-exist-flow-kernel"
        verdict = self._verdict_for_stamp("b" * 40, kernel=absent_kernel)

        self.assertEqual(self.freshness.SKIPPED, verdict.status)
        self.assertEqual("no-kernel-checkout", verdict.reason)
        self.assertIn(self.freshness.KERNEL_DIR_ENV, verdict.detail)

    def test_a_stamp_that_does_not_match_the_kernel_tree_is_stale(self) -> None:
        """The case the whole ticket exists for, driven against a real repo.

        A throwaway git repository stands in for the kernel: its `frontend/`
        tree id is real, and a bundle stamped with anything else is stale.
        Asserting this against a fabricated tree id rather than the live kernel
        keeps the test's meaning independent of whatever state the developer's
        kernel checkout happens to be in.
        """
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "kernel"
            (kernel / "frontend").mkdir(parents=True)
            (kernel / "frontend" / "app.ts").write_text("export const x = 1;\n")
            env = {
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@example.com",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@example.com",
                "PATH": "/usr/bin:/bin:/usr/local/bin",
            }
            run = lambda *a: subprocess.run(  # noqa: E731
                ["git", "-C", str(kernel), *a], check=True, capture_output=True, env=env
            )
            run("init", "-q")
            run("add", "-A")
            run("commit", "-qm", "init")
            tree = subprocess.run(
                ["git", "-C", str(kernel), "rev-parse", "HEAD:frontend"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()

            matched = self._verdict_for_stamp(tree, kernel=kernel)
            self.assertEqual(self.freshness.OK, matched.status, matched.report())

            moved = self._verdict_for_stamp("c" * 40, kernel=kernel)
            self.assertEqual(self.freshness.STALE, moved.status)
            self.assertEqual("tree-mismatch", moved.reason)

            # Source edited, bundle not rebuilt: the acceptance criterion.
            (kernel / "frontend" / "app.ts").write_text("export const x = 2;\n")
            unbuilt = self._verdict_for_stamp(tree, kernel=kernel)
            self.assertEqual(self.freshness.STALE, unbuilt.status)
            self.assertEqual("kernel-dirty", unbuilt.reason)

    def test_a_bundle_with_no_stamp_is_not_given_the_benefit_of_the_doubt(self) -> None:
        verdict = self._verdict_for_stamp(None, kernel=REPO_ROOT)
        self.assertEqual(self.freshness.STALE, verdict.status)
        self.assertEqual("no-stamp", verdict.reason)

    def _verdict_for_stamp(self, stamp: str | None, *, kernel: Path):
        """Run the checker over a synthetic bundle carrying ``stamp``."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self.freshness.bundle_dir(root) / "assets"
            bundle.mkdir(parents=True)
            body = "console.log(1);" if stamp is None else f"eO(`{stamp}`);"
            (bundle / "index-deadbeef.js").write_text(body, encoding="utf-8")
            return self.freshness.check(root, kernel)


class CanvasBundleTransferRulesTest(unittest.TestCase):
    """The bundle is deployed; frontend build leftovers next to it are not.

    `server/webui/canvas/` is copied in by hand from another repository, so a
    stray `node_modules/` or a `.map` is one careless `cp -r` away — and
    `SOURCE_EXCLUDES` has no re-include mechanism, which is why the guards are
    anchored to `/webui/` instead of written as bare `node_modules/` and
    `*.map`. `server/scrcpy/` is a bundled node app that ships WITH its
    dependencies under `--with-scrcpy`; a bare rule would strip them out of
    that opt-in transfer and deploy an unrunnable app with no error anywhere.
    Both halves are asserted, because only asserting the first would let a
    future "simplification" of the pattern pass.
    """

    @classmethod
    def setUpClass(cls) -> None:
        verify_path = REPO_ROOT / "install" / "verify_install.py"
        if not verify_path.is_file():
            raise unittest.SkipTest(f"{verify_path} not present")
        spec = importlib.util.spec_from_file_location("verify_install", verify_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["verify_install"] = module
        spec.loader.exec_module(module)
        cls.verify = module

    def _excluded(self, rel: str) -> bool:
        return self.verify.is_excluded(rel, self.verify.SOURCE_EXCLUDES)

    def test_frontend_leftovers_under_webui_are_never_transferred(self) -> None:
        for rel in (
            "webui/canvas/assets/index.js.map",
            "webui/canvas/assets/nested/app.css.map",
            "webui/node_modules/foo/index.js",
            "webui/canvas/node_modules/foo/index.js",
        ):
            with self.subTest(rel=rel):
                self.assertTrue(self._excluded(rel), f"{rel} would be deployed")

    def test_the_bundle_itself_is_transferred(self) -> None:
        """The guards must not swallow what the server actually serves."""
        for rel in ("webui/canvas/index.html", "webui/canvas/assets/index-abc123.js"):
            with self.subTest(rel=rel):
                self.assertFalse(self._excluded(rel), f"{rel} would not be deployed")

    def test_scrcpys_bundled_dependencies_are_untouched(self) -> None:
        for rel in (
            "scrcpy/node_modules/xml2js/lib/xml2js.js",
            "scrcpy/node_modules/agent-base/dist/index.js.map",
        ):
            with self.subTest(rel=rel):
                self.assertFalse(
                    self._excluded(rel),
                    f"{rel} is excluded — --with-scrcpy would deploy a node app "
                    "with no dependencies, and say nothing",
                )

    def test_the_real_bundle_appears_in_the_transfer_set(self) -> None:
        """Not a pattern argument: walk `server/` the way the deploy does.

        `SOURCE_MAP` and the walk are what actually decide the rsync payload,
        and this asserts the committed bundle survives all of it.
        """
        patterns = self.verify.build_patterns(with_scrcpy=False)
        transferred = self.verify.walk_tree(REPO_ROOT / "server", patterns)

        canvas = sorted(r for r in transferred if r.startswith("webui/canvas/"))
        self.assertIn("webui/canvas/index.html", canvas)
        self.assertTrue(
            any(r.startswith("webui/canvas/assets/") and r.endswith(".js") for r in canvas),
            f"no canvas JavaScript in the transfer set: {canvas}",
        )
        self.assertFalse([r for r in canvas if r.endswith(".map")])
        self.assertFalse([r for r in canvas if "node_modules/" in r])


if __name__ == "__main__":
    unittest.main()
