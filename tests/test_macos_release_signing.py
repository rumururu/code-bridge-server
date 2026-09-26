"""A distributable macOS build has to be signed, hardened and notarized.

The build script used to sign `--sign -` — an ad-hoc signature, hardcoded,
with no way to pass a real identity. That artifact satisfies the loader on the
machine that produced it and nothing else: Gatekeeper refuses it everywhere,
and notarization will not accept it. So the `.dmg` in `dist/` was never
distributable, and nothing said so.

`desktop_server_app/macos_entitlements.plist` had sat in the repo referenced by
nothing for the same reason — entitlements only apply under the hardened
runtime, and nothing was signing with one.

Four properties are held here, and each of them is a build that fails at the
last step if it slips:

* every nested Mach-O is found, including the ones with no extension;
* the hardened runtime carries the entitlements CPython actually needs;
* `--notarize` without an identity is refused before the build, not after;
* Gatekeeper is asked *after* stapling and never before.
"""

from __future__ import annotations

import ast
import importlib.util
import plistlib
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_desktop_server_app.py"
ENTITLEMENTS = REPO_ROOT / "desktop_server_app" / "macos_entitlements.plist"
SOURCE = SCRIPT_PATH.read_text(encoding="utf-8")

SPEC = importlib.util.spec_from_file_location(
    "build_desktop_server_app_signing", SCRIPT_PATH
)
packaging = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = packaging
SPEC.loader.exec_module(packaging)


class NestedBinariesAreFoundByContentTest(unittest.TestCase):
    """An extension allow-list would miss exactly the files that break a build."""

    def test_a_binary_with_no_extension_is_found(self):
        # The vendored `node` and `adb` have no suffix. Missing them shows up
        # as a notarization rejection naming an unsigned nested binary — long
        # after the build, and nowhere near the cause.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Contents").mkdir()
            source = root / "m.c"
            source.write_text("int main(){return 0;}")
            for name in ("nested", "thing.dylib"):
                subprocess.run(
                    ["clang", "-o", str(root / "Contents" / name), str(source)]
                    + (["-shared"] if name.endswith(".dylib") else []),
                    check=True,
                    capture_output=True,
                )
            # Plain files that are not Mach-O must not be picked up.
            (root / "Contents" / "config.yaml").write_text("a: 1\n")
            (root / "Contents" / "notes.txt").write_text("hello\n")

            found = {path.name for path in packaging.macho_files(root)}

        self.assertIn("nested", found)
        self.assertIn("thing.dylib", found)
        self.assertNotIn("config.yaml", found)
        self.assertNotIn("notes.txt", found)

    def test_signing_order_is_innermost_first(self):
        # Signing the bundle seals whatever is inside it at that moment, so an
        # outer-first order produces a bundle whose nested code is unsigned.
        function = next(
            node
            for node in ast.parse(SOURCE).body
            if isinstance(node, ast.FunctionDef) and node.name == "sign_macos_app"
        )
        body = ast.get_source_segment(SOURCE, function) or ""
        self.assertIn("reverse=True", body)
        self.assertIn("len(path.parts)", body)


class TheHardenedRuntimeCarriesWhatCPythonNeedsTest(unittest.TestCase):
    def test_the_entitlements_file_is_actually_referenced(self):
        # It was not, for as long as the build signed ad-hoc. A file nothing
        # reads is indistinguishable from a file that says the wrong thing.
        self.assertIn("macos_entitlements.plist", SOURCE)
        self.assertIn("--entitlements", SOURCE)

    def test_every_entitlement_cpython_needs_is_declared(self):
        with ENTITLEMENTS.open("rb") as handle:
            keys = set(plistlib.load(handle))
        self.assertEqual(
            keys,
            {
                # CPython allocates and marks executable memory.
                "com.apple.security.cs.allow-jit",
                "com.apple.security.cs.allow-unsigned-executable-memory",
                # The bundle loads native code signed by other people — every
                # extension .so, the vendored Node runtime, adb. Without this
                # the app passes notarization and crashes on first launch on
                # the first machine that is not the build machine.
                "com.apple.security.cs.disable-library-validation",
            },
        )

    def test_the_runtime_option_travels_with_the_identity(self):
        # `--options runtime` is what makes the entitlements apply, and
        # notarization refuses a bundle without it.
        self.assertIn('"--options",\n        "runtime",', SOURCE)


class ImpossibleCombinationsAreRefusedEarlyTest(unittest.TestCase):
    def test_notarize_without_an_identity_is_refused(self):
        # Apple rejects an unsigned or ad-hoc bundle. Discovering that from
        # Apple costs a build plus a round trip for something knowable before
        # anything is compiled.
        main = SOURCE[SOURCE.index("def main("):]
        guard = main.index("--notarize needs --sign-identity")
        build = main.index("run(command, cwd=REPO_ROOT)")
        self.assertLess(guard, build)

    def test_ad_hoc_builds_say_they_are_not_distributable(self):
        # The default is still ad-hoc, because that is right for development.
        # What changed is that it no longer passes for a release.
        self.assertIn("NOT distributable", SOURCE)


class GatekeeperIsAskedOnlyWhenItsAnswerMeansSomethingTest(unittest.TestCase):
    def test_the_assessment_runs_after_stapling_and_not_before(self):
        """A signed-but-unnotarized bundle is *supposed* to be rejected.

        Asking `spctl --assess` during signing failed every correct release
        build one step before the step that fixes it
        ("source=Unnotarized Developer ID"). Caught by signing a probe bundle
        with the real certificate; without that, the failure would have
        surfaced only on the first real release build.
        """
        sign = SOURCE.index("def sign_macos_app(")
        notarize = SOURCE.index("def notarize_macos_artifact(")
        signing_body = SOURCE[sign:notarize]
        self.assertNotIn('run(["spctl"', signing_body)
        self.assertIn('run(["codesign", "--verify", "--strict"', signing_body)

        after = SOURCE[notarize:]
        staple = after.index('"staple"')
        assess = after.index('"spctl"')
        self.assertLess(staple, assess)


class CredentialsStayOutOfThisScriptTest(unittest.TestCase):
    def test_notarization_reads_a_keychain_profile_and_never_a_key(self):
        # The App Store Connect key stays out of the script, out of its
        # arguments and out of shell history. `store-credentials` is a
        # one-time human step, which is why the failure path prints it.
        self.assertIn("--keychain-profile", SOURCE)
        self.assertIn("notarytool store-credentials", SOURCE)
        for leaked in ("--key-id", "--issuer"):
            # Named only inside the instructions, never passed by this script.
            occurrences = SOURCE.count(leaked)
            self.assertEqual(occurrences, 1, f"{leaked} appears {occurrences} times")


if __name__ == "__main__":
    unittest.main()
