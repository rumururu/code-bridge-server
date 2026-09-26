"""Code Bridge deployment requires the reviewed compiled kernel artifact."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = REPO_ROOT / "install" / "sync-local-install.sh"
REQUIREMENTS = REPO_ROOT / "server" / "requirements.txt"
KERNEL_REQUIREMENTS = REPO_ROOT / "server" / "requirements-kernel.txt"
INSTALLER = REPO_ROOT / "scripts" / "install_closed_kernel.py"
ARTIFACTS = REPO_ROOT / "server" / "vendor" / "agent-flow-core"


class FlowKernelInstallStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SYNC_SCRIPT.read_text(encoding="utf-8")

    def _function_body(self) -> str:
        match = re.search(
            r"^sync_flow_core\(\)\s*\{\n(.*?)^\}", self.script, re.M | re.S
        )
        self.assertIsNotNone(match, "sync_flow_core() is missing")
        return match.group(1)

    def test_step_is_invoked_and_uses_the_verifier(self) -> None:
        body = self._function_body()
        after_def = self.script.split("sync_flow_core() {", 1)[1]
        self.assertRegex(after_def, re.compile(r"^\s*sync_flow_core\s*$", re.M))
        self.assertIn("install_closed_kernel.py", body)
        self.assertIn("vendor/agent-flow-core", body)
        self.assertIn('"$python" "$installer" "$artifacts"', body)

    def test_missing_kernel_or_verifier_fails_deployment(self) -> None:
        body = self._function_body()
        self.assertIn("kernel verifier not found", body)
        self.assertIn("kernel artifacts missing", body)
        self.assertNotIn("skipped (expected", body)
        self.assertNotIn("CODE_BRIDGE_FLOW_CORE_DIR", body)

    def test_release_install_has_no_source_or_editable_path(self) -> None:
        body = self._function_body()
        self.assertNotRegex(body, r"(?:^|\s)(-e|--editable)(?:\s|$)")
        self.assertNotIn("agent-flow-core @ file:", body)
        self.assertNotIn("ALLOW_SOURCE", body)

    def test_dry_run_reports_but_does_not_install(self) -> None:
        body = self._function_body()
        dry = re.search(r'if \[ "\$APPLY" -ne 1 \];\s*then\n(.*?)\bfi\b', body, re.S)
        self.assertIsNotNone(dry)
        self.assertIn("return 0", dry.group(1))

    def test_no_pip_cannot_bypass_the_offline_kernel_verifier(self) -> None:
        invocation = self.script.split('echo "${CYAN}--- flow kernel${NC}"', 1)[1]
        invocation = invocation.split('echo ""', 1)[0]
        self.assertIn("sync_flow_core", invocation)
        self.assertNotIn("SKIP_PIP", invocation)


class FlowKernelArtifactDeclarationTest(unittest.TestCase):
    def test_declaration_and_verifier_exist(self) -> None:
        self.assertTrue(KERNEL_REQUIREMENTS.is_file())
        self.assertTrue(INSTALLER.is_file())
        text = KERNEL_REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("agent-flow-core 0.1.0", text)
        self.assertIn("017ca72f1da20fd6416e8fb92b34c267c709f44c", text)
        self.assertIn("install_closed_kernel.py", text)

    def test_mac_arm64_cp313_wheel_and_manifest_are_bundled(self) -> None:
        wheel = ARTIFACTS / "agent_flow_core-0.1.0-cp313-cp313-macosx_26_0_arm64.whl"
        manifest = ARTIFACTS / f"{wheel.name}.manifest.json"
        self.assertTrue(wheel.is_file())
        self.assertTrue(manifest.is_file())

    def test_base_requirements_points_to_the_separate_artifact_contract(self) -> None:
        text = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("requirements-kernel.txt", text)
        installable = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any("agent-flow-core" in line for line in installable))


if __name__ == "__main__":
    unittest.main()
