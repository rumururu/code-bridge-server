import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from system.llm_provider_install_jobs import (
    LlmProviderInstallJobManager,
    PROVIDER_CLI_MISSING,
    PROVIDER_CLI_UNAVAILABLE,
    PROVIDER_INSTALL_FAILED,
    PROVIDER_INSTALL_TIMEOUT,
)


INSTALL_SPEC = {
    "provider_id": "openai",
    "provider_name": "OpenAI (Codex)",
    "provider_command": "codex",
    "method": "npm",
    "command": ["npm", "install", "-g", "@openai/codex"],
}


class FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        delay: float = 0.0,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.delay = delay
        self.terminated = False
        self.killed = False

    async def communicate(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.stdout, self.stderr

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


class LlmProviderInstallJobManagerTest(unittest.IsolatedAsyncioTestCase):
    async def _start_and_wait(self, manager: LlmProviderInstallJobManager):
        job = await manager.start_install("openai", "npm")
        await manager._tasks[job.job_id]
        return manager.serialize(job)

    async def test_successful_install_completes_with_options(self):
        manager = LlmProviderInstallJobManager()
        with patch(
            "system.llm_provider_install_jobs.resolve_llm_provider_install",
            return_value=INSTALL_SPEC,
        ), patch(
            "system.llm_provider_install_jobs.shutil.which",
            return_value="/usr/local/bin/npm",
        ), patch(
            "system.llm_provider_install_jobs.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=FakeProcess(returncode=0, stdout=b"ok\n")),
        ), patch(
            "system.llm_provider_install_jobs.check_llm_provider_cli_available",
            return_value=(True, None),
        ), patch(
            "system.llm_provider_install_jobs.get_llm_options_snapshot",
            return_value={"companies": [{"id": "openai"}]},
        ):
            payload = await self._start_and_wait(manager)

        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["installed"])
        self.assertIsNone(payload["error_code"])
        self.assertEqual(payload["returncode"], 0)
        self.assertIn("options", payload)

    async def test_missing_installer_executable_fails_with_stable_code(self):
        manager = LlmProviderInstallJobManager()
        with patch(
            "system.llm_provider_install_jobs.resolve_llm_provider_install",
            return_value=INSTALL_SPEC,
        ), patch(
            "system.llm_provider_install_jobs.shutil.which",
            return_value=None,
        ), patch(
            "system.llm_provider_install_jobs.get_llm_options_snapshot",
            return_value={"companies": []},
        ):
            payload = await self._start_and_wait(manager)

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_code"], PROVIDER_CLI_MISSING)
        self.assertFalse(payload["installed"])

    async def test_nonzero_install_fails_with_stdout_stderr_tail(self):
        manager = LlmProviderInstallJobManager()
        with patch(
            "system.llm_provider_install_jobs.resolve_llm_provider_install",
            return_value=INSTALL_SPEC,
        ), patch(
            "system.llm_provider_install_jobs.shutil.which",
            return_value="/usr/local/bin/npm",
        ), patch(
            "system.llm_provider_install_jobs.asyncio.create_subprocess_exec",
            new=AsyncMock(
                return_value=FakeProcess(
                    returncode=1,
                    stdout=b"stdout detail",
                    stderr=b"stderr detail",
                )
            ),
        ), patch(
            "system.llm_provider_install_jobs.get_llm_options_snapshot",
            return_value={"companies": []},
        ):
            payload = await self._start_and_wait(manager)

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_code"], PROVIDER_INSTALL_FAILED)
        self.assertEqual(payload["returncode"], 1)
        self.assertEqual(payload["stdout_tail"], "stdout detail")
        self.assertEqual(payload["stderr_tail"], "stderr detail")

    async def test_timeout_fails_explicitly(self):
        manager = LlmProviderInstallJobManager(timeout_seconds=0.01)
        with patch(
            "system.llm_provider_install_jobs.resolve_llm_provider_install",
            return_value=INSTALL_SPEC,
        ), patch(
            "system.llm_provider_install_jobs.shutil.which",
            return_value="/usr/local/bin/npm",
        ), patch(
            "system.llm_provider_install_jobs.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=FakeProcess(returncode=0, delay=0.1)),
        ), patch(
            "system.llm_provider_install_jobs.get_llm_options_snapshot",
            return_value={"companies": []},
        ):
            payload = await self._start_and_wait(manager)

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_code"], PROVIDER_INSTALL_TIMEOUT)

    async def test_successful_process_but_cli_unavailable_fails(self):
        manager = LlmProviderInstallJobManager()
        with patch(
            "system.llm_provider_install_jobs.resolve_llm_provider_install",
            return_value=INSTALL_SPEC,
        ), patch(
            "system.llm_provider_install_jobs.shutil.which",
            return_value="/usr/local/bin/npm",
        ), patch(
            "system.llm_provider_install_jobs.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=FakeProcess(returncode=0)),
        ), patch(
            "system.llm_provider_install_jobs.check_llm_provider_cli_available",
            return_value=(False, "codex CLI not installed"),
        ), patch(
            "system.llm_provider_install_jobs.get_llm_options_snapshot",
            return_value={"companies": []},
        ):
            payload = await self._start_and_wait(manager)

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_code"], PROVIDER_CLI_UNAVAILABLE)
        self.assertFalse(payload["installed"])


if __name__ == "__main__":
    unittest.main()
