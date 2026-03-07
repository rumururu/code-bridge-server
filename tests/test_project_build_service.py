import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_build_service import build_flutter_web_project, build_nextjs_project


class _FakeProcess:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


class ProjectBuildServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_build_flutter_web_project_success(self):
        fake_process = _FakeProcess(returncode=0)
        with patch(
            "project_build_service.asyncio.create_subprocess_exec",
            return_value=fake_process,
        ) as mock_exec:
            result = await build_flutter_web_project("/tmp/demo")

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Build completed")
        self.assertEqual(result.build_path, "/tmp/demo/build/web")
        mock_exec.assert_awaited_once()

    async def test_build_flutter_web_project_failure(self):
        fake_process = _FakeProcess(returncode=1, stderr=b"flutter failed")
        with patch(
            "project_build_service.asyncio.create_subprocess_exec",
            return_value=fake_process,
        ):
            result = await build_flutter_web_project("/tmp/demo")

        self.assertFalse(result.success)
        self.assertEqual(result.message, "flutter failed")
        self.assertIsNone(result.build_path)

    async def test_build_flutter_web_project_missing_cli(self):
        with patch(
            "project_build_service.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError(),
        ):
            result = await build_flutter_web_project("/tmp/demo")

        self.assertFalse(result.success)
        self.assertEqual(result.message, "Flutter CLI not found. Is Flutter installed?")

    async def test_build_nextjs_project_success_with_out(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            fake_process = _FakeProcess(returncode=0)

            with patch(
                "project_build_service.asyncio.create_subprocess_exec",
                return_value=fake_process,
            ):
                result = await build_nextjs_project(str(root))

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Build completed")
        self.assertEqual(result.build_path, str(out_dir))

    async def test_build_nextjs_project_failure_when_output_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_process = _FakeProcess(returncode=0)
            with patch(
                "project_build_service.asyncio.create_subprocess_exec",
                return_value=fake_process,
            ):
                result = await build_nextjs_project(tmp_dir)

        self.assertFalse(result.success)
        self.assertEqual(result.message, "Build completed but no output directory found")
        self.assertIsNone(result.build_path)

    async def test_build_nextjs_project_missing_npm(self):
        with patch(
            "project_build_service.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError(),
        ):
            result = await build_nextjs_project("/tmp/demo")

        self.assertFalse(result.success)
        self.assertEqual(result.message, "npm not found. Is Node.js installed?")


if __name__ == "__main__":
    unittest.main()
