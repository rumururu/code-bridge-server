import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_models import ProjectType
from project_server_process import extract_process_error, wait_for_project_server_port


class _FakeProcess:
    def __init__(self, poll_result, stderr_bytes: bytes | None):
        self._poll_result = poll_result
        self.stderr = io.BytesIO(stderr_bytes) if stderr_bytes is not None else None

    def poll(self):
        return self._poll_result


class ProjectServerProcessTest(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_project_server_port_returns_detected_port(self):
        detect_mock = MagicMock(return_value=3000)

        result = await wait_for_project_server_port(
            "/tmp/demo",
            ProjectType.NEXTJS,
            detect_port=detect_mock,
            timeout_seconds=0.1,
        )

        self.assertEqual(result, 3000)
        detect_mock.assert_called_once_with("/tmp/demo", ProjectType.NEXTJS)

    async def test_wait_for_project_server_port_returns_none_when_process_exits(self):
        detect_mock = MagicMock(return_value=None)
        process = _FakeProcess(poll_result=1, stderr_bytes=b"boom")

        result = await wait_for_project_server_port(
            "/tmp/demo",
            ProjectType.NEXTJS,
            detect_port=detect_mock,
            process=process,
            timeout_seconds=0.1,
        )

        self.assertIsNone(result)
        detect_mock.assert_not_called()

    async def test_wait_for_project_server_port_times_out(self):
        detect_mock = MagicMock(return_value=None)

        result = await wait_for_project_server_port(
            "/tmp/demo",
            ProjectType.NEXTJS,
            detect_port=detect_mock,
            timeout_seconds=0.0,
        )

        self.assertIsNone(result)

    def test_extract_process_error_returns_last_line(self):
        process = _FakeProcess(
            poll_result=1,
            stderr_bytes=b"first line\nsecond line\n",
        )

        result = extract_process_error(process)
        self.assertEqual(result, "second line")

    def test_extract_process_error_returns_none_when_running(self):
        process = _FakeProcess(poll_result=None, stderr_bytes=b"error")
        self.assertIsNone(extract_process_error(process))

    def test_extract_process_error_returns_none_without_stderr(self):
        process = _FakeProcess(poll_result=1, stderr_bytes=None)
        self.assertIsNone(extract_process_error(process))


if __name__ == "__main__":
    unittest.main()
