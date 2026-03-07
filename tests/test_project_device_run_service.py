import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_device_run_service import (
    build_flutter_run_command,
    prepare_device_run_log,
    start_flutter_run_process,
    summarize_flutter_run_exit,
)


class _FakeProcess:
    pid = 12345


class ProjectDeviceRunServiceTest(unittest.TestCase):
    def test_build_flutter_run_command(self):
        self.assertEqual(
            build_flutter_run_command("emulator-5554"),
            [
                "flutter",
                "run",
                "-d",
                "emulator-5554",
                "--machine",
                "--target",
                "lib/main.dart",
            ],
        )

    def test_prepare_device_run_log_creates_and_clears_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "logs" / "run.log"
            prepare_device_run_log(log_path)
            self.assertTrue(log_path.exists())
            log_path.write_text("old", encoding="utf-8")
            prepare_device_run_log(log_path)

            result = log_path.read_text(encoding="utf-8")

        self.assertEqual(result, "")

    def test_start_flutter_run_process_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "run.log"
            fake_process = _FakeProcess()
            with patch(
                "project_device_run_service.subprocess.Popen",
                return_value=fake_process,
            ) as mock_popen:
                result = start_flutter_run_process(
                    tmp_dir,
                    device_id="emulator-5554",
                    log_path=log_path,
                )

        self.assertTrue(result.success)
        self.assertEqual(result.process, fake_process)
        self.assertEqual(result.command, build_flutter_run_command("emulator-5554"))
        mock_popen.assert_called_once()

    def test_start_flutter_run_process_handles_missing_flutter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "run.log"
            with patch(
                "project_device_run_service.subprocess.Popen",
                side_effect=FileNotFoundError(),
            ):
                result = start_flutter_run_process(
                    tmp_dir,
                    device_id="emulator-5554",
                    log_path=log_path,
                )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Flutter CLI not found on server")

    def test_summarize_flutter_run_exit_uses_last_non_empty_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "run.log"
            log_path.write_text("line-1\nline-2\n", encoding="utf-8")

            summary, tail = summarize_flutter_run_exit(log_path)

        self.assertEqual(summary, "line-2")
        self.assertIn("line-1", tail)
        self.assertIn("line-2", tail)


if __name__ == "__main__":
    unittest.main()
