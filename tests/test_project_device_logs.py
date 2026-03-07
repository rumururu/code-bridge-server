import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_device_logs import device_run_log_path, read_log_tail, sanitize_log_name


class ProjectDeviceLogsTest(unittest.TestCase):
    def test_sanitize_log_name_normalizes_symbols(self):
        self.assertEqual(sanitize_log_name("demo project"), "demo_project")
        self.assertEqual(sanitize_log_name("device:id"), "device_id")
        self.assertEqual(sanitize_log_name("___"), "project")

    def test_device_run_log_path_uses_tmp_and_sanitized_segments(self):
        path = device_run_log_path("my app", "emulator:5554")

        self.assertEqual(path.parent, Path("/tmp"))
        self.assertEqual(path.name, "code_bridge_device_run_my_app_emulator_5554.log")

    def test_read_log_tail_returns_empty_for_missing_file(self):
        result = read_log_tail("/tmp/this-file-should-not-exist.log")
        self.assertEqual(result, "")

    def test_read_log_tail_applies_line_and_char_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "run.log"
            lines = [f"line-{idx}" for idx in range(1, 11)]
            log_path.write_text("\n".join(lines), encoding="utf-8")

            by_line = read_log_tail(log_path, max_lines=3, max_chars=1000)
            by_char = read_log_tail(log_path, max_lines=20, max_chars=8)

        self.assertEqual(by_line, "line-8\nline-9\nline-10")
        self.assertEqual(by_char, "\nline-10")


if __name__ == "__main__":
    unittest.main()
