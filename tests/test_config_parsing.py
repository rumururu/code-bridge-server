import os
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from config import Config


class ConfigParsingTest(unittest.TestCase):
    def _write_config(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_numeric_settings_apply_fallbacks_and_minimums(self):
        path = self._write_config(
            """
server:
  weekly_budget_usd: invalid
  usage_window_days: -2
  heartbeat_interval_minutes: 3
"""
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        cfg = Config(config_path=path)

        self.assertEqual(cfg.weekly_budget_usd, 100.0)
        self.assertEqual(cfg.usage_window_days, 1)
        self.assertEqual(cfg.heartbeat_interval_minutes, 5)

    def test_port_prefers_environment_override(self):
        path = self._write_config(
            """
server:
  port: 8123
"""
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        previous = os.environ.get("CODEBRIDGE_PORT")
        os.environ["CODEBRIDGE_PORT"] = "9001"
        self.addCleanup(
            lambda: os.environ.__setitem__("CODEBRIDGE_PORT", previous)
            if previous is not None
            else os.environ.pop("CODEBRIDGE_PORT", None)
        )

        cfg = Config(config_path=path)
        self.assertEqual(cfg.port, 9001)

    def test_runtime_port_sets_environment(self):
        path = self._write_config("server: {}\n")
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        previous = os.environ.get("CODEBRIDGE_PORT")
        self.addCleanup(
            lambda: os.environ.__setitem__("CODEBRIDGE_PORT", previous)
            if previous is not None
            else os.environ.pop("CODEBRIDGE_PORT", None)
        )

        cfg = Config(config_path=path)
        cfg.set_runtime_port(7777)

        self.assertEqual(os.environ.get("CODEBRIDGE_PORT"), "7777")
        self.assertEqual(cfg.port, 7777)


if __name__ == "__main__":
    unittest.main()
