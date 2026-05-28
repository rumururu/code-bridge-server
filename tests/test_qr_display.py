import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from pairing import qr_display


class QrDisplayTest(unittest.TestCase):
    def test_ascii_renderer_uses_ascii_only(self):
        rendered = qr_display._render_ascii([[True, False], [False, True]])

        rendered.encode("ascii")
        self.assertIn("##", rendered)
        self.assertNotIn("\u2588", rendered)

    def test_stdout_unicode_support_returns_false_for_cp949(self):
        fake_stdout = type("FakeStdout", (), {"encoding": "cp949"})()

        with patch.object(qr_display.sys, "stdout", fake_stdout):
            self.assertFalse(qr_display._stdout_supports_unicode())


if __name__ == "__main__":
    unittest.main()
