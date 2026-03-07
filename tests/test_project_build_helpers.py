import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_build_helpers import decode_build_error, resolve_nextjs_build_output_path


class ProjectBuildHelpersTest(unittest.TestCase):
    def test_decode_build_error_prefers_stderr(self):
        result = decode_build_error(b"stderr error", b"stdout error")
        self.assertEqual(result, "stderr error")

    def test_decode_build_error_falls_back_to_stdout(self):
        result = decode_build_error(b"   ", b"stdout error")
        self.assertEqual(result, "stdout error")

    def test_resolve_nextjs_build_output_path_prefers_out_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "index.html").write_text("<html></html>", encoding="utf-8")
            (root / ".next" / "standalone").mkdir(parents=True, exist_ok=True)

            result = resolve_nextjs_build_output_path(root)

        self.assertEqual(result, str(out_dir))

    def test_resolve_nextjs_build_output_path_uses_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            standalone = root / ".next" / "standalone"
            standalone.mkdir(parents=True, exist_ok=True)

            result = resolve_nextjs_build_output_path(root)

        self.assertEqual(result, str(standalone))

    def test_resolve_nextjs_build_output_path_uses_next_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            next_dir = root / ".next"
            next_dir.mkdir(parents=True, exist_ok=True)

            result = resolve_nextjs_build_output_path(root)

        self.assertEqual(result, str(next_dir))

    def test_resolve_nextjs_build_output_path_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            result = resolve_nextjs_build_output_path(root)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
