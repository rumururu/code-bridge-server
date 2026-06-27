import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from core.runtime_paths import runtime_dir, runtime_path


class RuntimePathsTest(unittest.TestCase):
    def test_runtime_path_uses_legacy_path_without_app_support_env(self):
        legacy = Path("/tmp/code_bridge_legacy_state.json")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_path("state.json", legacy), legacy)

    def test_runtime_path_uses_app_support_env_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CODEBRIDGE_APP_SUPPORT_DIR": tmp}):
                self.assertEqual(runtime_path("state.json", Path("/legacy/state.json")), Path(tmp) / "state.json")

    def test_runtime_dir_creates_named_directory_under_app_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CODEBRIDGE_APP_SUPPORT_DIR": tmp}):
                path = runtime_dir("pairing", Path("/legacy/pairing"))

            self.assertEqual(path, Path(tmp) / "pairing")
            self.assertTrue(path.exists())

    def test_runtime_dir_creates_legacy_directory_without_app_support_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "global_chat"

            with patch.dict(os.environ, {}, clear=True):
                path = runtime_dir("global_chat", legacy)

            self.assertEqual(path, legacy)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
