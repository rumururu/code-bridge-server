import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from devices.scrcpy_manager import ScrcpyManager


class ScrcpyManagerNodePathTest(unittest.TestCase):
    def test_resolve_adb_path_prefers_explicit_codebridge_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit_adb = Path(tmp) / "custom-adb"
            explicit_adb.touch()

            manager = ScrcpyManager(scrcpy_path=str(Path(tmp) / "server" / "scrcpy"))
            with patch.dict(os.environ, {"CODEBRIDGE_ADB_PATH": str(explicit_adb)}), patch(
                "devices.scrcpy_manager.shutil.which",
                return_value=None,
            ):
                self.assertEqual(manager._resolve_adb_path(), str(explicit_adb))

    def test_resolve_adb_path_uses_bundled_platform_tools_before_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            server_dir = Path(tmp) / "server"
            bundled_adb = server_dir / "vendor" / "platform-tools" / "adb"
            bundled_adb.parent.mkdir(parents=True)
            bundled_adb.touch()

            manager = ScrcpyManager(scrcpy_path=str(server_dir / "scrcpy"))
            with patch.dict(os.environ, {}, clear=True), patch(
                "devices.scrcpy_manager.shutil.which",
                return_value="/usr/local/bin/adb",
            ):
                self.assertEqual(manager._resolve_adb_path(), str(bundled_adb))

    def test_subprocess_env_puts_resolved_adb_on_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            server_dir = Path(tmp) / "server"
            bundled_adb = server_dir / "vendor" / "platform-tools" / "adb"
            bundled_adb.parent.mkdir(parents=True)
            bundled_adb.touch()

            manager = ScrcpyManager(scrcpy_path=str(server_dir / "scrcpy"))
            with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True), patch(
                "devices.scrcpy_manager.shutil.which",
                return_value=None,
            ):
                env = manager._subprocess_env()

            self.assertEqual(env["CODEBRIDGE_ADB_PATH"], str(bundled_adb))
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(bundled_adb.parent))

    def test_resolve_node_path_prefers_explicit_codebridge_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit_node = Path(tmp) / "custom-node"
            explicit_node.touch()

            manager = ScrcpyManager(scrcpy_path=str(Path(tmp) / "server" / "scrcpy"))
            with patch.dict(os.environ, {"CODEBRIDGE_NODE_PATH": str(explicit_node)}), patch(
                "devices.scrcpy_manager.shutil.which",
                return_value=None,
            ):
                self.assertEqual(manager._resolve_node_path(), str(explicit_node))

    def test_resolve_node_path_uses_bundled_runtime_before_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            server_dir = Path(tmp) / "server"
            bundled_node = server_dir / "vendor" / "node" / "bin" / "node"
            bundled_node.parent.mkdir(parents=True)
            bundled_node.touch()

            manager = ScrcpyManager(scrcpy_path=str(server_dir / "scrcpy"))
            with patch.dict(os.environ, {}, clear=True), patch(
                "devices.scrcpy_manager.shutil.which",
                return_value="/usr/local/bin/node",
            ):
                self.assertEqual(manager._resolve_node_path(), str(bundled_node))


if __name__ == "__main__":
    unittest.main()
