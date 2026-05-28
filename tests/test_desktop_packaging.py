import importlib.util
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_desktop_server_app.py"
SPEC = importlib.util.spec_from_file_location("build_desktop_server_app", SCRIPT_PATH)
packaging = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = packaging
SPEC.loader.exec_module(packaging)


class DesktopPackagingFilterTest(unittest.TestCase):
    def test_stage_server_tree_excludes_runtime_state_without_dropping_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_dir = root / "server"
            build_dir = root / "build"
            (server_dir / "core").mkdir(parents=True)
            (server_dir / "firebase").mkdir()
            (server_dir / "tests").mkdir()
            (server_dir / "scrcpy" / "dist").mkdir(parents=True)
            (server_dir / "__pycache__").mkdir()

            for path in [
                server_dir / "main.py",
                server_dir / "core" / "config.py",
                server_dir / "firebase" / "paired_accounts.py",
                server_dir / "config.yaml",
                server_dir / ".env.local",
                server_dir / "api_keys.json",
                server_dir / "server_info.json.backup",
                server_dir / "device_info.sqlite3-wal",
                server_dir / "code_bridge.db.bak_before_review",
                server_dir / "server.log",
                server_dir / ".server.pid",
                server_dir / "tests" / "test_demo.py",
                server_dir / "scrcpy" / "dist" / "bundle.js",
                server_dir / "__pycache__" / "main.pyc",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            with patch.object(packaging, "SERVER_DIR", server_dir):
                staged = packaging.stage_server_tree(build_dir)

            self.assertTrue((staged / "main.py").exists())
            self.assertTrue((staged / "core" / "config.py").exists())
            self.assertTrue((staged / "firebase" / "paired_accounts.py").exists())
            self.assertFalse((staged / "config.yaml").exists())
            self.assertFalse((staged / ".env.local").exists())
            self.assertFalse((staged / "api_keys.json").exists())
            self.assertFalse((staged / "server_info.json.backup").exists())
            self.assertFalse((staged / "device_info.sqlite3-wal").exists())
            self.assertFalse((staged / "code_bridge.db.bak_before_review").exists())
            self.assertFalse((staged / "server.log").exists())
            self.assertFalse((staged / ".server.pid").exists())
            self.assertFalse((staged / "tests").exists())
            self.assertFalse((staged / "scrcpy").exists())
            self.assertFalse((staged / "__pycache__").exists())

    def test_stage_scrcpy_dist_uses_filtered_stage_for_add_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_dir = root / "server"
            build_dir = root / "build"
            dist = server_dir / "scrcpy" / "dist"
            (dist / "node_modules" / "module").mkdir(parents=True)
            (dist / "__pycache__").mkdir()

            for path in [
                dist / "bundle.js",
                dist / "node_modules" / "module" / "runtime.node",
                dist / ".env.production",
                dist / "paired_accounts.json",
                dist / "debug.log",
                dist / "cache.sqlite3-shm",
                dist / "__pycache__" / "bundle.pyc",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            with patch.object(packaging, "SERVER_DIR", server_dir):
                staged = packaging.stage_scrcpy_dist(build_dir)
                items = packaging.collect_data_items(True, build_dir / packaging.SERVER_STAGE_DIRNAME)

            self.assertEqual(items[1].source, build_dir / packaging.SCRCPY_DIST_STAGE_DIRNAME)
            self.assertTrue((staged / "bundle.js").exists())
            self.assertTrue((staged / "node_modules" / "module" / "runtime.node").exists())
            self.assertFalse((staged / ".env.production").exists())
            self.assertFalse((staged / "paired_accounts.json").exists())
            self.assertFalse((staged / "debug.log").exists())
            self.assertFalse((staged / "cache.sqlite3-shm").exists())
            self.assertFalse((staged / "__pycache__").exists())


class DesktopPackagingCacheTest(unittest.TestCase):
    def test_node_stage_marker_mismatch_reextracts_and_rewrites_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            archive_path = build_dir / packaging.NODE_DOWNLOAD_DIRNAME / "node-v1-linux-x64.tar.xz"
            source_node = root / "extracted" / "bin" / "node"
            stage_node = build_dir / packaging.NODE_RUNTIME_STAGE_DIRNAME / "bin" / "node"
            marker = build_dir / packaging.NODE_RUNTIME_STAGE_DIRNAME / ".codebridge-node-runtime"

            archive_path.parent.mkdir(parents=True)
            archive_path.write_text("fresh archive", encoding="utf-8")
            source_node.parent.mkdir(parents=True)
            source_node.write_text("fresh node", encoding="utf-8")
            stage_node.parent.mkdir(parents=True)
            stage_node.write_text("stale node", encoding="utf-8")
            marker.write_text(json.dumps({"schema": 1, "archive_sha256": "wrong"}) + "\n", encoding="utf-8")

            spec = packaging.NodeRuntimeSpec("linux-x64", "tar.xz", "bin/node", "bin/node")
            with patch.object(packaging, "node_runtime_spec", return_value=spec), patch.object(
                packaging,
                "download_node_archive",
                return_value=archive_path,
            ) as download, patch.object(packaging, "extract_node_archive", return_value=root / "extracted") as extract:
                staged = packaging.stage_node_runtime(build_dir, "v1", "linux")

            self.assertEqual(staged, (build_dir / packaging.NODE_RUNTIME_STAGE_DIRNAME).resolve())
            download.assert_called_once()
            extract.assert_called_once()
            self.assertEqual(stage_node.read_text(encoding="utf-8"), "fresh node")
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(marker_data["kind"], "node-runtime")
            self.assertEqual(marker_data["archive_name"], archive_path.name)
            self.assertEqual(marker_data["archive_sha256"], packaging.hash_file(archive_path, "sha256"))


class DesktopPackagingMsiTest(unittest.TestCase):
    def test_generated_msi_uses_64_bit_program_files_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "Code Bridge Server"
            wxs_path = root / "installer.wxs"
            app_dir.mkdir()
            (app_dir / "Code Bridge Server.exe").write_text("x", encoding="utf-8")

            packaging.generate_wix_source(app_dir, wxs_path, "Code Bridge Server", "1.0.0")

            ns = {"wix": packaging.WIX_NAMESPACE}
            tree = ET.parse(wxs_path)
            standard_dir = tree.find(".//wix:StandardDirectory", ns)
            self.assertIsNotNone(standard_dir)
            self.assertEqual(standard_dir.attrib["Id"], "ProgramFiles64Folder")

    def test_generated_msi_includes_start_menu_shortcut(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "Code Bridge Server"
            wxs_path = root / "installer.wxs"
            app_dir.mkdir()
            (app_dir / "Code Bridge Server.exe").write_text("x", encoding="utf-8")

            packaging.generate_wix_source(app_dir, wxs_path, "Code Bridge Server", "1.0.0")

            ns = {"wix": packaging.WIX_NAMESPACE}
            tree = ET.parse(wxs_path)
            shortcut = tree.find(".//wix:Shortcut", ns)
            self.assertIsNotNone(shortcut)
            self.assertEqual(shortcut.attrib["Name"], "Code Bridge Server")
            self.assertEqual(shortcut.attrib["Target"], "[INSTALLFOLDER]Code Bridge Server.exe")


if __name__ == "__main__":
    unittest.main()
