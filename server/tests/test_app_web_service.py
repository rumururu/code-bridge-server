import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import app_web_service


class AppWebServiceTest(unittest.TestCase):
    def test_resolve_flutter_web_file_for_current_server_missing_build(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_build = Path(tmp_dir) / "missing"
            result = app_web_service.resolve_flutter_web_file_for_current_server(
                "",
                build_root=missing_build,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.error_payload.get("error"), "Flutter web build not found")

    def test_resolve_flutter_web_file_for_current_server_index_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_root = Path(tmp_dir)
            index_file = build_root / "index.html"
            index_file.write_text('<base href="/">hello')

            result = app_web_service.resolve_flutter_web_file_for_current_server(
                "missing/path.js",
                build_root=build_root,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.file_path, index_file)

    def test_render_flutter_index_for_current_server_replaces_base_href(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "index.html"
            file_path.write_text('<base href="/">')

            content = app_web_service.render_flutter_index_for_current_server(file_path)

        self.assertEqual(content, '<base href="/app/">')

    def test_get_flutter_media_type_for_current_server(self):
        media_type = app_web_service.get_flutter_media_type_for_current_server(Path("main.js"))
        self.assertEqual(media_type, "application/javascript")


if __name__ == "__main__":
    unittest.main()
