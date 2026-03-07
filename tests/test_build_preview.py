import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from build_preview import serve_build_preview


class BuildPreviewTest(unittest.TestCase):
    def test_serve_build_preview_returns_404_when_missing_build(self):
        manager = SimpleNamespace(
            get_build_path=lambda _: None,
            get_build_status=lambda _: {"project_type": "flutter"},
        )

        response = serve_build_preview("demo", "", manager)

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 404)
        self.assertIn("No build available", response.body.decode())

    def test_serve_build_preview_flutter_rewrites_base_href(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_dir = Path(tmp_dir)
            (build_dir / "index.html").write_text('<base href="/">', encoding="utf-8")

            manager = SimpleNamespace(
                get_build_path=lambda _: str(build_dir),
                get_build_status=lambda _: {"project_type": "flutter"},
            )

            response = serve_build_preview("demo", "", manager)

        self.assertIsInstance(response, HTMLResponse)
        self.assertEqual(response.status_code, 200)
        self.assertIn('<base href="/build-preview/demo/">', response.body.decode())

    def test_serve_build_preview_nextjs_public_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            build_dir = project_root / ".next"
            build_dir.mkdir(parents=True, exist_ok=True)
            public_dir = project_root / "public"
            public_dir.mkdir(parents=True, exist_ok=True)
            (public_dir / "robots.txt").write_text("ok", encoding="utf-8")

            manager = SimpleNamespace(
                get_build_path=lambda _: str(build_dir),
                get_build_status=lambda _: {"project_type": "nextjs"},
            )

            response = serve_build_preview("demo", "robots.txt", manager)

        self.assertIsInstance(response, FileResponse)
        self.assertTrue(str(response.path).endswith("robots.txt"))

    def test_serve_build_preview_blocks_path_outside_allowed_base(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            build_dir = project_root / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            outside_file = project_root / "outside.txt"
            outside_file.write_text("secret", encoding="utf-8")
            symlink_path = build_dir / "link.txt"
            symlink_path.symlink_to(outside_file)

            manager = SimpleNamespace(
                get_build_path=lambda _: str(build_dir),
                get_build_status=lambda _: {"project_type": "flutter"},
            )

            response = serve_build_preview("demo", "link.txt", manager)

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.body.decode())


if __name__ == "__main__":
    unittest.main()
