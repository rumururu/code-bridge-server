import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from routes import dashboard


class DashboardDownloadRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(dashboard.router)
        self.client = TestClient(app)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.artifacts_dir = Path(self.tmpdir.name) / "dist" / "desktop_server_app"
        self.artifacts_dir.mkdir(parents=True)

    def tearDown(self):
        self.client.close()
        self.tmpdir.cleanup()

    def test_downloads_macos_desktop_installer(self):
        artifact = self.artifacts_dir / "Code Bridge Server.dmg"
        artifact.write_bytes(b"dmg-bytes")

        with patch.object(dashboard, "DESKTOP_ARTIFACTS_DIR", self.artifacts_dir):
            response = self.client.get("/downloads/desktop-server/macos")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"dmg-bytes")
        self.assertIn("Code%20Bridge%20Server.dmg", response.headers["content-disposition"])

    def test_downloads_windows_desktop_installer(self):
        artifact = self.artifacts_dir / "Code Bridge Server.msi"
        artifact.write_bytes(b"msi-bytes")

        with patch.object(dashboard, "DESKTOP_ARTIFACTS_DIR", self.artifacts_dir):
            response = self.client.get("/downloads/desktop-server/windows")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"msi-bytes")
        self.assertIn("Code%20Bridge%20Server.msi", response.headers["content-disposition"])

    def test_missing_desktop_installer_returns_404(self):
        with patch.object(dashboard, "DESKTOP_ARTIFACTS_DIR", self.artifacts_dir):
            response = self.client.get("/downloads/desktop-server/windows")

        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
