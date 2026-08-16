"""The overview must say whether cloudflared exists, not just that no tunnel runs.

Without this, a machine with no cloudflared binary and a machine with a
stopped tunnel look identical in the dashboard: "External Access" simply does
not turn on, and the reason lives only in the server log. The distinction only
became trustworthy after the PATH bootstrap — before it, a Homebrew-installed
cloudflared probed as missing, so surfacing the probe would have been
surfacing a lie.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from dashboard import dashboard_service  # noqa: E402
from dashboard.dashboard_service import DashboardTunnelStatus, _build_tunnel_status  # noqa: E402
from remote.tunnel_service import TunnelService  # noqa: E402
from system import system_status_service  # noqa: E402
from system.system_status_service import is_cloudflared_installed  # noqa: E402


class CloudflaredProbeTest(unittest.TestCase):
    def test_probe_follows_which(self):
        with patch("remote.tunnel_service.shutil.which", return_value="/opt/homebrew/bin/cloudflared"):
            self.assertTrue(is_cloudflared_installed())

    def test_probe_reports_false_when_binary_is_absent(self):
        with patch("remote.tunnel_service.shutil.which", return_value=None):
            self.assertFalse(is_cloudflared_installed())

    def test_probe_matches_the_tunnel_service_definition(self):
        """One definition of "installed", so banner and start path agree."""
        for which_result in ("/usr/local/bin/cloudflared", None):
            with self.subTest(which=which_result):
                with patch("remote.tunnel_service.shutil.which", return_value=which_result):
                    self.assertEqual(
                        is_cloudflared_installed(),
                        TunnelService.is_cloudflared_installed(),
                    )

    def test_probe_never_raises_into_the_overview(self):
        with patch("remote.tunnel_service.shutil.which", side_effect=OSError("boom")):
            self.assertFalse(is_cloudflared_installed())


class TunnelOverviewPayloadTest(unittest.TestCase):
    def test_payload_exposes_the_flag(self):
        payload = DashboardTunnelStatus(
            available=True, running=False, url=None, cloudflared_installed=False
        ).as_dict()

        self.assertIn("cloudflared_installed", payload)
        self.assertIs(payload["cloudflared_installed"], False)

    def test_flag_is_reported_even_when_no_tunnel_service_exists(self):
        """The case worth telling the user about: no tunnel ever started."""
        with patch.object(dashboard_service, "TUNNEL_AVAILABLE", True), \
             patch.object(dashboard_service, "get_tunnel_service", return_value=None), \
             patch.object(dashboard_service, "is_cloudflared_installed", return_value=False):
            status = _build_tunnel_status()

        self.assertFalse(status.cloudflared_installed)
        self.assertFalse(status.running)

    def test_flag_is_reported_when_the_tunnel_integration_is_unavailable(self):
        with patch.object(dashboard_service, "TUNNEL_AVAILABLE", False), \
             patch.object(dashboard_service, "is_cloudflared_installed", return_value=True):
            status = _build_tunnel_status()

        self.assertFalse(status.available)
        self.assertTrue(status.cloudflared_installed)

    def test_running_tunnel_reports_installed_from_the_service_status(self):
        class _Service:
            def get_status(self):
                return {
                    "installed": True,
                    "running": True,
                    "url": "https://example.trycloudflare.com",
                }

        with patch.object(dashboard_service, "TUNNEL_AVAILABLE", True), \
             patch.object(dashboard_service, "get_tunnel_service", return_value=_Service()), \
             patch.object(dashboard_service, "is_cloudflared_installed", return_value=False):
            status = _build_tunnel_status()

        self.assertTrue(status.cloudflared_installed)
        self.assertTrue(status.running)
        self.assertEqual(status.url, "https://example.trycloudflare.com")


class DashboardBannerTemplateTest(unittest.TestCase):
    """The payload only helps if the page actually reads it."""

    def setUp(self):
        self.markup = (SERVER_DIR / "dashboard" / "templates" / "dashboard.html").read_text(
            encoding="utf-8"
        )

    def test_banner_element_and_copyable_command_exist(self):
        self.assertIn('id="cloudflaredMissingBanner"', self.markup)
        self.assertIn("brew install cloudflared", self.markup)
        self.assertIn("copyCloudflaredCommand", self.markup)

    def test_banner_reads_the_overview_flag(self):
        self.assertIn("cloudflared_installed", self.markup)

    def test_banner_shows_only_on_an_explicit_false(self):
        """An absent flag is unknown; asserting "not installed" would be a claim."""
        self.assertIn("installed !== false", self.markup)


if __name__ == "__main__":
    unittest.main()
