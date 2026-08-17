"""A persistent browser profile must actually keep the logins it is for.

Playwright appends `--use-mock-keychain` to every Chromium launch so an
automated run never blocks on an OS keychain prompt. On macOS that argument
also swaps the key Chrome encrypts its cookie store with — and Chrome deletes
any cookie it cannot decrypt. A profile opened alternately by a person's
Chrome (real keychain) and by Playwright (mock keychain) therefore loses every
cookie at each handover.

Measured on the development machine before the fix: a person's Chrome visited
naver.com and wrote 10 cookies; the very next Playwright launch on the same
`user_data_dir` left 0. With `--use-mock-keychain` dropped, all 10 survived
with their creation timestamps intact.

Nothing raises when this regresses. The browser starts, the run proceeds, and
the site simply shows a logged-out page — which is why it is pinned here.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from system.browser_preferences import (  # noqa: E402
    MOCK_KEYCHAIN_ARG,
    BrowserLaunchPlan,
    persistent_profile_launch_overrides,
)


def _plan(**overrides) -> BrowserLaunchPlan:
    base = dict(
        browser="chrome",
        label="Google Chrome",
        channel="chrome",
        executable_path=None,
        headless=True,
        persistent=True,
        user_data_dir="/tmp/code-bridge-tests/profile",
        profile="dedicated",
        install_required=False,
    )
    base.update(overrides)
    return BrowserLaunchPlan(**base)


class PersistentProfileKeepsLoginsTest(unittest.TestCase):
    def test_installed_chrome_profile_uses_the_real_keychain(self) -> None:
        overrides = persistent_profile_launch_overrides(_plan())
        self.assertIn(
            MOCK_KEYCHAIN_ARG,
            overrides.get("ignore_default_args", []),
            "Without dropping this argument the profile cannot decrypt the "
            "cookies a person's Chrome wrote, and Chrome deletes them. The "
            "sign-in a user performs in the browser handoff is then gone by "
            "the next scheduled run, with no error anywhere.",
        )

    def test_bundled_chromium_keeps_the_mock_keychain(self) -> None:
        """No channel means Playwright's own Chromium: no keychain entry of its
        own, so the real keychain would raise a permission dialog that an
        unattended run has nobody to answer."""
        self.assertEqual(persistent_profile_launch_overrides(_plan(channel=None)), {})

    def test_a_throwaway_profile_needs_no_override(self) -> None:
        """Non-persistent launches carry no logins, so there is nothing to lose
        and no reason to touch the keychain."""
        self.assertEqual(persistent_profile_launch_overrides(_plan(persistent=False)), {})
        self.assertEqual(persistent_profile_launch_overrides(_plan(user_data_dir=None)), {})


class BothLaunchSitesApplyItTest(unittest.TestCase):
    """Two modules launch persistent contexts — the step adapter and the
    interactive handoff. A login is only carried forward if *both* use the same
    encryption key, so neither may be left behind."""

    LAUNCH_SITES = (
        "agent/browser_action_adapter.py",
        "agent/browser_runtime_manager.py",
    )

    def test_every_persistent_launch_applies_the_overrides(self) -> None:
        server_root = SERVER_DIR
        for rel in self.LAUNCH_SITES:
            with self.subTest(module=rel):
                source = (server_root / rel).read_text(encoding="utf-8")
                self.assertIn("launch_persistent_context", source)
                self.assertIn(
                    "persistent_profile_launch_overrides",
                    source,
                    f"{rel} opens the shared profile without the keychain "
                    "override; whichever side runs second deletes the other's "
                    "cookies",
                )


if __name__ == "__main__":
    unittest.main()
