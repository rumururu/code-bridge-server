"""Deployment must not treat the user's runtime state as code.

`install/verify_install.py` holds the single exclude/protect list that
`sync-local-install.sh` reads (`--print-rsync-filters`). Every path in it is
something the install directory owns and the repository never authors: the
database, the paired accounts, the API keys, the tunnel config.

The list is maintained by hand, so it goes stale the moment a feature adds a
new runtime directory — and it does so quietly, because nothing fails. When
browser steps gained a persistent Chrome profile under `core/browser_profile/`,
the very next deployment reported **194 orphan candidates** instead of 3: a
Chrome profile is thousands of small files, and the report is what a person
reads to decide whether a deployment did what they expected. Worse, that
directory is where the agent's logins live, so a future `--delete` — which the
script refuses today, but the list is also the answer to "what must never be
deleted" — would have taken them.

These tests read the list rather than the filesystem, so they hold on a machine
where the feature has never run.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_INSTALL = REPO_ROOT / "install" / "verify_install.py"


def _load_verify_install():
    spec = importlib.util.spec_from_file_location("verify_install", VERIFY_INSTALL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_install"] = module
    spec.loader.exec_module(module)
    return module


class RuntimeDirectoriesAreExcludedTest(unittest.TestCase):
    """Every directory the server writes user state into must be in the list."""

    #: Written by the server at runtime, never by the repository. Keep this in
    #: step with the code that creates them — that is the point of the test.
    RUNTIME_DIRS = (
        "/core/browser_sessions/",
        "/core/browser_profile/",
    )

    def setUp(self) -> None:
        self.module = _load_verify_install()
        self.patterns = set(self._exclude_patterns())
        self.protected = set(getattr(self.module, "PROTECTED_TARGETS", ()))

    def _exclude_patterns(self):
        value = getattr(self.module, "SOURCE_EXCLUDES", None)
        if not value:
            self.fail(
                "verify_install.SOURCE_EXCLUDES is gone; this test guards that "
                "list and must be pointed at whatever replaced it"
            )
        return value

    def test_every_runtime_directory_is_excluded(self) -> None:
        for path in self.RUNTIME_DIRS:
            with self.subTest(path=path):
                self.assertIn(
                    path,
                    self.patterns,
                    f"{path} is written by the server at runtime. Left out of the "
                    "exclude list it is reported as orphaned code on every "
                    "deployment, and a Chrome profile alone is ~190 files.",
                )

    def test_the_login_profile_is_also_protected_from_deletion(self) -> None:
        """Excluding keeps it out of the transfer; protecting keeps it alive."""
        self.assertIn(
            "/core/browser_profile/",
            self.protected,
            "PROTECTED_TARGETS is what a delete must never reach, and this "
            "directory holds the logins an unattended browser agent runs on",
        )

    def test_the_browser_profile_is_not_something_the_repo_ships(self) -> None:
        """If the repo ever authored it, excluding it would hide a real file."""
        self.assertFalse(
            (REPO_ROOT / "server" / "core" / "browser_profile").exists(),
            "server/core/browser_profile exists in the checkout — it is runtime "
            "state and must not be committed; check .gitignore and remove it",
        )


if __name__ == "__main__":
    unittest.main()
