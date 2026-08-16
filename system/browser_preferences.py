"""What browser the agent drives, where it shows up, and whose logins it has.

Three questions decide whether "check the board every morning" actually works,
and the product used to answer all three by itself, in code:

*   **Which browser.** `playwright.chromium.launch()` with no `channel` uses the
    bundled Chromium — a second browser, downloaded per Playwright minor
    version, that nobody is logged into. A machine with Google Chrome already
    installed needs no download at all: Playwright drives it through
    `channel="chrome"`.
*   **Headed or headless.** Was a hardcoded `headless=True`.
*   **Whose profile.** Was none: a fresh context every launch, so no cookies, so
    no logins, so the one thing a daily site check is for could not happen.

None of these have a right answer that holds for everybody, so this module
stores the operator's answer and resolves it into a launch plan. Nothing here
picks the option that shares the user's own browser sessions with the agent —
that one is only ever reached by someone choosing it after reading what it
costs.

The settings live in the same `app_settings` key-value table as `allow_ip_login`
and the LLM selection (`core/database.py::SettingsDB`), under one JSON key, so
an unconfigured install has no row and gets the documented defaults.

Deliberately separate from `system/browser_runtime_setup.py`: that module is run
by the installers as a bare file with only `requirements.txt` behind it and must
not import the server. This one is server-side and reads the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime_paths import SERVER_DIR, runtime_dir
from system.browser_runtime_setup import (
    CHROME_CHANNEL,
    InstalledBrowser,
    detect_installed_chrome,
)

logger = logging.getLogger(__name__)

SETTING_KEY = "browser_automation"

# --- which browser -----------------------------------------------------------
BROWSER_AUTO = "auto"
BROWSER_CHROME = "chrome"
BROWSER_CHROMIUM = "chromium"
BROWSER_CHOICES = (BROWSER_AUTO, BROWSER_CHROME, BROWSER_CHROMIUM)

# --- headed or headless ------------------------------------------------------
HEADLESS_AUTO = "auto"
HEADLESS_ON = "headless"
HEADLESS_OFF = "headed"
HEADLESS_MODES = (HEADLESS_AUTO, HEADLESS_ON, HEADLESS_OFF)

# --- whose profile -----------------------------------------------------------
PROFILE_DEDICATED = "dedicated"
PROFILE_SHARED_CHROME = "shared_chrome"
PROFILE_EPHEMERAL = "ephemeral"
PROFILE_MODES = (PROFILE_DEDICATED, PROFILE_SHARED_CHROME, PROFILE_EPHEMERAL)

DEFAULT_BROWSER = BROWSER_AUTO
DEFAULT_HEADLESS_MODE = HEADLESS_AUTO
DEFAULT_PROFILE = PROFILE_DEDICATED

# Blocked reasons. Each one is a state the operator asked for that this machine
# cannot deliver; the adapter turns them into a `waiting_for_user` naming the
# reason rather than quietly running with different settings.
BLOCKED_CHROME_NOT_INSTALLED = "browser_chrome_not_installed"
BLOCKED_SHARED_PROFILE_MISSING = "browser_shared_profile_missing"
BLOCKED_SHARED_PROFILE_NEEDS_CHROME = "browser_shared_profile_needs_chrome"

WARN_SHARED_PROFILE_SCOPE = "shared_profile_reaches_every_login"
WARN_SHARED_PROFILE_LOCK = "shared_profile_conflicts_with_open_chrome"
WARN_HEADED_NEEDS_DISPLAY = "headed_needs_a_display"


@dataclass(frozen=True)
class BrowserPreferences:
    """The operator's three answers. Defaults are what an unconfigured install runs."""

    browser: str = DEFAULT_BROWSER
    headless_mode: str = DEFAULT_HEADLESS_MODE
    profile: str = DEFAULT_PROFILE

    def as_dict(self) -> dict[str, str]:
        return {
            "browser": self.browser,
            "headless_mode": self.headless_mode,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class BrowserLaunchPlan:
    """Exactly how the next browser launch will happen, and what it costs."""

    browser: str
    label: str
    channel: str | None
    executable_path: str | None
    headless: bool
    persistent: bool
    user_data_dir: str | None
    profile: str
    install_required: bool
    blocked_reason: str | None = None
    blocked_message: str | None = None
    warnings: list[dict[str, str]] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.blocked_reason is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "browser": self.browser,
            "label": self.label,
            "channel": self.channel,
            "executable_path": self.executable_path,
            "headless": self.headless,
            "persistent": self.persistent,
            "user_data_dir": self.user_data_dir,
            "profile": self.profile,
            "install_required": self.install_required,
            "blocked_reason": self.blocked_reason,
            "blocked_message": self.blocked_message,
            "warnings": [dict(item) for item in self.warnings],
        }


def dedicated_profile_dir() -> Path:
    """The agent's own browser profile — created on first use, kept between runs.

    This is what makes a login survive: a persistent user-data-dir holds the
    cookies, so tomorrow's scheduled run opens already signed in. It is not the
    user's Chrome profile, so a person has to sign in once, in the browser
    handoff, and only once.
    """
    return runtime_dir("browser_profile", SERVER_DIR / "core" / "browser_profile")


def _coerce(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def get_browser_preferences() -> BrowserPreferences:
    """Read the stored answers. An unreadable store means defaults, never a crash."""
    raw: Any = {}
    try:
        from core.database import get_settings_db

        raw = get_settings_db().get_json(SETTING_KEY, {})
    except Exception:  # noqa: BLE001 - a browser setting must not take the server down
        logger.warning("browser preferences could not be read; using defaults", exc_info=True)
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return BrowserPreferences(
        browser=_coerce(raw.get("browser"), BROWSER_CHOICES, DEFAULT_BROWSER),
        headless_mode=_coerce(
            raw.get("headless_mode"), HEADLESS_MODES, DEFAULT_HEADLESS_MODE
        ),
        profile=_coerce(raw.get("profile"), PROFILE_MODES, DEFAULT_PROFILE),
    )


def set_browser_preferences(
    *,
    browser: str | None = None,
    headless_mode: str | None = None,
    profile: str | None = None,
) -> BrowserPreferences:
    """Write the answers the operator gave. Rejects values it does not implement.

    Unknown values raise rather than silently degrading to a default: an
    operator who asked for the shared Chrome profile and got the dedicated one
    without being told would believe the agent is logged in when it is not.
    """
    current = get_browser_preferences()
    updated = BrowserPreferences(
        browser=_require(browser, BROWSER_CHOICES, "browser", current.browser),
        headless_mode=_require(
            headless_mode, HEADLESS_MODES, "headless_mode", current.headless_mode
        ),
        profile=_require(profile, PROFILE_MODES, "profile", current.profile),
    )
    from core.database import get_settings_db

    get_settings_db().set_json(SETTING_KEY, updated.as_dict())
    return updated


def _require(value: Any, allowed: tuple[str, ...], field_name: str, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValueError(
            f"{field_name} must be one of {', '.join(allowed)} (got {value!r})"
        )
    return text


def resolve_browser_launch_plan(
    preferences: BrowserPreferences | None = None,
    *,
    chrome: InstalledBrowser | None = None,
    detect_chrome: bool = True,
    chromium_executable_path: str | None = None,
    chromium_present: bool | None = None,
) -> BrowserLaunchPlan:
    """Turn the three answers into one launch, with its costs stated.

    ``chrome`` may be passed in by a caller that already probed (readiness does)
    so this does not stat the filesystem twice. ``detect_chrome=False`` with no
    ``chrome`` means "behave as if no Chrome is installed" — used by tests, and
    by nothing else.
    """
    prefs = preferences or get_browser_preferences()
    if chrome is None and detect_chrome:
        chrome = detect_installed_chrome()

    warnings: list[dict[str, str]] = []

    # --- which browser -------------------------------------------------------
    if prefs.browser == BROWSER_CHROME and chrome is None:
        return _blocked(
            prefs,
            BLOCKED_CHROME_NOT_INSTALLED,
            "This server is set to use installed Google Chrome, but no Chrome "
            "was found on it. Install Chrome, or change the browser setting to "
            "auto so the bundled Chromium is used instead.",
        )

    use_chrome = chrome is not None and prefs.browser in (BROWSER_AUTO, BROWSER_CHROME)
    if use_chrome:
        assert chrome is not None
        browser_key = BROWSER_CHROME
        label = f"{chrome.name} (installed)"
        channel: str | None = CHROME_CHANNEL
        executable_path: str | None = chrome.executable_path
        install_required = False
    else:
        browser_key = BROWSER_CHROMIUM
        label = "Bundled Chromium"
        channel = None
        executable_path = chromium_executable_path
        install_required = chromium_present is False

    # --- whose profile -------------------------------------------------------
    persistent = prefs.profile in (PROFILE_DEDICATED, PROFILE_SHARED_CHROME)
    user_data_dir: str | None = None
    if prefs.profile == PROFILE_DEDICATED:
        user_data_dir = str(dedicated_profile_dir())
    elif prefs.profile == PROFILE_SHARED_CHROME:
        if browser_key != BROWSER_CHROME:
            return _blocked(
                prefs,
                BLOCKED_SHARED_PROFILE_NEEDS_CHROME,
                "Sharing your Chrome profile needs the installed Chrome, but "
                "this server is set to use the bundled Chromium. Set the "
                "browser to auto or Chrome, or pick a different profile.",
            )
        shared_dir = chrome.user_data_dir if chrome is not None else None
        if not shared_dir:
            return _blocked(
                prefs,
                BLOCKED_SHARED_PROFILE_MISSING,
                "This server is set to share your Chrome profile, but Chrome's "
                "profile directory was not found on this machine. Pick the "
                "agent's own profile instead.",
            )
        user_data_dir = shared_dir
        warnings.append(
            {
                "code": WARN_SHARED_PROFILE_SCOPE,
                "message": (
                    "The agent is using your own Chrome profile. It can reach "
                    "every site that browser is signed into — mail, banking, "
                    "work accounts — not just the site the step names."
                ),
            }
        )
        warnings.append(
            {
                "code": WARN_SHARED_PROFILE_LOCK,
                "message": (
                    "Chrome must be closed while a run uses this profile. Two "
                    "processes cannot hold one profile, so a run started while "
                    "Chrome is open fails on the profile lock."
                ),
            }
        )

    # --- headed or headless --------------------------------------------------
    # `auto` is headless. Not because headed is worse at driving sites — it is
    # better, and some sites refuse headless outright — but because the default
    # has to hold for the run nobody is watching, which is the whole point of a
    # scheduled agent: a window that grabs focus at 03:00 is hostile, and a
    # server with no display cannot open one at all. The login problem that
    # headed mode used to be the answer to is solved by the profile above
    # instead. `headed` stays one setting away for sites that block headless.
    headless = prefs.headless_mode != HEADLESS_OFF
    if not headless:
        warnings.append(
            {
                "code": WARN_HEADED_NEEDS_DISPLAY,
                "message": (
                    "Headed runs need a desktop session on the machine hosting "
                    "the server. On a headless host the launch fails and the "
                    "step reports why rather than falling back."
                ),
            }
        )

    return BrowserLaunchPlan(
        browser=browser_key,
        label=label,
        channel=channel,
        executable_path=executable_path,
        headless=headless,
        persistent=persistent,
        user_data_dir=user_data_dir,
        profile=prefs.profile,
        install_required=install_required,
        warnings=warnings,
    )


def _blocked(
    prefs: BrowserPreferences,
    reason: str,
    message: str,
) -> BrowserLaunchPlan:
    return BrowserLaunchPlan(
        browser=prefs.browser,
        label="",
        channel=None,
        executable_path=None,
        headless=prefs.headless_mode != HEADLESS_OFF,
        persistent=prefs.profile in (PROFILE_DEDICATED, PROFILE_SHARED_CHROME),
        user_data_dir=None,
        profile=prefs.profile,
        install_required=False,
        blocked_reason=reason,
        blocked_message=message,
    )


def preference_options() -> dict[str, Any]:
    """The choices, and what each one costs — for whatever surface asks.

    Kept here rather than in the template so the dashboard cannot describe an
    option differently from the way the server implements it.
    """
    return {
        "browser": [
            {
                "value": BROWSER_AUTO,
                "label": "Use installed Chrome when there is one",
                "detail": (
                    "Prefers the Google Chrome already on this machine, and "
                    "falls back to the bundled Chromium when there is none."
                ),
            },
            {
                "value": BROWSER_CHROME,
                "label": "Installed Google Chrome only",
                "detail": "Runs stop and say so if Chrome is not on this machine.",
            },
            {
                "value": BROWSER_CHROMIUM,
                "label": "Bundled Chromium only",
                "detail": (
                    "Needs a ~200MB download per Playwright version, and is "
                    "easier for sites to detect."
                ),
            },
        ],
        "headless_mode": [
            {
                "value": HEADLESS_AUTO,
                "label": "Let the server choose (headless)",
                "detail": (
                    "Today that is headless for every run: scheduled runs have "
                    "nobody watching, and a server may have no display at all."
                ),
            },
            {
                "value": HEADLESS_ON,
                "label": "Always headless",
                "detail": "No window. Fastest, and works on a machine with no display.",
            },
            {
                "value": HEADLESS_OFF,
                "label": "Always show a window",
                "detail": (
                    "For sites that refuse headless browsers. Needs a desktop "
                    "session on the host; a headless host fails the launch and says so."
                ),
            },
        ],
        "profile": [
            {
                "value": PROFILE_DEDICATED,
                "label": "The agent's own profile",
                "detail": (
                    "Sign in once, in the browser handoff, and the login is "
                    "kept for every later run. The agent sees only what it "
                    "signed into itself."
                ),
            },
            {
                "value": PROFILE_SHARED_CHROME,
                "label": "Your own Chrome profile",
                "detail": (
                    "Signed in immediately — and the agent can reach every site "
                    "that browser is signed into, not just this one. Chrome must "
                    "be closed while a run uses it."
                ),
                "consequence": "shared",
            },
            {
                "value": PROFILE_EPHEMERAL,
                "label": "No profile",
                "detail": (
                    "Nothing is kept. Every run starts signed out — fine for "
                    "public pages, useless for anything behind a login."
                ),
            },
        ],
    }
