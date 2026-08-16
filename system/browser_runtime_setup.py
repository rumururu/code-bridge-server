"""One implementation of "make sure this machine can drive a browser".

`browser_action` workflow steps run on Chromium. `pip install playwright`
installs the Python package but never the browser binary, so a server can have
the import and still park every browser step waiting for a human — which is
exactly what happened here: the installer grew a Chromium step, the deploy path
actually in use (`install/sync-local-install.sh`) never called it, and the
machine ran for months with `playwright` importable and no Chromium behind it.

The lesson is that three copies of "is Chromium here?" in three scripts drift.
So the decision lives here, once, and everything else calls it:

    install/install.sh            ->  --ensure     (fresh install)
    install/install.ps1           ->  --ensure     (fresh install, Windows)
    install/sync-local-install.sh ->  --ensure     (already-installed machines)
    the dashboard install job     ->  imports `chromium_install_argv()`

This module deliberately imports nothing from the rest of the server: the
installers run it with a bare venv that has requirements.txt and nothing
configured yet, and it is also run as a plain file path
(`$INSTALL_DIR/system/browser_runtime_setup.py`) rather than as a package.

The cheapest browser is the one already installed. `detect_installed_chrome()`
looks for a real Google Chrome, and when it finds one nothing is downloaded at
all: Playwright drives it through `channel="chrome"`. The bundled Chromium is
the fallback for machines without Chrome — and it is the expensive path, since
each Playwright minor pins a different revision (measured on one machine: 1.0GB
of `ms-playwright` holding revisions 1200 and 1228 while the two installed
Playwrights wanted 1208 and 1234, so none of the three shared a build).

Three invariants that must not be softened:

*   **Present means present.** Readiness is decided by looking for an executable
    on disk — Chrome's or Chromium's — never by an install command having
    exited 0, and never by inferring Chrome from the platform. A download that
    half-succeeds must report as missing, not as ready.
*   **Never fatal.** A machine without a browser is a working Code Bridge
    server that says so. Nothing here may abort an install or a server start.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Stated wherever the download is offered. A user who is not told that a
# progress-less step is fetching 200MB reads the silence as a hang and kills it.
CHROMIUM_DOWNLOAD_MB = 200
CHROMIUM_DISK_MB = 450

#: Playwright's name for "the Google Chrome that is already on this machine".
#: Passing it to `launch(channel=...)` uses that install instead of the
#: bundled build, which is why a machine with Chrome needs no download at all.
CHROME_CHANNEL = "chrome"

#: Set to "0" to keep Chromium off this machine entirely. Honoured by the
#: installers, by the sync script and by the dashboard install endpoint, so
#: there is one switch rather than one per surface.
OPT_OUT_ENV = "CODE_BRIDGE_BROWSER"

STATUS_READY = "ready"
STATUS_CHROMIUM_MISSING = "chromium_missing"
STATUS_PLAYWRIGHT_MISSING = "playwright_missing"
STATUS_PROBE_FAILED = "probe_failed"

#: Where a real Google Chrome lives, per platform. Probed as files on disk —
#: never assumed from the platform alone, because "macOS therefore Chrome" is
#: the kind of guess that makes readiness lie.
_CHROME_EXECUTABLES: dict[str, tuple[str, ...]] = {
    "darwin": (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ),
    "win32": (
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    ),
    "linux": (
        "/opt/google/chrome/chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ),
}

#: The profile directory a normal Chrome keeps its logins in. Only ever used
#: when the operator explicitly asks for the sharing option; see
#: `system/browser_preferences.py` for why it is never a default.
_CHROME_USER_DATA_DIRS: dict[str, tuple[str, ...]] = {
    "darwin": ("~/Library/Application Support/Google/Chrome",),
    "win32": (r"%LOCALAPPDATA%\Google\Chrome\User Data",),
    "linux": ("~/.config/google-chrome",),
}

# Exit codes. `--ensure` returns EXIT_OK for both "installed it" and "already
# had it" and for an explicit opt-out; anything non-zero means the runtime is
# not usable and the caller should say so without failing the install.
EXIT_OK = 0
EXIT_PLAYWRIGHT_MISSING = 2
EXIT_CHROMIUM_MISSING = 3
EXIT_PROBE_FAILED = 4
EXIT_INSTALL_FAILED = 5


@dataclass(frozen=True)
class ProbeResult:
    """What the machine actually has, right now."""

    status: str
    ready: bool
    message: str
    executable_path: str | None = None
    #: Playwright channel that will be used, when it is not the bundled build.
    #: ``"chrome"`` means "an installed Google Chrome answered the probe".
    channel: str | None = None
    browser_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "message": self.message,
            "executable_path": self.executable_path,
            "channel": self.channel,
            "browser_name": self.browser_name,
        }


@dataclass(frozen=True)
class InstalledBrowser:
    """A browser that is on this machine right now, found by looking."""

    channel: str
    name: str
    executable_path: str
    user_data_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "name": self.name,
            "executable_path": self.executable_path,
            "user_data_dir": self.user_data_dir,
        }


def _expand(candidate: str) -> Path:
    return Path(os.path.expandvars(candidate)).expanduser()


def detect_installed_chrome(
    *,
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
) -> InstalledBrowser | None:
    """Return the Google Chrome on this machine, or ``None``.

    Every answer comes from a file that exists. The alternative — assuming
    Chrome because this is a Mac — would make the readiness surface claim no
    download is needed and then park every browser step at run time, which is
    the exact failure this whole surface exists to prevent.
    """
    system = platform or sys.platform
    if system.startswith("linux"):
        system = "linux"
    candidates = list(_CHROME_EXECUTABLES.get(system, ()))

    source = os.environ if env is None else env
    override = str(source.get("CODE_BRIDGE_CHROME_PATH", "")).strip()
    if override:
        candidates.insert(0, override)

    found: Path | None = None
    for candidate in candidates:
        path = _expand(candidate)
        if path.is_file():
            found = path
            break

    if found is None and system == "linux":
        for name in ("google-chrome", "google-chrome-stable"):
            located = shutil.which(name)
            if located:
                found = Path(located)
                break

    if found is None:
        return None

    return InstalledBrowser(
        channel=CHROME_CHANNEL,
        name="Google Chrome",
        executable_path=str(found),
        user_data_dir=_detect_chrome_user_data_dir(platform=system),
    )


def _detect_chrome_user_data_dir(*, platform: str | None = None) -> str | None:
    system = platform or sys.platform
    if system.startswith("linux"):
        system = "linux"
    for candidate in _CHROME_USER_DATA_DIRS.get(system, ()):
        path = _expand(candidate)
        if path.is_dir():
            return str(path)
    return None


def chromium_install_argv(python_executable: str | None = None) -> list[str]:
    """The command that downloads Chromium, as argv.

    Derived from the interpreter that is actually running, not from a guessed
    repo-relative path: the dev tree uses `server/venv` and a real install uses
    `~/.code-bridge/venv`, so a hardcoded path named a directory that exists in
    no install and told the operator to run a command that could not work.
    """
    return [python_executable or sys.executable, "-m", "playwright", "install", "chromium"]


def chromium_install_command(python_executable: str | None = None) -> str:
    """The same command as a copy-pasteable string (quoted for paths with spaces)."""
    return shlex.join(chromium_install_argv(python_executable))


def cost_notice() -> str:
    return (
        f"Download: about {CHROMIUM_DOWNLOAD_MB}MB. "
        f"On disk: about {CHROMIUM_DISK_MB}MB. Takes 1-5 minutes."
    )


def browser_runtime_opt_out(env: Mapping[str, str] | None = None) -> bool:
    """True when the operator has said this machine must not have Chromium."""
    source = os.environ if env is None else env
    return str(source.get(OPT_OUT_ENV, "1")).strip() == "0"


def probe_chromium(*, python_executable: str | None = None) -> ProbeResult:
    """Look for the Chromium build Playwright would launch.

    Synchronous on purpose: the installers call this in a subprocess where no
    event loop exists. The server has an async twin in
    `agent/browser_action_adapter.py::_probe_browser_runtime_readiness`; both
    answer the same question the same way — does the executable file exist.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001 - any import failure means "not usable"
        return ProbeResult(
            status=STATUS_PLAYWRIGHT_MISSING,
            ready=False,
            message=f"Python Playwright package is not available: {exc}",
        )

    try:
        with sync_playwright() as playwright:
            executable_path = str(playwright.chromium.executable_path)
    except Exception as exc:  # noqa: BLE001 - driver availability is host-specific
        return ProbeResult(
            status=STATUS_PROBE_FAILED,
            ready=False,
            message=f"Playwright driver is not available: {exc}",
        )

    if Path(executable_path).is_file():
        return ProbeResult(
            status=STATUS_READY,
            ready=True,
            message="Playwright Chromium is ready.",
            executable_path=executable_path,
        )
    return ProbeResult(
        status=STATUS_CHROMIUM_MISSING,
        ready=False,
        message=(
            "Playwright Chromium executable is missing. Run "
            f"`{chromium_install_command(python_executable)}`."
        ),
        executable_path=executable_path,
    )


def _default_runner(argv: Sequence[str]) -> int:
    # Inherits stdout/stderr so the download's own progress reaches the
    # operator's terminal. Silence is what makes a long step read as a hang.
    return subprocess.call(list(argv))


def probe_browser_runtime(
    *,
    python_executable: str | None = None,
    allow_installed_chrome: bool = True,
    chrome_detector: Callable[[], InstalledBrowser | None] | None = None,
) -> ProbeResult:
    """Can this machine drive a browser at all — by any route?

    An installed Chrome counts. Playwright drives it through
    ``channel="chrome"`` and never downloads anything for it, so a machine with
    Chrome is ready even with an empty `ms-playwright` cache. The bundled build
    is the fallback, not the definition of readiness.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure means "not usable"
        return ProbeResult(
            status=STATUS_PLAYWRIGHT_MISSING,
            ready=False,
            message=f"Python Playwright package is not available: {exc}",
        )

    if allow_installed_chrome:
        detect = chrome_detector or detect_installed_chrome
        chrome = detect()
        if chrome is not None:
            return ProbeResult(
                status=STATUS_READY,
                ready=True,
                message=(
                    f"{chrome.name} is installed at {chrome.executable_path} — "
                    "browser steps use it, so no download is needed."
                ),
                executable_path=chrome.executable_path,
                channel=chrome.channel,
                browser_name=chrome.name,
            )

    return probe_chromium(python_executable=python_executable)


def ensure_chromium(
    *,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
    probe: Callable[[], ProbeResult] | None = None,
    runner: Callable[[Sequence[str]], int] | None = None,
    log: Callable[[str], None] = print,
    allow_installed_chrome: bool = True,
    chrome_detector: Callable[[], InstalledBrowser | None] | None = None,
) -> tuple[int, ProbeResult]:
    """Install Chromium only if it is genuinely not here.

    Returns ``(exit_code, final_probe)``. The final probe is what decides the
    verdict: a command that exits 0 without leaving an executable behind is
    reported as a failure, never as success.

    An installed Google Chrome short-circuits the whole thing. Playwright can
    drive it directly, so downloading 200MB of a second Chromium onto a machine
    that already has a browser is cost with no benefit. ``--force-chromium``
    (``allow_installed_chrome=False``) is there for anyone who wants the
    bundled build anyway.
    """
    do_probe = probe or (lambda: probe_chromium(python_executable=python_executable))
    do_run = runner or _default_runner

    if browser_runtime_opt_out(env):
        result = ProbeResult(
            status=STATUS_CHROMIUM_MISSING,
            ready=False,
            message=f"Skipped: {OPT_OUT_ENV}=0.",
        )
        log(f"Browser runtime skipped ({OPT_OUT_ENV}=0).")
        log(f"  Browser steps will wait for a person until you run: "
            f"{chromium_install_command(python_executable)}")
        return EXIT_OK, result

    if allow_installed_chrome:
        detect = chrome_detector or detect_installed_chrome
        chrome = detect()
        if chrome is not None:
            result = ProbeResult(
                status=STATUS_READY,
                ready=True,
                message=(
                    f"{chrome.name} is installed at {chrome.executable_path} — "
                    "browser steps use it, so no download is needed."
                ),
                executable_path=chrome.executable_path,
                channel=chrome.channel,
                browser_name=chrome.name,
            )
            log(f"Browser runtime: using installed {chrome.name} ({chrome.executable_path}).")
            log(f"  No download needed (this would otherwise fetch ~{CHROMIUM_DOWNLOAD_MB}MB).")
            log("  Want the bundled Chromium as well? Re-run with --force-chromium.")
            return EXIT_OK, result

    before = do_probe()
    if before.ready:
        # The whole point of probing first: re-running the installer or the
        # sync script on a machine that already has Chromium must not spend
        # another 450MB, and must not even reach for the network.
        log(f"Browser runtime already present ({before.executable_path}).")
        return EXIT_OK, before

    if before.status == STATUS_PLAYWRIGHT_MISSING:
        # pip's job, not ours. Saying so beats running a command that cannot work.
        log(before.message)
        log("  Install Python dependencies first (pip install -r requirements.txt).")
        return EXIT_PLAYWRIGHT_MISSING, before

    if before.status == STATUS_PROBE_FAILED:
        log(before.message)
        return EXIT_PROBE_FAILED, before

    argv = chromium_install_argv(python_executable)
    if dry_run:
        log("Browser runtime missing.")
        log(f"  would run: {shlex.join(argv)}")
        log(f"  {cost_notice()}")
        return EXIT_CHROMIUM_MISSING, before

    log("Installing browser runtime (Chromium for browser workflow steps)...")
    log(f"  {cost_notice()}")
    log(f"  Skip with {OPT_OUT_ENV}=0 (browser steps then wait for a person).")

    returncode = do_run(argv)

    # Re-probe unconditionally. `playwright install` has exited 0 with an
    # unusable download before, and "the command succeeded" is not evidence
    # that the binary is on disk.
    after = do_probe()
    if after.ready:
        log("Browser runtime ready.")
        return EXIT_OK, after

    if returncode != 0:
        log(f"Chromium download failed (exit {returncode}) — the server still starts.")
    else:
        log("Chromium download reported success but no executable is present — "
            "the server still starts.")
    log(f"  Browser steps will wait for a person until you run: {shlex.join(argv)}")
    return EXIT_INSTALL_FAILED, after


def _probe_exit_code(result: ProbeResult) -> int:
    return {
        STATUS_READY: EXIT_OK,
        STATUS_CHROMIUM_MISSING: EXIT_CHROMIUM_MISSING,
        STATUS_PLAYWRIGHT_MISSING: EXIT_PLAYWRIGHT_MISSING,
        STATUS_PROBE_FAILED: EXIT_PROBE_FAILED,
    }.get(result.status, EXIT_PROBE_FAILED)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="browser_runtime_setup",
        description="Report or install the Chromium build browser_action steps run on.",
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Install Chromium when it is missing (default: report only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --ensure: say what would be downloaded, download nothing.",
    )
    parser.add_argument(
        "--force-chromium",
        action="store_true",
        help="Download the bundled Chromium even when Google Chrome is installed.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the probe result as JSON.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    lines: list[str] = []

    def log(message: str) -> None:
        lines.append(message)
        if not args.json:
            print(message)

    allow_chrome = not args.force_chromium
    if args.ensure:
        code, result = ensure_chromium(
            dry_run=args.dry_run,
            log=log,
            allow_installed_chrome=allow_chrome,
        )
    else:
        result = probe_browser_runtime(allow_installed_chrome=allow_chrome)
        code = _probe_exit_code(result)
        log(result.message)

    if args.json:
        import json

        payload = result.as_dict()
        payload["exit_code"] = code
        payload["log"] = lines
        payload["install_command"] = chromium_install_command()
        chrome = detect_installed_chrome()
        payload["installed_chrome"] = chrome.as_dict() if chrome else None
        print(json.dumps(payload))
    return code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
