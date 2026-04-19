"""Suppress Chrome's first-run interstitials on emulator devices.

Stock Chrome on Android shows a "Welcome to Chrome" / sign-in flow on first
launch that blocks the dev server URL from rendering. On userdebug emulator
images we can drop a Chromium command-line file under /data/local/tmp/ which
Chrome reads at startup and which honours --no-first-run / --disable-fre.

This module is best-effort: if writing the file fails (production-flavoured
device, missing chrome, etc.) the caller should still launch Chrome — the
worst case is just the original first-run experience.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CHROME_PACKAGE = "com.android.chrome"
COMMAND_LINE_PATH = "/data/local/tmp/chrome-command-line"
COMMAND_LINE_BODY = (
    "_ --disable-fre --no-first-run --no-default-browser-check "
    "--disable-domain-reliability --disable-notifications "
    "--disable-features=FeatureNotificationGuide,DefaultBrowserPromptAndroid"
)

# Devices we have already prepared this server run.
_prepared_devices: set[str] = set()


async def _run_adb(adb_path: str, *args: str, timeout: float = 10.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        adb_path,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "timed out"
    output = (stdout + stderr).decode("utf-8", errors="replace").strip()
    return proc.returncode if proc.returncode is not None else -1, output


async def _chrome_installed(adb_path: str, device_id: str) -> bool:
    code, output = await _run_adb(
        adb_path, "-s", device_id, "shell", "pm", "list", "packages", CHROME_PACKAGE
    )
    return code == 0 and CHROME_PACKAGE in output


def _extract_bounds_after(dump: str, marker: str) -> Optional[tuple[int, int]]:
    """Find the first occurrence of `marker` in `dump` and return the center
    (x, y) of the next `bounds="[L,T][R,B]"` attribute that follows it."""
    idx = dump.find(marker)
    if idx == -1:
        return None
    tail = dump[idx : idx + 800]
    bstart = tail.find('bounds="[')
    if bstart == -1:
        return None
    raw = tail[bstart + len('bounds="[') :]
    try:
        left_top, _, rest = raw.partition("][")
        right_bot, _, _ = rest.partition("]")
        l, t = (int(x) for x in left_top.split(","))
        r, b = (int(x) for x in right_bot.split(","))
    except ValueError:
        return None
    return (l + r) // 2, (t + b) // 2


# Markers to look for in uiautomator dumps. Order matters: Welcome ToS comes
# first when Chrome data is fresh, then sign-in, then notifications. Each
# marker is associated with the resource-id of the button that dismisses the
# corresponding prompt.
_KNOWN_INTERSTITIALS = (
    # Chrome ToS accept (older Chrome builds).
    ("com.android.chrome:id/terms_accept", "com.android.chrome:id/terms_accept"),
    # First-run sign-in promo (Chrome >=120). Tap "Use without an account".
    (
        "com.android.chrome:id/signin_fre_dismiss_button",
        "com.android.chrome:id/signin_fre_dismiss_button",
    ),
    # Generic dismiss for sync / sign-in / notifications dialogs.
    ("com.android.chrome:id/negative_button", "com.android.chrome:id/negative_button"),
)


async def _dismiss_chrome_interstitials(adb_path: str, device_id: str) -> None:
    """Best-effort: tap through Chrome's first-run / sign-in / notifications
    prompts so the dev server URL becomes visible immediately. Each iteration
    dumps the UI, finds the first matching dismiss button, and taps it. Runs
    a few cycles because Chrome chains prompts (ToS → sign-in → notifications).
    """
    logger.info("[chrome] auto-dismiss interstitials starting on %s", device_id)
    for attempt in range(10):
        dump_code, _ = await _run_adb(
            adb_path,
            "-s",
            device_id,
            "shell",
            "uiautomator",
            "dump",
            "/sdcard/_cb_ui.xml",
            timeout=8.0,
        )
        if dump_code != 0:
            await asyncio.sleep(1.0)
            continue

        _, dump = await _run_adb(
            adb_path, "-s", device_id, "shell", "cat", "/sdcard/_cb_ui.xml"
        )
        # Stop once we see the Chrome address bar — that means the page is
        # actually rendering and no interstitial is in the way.
        if (
            "com.android.chrome:id/url_bar" in dump
            and "com.android.chrome:id/negative_button" not in dump
            and "com.android.chrome:id/terms_accept" not in dump
        ):
            return

        tapped = False
        for marker, dismiss_id in _KNOWN_INTERSTITIALS:
            if marker not in dump:
                continue
            center = _extract_bounds_after(dump, dismiss_id)
            if center is None:
                continue
            cx, cy = center
            logger.info(
                "[chrome] dismissing %s at (%d,%d) on %s",
                dismiss_id,
                cx,
                cy,
                device_id,
            )
            await _run_adb(
                adb_path, "-s", device_id, "shell", "input", "tap", str(cx), str(cy)
            )
            tapped = True
            await asyncio.sleep(1.5)
            break

        if not tapped:
            # Nothing matched — give Chrome a beat to render and retry.
            await asyncio.sleep(1.5)


# Backwards-compat alias used by scrcpy_manager.
_dismiss_notification_prompt = _dismiss_chrome_interstitials


async def ensure_chrome_first_run_skipped(
    adb_path: str, device_id: str
) -> Optional[str]:
    """Drop a Chrome command-line file so Chrome skips its onboarding flow.

    Returns None on success or when nothing to do, or an error string for
    diagnostics. The caller can ignore the return value: a failed prepare
    just means Chrome will show its normal first-run UI.
    """
    if device_id in _prepared_devices:
        return None

    if not await _chrome_installed(adb_path, device_id):
        # No Chrome — nothing to prepare. Mark as handled so we don't retry.
        _prepared_devices.add(device_id)
        return None

    # Write the command-line file Chrome reads at startup.
    code, output = await _run_adb(
        adb_path,
        "-s",
        device_id,
        "shell",
        f"echo '{COMMAND_LINE_BODY}' > {COMMAND_LINE_PATH}",
    )
    if code != 0:
        logger.warning(
            "[chrome] writing %s failed on %s: %s", COMMAND_LINE_PATH, device_id, output
        )
        return output or "failed to write chrome command line"

    # World-readable so Chrome (running as its own UID) can read it.
    await _run_adb(
        adb_path,
        "-s",
        device_id,
        "shell",
        "chmod",
        "644",
        COMMAND_LINE_PATH,
    )

    _prepared_devices.add(device_id)
    logger.info(
        "[chrome] prepared %s on %s (skip first-run)", COMMAND_LINE_PATH, device_id
    )
    return None
