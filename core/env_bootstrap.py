"""PATH bootstrap for server processes started outside a login shell.

launchd / a macOS login item starts the server with ``PATH=/usr/bin:/bin:...``
— none of the directories a Homebrew, npm, or Android SDK install actually
lives in. Every ``shutil.which`` consumer in this codebase silently reports
"not installed" under that PATH even though the tool is on disk: CLI
detection in ``llm/llm_settings.py``, cloudflared in
``remote/tunnel_service.py``, the claude/gemini/codex/agy sessions, mmdc in
``core/mermaid_service.py``, and the adb fallback in
``agent/app_action_executor.py``.

:func:`bootstrap_path` repairs this before anything else runs: it prepends a
static list of well-known install directories (each only if it actually
exists) and merges in whatever a real login shell resolves for ``$PATH``. The
shell probe is a *secondary*, best-effort source, never a replacement for the
static list — a login shell can fail to spawn under launchd's early-boot
environment, and even when it succeeds, ``-l`` only sources
``.zprofile``/``.bash_profile`` (not ``.zshrc``/``.bashrc``), so it cannot be
trusted alone to carry everything a normal interactive shell would have.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Checked for existence before being added — never trusted blindly.
_STATIC_PATH_CANDIDATES: tuple[str, ...] = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "~/.local/bin",
    "~/bin",
    "~/Library/Android/sdk/platform-tools",
    "~/Library/Android/sdk/emulator",
    "~/.npm-global/bin",
)

_SHELL_PROBE_TIMEOUT = 5


def _static_candidates() -> list[str]:
    """Well-known install dirs that actually exist on this machine."""
    found: list[str] = []
    for raw in _STATIC_PATH_CANDIDATES:
        expanded = os.path.expanduser(raw)
        if os.path.isdir(expanded):
            found.append(expanded)
    return found


def _login_shell_path() -> list[str]:
    """Ask a real login shell what PATH it resolves to.

    Best-effort only. rc files can print banners/warnings to stdout ahead of
    the PATH line, so only the last non-empty line of stdout is trusted as
    the PATH value. Any failure — missing ``$SHELL``, a non-zero exit, a
    timeout, or empty output — returns an empty list; this must never raise
    and must never fabricate a PATH out of a failed probe.
    """
    shell = os.environ.get("SHELL")
    if not shell:
        return []
    try:
        proc = subprocess.run(
            [shell, "-l", "-c", 'command printf "%s" "$PATH"'],
            capture_output=True,
            text=True,
            timeout=_SHELL_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if proc.returncode != 0:
        return []

    non_empty_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not non_empty_lines:
        return []

    last_line = non_empty_lines[-1]
    return [entry for entry in last_line.split(os.pathsep) if entry]


def bootstrap_path(env: dict[str, str] | None = None) -> None:
    """Prepend well-known tool directories to ``env``'s PATH, in place.

    ``env`` defaults to ``os.environ`` so most callers just fix the current
    process's PATH with no arguments. A caller building the environment for
    a *child* process (e.g. the desktop launcher spawning the server
    subprocess) passes its own env dict so the merge lands there instead of
    (or in addition to) ``os.environ``.

    Never raises: any failure degrades to "PATH left unchanged" so a broken
    shell probe, an unreadable filesystem, or anything else unexpected can
    never block server startup.
    """
    target = os.environ if env is None else env

    try:
        existing_raw = target.get("PATH", "")
        existing_entries = [entry for entry in existing_raw.split(os.pathsep) if entry]
        existing_set = set(existing_entries)

        candidates = [*_static_candidates(), *_login_shell_path()]

        added: list[str] = []
        seen = set(existing_set)
        for entry in candidates:
            if entry in seen:
                continue
            seen.add(entry)
            added.append(entry)

        if added:
            target["PATH"] = os.pathsep.join([*added, *existing_entries])

        logger.info(
            "PATH bootstrapped: added=%s existing_count=%d",
            added,
            len(existing_entries),
        )
    except Exception:  # noqa: BLE001 - PATH bootstrap must never block startup
        logger.exception("PATH bootstrap failed; continuing with existing PATH")
