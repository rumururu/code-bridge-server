"""Filesystem path guard.

Decides whether a path is inside the active workspace, escapes into the
user's home, or touches a system-critical location. Like
:mod:`policy.command_classifier`, this module returns an *effect* hint that
the policy engine escalates over the static operation decision — it never
downgrades.

Rules of thumb:

- ``WORKSPACE_INTERNAL``  → ``allow`` (caller-provided base decision wins).
- ``WORKSPACE_EXTERNAL``  → at least ``confirm_each``.
- ``HOME_SENSITIVE``      → ``desktop_only`` (SSH keys, AWS creds, browser
  profiles, password stores, etc.).
- ``SYSTEM_CRITICAL``     → ``forbidden`` (kernel, system bootloader, init
  scripts, package manager root).

Symlinks are resolved with ``os.path.realpath`` before classification so
``${workspace}/link-to-/etc/passwd`` reads as ``SYSTEM_CRITICAL``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .command_classifier import (
    EFFECT_ALLOW,
    EFFECT_CONFIRM_EACH,
    EFFECT_DESKTOP_ONLY,
    EFFECT_FORBIDDEN,
    escalate,
)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

CATEGORY_WORKSPACE_INTERNAL = "workspace_internal"
CATEGORY_WORKSPACE_EXTERNAL = "workspace_external"
CATEGORY_HOME_SENSITIVE = "home_sensitive"
CATEGORY_SYSTEM_CRITICAL = "system_critical"
CATEGORY_UNKNOWN = "unknown"


_CATEGORY_EFFECT = {
    CATEGORY_WORKSPACE_INTERNAL: EFFECT_ALLOW,
    CATEGORY_WORKSPACE_EXTERNAL: EFFECT_CONFIRM_EACH,
    CATEGORY_HOME_SENSITIVE: EFFECT_DESKTOP_ONLY,
    CATEGORY_SYSTEM_CRITICAL: EFFECT_FORBIDDEN,
    CATEGORY_UNKNOWN: EFFECT_CONFIRM_EACH,
}


# ---------------------------------------------------------------------------
# Sensitive locations
# ---------------------------------------------------------------------------

# Paths relative to the user's home directory that hold long-lived credentials,
# tokens, OAuth refresh state, or browser sessions. Any access — read or write —
# requires desktop approval.
_HOME_SENSITIVE_RELATIVE = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".gpg",
    ".kube",
    ".docker/config.json",
    ".netrc",
    ".pypirc",
    ".npmrc",
    ".pgpass",
    ".bash_history",
    ".zsh_history",
    ".fish_history",
    ".python_history",
    ".lesshst",
    ".viminfo",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox/Profiles",
    "Library/Application Support/com.apple.TCC",
    "Library/Application Support/Slack",
    "Library/Keychains",
    "Library/Cookies",
    "Library/Mail",
    "Library/Messages",
    ".config/gh/hosts.yml",
    ".config/git/credentials",
    ".cache/keyring",
    "AppData/Roaming/Microsoft/Credentials",
    "AppData/Roaming/Microsoft/Vault",
    "AppData/Local/Google/Chrome/User Data",
    "AppData/Local/Microsoft/Edge/User Data",
    "AppData/Roaming/Mozilla/Firefox/Profiles",
)

# Absolute prefixes that are system-owned. Mutations here can brick the host.
_SYSTEM_CRITICAL_PREFIXES = (
    "/System",
    "/usr/bin",
    "/usr/sbin",
    "/usr/libexec",
    "/sbin",
    "/bin",
    "/private/var/db",
    "/private/etc",
    "/etc",
    "/boot",
    "/Library/LaunchDaemons",
    "/Library/Security",
    "C:\\Windows\\System32",
    "C:\\Windows\\SysWOW64",
    "C:\\Program Files\\Windows Defender",
)

# Filename markers that look like secrets/credentials irrespective of directory.
# These get HOME_SENSITIVE treatment.
_SECRET_FILENAMES = (
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "known_hosts",
    "credentials",
    "config.json.enc",
    "keystore.jks",
    "google-services.json",
    "GoogleService-Info.plist",
    "firebase_config.json",
    # Firebase Admin SDK key: push-send authority, and per this repo's own
    # do-not-commit list, not reissuable. It was slipping through — the name
    # matches neither `firebase_config.json` above nor any fragment below
    # ("service_account" is not "secret") — which mattered little while every
    # tool read needed confirmation and matters a great deal now that an
    # in-workspace read is allowed by default.
    "firebase_service_account.json",
    "server_info.json",
    "AuthKey",  # prefix match
)

# Filename glob fragments meaning "definitely a secret"; case-insensitive.
_SECRET_NAME_FRAGMENTS = (
    ".pem",
    ".p12",
    ".pfx",
    ".key",
    ".keystore",
    ".jks",
    "secret",
    "private_key",
    # Dotenv files are the one credential store that normally lives *inside*
    # the workspace, so unlike an SSH key they are not caught by the
    # home-sensitive rules above and would otherwise classify as
    # `workspace_internal` (allow). Matching the fragment covers `.env`,
    # `.env.local`, `.env.production` and `.envrc` alike.
    ".env",
)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class PathClassification:
    """Outcome of classifying one filesystem path."""

    effect: str
    category: str
    resolved_path: str
    workspace_root: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "effect": self.effect,
            "category": self.category,
            "resolved_path": self.resolved_path,
            "workspace_root": self.workspace_root,
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _safe_realpath(path: str) -> str:
    """Resolve ``path`` to an absolute, symlink-resolved string. Never raises.

    Returns the cleaned absolute path even for paths that do not currently
    exist on disk; that's important because policy must decide *before* the
    filesystem operation actually executes.
    """
    try:
        absolute = os.path.abspath(os.path.expanduser(path))
    except (TypeError, ValueError):
        return path or ""
    try:
        return os.path.realpath(absolute)
    except OSError:
        return absolute


def _path_starts_with(path: str, prefix: str) -> bool:
    """``os.path.commonpath``-style "is ``path`` inside ``prefix``?" check.

    Works regardless of trailing slashes and on Windows-style backslashes by
    delegating to :class:`pathlib.PurePath`.
    """
    if not path or not prefix:
        return False
    try:
        norm_path = Path(path)
        norm_prefix = Path(prefix)
        return norm_prefix == norm_path or norm_prefix in norm_path.parents
    except (TypeError, ValueError):
        return False


def _home_dir() -> str:
    return os.path.expanduser("~")


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _matches_sensitive_filename(path: str) -> tuple[bool, str | None]:
    name = os.path.basename(path)
    if not name:
        return False, None
    lower = name.lower()
    for marker in _SECRET_FILENAMES:
        if name.startswith(marker):
            return True, marker
    for fragment in _SECRET_NAME_FRAGMENTS:
        if fragment in lower:
            return True, fragment
    return False, None


def _matches_home_sensitive_dir(path: str) -> tuple[bool, str | None]:
    home = _home_dir()
    if not _path_starts_with(path, home):
        return False, None
    for relative in _HOME_SENSITIVE_RELATIVE:
        candidate = os.path.join(home, relative)
        if _path_starts_with(path, candidate):
            return True, relative
    return False, None


def _matches_system_critical(path: str) -> tuple[bool, str | None]:
    for prefix in _SYSTEM_CRITICAL_PREFIXES:
        if _path_starts_with(path, prefix):
            return True, prefix
    return False, None


def classify_path(
    path: str | None,
    *,
    workspace_root: str | None = None,
) -> PathClassification:
    """Classify a single filesystem path against an optional workspace root.

    ``workspace_root`` should be the **registered** workspace root from the
    policy details, not an arbitrary CWD. When omitted, the classifier still
    flags system-critical and home-sensitive locations but treats other paths
    as ``workspace_external`` (``confirm_each``).
    """
    raw = (path or "").strip()
    if not raw:
        return PathClassification(
            effect=EFFECT_FORBIDDEN,
            category=CATEGORY_UNKNOWN,
            resolved_path="",
            workspace_root=workspace_root,
            reasons=["empty path"],
        )

    resolved = _safe_realpath(raw)
    resolved_root = _safe_realpath(workspace_root) if workspace_root else None

    reasons: list[str] = []
    candidate_effects: list[str] = []
    category = CATEGORY_UNKNOWN

    # 1) System-critical wins over everything else.
    is_system, marker = _matches_system_critical(resolved)
    if is_system:
        category = CATEGORY_SYSTEM_CRITICAL
        candidate_effects.append(EFFECT_FORBIDDEN)
        reasons.append(f"path is inside system-critical prefix {marker}")

    # 2) Home-sensitive directory or credential filename.
    is_home_dir, dir_marker = _matches_home_sensitive_dir(resolved)
    if is_home_dir:
        if category == CATEGORY_UNKNOWN:
            category = CATEGORY_HOME_SENSITIVE
        candidate_effects.append(EFFECT_DESKTOP_ONLY)
        reasons.append(f"path is inside sensitive home directory ~/{dir_marker}")

    is_secret_name, name_marker = _matches_sensitive_filename(resolved)
    if is_secret_name:
        if category == CATEGORY_UNKNOWN:
            category = CATEGORY_HOME_SENSITIVE
        candidate_effects.append(EFFECT_DESKTOP_ONLY)
        reasons.append(
            f"filename matches credential pattern '{name_marker}'"
        )

    # 3) Workspace boundary check.
    if resolved_root:
        if _path_starts_with(resolved, resolved_root):
            if category == CATEGORY_UNKNOWN:
                category = CATEGORY_WORKSPACE_INTERNAL
                # Don't downgrade existing escalations; just provide a hint.
            reasons.append(f"path is inside workspace root {resolved_root}")
            # If the raw path used '..' to escape and then realpath brought it
            # back inside, that's worth flagging as a reason — but not
            # restricting beyond what the absolute resolved path already
            # implies.
            if ".." in Path(raw).parts:
                reasons.append("path used '..' traversal segments")
        else:
            if category == CATEGORY_UNKNOWN:
                category = CATEGORY_WORKSPACE_EXTERNAL
            candidate_effects.append(EFFECT_CONFIRM_EACH)
            reasons.append(
                f"path is outside workspace root {resolved_root}"
            )
    else:
        if category == CATEGORY_UNKNOWN:
            category = CATEGORY_WORKSPACE_EXTERNAL
        candidate_effects.append(EFFECT_CONFIRM_EACH)
        reasons.append("no workspace root supplied; treating as external")

    base_effect = _CATEGORY_EFFECT[category]
    final = escalate(base_effect, *candidate_effects) if candidate_effects else base_effect

    return PathClassification(
        effect=final,
        category=category,
        resolved_path=resolved,
        workspace_root=resolved_root,
        reasons=reasons,
    )


def classify_paths(
    paths: list[str] | tuple[str, ...],
    *,
    workspace_root: str | None = None,
) -> PathClassification:
    """Classify a list of paths; returns the most restrictive result."""
    if not paths:
        return PathClassification(
            effect=EFFECT_CONFIRM_EACH,
            category=CATEGORY_UNKNOWN,
            resolved_path="",
            workspace_root=workspace_root,
            reasons=["no paths provided"],
        )
    classifications = [classify_path(p, workspace_root=workspace_root) for p in paths]
    classifications.sort(key=lambda r: -_RANK_BY_EFFECT[r.effect])
    return classifications[0]


_RANK_BY_EFFECT = {
    EFFECT_ALLOW: 0,
    "confirm_once": 1,
    EFFECT_CONFIRM_EACH: 2,
    EFFECT_DESKTOP_ONLY: 3,
    EFFECT_FORBIDDEN: 4,
}
