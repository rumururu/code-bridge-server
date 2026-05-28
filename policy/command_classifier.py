"""Pattern-based shell-command classifier.

Returns a policy *effect* hint plus the list of matched rules. The result is
consumed by :func:`policy.policy_engine.decide_policy`, which combines it
with the static operation-name decision and **escalates** (never downgrades)
the final effect.

Effect ordering (most permissive to most restrictive):

    allow < confirm_once < confirm_each < desktop_only < forbidden

A classifier finding of ``desktop_only`` cannot make a ``forbidden`` operation
permissive, but it *can* make a ``confirm_each`` shell call go to
``desktop_only``. This module never produces ``allow`` for an unknown
command — the floor is ``confirm_each`` because shell execution is itself
side-effectful.

Patterns favor false-positives over false-negatives. If a pattern fires by
accident the user sees an approval prompt; if a real destructive command
slips past, we have a much worse problem.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

EFFECT_ALLOW = "allow"
EFFECT_CONFIRM_ONCE = "confirm_once"
EFFECT_CONFIRM_EACH = "confirm_each"
EFFECT_DESKTOP_ONLY = "desktop_only"
EFFECT_FORBIDDEN = "forbidden"

_EFFECT_RANK = {
    EFFECT_ALLOW: 0,
    EFFECT_CONFIRM_ONCE: 1,
    EFFECT_CONFIRM_EACH: 2,
    EFFECT_DESKTOP_ONLY: 3,
    EFFECT_FORBIDDEN: 4,
}


def escalate(*effects: str) -> str:
    """Return the most restrictive of the given effects."""
    if not effects:
        return EFFECT_ALLOW
    return max(effects, key=lambda effect: _EFFECT_RANK.get(effect, 0))


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pattern:
    name: str
    effect: str
    description: str
    pattern: re.Pattern[str]


def _compile(rules: Iterable[tuple[str, str, str, str]]) -> list[_Pattern]:
    return [
        _Pattern(name=name, effect=effect, description=description, pattern=re.compile(regex, re.IGNORECASE))
        for name, effect, description, regex in rules
    ]


# Patterns are checked top-to-bottom; the first match wins per category. Multiple
# matches across categories accumulate and the final effect escalates over all.

_FORBIDDEN: list[_Pattern] = _compile(
    [
        (
            "rm_rf_root",
            EFFECT_FORBIDDEN,
            "Recursive delete targeting filesystem root",
            r"\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*|--recursive\s+--force)\s+/\s*(?:\*|$|\s|;)",
        ),
        (
            "rm_no_preserve_root",
            EFFECT_FORBIDDEN,
            "rm with --no-preserve-root",
            r"\brm\b[^|;&]*--no-preserve-root",
        ),
        (
            "disk_write_dd",
            EFFECT_FORBIDDEN,
            "dd writing to a block device",
            r"\bdd\b[^|;&]*\bof=/dev/(?:disk|sd|nvme|hd|rdisk)\w*",
        ),
        (
            "mkfs",
            EFFECT_FORBIDDEN,
            "Format a filesystem",
            r"\bmkfs(?:\.[a-z0-9]+)?\b",
        ),
        (
            "fork_bomb",
            EFFECT_FORBIDDEN,
            "Classic shell fork bomb",
            r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        ),
        (
            "curl_pipe_shell",
            EFFECT_FORBIDDEN,
            "Piping a remote download directly into a shell",
            r"\b(?:curl|wget|fetch)\b[^|;&]*\|\s*(?:sudo\s+)?(?:bash|sh|zsh|ksh|fish|python|python3|node|perl|ruby)\b",
        ),
        (
            "chmod_777_root",
            EFFECT_FORBIDDEN,
            "World-writable mode on filesystem root",
            r"\bchmod\b[^|;&]*\s(?:-[a-zA-Z]*R[a-zA-Z]*|--recursive)\s[^|;&]*(?:0?777|a\+rwx)[^|;&]*\s/(?:\s|$|;)",
        ),
        (
            "shutdown_or_reboot",
            EFFECT_FORBIDDEN,
            "Halt the host",
            r"\b(?:shutdown\b[^|;&]*-h|halt|poweroff|init\s+0|init\s+6|reboot\s+-f)\b",
        ),
        (
            "kernel_panic",
            EFFECT_FORBIDDEN,
            "Triggering kernel panic via sysrq",
            r"sysrq-trigger",
        ),
    ]
)

_DESKTOP_ONLY: list[_Pattern] = _compile(
    [
        (
            "sudo",
            EFFECT_DESKTOP_ONLY,
            "Elevated privileges via sudo",
            r"(?:^|[\s|;&])sudo\b",
        ),
        (
            "su",
            EFFECT_DESKTOP_ONLY,
            "Switch user",
            r"(?:^|[\s|;&])su\b(?:\s+-|\s+\w|\s*$)",
        ),
        (
            "git_force_push",
            EFFECT_DESKTOP_ONLY,
            "git push --force on remote",
            r"\bgit\s+push\b[^|;&]*(?:--force\b|-f\b)",
        ),
        (
            "git_reset_hard",
            EFFECT_DESKTOP_ONLY,
            "git reset --hard discards local changes",
            r"\bgit\s+reset\b[^|;&]*--hard",
        ),
        (
            "system_paths_rm",
            EFFECT_DESKTOP_ONLY,
            "Recursive delete inside a system path",
            r"\brm\s+(?:-[a-zA-Z]*r\w*|--recursive)\b[^|;&]*\s+(?:/etc|/usr|/var|/System|/Library|/private/etc|/private/var)(?:/|\s|$)",
        ),
        (
            "credential_dump",
            EFFECT_DESKTOP_ONLY,
            "Reading credential or secret stores",
            r"(?:/\.ssh/|/\.aws/credentials|/\.netrc|/etc/shadow|security\s+find-(?:internet|generic)-password)",
        ),
        (
            "firewall_change",
            EFFECT_DESKTOP_ONLY,
            "Firewall modification",
            r"\b(?:iptables|nft|pfctl|netsh\s+advfirewall|ufw\s+(?:enable|disable|allow|deny))\b",
        ),
        (
            "launchctl_load",
            EFFECT_DESKTOP_ONLY,
            "macOS LaunchAgent install",
            r"\blaunchctl\s+(?:load|bootstrap)\b",
        ),
        (
            "powershell_executionpolicy",
            EFFECT_DESKTOP_ONLY,
            "Lowering PowerShell execution policy",
            r"Set-ExecutionPolicy\s+(?:Bypass|Unrestricted)",
        ),
    ]
)

_CONFIRM_EACH: list[_Pattern] = _compile(
    [
        (
            "package_install_global",
            EFFECT_CONFIRM_EACH,
            "Global package install",
            r"\b(?:npm\s+install\s+-g|yarn\s+global\s+add|pnpm\s+add\s+-g|pip\s+install(?:\s+--user)?|pip3\s+install|brew\s+install|apt-get\s+install|apt\s+install|dnf\s+install|yum\s+install)\b",
        ),
        (
            "git_push",
            EFFECT_CONFIRM_EACH,
            "git push to remote",
            r"\bgit\s+push\b",
        ),
        (
            "git_commit",
            EFFECT_CONFIRM_EACH,
            "git commit",
            r"\bgit\s+commit\b",
        ),
        (
            "rm_recursive",
            EFFECT_CONFIRM_EACH,
            "Recursive delete",
            r"\brm\s+(?:-[a-zA-Z]*r\w*|--recursive)\b",
        ),
        (
            "network_send",
            EFFECT_CONFIRM_EACH,
            "External network send",
            r"\b(?:curl|wget|http|httpie)\b[^|;&]*-X\s*(?:POST|PUT|PATCH|DELETE)\b",
        ),
        (
            "ssh_or_scp",
            EFFECT_CONFIRM_EACH,
            "SSH/SCP to a remote host",
            r"\b(?:ssh|scp|sftp|rsync)\b[^|;&]*\s\S+@\S",
        ),
        (
            "docker_run_priv",
            EFFECT_CONFIRM_EACH,
            "Privileged docker run",
            r"\bdocker\s+run\b[^|;&]*(?:--privileged|--cap-add|--device|-v\s+/:/)",
        ),
        (
            "process_kill_all",
            EFFECT_CONFIRM_EACH,
            "Broad process kill",
            r"\b(?:pkill|killall)\b|\bkill\s+-9\s+1\b",
        ),
    ]
)


@dataclass
class ClassificationResult:
    """Outcome of classifying a single shell command."""

    effect: str
    matches: list[dict[str, str]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return bool(self.matches)

    def to_dict(self) -> dict[str, object]:
        return {
            "effect": self.effect,
            "matched": self.matched,
            "matches": list(self.matches),
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_command(command: str | None) -> ClassificationResult:
    """Classify a single command string.

    Pass a raw command line, not a pre-split argv. The classifier is intentionally
    string-based — it must catch dangerous patterns even when arguments are passed
    through a shell.
    """
    if command is None:
        return ClassificationResult(effect=EFFECT_CONFIRM_EACH, reasons=["empty command"])

    text = command.strip()
    if not text:
        return ClassificationResult(effect=EFFECT_CONFIRM_EACH, reasons=["empty command"])

    matches: list[dict[str, str]] = []
    effects: list[str] = []
    reasons: list[str] = []

    for bucket in (_FORBIDDEN, _DESKTOP_ONLY, _CONFIRM_EACH):
        for pat in bucket:
            if pat.pattern.search(text):
                matches.append(
                    {
                        "name": pat.name,
                        "effect": pat.effect,
                        "description": pat.description,
                    }
                )
                effects.append(pat.effect)
                reasons.append(pat.description)

    # If nothing matched, the command is still side-effectful (it's a shell
    # call), so floor at confirm_each. Static policy on ``process.terminal``
    # already declares this, but classify_command must never *downgrade* its
    # own opinion below that floor — otherwise wrappers that call it for
    # advisory purposes would be tempted to grant ``allow`` incorrectly.
    if not effects:
        return ClassificationResult(
            effect=EFFECT_CONFIRM_EACH,
            reasons=["no dangerous pattern matched; default confirm_each for shell execution"],
        )

    return ClassificationResult(
        effect=escalate(*effects),
        matches=matches,
        reasons=reasons,
    )


def classify_argv(argv: list[str] | tuple[str, ...]) -> ClassificationResult:
    """Convenience wrapper that joins argv using ``shlex.join`` before classifying."""
    if not argv:
        return ClassificationResult(effect=EFFECT_CONFIRM_EACH, reasons=["empty command"])
    return classify_command(shlex.join(argv))


def list_patterns() -> list[dict[str, str]]:
    """Expose the rule list for the Cockpit policy view / diagnostics."""
    out: list[dict[str, str]] = []
    for bucket in (_FORBIDDEN, _DESKTOP_ONLY, _CONFIRM_EACH):
        for pat in bucket:
            out.append(
                {
                    "name": pat.name,
                    "effect": pat.effect,
                    "description": pat.description,
                    "pattern": pat.pattern.pattern,
                }
            )
    return out
