#!/usr/bin/env python3
"""Read-only integrity report for a Code Bridge local install.

Compares the repository sources against the deployed install directory
(default ``~/.code-bridge``) and prints three sections:

  1. MISSING FROM INSTALL   — source file has no counterpart deployed
  2. HASH MISMATCH          — deployed file differs from the source
  3. PRESENT ONLY IN INSTALL — orphan candidates (reported, never removed)
  4. CANVAS BUNDLE FRESHNESS — is the committed bundle built from the
     kernel source we have? (see canvas_bundle_freshness.py)

This script NEVER writes, moves or deletes anything. It only reads and
hashes files. Sections 1, 2 and 4 set a non-zero exit status; section 3 is
informational because the install directory legitimately accumulates
leftovers from the old flat layout, and deleting them is a human decision.

Section 4 exists because sections 1 and 2 cannot see the problem it catches:
they compare repository bytes to install bytes, and a canvas bundle that is
weeks behind its source is byte-identical on both sides. The deploy passes
and the screen is old. That check needs the flow kernel checkout, and says
so explicitly when it is absent rather than passing.

This module is also the single source of truth for the transfer rules used
by ``sync-local-install.sh``: that script obtains its rsync filters from
``--print-rsync-filters`` so the two tools can never drift apart.

Usage:
    verify_install.py [--install-dir DIR] [--repo-root DIR]
                      [--with-scrcpy] [--strict] [--quiet]
    verify_install.py --print-rsync-filters [--with-scrcpy]
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
from pathlib import Path

# Sibling script, not a package: `install/` is deliberately never deployed
# (it is in PROTECTED_TARGETS below), so there is nothing to import normally.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import canvas_bundle_freshness  # noqa: E402

# ---------------------------------------------------------------------------
# Transfer rules
# ---------------------------------------------------------------------------
# Pattern syntax is the rsync subset we actually rely on:
#   "name/"        directory, matched at any depth
#   "name"         basename, matched at any depth
#   "*.log"        glob on the basename, matched at any depth
#   "/a/b"         path anchored to the transfer root
#   "/a/b/"        directory path anchored to the transfer root

# Never copied OUT of the repository: build artifacts, virtualenvs, local
# databases, scratch state, and the per-machine files the installer or the
# running server owns (server_info.json, firebase_config.json, config.yaml,
# start.sh). Copying those from a dev checkout would overwrite the deployed
# machine's identity and credentials.
SOURCE_EXCLUDES: tuple[str, ...] = (
    "venv/",
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".DS_Store",
    "*.log",
    "code_bridge.db*",
    "/core/code_bridge.db",
    "/core/browser_sessions/",
    # The browser profile the agent signs in with. It appeared when browser
    # steps gained a persistent `user_data_dir`, and without this line a
    # deployment reported ~190 of its files as orphans — a Chrome profile is
    # thousands of small runtime files, and it is where the user's logins live.
    # Nothing here is ever authored in the repo, so it can only be runtime state.
    "/core/browser_profile/",
    "/server_info.json",
    "/firebase_config.json",
    "/config.yaml",
    "/start.sh",
    ".initialized",
    "/workspaces/",
    "/resources/",
    # Addition beyond the TRACK_D list, deliberately: server/.server.pid exists
    # in the repo checkout and holds a *dev machine's* pid. Copying it over the
    # deployed .server.pid would point the launcher's liveness check at a pid
    # that means nothing on that host. It is runtime state, never source.
    "/.server.pid",
    # Frontend build leftovers under server/webui/. Neither exists there today,
    # and that is exactly why they are listed: server/webui/canvas/ is a build
    # artifact copied in from another repository by hand (T-I1-07, see
    # docs/guide/CANVAS_BUNDLE.md), so the accident these guard against is one
    # careless `cp -r` away. A copied node_modules/ is tens of thousands of
    # files rsynced onto every deployment; a copied .map is the bulk of the
    # payload (in the kernel's sibling build the .map files are most of a 25MB
    # dist/) and it hands a readable copy of the source to anyone who opens the
    # page.
    #
    # Anchored to /webui/ rather than written as bare "node_modules/" and
    # "*.map", and that is not fussiness. server/scrcpy/ is a bundled node app
    # that ships WITH its dependencies, opt-in via --with-scrcpy, and this list
    # has no re-include mechanism: a bare "node_modules/" would strip every
    # dependency out of that opt-in transfer and deploy an unrunnable app with
    # no error anywhere. Excluding what should never exist is free; excluding
    # what something needs is a silent breakage of exactly the kind this ticket
    # exists to prevent.
    "/webui/*node_modules/*",
    "/webui/*.map",
)

# scrcpy/ is a bundled node app: large, rarely changed, opt-in via --with-scrcpy.
SCRCPY_EXCLUDE = "/scrcpy/"

# Never touched IN the install directory. These hold the deployed machine's
# git checkout, API keys, paired user accounts, the Firebase service account,
# the Apple signing key, logs and user-generated content. Losing any of them
# is unrecoverable, so they are excluded from the transfer *and* emitted as
# rsync protect ("P") rules — protect rules survive even if somebody later
# adds a --delete flag by hand.
PROTECTED_TARGETS: tuple[str, ...] = (
    "/.git/",
    "/.env",
    "/.github/",
    "/.gitignore",
    "/LICENSE",
    "/api_keys.json",
    "/paired_accounts.json",
    "/firebase_service_account.json",
    "/AuthKey*.p8",
    "/logs/",
    "/generated_scripts/",
    "/global_chat/",
    # The browser profile the agent signs in with. Protected as well as
    # excluded: excluding keeps it out of the transfer, but this list is the
    # answer to "what must a delete never reach", and a persistent Chrome
    # profile holds the logins that make an unattended browser agent work at
    # all. Losing it silently signs the agent out of every site.
    "/core/browser_profile/",
    "/start-menubar.sh",
    "/install/",
    "/scripts/",
    "/docs/",
)

# Runtime artifacts the server writes into the install directory. They are not
# orphan modules, so listing them in section 3 would be noise. Excluded from
# the orphan report only — they are not part of the transfer rules.
RUNTIME_ARTIFACTS: tuple[str, ...] = ("*.pyc",)

# The two source trees and where each lands inside the install directory.
SOURCE_MAP: tuple[tuple[str, str], ...] = (
    ("server", ""),
    ("desktop_server_app", "desktop_server_app"),
)


def _match(pattern: str, rel_path: str, is_dir: bool) -> bool:
    """Return True if ``rel_path`` matches an rsync-style ``pattern``."""
    dir_only = pattern.endswith("/")
    pat = pattern.rstrip("/")
    anchored = pat.startswith("/")
    if anchored:
        pat = pat[1:]

    if dir_only and not is_dir:
        # A directory pattern also suppresses everything beneath it; the
        # caller prunes directories, so here we only need the prefix test.
        segments = rel_path.split("/")
        if anchored:
            depth = pat.count("/") + 1
            return fnmatch.fnmatch("/".join(segments[:depth]), pat)
        return any(fnmatch.fnmatch(seg, pat) for seg in segments[:-1])

    if anchored or "/" in pat:
        return fnmatch.fnmatch(rel_path, pat)
    return fnmatch.fnmatch(os.path.basename(rel_path), pat)


def is_excluded(rel_path: str, patterns: tuple[str, ...], is_dir: bool = False) -> bool:
    return any(_match(p, rel_path, is_dir) for p in patterns)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_tree(root: Path, patterns: tuple[str, ...]) -> dict[str, Path]:
    """Collect ``{relative_path: absolute_path}`` for every non-excluded file."""
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir
        # Prune excluded directories so we never descend into venv/ or .git/.
        dirnames[:] = [
            d
            for d in dirnames
            if not is_excluded(f"{rel_dir}/{d}".lstrip("/"), patterns, is_dir=True)
        ]
        for name in filenames:
            rel = f"{rel_dir}/{name}".lstrip("/")
            if is_excluded(rel, patterns):
                continue
            if os.path.islink(os.path.join(dirpath, name)):
                continue
            found[rel] = Path(dirpath) / name
    return found


def build_patterns(with_scrcpy: bool) -> tuple[str, ...]:
    patterns = SOURCE_EXCLUDES + PROTECTED_TARGETS
    if not with_scrcpy:
        patterns = patterns + (SCRCPY_EXCLUDE,)
    return patterns


def print_rsync_filters(with_scrcpy: bool) -> None:
    """Emit rsync --filter rules, one per line, for sync-local-install.sh.

    Protect rules come first so they win the first-match-wins ordering.
    """
    for pattern in PROTECTED_TARGETS:
        print(f"P {pattern}")
        print(f"- {pattern}")
    for pattern in SOURCE_EXCLUDES:
        print(f"- {pattern}")
    if not with_scrcpy:
        print(f"- {SCRCPY_EXCLUDE}")


def main(argv: list[str] | None = None) -> int:
    default_repo = Path(__file__).resolve().parent.parent
    default_install = Path(
        os.environ.get("CODE_BRIDGE_INSTALL_DIR", Path.home() / ".code-bridge")
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=default_repo)
    ap.add_argument("--install-dir", type=Path, default=default_install)
    ap.add_argument("--with-scrcpy", action="store_true", help="include the bundled scrcpy/ node app")
    ap.add_argument("--strict", action="store_true", help="also fail when orphan candidates exist")
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    ap.add_argument("--print-rsync-filters", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.print_rsync_filters:
        print_rsync_filters(args.with_scrcpy)
        return 0

    repo_root: Path = args.repo_root.expanduser().resolve()
    install_dir: Path = args.install_dir.expanduser()
    if not install_dir.is_dir():
        print(f"error: install directory not found: {install_dir}", file=sys.stderr)
        return 2

    patterns = build_patterns(args.with_scrcpy)

    # Map every source file to the install path it should occupy.
    expected: dict[str, Path] = {}
    for src_name, dest_prefix in SOURCE_MAP:
        src_root = repo_root / src_name
        if not src_root.is_dir():
            print(f"error: source tree not found: {src_root}", file=sys.stderr)
            return 2
        for rel, abs_path in walk_tree(src_root, patterns).items():
            dest_rel = f"{dest_prefix}/{rel}".lstrip("/") if dest_prefix else rel
            expected[dest_rel] = abs_path

    installed = walk_tree(install_dir, patterns + RUNTIME_ARTIFACTS)

    missing: list[str] = []
    mismatched: list[str] = []
    for dest_rel, src_path in sorted(expected.items()):
        target = install_dir / dest_rel
        if not target.is_file():
            missing.append(dest_rel)
        elif sha256(src_path) != sha256(target):
            mismatched.append(dest_rel)

    orphans = sorted(set(installed) - set(expected))

    if not args.quiet:
        print(f"repo    : {repo_root}")
        print(f"install : {install_dir}")
        print(f"compared: {len(expected)} source files\n")

        print(f"[1] MISSING FROM INSTALL ({len(missing)})")
        for rel in missing:
            print(f"    - {rel}")
        if not missing:
            print("    (none)")

        print(f"\n[2] HASH MISMATCH ({len(mismatched)})")
        for rel in mismatched:
            print(f"    ~ {rel}")
        if not mismatched:
            print("    (none)")

        print(f"\n[3] PRESENT ONLY IN INSTALL — orphan candidates ({len(orphans)})")
        print("    reported only; this tool never removes anything")
        for rel in orphans:
            print(f"    ? {rel}")
        if not orphans:
            print("    (none)")
        print()

    canvas = canvas_bundle_freshness.check(repo_root)
    if not args.quiet:
        print("[4] CANVAS BUNDLE FRESHNESS")
        for line in canvas.report().splitlines():
            print(f"    {line}")
        print()

    drift = len(missing) + len(mismatched)
    print(
        f"summary: {len(missing)} missing, {len(mismatched)} mismatched, "
        f"{len(orphans)} orphan candidate(s), canvas bundle {canvas.status}"
    )
    if drift or canvas.failed or (args.strict and orphans):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
