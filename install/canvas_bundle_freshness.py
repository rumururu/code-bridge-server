#!/usr/bin/env python3
"""Is the committed canvas bundle built from the kernel source we have?

`server/webui/canvas/` is a **build artifact committed into this repository**
(agent-flow-core T-I1-07). It is produced in a different repository — the flow
kernel checkout, `frontend/packages/flow-canvas-standalone` — because the
machines Code Bridge installs onto have no node and cannot build at deploy time.

That arrangement has one failure mode, and it is silent. `verify_install.py`
compares repository bytes against install-directory bytes: a bundle that is six
weeks behind its source is byte-identical on both sides, so the deploy reports
success and the screen shows the old canvas. Nobody is told. The person who
edited the canvas source, forgot to rebuild, and committed is not told either.

This module is the missing question: *does the committed artifact correspond to
the source that produced it?* It can be asked because the bundle carries its own
provenance — the Vite build stamps `__CANVAS_BUILD__` with the **git tree id of
`frontend/`** and the bundle publishes it as `window.flowCanvas.version`
(kernel `frontend/packages/flow-canvas-standalone/vite.config.ts`, `sourceStamp()`).

A tree id, not a commit id, and that choice is what makes the check usable:
a commit id changes on every commit to the kernel repository, including commits
that never touch the canvas, so a commit-stamped artifact would have to be
rebuilt and re-committed *here* for changes that happened *there* and mean
nothing to this bundle. A tree id changes exactly when the canvas source
changes. Unchanged source therefore rebuilds to the same stamp, and the gate
stays quiet until something real moved.

Two stamps are rejected without consulting the kernel at all, because neither
can ever be correct in a committed artifact:

* `<tree>-dirty` — built from a working tree. No tree id identifies it, so
  nobody can ever reproduce it, including the person who built it.
* `unknown` — built with no git available. Same problem, less information.

Everything else needs the kernel checkout to compare against, and when the
checkout is absent this reports **skipped with a reason** rather than passing.
A gate that goes quiet on the machines that do not have the material is
indistinguishable from a gate that passed, which is how stale artifacts get
through in the first place.

Used by `server/tests/test_canvas_bundle_freshness.py` (fast local red light,
runs in the ordinary test suite) and by `install/verify_install.py` section [4]
(the deploy-time report). One implementation, so the two cannot disagree.

Runnable on its own for a quick answer:

    python3 install/canvas_bundle_freshness.py
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# The kernel checkout has no package index and no git remote, so the only
# reference that can exist is a path on this machine. Same env var and same
# default as `install/sync-local-install.sh:299` — one name for one checkout.
KERNEL_DIR_ENV = "CODE_BRIDGE_FLOW_CORE_DIR"
DEFAULT_KERNEL_DIR = Path.home() / "VSCodeProject" / "agent-flow-core"

# Where the artifact lives in this repository, and which subtree of the kernel
# repository produces it. `canvas_static.DEFAULT_CANVAS_BUNDLE_DIR` resolves to
# the same place at runtime; this module must not import server code, because
# `verify_install.py` runs under a bare `python3` with no venv.
BUNDLE_REL = Path("server") / "webui" / "canvas"
KERNEL_SOURCE_SUBTREE = "frontend"

# Status values. Only three, because only three decisions follow from them.
OK = "ok"
STALE = "stale"
SKIPPED = "skipped"

# What the stamp can look like in the emitted JavaScript. The minifier picks
# its own quote character (this bundle came out with backticks), so the quote
# is captured and back-referenced rather than assumed.
_STAMP_RE = re.compile(r"""(["'`])([0-9a-f]{40}(?:-dirty)?)\1""")
_UNKNOWN_RE = re.compile(r"""(["'`])unknown\1""")

REMEDY = (
    "Whoever changes the canvas source owns all four steps — the bundle is a\n"
    "committed artifact, so leaving any of them undone ships the old screen:\n"
    "  1. commit the change in the kernel repo first (an uncommitted tree\n"
    "     stamps the bundle '-dirty', which this gate rejects)\n"
    "  2. cd <kernel>/frontend && npm run build\n"
    "  3. rsync -a --delete <kernel>/frontend/packages/flow-canvas-standalone/\n"
    "     dist/ server/webui/canvas/   (--delete matters: Vite content-hashes\n"
    "     filenames, so a stale chunk left behind is never overwritten)\n"
    "  4. commit the rebuilt bundle here as well\n"
    "See docs/guide/CANVAS_BUNDLE.md"
)


@dataclass(frozen=True)
class Freshness:
    """One verdict, with enough detail to act on without re-investigating."""

    status: str
    reason: str
    detail: str
    bundle_stamp: str | None = None
    kernel_tree: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == STALE

    def report(self) -> str:
        """The multi-line text both consumers print. Remedy only when useful."""
        head = f"{self.status.upper()} ({self.reason}): {self.detail}"
        return f"{head}\n{REMEDY}" if self.failed else head


def kernel_dir() -> Path:
    """The flow kernel checkout: env override, then the conventional path."""
    override = os.environ.get(KERNEL_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_KERNEL_DIR


def bundle_dir(repo_root: Path) -> Path:
    return repo_root / BUNDLE_REL


def read_bundle_stamp(directory: Path) -> tuple[str | None, str]:
    """Extract the build stamp from the bundle's JavaScript.

    Returns ``(stamp, reason)``. ``stamp`` is ``None`` when no single stamp can
    be named, and ``reason`` says which of the several ways that happened —
    they have different fixes, so collapsing them into one message would send
    the reader looking in the wrong place.
    """
    if not directory.is_dir():
        return None, "bundle-missing"

    scripts = sorted(directory.rglob("*.js"))
    if not scripts:
        return None, "bundle-empty"

    hex_stamps: set[str] = set()
    saw_unknown = False
    for script in scripts:
        try:
            text = script.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hex_stamps.update(match.group(2) for match in _STAMP_RE.finditer(text))
        saw_unknown = saw_unknown or bool(_UNKNOWN_RE.search(text))

    if len(hex_stamps) == 1:
        return hex_stamps.pop(), "found"
    if len(hex_stamps) > 1:
        # Two 40-hex literals in one bundle. Most likely a half-replaced
        # directory: Vite content-hashes filenames, so an old chunk that was
        # not deleted sits next to the new one and both are served.
        return None, "ambiguous-stamp:" + ",".join(sorted(hex_stamps))
    if saw_unknown:
        # Only trusted when no hex stamp exists anywhere — `unknown` is a
        # common enough word in a JavaScript bundle to be a bad positive
        # signal, but a safe last resort.
        return "unknown", "found"
    return None, "no-stamp"


def _git(directory: Path, *args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def kernel_source_state(directory: Path) -> tuple[str | None, bool, str]:
    """``(tree_id, dirty, reason)`` for the kernel's canvas source subtree.

    ``dirty`` mirrors the build's own definition exactly (``git status
    --porcelain`` over the subtree, untracked files included), so this module
    and `sourceStamp()` can never disagree about what "dirty" means.
    """
    if not directory.is_dir():
        return None, False, "no-kernel-checkout"

    code, tree = _git(directory, "rev-parse", f"HEAD:{KERNEL_SOURCE_SUBTREE}")
    if code != 0 or not re.fullmatch(r"[0-9a-f]{40}", tree):
        return None, False, "git-unavailable"

    code, porcelain = _git(directory, "status", "--porcelain", "--", KERNEL_SOURCE_SUBTREE)
    if code != 0:
        return None, False, "git-unavailable"
    return tree, porcelain != "", "found"


def check(repo_root: Path, kernel: Path | None = None) -> Freshness:
    """Compare the committed bundle's stamp against the kernel source."""
    directory = bundle_dir(repo_root)
    stamp, stamp_reason = read_bundle_stamp(directory)

    if stamp is None:
        detail = {
            "bundle-missing": f"no canvas bundle at {directory}",
            "bundle-empty": f"{directory} holds no JavaScript — the copy is incomplete",
            "no-stamp": (
                f"the bundle at {directory} carries no build stamp; it predates"
                " __CANVAS_BUILD__ or was not produced by this build"
            ),
        }.get(stamp_reason)
        if detail is None and stamp_reason.startswith("ambiguous-stamp:"):
            found = stamp_reason.split(":", 1)[1]
            detail = (
                f"{directory} carries more than one build stamp ({found});"
                " old chunks were left in place instead of replaced"
            )
        return Freshness(STALE, stamp_reason.split(":", 1)[0], detail or stamp_reason)

    # Rejected without the kernel: no tree id identifies either of these, so no
    # comparison could ever make them right.
    if stamp == "unknown":
        return Freshness(
            STALE,
            "unknown-stamp",
            "the committed bundle was built with no git available, so the"
            " source that produced it cannot be named",
            bundle_stamp=stamp,
        )
    if stamp.endswith("-dirty"):
        return Freshness(
            STALE,
            "dirty-stamp",
            f"the committed bundle was built from an uncommitted working tree"
            f" ({stamp}); no commit in the kernel repository reproduces it",
            bundle_stamp=stamp,
        )

    kernel_path = (kernel or kernel_dir()).expanduser()
    tree, dirty, kernel_reason = kernel_source_state(kernel_path)
    if tree is None:
        detail = {
            "no-kernel-checkout": (
                f"no flow kernel checkout at {kernel_path}, so the bundle stamp"
                f" {stamp} has nothing to be compared against"
                f" (set {KERNEL_DIR_ENV} if it lives elsewhere)"
            ),
            "git-unavailable": (
                f"{kernel_path} is not a usable git checkout of the flow kernel;"
                f" the bundle stamp {stamp} cannot be compared"
            ),
        }[kernel_reason]
        return Freshness(SKIPPED, kernel_reason, detail, bundle_stamp=stamp)

    if dirty:
        return Freshness(
            STALE,
            "kernel-dirty",
            f"the kernel has uncommitted changes under {KERNEL_SOURCE_SUBTREE}/,"
            f" so its source is not the committed tree {tree} that the bundle"
            f" claims ({stamp}). Any canvas edit sitting there is not in the"
            " artifact this repository would deploy",
            bundle_stamp=stamp,
            kernel_tree=tree,
        )

    if stamp != tree:
        return Freshness(
            STALE,
            "tree-mismatch",
            f"the committed bundle was built from {KERNEL_SOURCE_SUBTREE}/ tree"
            f" {stamp}, but the kernel's current tree is {tree}. The canvas"
            " source moved and the bundle did not",
            bundle_stamp=stamp,
            kernel_tree=tree,
        )

    return Freshness(
        OK,
        "match",
        f"the committed bundle was built from {KERNEL_SOURCE_SUBTREE}/ tree {tree}",
        bundle_stamp=stamp,
        kernel_tree=tree,
    )


def main(argv: list[str] | None = None) -> int:
    default_repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Check that server/webui/canvas/ matches the kernel source it was built from.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=REMEDY,
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=None,
        help=f"flow kernel checkout (default: ${KERNEL_DIR_ENV} or {DEFAULT_KERNEL_DIR})",
    )
    args = parser.parse_args(argv)

    result = check(args.repo_root.expanduser().resolve(), args.kernel_dir)
    stream = sys.stderr if result.failed else sys.stdout
    print(result.report(), file=stream)
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
