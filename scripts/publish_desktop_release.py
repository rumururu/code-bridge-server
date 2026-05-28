#!/usr/bin/env python3
"""Upload desktop server app artifacts to a GitHub Release.

Requires GitHub CLI (`gh`) authenticated with permission to create releases.
This script does not build artifacts; run scripts/build_desktop_server_app.py
on each target OS first, then upload the produced .dmg/.msi/.zip files.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "dist" / "desktop_server_app"


def run(command: list[str]) -> None:
    print("$ " + shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish desktop app artifacts to GitHub Releases.")
    parser.add_argument("tag", help="Release tag, e.g. desktop-server-v1.0.0")
    parser.add_argument("--title", default=None, help="Release title. Defaults to the tag.")
    parser.add_argument("--notes", default=None, help="Release notes text.")
    parser.add_argument("--draft", action="store_true", help="Create/update the release as a draft.")
    parser.add_argument("--prerelease", action="store_true", help="Mark the release as prerelease.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing built artifacts.",
    )
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        help="Explicit artifact paths. Defaults to common desktop package extensions in artifact-dir.",
    )
    return parser.parse_args()


def discover_artifacts(directory: Path) -> list[Path]:
    patterns = ("*.dmg", "*.pkg", "*.msi", "*.zip", "*.AppImage", "*.deb", "*.rpm")
    artifacts: list[Path] = []
    for pattern in patterns:
        artifacts.extend(sorted(directory.glob(pattern)))
    return artifacts


def ensure_release(args: argparse.Namespace) -> None:
    view = subprocess.run(
        ["gh", "release", "view", args.tag],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if view.returncode == 0:
        return

    command = ["gh", "release", "create", args.tag, "--title", args.title or args.tag]
    if args.notes:
        command.extend(["--notes", args.notes])
    else:
        command.extend(["--notes", "Code Bridge desktop server app release."])
    if args.draft:
        command.append("--draft")
    if args.prerelease:
        command.append("--prerelease")
    run(command)


def main() -> None:
    args = parse_args()
    if shutil.which("gh") is None:
        raise SystemExit("GitHub CLI `gh` is required. Install it and run `gh auth login` first.")

    artifacts = [path.resolve() for path in args.artifacts] if args.artifacts else discover_artifacts(args.artifact_dir)
    artifacts = [path for path in artifacts if path.exists() and path.is_file()]
    if not artifacts:
        raise SystemExit(f"No release artifacts found in {args.artifact_dir}")

    ensure_release(args)
    run(["gh", "release", "upload", args.tag, *[str(path) for path in artifacts], "--clobber"])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
