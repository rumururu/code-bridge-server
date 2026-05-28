#!/usr/bin/env python3
"""Sign, notarize, and staple the macOS desktop server DMG."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
APP_NAME = "Code Bridge Server"
DEFAULT_DIST_DIR = REPO_ROOT / "dist" / "desktop_server_app"
DEFAULT_APP_PATH = DEFAULT_DIST_DIR / f"{APP_NAME}.app"
DEFAULT_DMG_PATH = DEFAULT_DIST_DIR / f"{APP_NAME}.dmg"
DEFAULT_ENTITLEMENTS = REPO_ROOT / "desktop_server_app" / "macos_entitlements.plist"
REDACTED_FLAGS = {"--key", "--key-id", "--issuer"}


def printable_command(command: list[str]) -> str:
    printable: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            printable.append("<redacted>")
            redact_next = False
            continue
        printable.append(part)
        if part in REDACTED_FLAGS:
            redact_next = True
    return shlex.join(printable)


def run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print(f"\n$ {printable_command(command)}\n")
    subprocess.run(command, cwd=cwd, check=True)


def resolve_required(value: str | None, env_name: str, label: str) -> str:
    resolved = value or os.environ.get(env_name)
    if not resolved:
        raise SystemExit(f"{label} is required. Pass it as an argument or set {env_name}.")
    return resolved


def create_dmg(app_path: Path, dmg_path: Path, volume_name: str) -> None:
    if not app_path.exists():
        raise SystemExit(f"App bundle does not exist: {app_path}")
    if dmg_path.exists():
        dmg_path.unlink()
    run(
        [
            "hdiutil",
            "create",
            "-volname",
            volume_name,
            "-srcfolder",
            str(app_path),
            "-ov",
            "-format",
            "UDZO",
            str(dmg_path),
        ]
    )


def sign_app(app_path: Path, identity: str, entitlements: Path) -> None:
    if not entitlements.exists():
        raise SystemExit(f"Entitlements file does not exist: {entitlements}")
    run(
        [
            "codesign",
            "--force",
            "--deep",
            "--options",
            "runtime",
            "--timestamp",
            "--entitlements",
            str(entitlements),
            "--sign",
            identity,
            str(app_path),
        ]
    )
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app_path)])


def sign_dmg(dmg_path: Path, identity: str) -> None:
    run(["codesign", "--force", "--timestamp", "--sign", identity, str(dmg_path)])
    run(["codesign", "--verify", "--verbose=2", str(dmg_path)])
    run(["hdiutil", "verify", str(dmg_path)])


def notarize_dmg(dmg_path: Path, key: str, key_id: str, issuer: str) -> None:
    run(
        [
            "xcrun",
            "notarytool",
            "submit",
            str(dmg_path),
            "--key",
            key,
            "--key-id",
            key_id,
            "--issuer",
            issuer,
            "--wait",
            "--output-format",
            "json",
        ]
    )
    run(["xcrun", "stapler", "staple", str(dmg_path)])
    run(["xcrun", "stapler", "validate", str(dmg_path)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sign and notarize the macOS Code Bridge Server DMG.")
    parser.add_argument("--app", type=Path, default=DEFAULT_APP_PATH, help="Path to the .app bundle.")
    parser.add_argument("--dmg", type=Path, default=DEFAULT_DMG_PATH, help="Path to create/sign/notarize.")
    parser.add_argument("--volume-name", default=APP_NAME, help="DMG volume name.")
    parser.add_argument("--identity", default=None, help="Developer ID Application identity.")
    parser.add_argument("--entitlements", type=Path, default=DEFAULT_ENTITLEMENTS, help="macOS entitlements plist.")
    parser.add_argument("--skip-dmg-create", action="store_true", help="Use the existing DMG instead of recreating it.")
    parser.add_argument("--skip-notarize", action="store_true", help="Only sign and verify; do not notarize/staple.")
    parser.add_argument("--notary-key", default=None, help="App Store Connect API key .p8 path.")
    parser.add_argument("--notary-key-id", default=None, help="App Store Connect API key id.")
    parser.add_argument("--notary-issuer", default=None, help="App Store Connect issuer id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = resolve_required(args.identity, "CODEBRIDGE_CODESIGN_IDENTITY", "Code signing identity")

    sign_app(args.app.resolve(), identity, args.entitlements.resolve())
    if not args.skip_dmg_create:
        create_dmg(args.app.resolve(), args.dmg.resolve(), args.volume_name)
    sign_dmg(args.dmg.resolve(), identity)

    if not args.skip_notarize:
        key = resolve_required(args.notary_key, "CODEBRIDGE_NOTARY_KEY", "Notary API key path")
        key_id = resolve_required(args.notary_key_id, "CODEBRIDGE_NOTARY_KEY_ID", "Notary API key id")
        issuer = resolve_required(args.notary_issuer, "CODEBRIDGE_NOTARY_ISSUER", "Notary issuer id")
        notarize_dmg(args.dmg.resolve(), key, key_id, issuer)

    run(["shasum", "-a", "256", str(args.dmg.resolve())])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
