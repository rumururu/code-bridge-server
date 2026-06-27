"""Runtime data path helpers for packaged and development server modes."""

from __future__ import annotations

import os
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]
APP_SUPPORT_ENV = "CODEBRIDGE_APP_SUPPORT_DIR"


def app_support_dir() -> Path | None:
    """Return the configured app support directory, if desktop mode set one."""
    raw_path = os.environ.get(APP_SUPPORT_ENV)
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_path(filename: str, legacy_path: Path) -> Path:
    """Resolve a runtime state path.

    Development keeps existing in-repo paths for compatibility. Packaged
    desktop launches set CODEBRIDGE_APP_SUPPORT_DIR so mutable state is kept
    outside the signed app bundle and survives app upgrades.
    """
    support_dir = app_support_dir()
    if support_dir is None:
        return legacy_path
    return support_dir / filename


def runtime_dir(dirname: str, legacy_path: Path) -> Path:
    support_dir = app_support_dir()
    if support_dir is None:
        legacy_path.mkdir(parents=True, exist_ok=True)
        return legacy_path
    path = support_dir / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path
