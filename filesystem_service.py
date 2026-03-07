"""Filesystem browsing service for cross-platform folder selection."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from config import get_config


def get_home_directory() -> str:
    """Get the user's home directory."""
    return str(Path.home())


def validate_accessible_path(path: str) -> bool:
    """Check if path is within accessible_folders.

    Args:
        path: Path to validate

    Returns:
        True if path is within any accessible folder
    """
    config = get_config()
    target = Path(path).resolve()

    for folder in config.accessible_folders:
        folder_path = Path(folder).resolve()
        try:
            target.relative_to(folder_path)
            return True
        except ValueError:
            continue
    return False


def get_allowed_roots() -> list[dict[str, str]]:
    """Get list of accessible root folders.

    Returns:
        List of accessible folders with name and path
    """
    config = get_config()
    roots = []
    for folder in config.accessible_folders:
        p = Path(folder)
        if p.exists() and p.is_dir():
            roots.append({"name": p.name, "path": str(p.resolve())})
    return roots


def list_drives() -> list[str] | None:
    """List available drives on Windows.

    Returns:
        List of drive letters (e.g., ['C:', 'D:']) on Windows,
        None on other platforms.
    """
    if platform.system() != "Windows":
        return None

    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:\\"
        if Path(drive).exists():
            drives.append(f"{letter}:")
    return drives if drives else None


def browse_directory(path: str | None = None) -> dict[str, Any]:
    """Browse a directory within accessible folders only.

    Args:
        path: Directory path to browse. If None, returns accessible_folders list.

    Returns:
        Dictionary containing:
        - current_path: The absolute path being browsed (None if showing roots)
        - parent_path: Path to parent directory (None if at accessible root)
        - is_root: Whether current path is an accessible root
        - is_accessible_roots: True if showing accessible_folders selection
        - platform: Operating system (darwin, linux, windows)
        - folders: List of subdirectory info
        - drives: List of available drives (Windows only)

    Raises:
        FileNotFoundError: If the path does not exist
        NotADirectoryError: If the path is not a directory
        PermissionError: If path is outside accessible folders
    """
    config = get_config()
    roots = get_allowed_roots()

    # No path specified - return accessible_folders list
    if not path:
        if len(roots) == 1:
            # Single root - browse it directly
            path = roots[0]["path"]
        else:
            # Multiple roots - show selection screen
            return {
                "current_path": None,
                "parent_path": None,
                "is_root": True,
                "is_accessible_roots": True,
                "platform": platform.system().lower(),
                "folders": roots,
                "drives": None,
            }

    target = Path(path).resolve()

    # Validate path exists
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    # CRITICAL: Validate path is within accessible_folders
    if not validate_accessible_path(str(target)):
        raise PermissionError(
            f"Access denied: {path} is outside accessible folders"
        )

    # Calculate parent path with accessible_folders boundary check
    parent_path: str | None = None
    is_root = False

    parent = target.parent
    if validate_accessible_path(str(parent)):
        parent_path = str(parent)
    else:
        # At accessible_folders boundary - can't go higher
        is_root = True

    # Check if we're exactly at an accessible root
    target_str = str(target)
    for folder in config.accessible_folders:
        if str(Path(folder).resolve()) == target_str:
            is_root = True
            # If multiple roots, allow going "up" to root selection
            if len(roots) > 1:
                parent_path = None  # Will trigger is_accessible_roots on client
            break

    # List subdirectories (excluding hidden folders)
    folders = []
    try:
        for item in sorted(target.iterdir(), key=lambda x: x.name.lower()):
            # Skip hidden files/folders
            if item.name.startswith("."):
                continue
            if item.is_dir():
                is_accessible = os.access(item, os.R_OK)
                folders.append({
                    "name": item.name,
                    "path": str(item),
                    "is_accessible": is_accessible,
                })
    except PermissionError:
        # Can't list directory contents
        pass

    return {
        "current_path": str(target),
        "parent_path": parent_path,
        "is_root": is_root,
        "is_accessible_roots": False,
        "platform": platform.system().lower(),
        "folders": folders,
        "drives": list_drives(),
    }


def get_quick_access_paths() -> list[dict[str, str]]:
    """Get quick access paths within accessible folders only.

    Returns only paths that are within the configured accessible_folders.
    This ensures security by not exposing paths outside the whitelist.

    Returns:
        List of dictionaries with 'name' and 'path' keys.
    """
    # Return accessible_folders as quick access paths
    return get_allowed_roots()
