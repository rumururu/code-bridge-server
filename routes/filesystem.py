"""Filesystem browsing routes for folder selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.database import get_accessible_folder_db
from files.filesystem_service import browse_directory, get_quick_access_paths

from .deps import require_local_access, require_localhost_only

router = APIRouter(
    prefix="/api/filesystem",
    tags=["filesystem"],
    dependencies=[Depends(require_local_access)],
)


# --- Accessible Folders Endpoints ---


class AddFolderRequest(BaseModel):
    """Request body for adding an accessible folder."""

    path: str


@router.get("/accessible-folders")
async def list_accessible_folders() -> dict[str, Any]:
    """List all accessible folders.

    These folders define the security boundary for file system access.
    Note: DB is auto-initialized with home folder on first startup.
    """
    db = get_accessible_folder_db()
    folders = db.get_all()

    # Return with folder metadata
    result = []
    for path in folders:
        p = Path(path)
        result.append({
            "path": path,
            "name": p.name or path,
            "exists": p.exists(),
        })

    return {"folders": result}


@router.post("/accessible-folders")
async def add_accessible_folder(data: AddFolderRequest) -> dict[str, Any]:
    """Add a new accessible folder.

    This defines the security boundary for file system access.
    """
    path = Path(data.path).expanduser().resolve()

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {data.path}")

    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {data.path}")

    db = get_accessible_folder_db()
    added = db.add(str(path))

    if not added:
        raise HTTPException(status_code=409, detail="Folder already exists")

    return {"success": True, "path": str(path)}


@router.delete("/accessible-folders")
async def remove_accessible_folder(path: str = Query(...)) -> dict[str, Any]:
    """Remove an accessible folder.

    Note: This only removes the folder from the accessible list,
    it does NOT delete the actual folder.
    """
    db = get_accessible_folder_db()
    removed = db.remove(path)

    if not removed:
        raise HTTPException(status_code=404, detail="Folder not found")

    return {"success": True, "path": path}


@router.get(
    "/browse-unrestricted",
    dependencies=[Depends(require_localhost_only)],
)
async def browse_filesystem_unrestricted(path: str | None = Query(None)) -> dict[str, Any]:
    """Browse filesystem without ``accessible_folders`` restrictions.

    This endpoint is used by the dashboard's folder picker to add new
    accessible folders. It allows browsing any user-accessible directory,
    so it is restricted to localhost (127.0.0.1 / ::1). The router-level
    ``require_local_access`` blocks tunnel access but still allows the
    rest of the local network — the extra ``require_localhost_only``
    here ensures that even if the dashboard binding is ever widened from
    127.0.0.1, only loopback clients can enumerate the host filesystem.
    """
    from pathlib import Path as PathLib
    import platform
    import os

    # Default to home directory
    if not path:
        path = str(PathLib.home())

    target = PathLib(path).resolve()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    # Calculate parent path
    parent = target.parent
    parent_path = str(parent) if parent != target else None

    # List subdirectories
    folders = []
    try:
        for item in sorted(target.iterdir(), key=lambda x: x.name.lower()):
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
        pass

    # List drives on Windows
    drives = None
    if platform.system() == "Windows":
        drives = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            drive = f"{letter}:\\"
            if PathLib(drive).exists():
                drives.append(f"{letter}:")

    return {
        "current_path": str(target),
        "parent_path": parent_path,
        "is_root": parent_path is None,
        "is_accessible_roots": False,
        "platform": platform.system().lower(),
        "folders": folders,
        "drives": drives,
    }


# --- Filesystem Browsing Endpoints (with accessible_folders restrictions) ---


@router.get("/browse")
async def browse_filesystem(path: str | None = Query(None)) -> dict[str, Any]:
    """Browse a directory within accessible folders only.

    Security: Only paths within server's accessible_folders config are allowed.
    Attempting to access paths outside this whitelist returns 403 Forbidden.

    Args:
        path: Directory path to browse. If not provided, returns accessible
              folders list (or browses single root if only one configured).

    Returns:
        JSON object with:
        - current_path: Absolute path being browsed (null if showing roots)
        - parent_path: Parent directory path (null if at accessible root)
        - is_root: Whether at an accessible root directory
        - is_accessible_roots: True if showing accessible_folders selection
        - platform: Operating system
        - folders: List of subdirectory info
        - drives: Available drives (Windows only)
    """
    try:
        return browse_directory(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/quick-access")
async def get_quick_access() -> dict[str, Any]:
    """Get quick access paths for common directories.

    Returns:
        List of common directories like Home, Documents, Downloads, etc.
    """
    return {"paths": get_quick_access_paths()}
