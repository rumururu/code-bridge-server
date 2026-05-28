"""Inspect packaged build outputs (APK, IPA) for manifest metadata.

Used by :func:`agent.tool_artifacts._build_output_metadata` so every
``build_output`` artifact the Cockpit records carries human-meaningful
package/version/signing info, not just a file size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .apk_inspector import inspect_apk
from .ipa_inspector import inspect_ipa

__all__ = ["inspect_apk", "inspect_ipa", "inspect_build_artifact"]


def inspect_build_artifact(path: str | Path) -> dict[str, Any] | None:
    """Dispatch to the right inspector based on file extension.

    Returns ``None`` for unsupported file types so the caller can fall back
    to size-only metadata without raising.
    """
    p = Path(path)
    if not p.is_file():
        return None
    suffix = p.suffix.lower()
    if suffix == ".apk":
        return {"manifest_kind": "apk", "manifest": inspect_apk(p).to_dict()}
    if suffix == ".ipa":
        return {"manifest_kind": "ipa", "manifest": inspect_ipa(p).to_dict()}
    return None
