"""Inspect an IPA build artifact for human-meaningful manifest metadata.

An IPA is a ZIP archive shaped as::

    Payload/
        MyApp.app/
            Info.plist          # binary or XML plist
            embedded.mobileprovision   # CMS-wrapped XML
            ...

We pull:

- Bundle id, version, build number, display name, minimum OS, supported
  device families, supported orientations (from ``Info.plist``).
- Team identifier, provisioning profile name, expiration date, and the
  entitlements that the profile authorises (from
  ``embedded.mobileprovision``).
- Total payload size and SHA-256 fingerprint.

The inspector never raises on a malformed IPA and finishes in well under a
second for a typical 60 MB release IPA. Heavy work (full SHA-256) is gated
behind a size cap.
"""

from __future__ import annotations

import hashlib
import logging
import plistlib
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Cap for fingerprinting. Bigger IPAs still get manifest metadata but no
# SHA-256, to keep inspection within a couple of seconds on slow disks.
_HASH_SIZE_CAP_BYTES = 250 * 1024 * 1024  # 250 MB


@dataclass
class IpaInspectionResult:
    """Manifest summary for one IPA file."""

    bundle_id: str | None = None
    bundle_name: str | None = None
    display_name: str | None = None
    version: str | None = None
    build: str | None = None
    minimum_os_version: str | None = None
    supported_devices: list[str] = field(default_factory=list)
    supported_orientations: list[str] = field(default_factory=list)
    team_id: str | None = None
    provisioning_profile_name: str | None = None
    provisioning_profile_uuid: str | None = None
    provisioning_profile_expiration: str | None = None
    entitlements: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None
    inspected_with: str = "plistlib"
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inspect_ipa(path: str | Path) -> IpaInspectionResult:
    """Return a best-effort manifest summary for the IPA at ``path``."""
    ipa_path = Path(path)
    if not ipa_path.exists() or not ipa_path.is_file():
        return IpaInspectionResult(error=f"IPA not found: {ipa_path}")

    result = IpaInspectionResult()
    try:
        with zipfile.ZipFile(ipa_path, "r") as zf:
            app_prefix = _find_app_prefix(zf)
            if app_prefix is None:
                result.error = "no Payload/*.app entry found"
                return result

            info_plist = _read_plist(zf, f"{app_prefix}Info.plist")
            if info_plist:
                _populate_from_info_plist(info_plist, result)
            else:
                result.error = "Info.plist not found or unreadable"

            mobileprovision = _read_zip_entry(zf, f"{app_prefix}embedded.mobileprovision")
            if mobileprovision:
                _populate_from_mobileprovision(mobileprovision, result)
    except zipfile.BadZipFile as exc:
        result.error = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 — must never raise to caller
        logger.warning("ipa_inspector: failed to inspect %s: %s", ipa_path, exc)
        result.error = str(exc)
        return result

    try:
        file_size = ipa_path.stat().st_size
    except OSError:
        file_size = 0
    if 0 < file_size <= _HASH_SIZE_CAP_BYTES:
        result.sha256 = _sha256_of(ipa_path)

    return result


# ---------------------------------------------------------------------------
# Info.plist
# ---------------------------------------------------------------------------


def _find_app_prefix(zf: zipfile.ZipFile) -> str | None:
    """Return the ``Payload/<Name>.app/`` prefix used inside the IPA."""
    for name in zf.namelist():
        # Match Payload/<Name>.app/ but skip nested .app bundles such as
        # extensions and watchOS apps. We want the top-level one only.
        match = re.match(r"^(Payload/[^/]+\.app/)", name)
        if match:
            return match.group(1)
    return None


def _read_zip_entry(zf: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        return zf.read(name)
    except KeyError:
        return None
    except (zipfile.BadZipFile, OSError) as exc:
        logger.debug("ipa_inspector: cannot read %s: %s", name, exc)
        return None


def _read_plist(zf: zipfile.ZipFile, name: str) -> dict[str, Any] | None:
    raw = _read_zip_entry(zf, name)
    if raw is None:
        return None
    try:
        parsed = plistlib.loads(raw)
    except (plistlib.InvalidFileException, ValueError) as exc:
        logger.debug("ipa_inspector: invalid plist %s: %s", name, exc)
        return None
    return parsed if isinstance(parsed, dict) else None


_DEVICE_FAMILY_NAMES = {
    1: "iPhone",
    2: "iPad",
    3: "AppleTV",
    4: "AppleWatch",
    6: "Mac",
    7: "Vision",
}


def _populate_from_info_plist(info: dict[str, Any], result: IpaInspectionResult) -> None:
    result.bundle_id = info.get("CFBundleIdentifier") or None
    result.bundle_name = info.get("CFBundleName") or None
    result.display_name = info.get("CFBundleDisplayName") or None
    result.version = info.get("CFBundleShortVersionString") or None
    result.build = info.get("CFBundleVersion") or None
    result.minimum_os_version = info.get("MinimumOSVersion") or info.get("LSMinimumSystemVersion") or None

    devices: list[str] = []
    raw_devices = info.get("UIDeviceFamily")
    if isinstance(raw_devices, list):
        for entry in raw_devices:
            if isinstance(entry, int):
                devices.append(_DEVICE_FAMILY_NAMES.get(entry, f"family_{entry}"))
    result.supported_devices = devices

    orientations: list[str] = []
    for key in (
        "UISupportedInterfaceOrientations",
        "UISupportedInterfaceOrientations~ipad",
    ):
        raw_or = info.get(key)
        if isinstance(raw_or, list):
            orientations.extend(str(item) for item in raw_or)
    result.supported_orientations = sorted(set(orientations))


# ---------------------------------------------------------------------------
# embedded.mobileprovision
# ---------------------------------------------------------------------------


_PLIST_BEGIN = b"<?xml"
_PLIST_END = b"</plist>"


def _populate_from_mobileprovision(data: bytes, result: IpaInspectionResult) -> None:
    """Pull the inner XML plist out of the CMS-wrapped ``mobileprovision``.

    ``embedded.mobileprovision`` is a CMS / PKCS#7 envelope; we don't validate
    the signature here, only extract the payload to read its plist contents.
    The payload always starts with ``<?xml`` and ends with ``</plist>``.
    """
    begin = data.find(_PLIST_BEGIN)
    end = data.find(_PLIST_END)
    if begin == -1 or end == -1 or end < begin:
        return
    xml = data[begin : end + len(_PLIST_END)]
    try:
        parsed = plistlib.loads(xml)
    except (plistlib.InvalidFileException, ValueError) as exc:
        logger.debug("ipa_inspector: invalid embedded.mobileprovision: %s", exc)
        return
    if not isinstance(parsed, dict):
        return

    team_ids = parsed.get("TeamIdentifier")
    if isinstance(team_ids, list) and team_ids:
        result.team_id = str(team_ids[0])

    result.provisioning_profile_name = parsed.get("Name") or None
    result.provisioning_profile_uuid = parsed.get("UUID") or None

    expiration = parsed.get("ExpirationDate")
    if isinstance(expiration, datetime):
        result.provisioning_profile_expiration = expiration.isoformat()
    elif isinstance(expiration, str):
        result.provisioning_profile_expiration = expiration

    entitlements = parsed.get("Entitlements")
    if isinstance(entitlements, dict):
        # Only keep scalar/list/dict entitlement values; drop binary blobs.
        clean: dict[str, Any] = {}
        for key, value in entitlements.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                clean[str(key)] = value
        result.entitlements = clean


def _sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
    except OSError:
        return ""
    return sha.hexdigest()
