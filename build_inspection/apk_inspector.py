"""Inspect an APK build artifact for human-meaningful manifest metadata.

The Cockpit shows artifact rows after a `flutter build apk` or
`./gradlew assembleRelease` runs. Without this inspector the artifact carries
only ``size_bytes`` and ``suffix`` — useful for "yes a file was produced" but
useless for "what package id, what version, what permissions, which keystore
signed it".

Two extraction backends, in order of preference:

1. **aapt2 / aapt** — when the Android SDK is installed on the host, this
   gives the canonical answer (parsed by Google's own AXML reader).
2. **ZIP fallback** — when no SDK is present, read the APK as a plain ZIP and
   extract what is decodable from raw bytes: APK signing block presence,
   total entry count, presence of the META-INF signature, optional native
   ABI detection from the ``lib/`` directory layout.

Both paths return the same :class:`ApkInspectionResult` shape so the caller
does not branch.

The inspector is **read-only**, must not raise on a malformed APK, and must
finish in < 2 seconds for a typical 30 MB release APK. Slow operations
(e.g. computing SHA-256) are guarded by a size cap.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Max APK size for the SHA-256 fingerprint. APKs above this are still
# inspected for manifest metadata, but the fingerprint is reported as None
# to keep inspection well under the 2-second budget on slow disks.
_HASH_SIZE_CAP_BYTES = 200 * 1024 * 1024  # 200 MB


@dataclass
class ApkInspectionResult:
    """Manifest summary for one APK file."""

    package: str | None = None
    version_code: int | None = None
    version_name: str | None = None
    min_sdk: int | None = None
    target_sdk: int | None = None
    application_label: str | None = None
    permissions: list[str] = field(default_factory=list)
    native_abis: list[str] = field(default_factory=list)
    signing_block_present: bool | None = None
    sha256: str | None = None
    inspected_with: str = "none"
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inspect_apk(path: str | Path) -> ApkInspectionResult:
    """Return a best-effort manifest summary for the APK at ``path``."""
    apk_path = Path(path)
    if not apk_path.exists() or not apk_path.is_file():
        return ApkInspectionResult(error=f"APK not found: {apk_path}")

    aapt_binary = _resolve_aapt_binary()
    result = ApkInspectionResult(inspected_with="aapt" if aapt_binary else "zip")

    try:
        if aapt_binary:
            _populate_with_aapt(aapt_binary, apk_path, result)
        _populate_with_zip(apk_path, result)
    except Exception as exc:  # noqa: BLE001 — must never raise to caller
        logger.warning("apk_inspector: failed to inspect %s: %s", apk_path, exc)
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# aapt path
# ---------------------------------------------------------------------------


def _resolve_aapt_binary() -> str | None:
    for candidate in ("aapt2", "aapt"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _populate_with_aapt(binary: str, apk_path: Path, result: ApkInspectionResult) -> None:
    """Best-effort populate using ``aapt dump badging``. Failures are recorded but not raised."""
    if Path(binary).name == "aapt2":
        argv = [binary, "dump", "badging", str(apk_path)]
    else:
        argv = [binary, "dump", "badging", str(apk_path)]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        result.error = f"aapt invocation failed: {exc}"
        return

    output = completed.stdout or ""
    if completed.returncode != 0 and not output:
        result.error = (completed.stderr or "aapt returned no output").strip()
        return

    package_match = re.search(
        r"package: name='([^']+)' versionCode='(\d+)' versionName='([^']*)'",
        output,
    )
    if package_match:
        result.package = package_match.group(1)
        try:
            result.version_code = int(package_match.group(2))
        except ValueError:
            pass
        result.version_name = package_match.group(3) or None

    min_match = re.search(r"sdkVersion:'(\d+)'", output)
    if min_match:
        try:
            result.min_sdk = int(min_match.group(1))
        except ValueError:
            pass

    target_match = re.search(r"targetSdkVersion:'(\d+)'", output)
    if target_match:
        try:
            result.target_sdk = int(target_match.group(1))
        except ValueError:
            pass

    label_match = re.search(r"application-label(?:-[\w-]+)?:'([^']*)'", output)
    if label_match:
        result.application_label = label_match.group(1) or None

    perms = re.findall(r"uses-permission(?:-sdk-23)?: name='([^']+)'", output)
    result.permissions = sorted(set(perms))


# ---------------------------------------------------------------------------
# ZIP fallback
# ---------------------------------------------------------------------------


def _populate_with_zip(apk_path: Path, result: ApkInspectionResult) -> None:
    """Read what we can from the raw APK ZIP. Cheap and always available."""
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        result.error = (result.error or str(exc))
        return

    # Native ABIs are encoded as lib/<abi>/*.so directory entries.
    abis: set[str] = set()
    has_meta_inf_sig = False
    for name in names:
        if name.startswith("lib/") and name.endswith(".so"):
            parts = name.split("/")
            if len(parts) >= 3:
                abis.add(parts[1])
        if name.startswith("META-INF/") and (name.endswith(".RSA") or name.endswith(".DSA") or name.endswith(".EC")):
            has_meta_inf_sig = True
    result.native_abis = sorted(abis)

    # APK Signing Block sits between the central directory and the ZIP EOCD
    # record. The magic string "APK Sig Block 42" is required by v2/v3 signers.
    # We only need to look at the last ~64 KB.
    result.signing_block_present = _detect_apk_signing_block(apk_path) or has_meta_inf_sig

    # Cheap fingerprint for artifact correlation.
    try:
        file_size = apk_path.stat().st_size
    except OSError:
        file_size = 0
    if 0 < file_size <= _HASH_SIZE_CAP_BYTES:
        result.sha256 = _sha256_of(apk_path)


def _detect_apk_signing_block(apk_path: Path) -> bool:
    """Return True if the APK Signing Block v2/v3 magic appears near EOCD."""
    try:
        with apk_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size < 32:
                return False
            scan = min(size, 128 * 1024)
            f.seek(size - scan)
            data = f.read(scan)
    except OSError:
        return False
    return b"APK Sig Block 42" in data


def _sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
    except OSError:
        return ""
    return sha.hexdigest()
