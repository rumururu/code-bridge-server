"""Tests for APK / IPA manifest inspection.

We synthesize minimal-but-valid APK and IPA ZIP archives at test time so the
suite stays hermetic (no checked-in binary fixtures, no Android SDK / Xcode
required on the runner).
"""

from __future__ import annotations

import io
import os
import plistlib
import struct
import sys
import zipfile
from pathlib import Path

import pytest

SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from build_inspection import (  # noqa: E402
    inspect_apk,
    inspect_build_artifact,
    inspect_ipa,
)


# ---------------------------------------------------------------------------
# APK fixtures
# ---------------------------------------------------------------------------


def _write_minimal_apk(
    path: Path,
    *,
    abis: tuple[str, ...] = ("arm64-v8a", "armeabi-v7a"),
    with_signing_block: bool = True,
    with_meta_inf_sig: bool = True,
) -> None:
    """Write a ZIP that the inspector will treat as an APK.

    We don't bother with a real AXML AndroidManifest.xml. The ZIP-fallback
    path only needs the lib/<abi>/*.so layout, optional META-INF signature
    file, and (optionally) the APK Signing Block magic somewhere near EOCD.
    """
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("AndroidManifest.xml", b"placeholder")
        zf.writestr("classes.dex", b"\x00" * 32)
        for abi in abis:
            zf.writestr(f"lib/{abi}/libapp.so", b"\x7fELF" + b"\x00" * 64)
        if with_meta_inf_sig:
            zf.writestr("META-INF/CERT.RSA", b"fake-rsa")
            zf.writestr("META-INF/CERT.SF", b"fake-sf")
            zf.writestr("META-INF/MANIFEST.MF", b"Manifest-Version: 1.0\n")

    if with_signing_block:
        # Append the APK Signing Block magic so the inspector reports
        # signing_block_present=True. We append it just before the
        # zip-trailing bytes by writing it as an extra ZIP comment.
        with path.open("rb") as f:
            content = f.read()
        # Re-open in append mode and append the magic. Strictly invalid ZIP
        # structure but the inspector only scans the last 128 KB for the
        # magic string, which is enough.
        with path.open("ab") as f:
            f.write(b"APK Sig Block 42" + b"\x00" * 8)


def test_inspect_apk_reports_native_abis_and_signing(tmp_path):
    apk = tmp_path / "app-release.apk"
    _write_minimal_apk(apk)

    result = inspect_apk(apk).to_dict()
    assert result["native_abis"] == ["arm64-v8a", "armeabi-v7a"]
    assert result["signing_block_present"] is True
    assert result["sha256"] is not None
    assert len(result["sha256"]) == 64
    assert result["error"] is None


def test_inspect_apk_without_signing_block(tmp_path):
    apk = tmp_path / "app-unsigned.apk"
    _write_minimal_apk(
        apk,
        with_signing_block=False,
        with_meta_inf_sig=False,
    )

    result = inspect_apk(apk).to_dict()
    assert result["signing_block_present"] is False


def test_inspect_apk_missing_file_returns_error(tmp_path):
    missing = tmp_path / "ghost.apk"
    result = inspect_apk(missing).to_dict()
    assert result["error"] is not None
    assert "not found" in result["error"]


def test_inspect_apk_malformed_zip_does_not_raise(tmp_path):
    bogus = tmp_path / "bad.apk"
    bogus.write_bytes(b"this is not a zip")
    result = inspect_apk(bogus).to_dict()
    # Inspector must NOT raise. Either signing_block_present is None or the
    # error field is populated; both are acceptable.
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# IPA fixtures
# ---------------------------------------------------------------------------


def _build_mobileprovision(payload: dict) -> bytes:
    """Wrap an inner plist in a fake CMS envelope.

    The inspector locates the inner XML by searching for ``<?xml`` and
    ``</plist>`` markers, so a real CMS wrapper is not necessary.
    """
    xml = plistlib.dumps(payload, fmt=plistlib.FMT_XML)
    return b"FAKE-CMS-HEADER\n" + xml + b"\nFAKE-CMS-FOOTER"


def _write_minimal_ipa(
    path: Path,
    *,
    bundle_id: str = "com.example.app",
    version: str = "1.2.3",
    build_number: str = "45",
    min_os: str = "16.0",
    devices: list[int] | None = None,
    team_id: str = "ABCDE12345",
) -> None:
    if devices is None:
        devices = [1, 2]  # iPhone + iPad
    info_plist = {
        "CFBundleIdentifier": bundle_id,
        "CFBundleName": "ExampleApp",
        "CFBundleDisplayName": "Example",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": build_number,
        "MinimumOSVersion": min_os,
        "UIDeviceFamily": devices,
        "UISupportedInterfaceOrientations": [
            "UIInterfaceOrientationPortrait",
            "UIInterfaceOrientationLandscapeLeft",
        ],
    }
    info_xml = plistlib.dumps(info_plist, fmt=plistlib.FMT_XML)

    mobileprovision = _build_mobileprovision(
        {
            "Name": "Test Provisioning Profile",
            "UUID": "00000000-0000-0000-0000-000000000001",
            "TeamIdentifier": [team_id],
            "ExpirationDate": "2099-01-01T00:00:00Z",
            "Entitlements": {
                "application-identifier": f"{team_id}.{bundle_id}",
                "com.apple.developer.team-identifier": team_id,
                "get-task-allow": False,
            },
        }
    )

    with zipfile.ZipFile(path, "w") as zf:
        app = "Payload/ExampleApp.app/"
        zf.writestr(f"{app}Info.plist", info_xml)
        zf.writestr(f"{app}embedded.mobileprovision", mobileprovision)
        zf.writestr(f"{app}ExampleApp", b"\x00binary\x00")


def test_inspect_ipa_reports_bundle_and_version(tmp_path):
    ipa = tmp_path / "Runner.ipa"
    _write_minimal_ipa(ipa)

    result = inspect_ipa(ipa).to_dict()
    assert result["bundle_id"] == "com.example.app"
    assert result["bundle_name"] == "ExampleApp"
    assert result["display_name"] == "Example"
    assert result["version"] == "1.2.3"
    assert result["build"] == "45"
    assert result["minimum_os_version"] == "16.0"
    assert result["supported_devices"] == ["iPhone", "iPad"]
    assert "UIInterfaceOrientationPortrait" in result["supported_orientations"]
    assert result["error"] is None


def test_inspect_ipa_reports_team_and_entitlements(tmp_path):
    ipa = tmp_path / "Runner.ipa"
    _write_minimal_ipa(ipa, team_id="TEAMABC123")

    result = inspect_ipa(ipa).to_dict()
    assert result["team_id"] == "TEAMABC123"
    assert result["provisioning_profile_name"] == "Test Provisioning Profile"
    assert result["provisioning_profile_uuid"] == "00000000-0000-0000-0000-000000000001"
    assert result["entitlements"]["com.apple.developer.team-identifier"] == "TEAMABC123"
    assert result["entitlements"]["get-task-allow"] is False


def test_inspect_ipa_sha256(tmp_path):
    ipa = tmp_path / "Runner.ipa"
    _write_minimal_ipa(ipa)
    result = inspect_ipa(ipa).to_dict()
    assert result["sha256"] is not None
    assert len(result["sha256"]) == 64


def test_inspect_ipa_missing_file_returns_error(tmp_path):
    missing = tmp_path / "ghost.ipa"
    result = inspect_ipa(missing).to_dict()
    assert result["error"] is not None
    assert "not found" in result["error"]


def test_inspect_ipa_without_payload_returns_error(tmp_path):
    bad = tmp_path / "noapp.ipa"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("README.txt", b"this is not an ipa")
    result = inspect_ipa(bad).to_dict()
    assert result["error"] == "no Payload/*.app entry found"


def test_inspect_ipa_handles_malformed_plist(tmp_path):
    ipa = tmp_path / "Runner.ipa"
    with zipfile.ZipFile(ipa, "w") as zf:
        app = "Payload/ExampleApp.app/"
        zf.writestr(f"{app}Info.plist", b"not a real plist")
    result = inspect_ipa(ipa).to_dict()
    # No raise; error populated.
    assert result["bundle_id"] is None
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatch_apk(tmp_path):
    apk = tmp_path / "app.apk"
    _write_minimal_apk(apk)
    result = inspect_build_artifact(apk)
    assert result is not None
    assert result["manifest_kind"] == "apk"
    assert "manifest" in result
    assert result["manifest"]["native_abis"]


def test_dispatch_ipa(tmp_path):
    ipa = tmp_path / "Runner.ipa"
    _write_minimal_ipa(ipa)
    result = inspect_build_artifact(ipa)
    assert result is not None
    assert result["manifest_kind"] == "ipa"
    assert result["manifest"]["bundle_id"] == "com.example.app"


def test_dispatch_returns_none_for_unsupported(tmp_path):
    other = tmp_path / "build.zip"
    other.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip
    assert inspect_build_artifact(other) is None


def test_dispatch_returns_none_for_directory(tmp_path):
    d = tmp_path / "build_output"
    d.mkdir()
    assert inspect_build_artifact(d) is None


# ---------------------------------------------------------------------------
# Integration with tool_artifacts._build_output_metadata
# ---------------------------------------------------------------------------


def test_tool_artifacts_includes_manifest_for_apk(tmp_path):
    from agent.tool_artifacts import _build_output_metadata

    apk = tmp_path / "app-release.apk"
    _write_minimal_apk(apk)

    metadata = _build_output_metadata(apk)
    assert metadata["suffix"] == ".apk"
    assert metadata["size_bytes"] is not None
    assert metadata["manifest_kind"] == "apk"
    assert "manifest" in metadata


def test_tool_artifacts_includes_manifest_for_ipa(tmp_path):
    from agent.tool_artifacts import _build_output_metadata

    ipa = tmp_path / "Runner.ipa"
    _write_minimal_ipa(ipa)

    metadata = _build_output_metadata(ipa)
    assert metadata["suffix"] == ".ipa"
    assert metadata["manifest_kind"] == "ipa"
    assert metadata["manifest"]["bundle_id"] == "com.example.app"


def test_tool_artifacts_unaffected_for_directory(tmp_path):
    from agent.tool_artifacts import _build_output_metadata

    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("ok")
    (d / "main.js").write_text("ok")

    metadata = _build_output_metadata(d)
    assert "entry_count_sampled" in metadata
    assert "manifest_kind" not in metadata
