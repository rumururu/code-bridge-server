"""Release artifact checks for the mandatory compiled workflow kernel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_closed_kernel.py"
SPEC = importlib.util.spec_from_file_location("install_closed_kernel", SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(installer)

TARGET = ("macos", "arm64", "cp313", "cp313-cp313-macosx_26_0_arm64")
WHEEL_NAME = "agent_flow_core-0.1.0-cp313-cp313-macosx_26_0_arm64.whl"


def _target(monkeypatch) -> None:
    monkeypatch.setattr(installer, "target_os", lambda: TARGET[0])
    monkeypatch.setattr(installer, "target_architecture", lambda: TARGET[1])
    monkeypatch.setattr(installer, "python_abi", lambda: TARGET[2])
    monkeypatch.setattr(installer, "wheel_tag", lambda: TARGET[3])


def _artifact(
    root: Path,
    *,
    build_updates: dict | None = None,
    wheel_entries: dict[str, bytes] | None = None,
) -> tuple[Path, Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    wheel = root / WHEEL_NAME
    entries = wheel_entries or {
        "agent_flow_core.cpython-313-darwin.so": b"compiled extension",
        "agent_flow_core-0.1.0.dist-info/METADATA": b"Name: agent-flow-core\nVersion: 0.1.0\n",
        "agent_flow_core-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    build = {
        "tool": "Nuitka",
        "tool_version": "2.8.10",
        "python_version": "3.13.13",
        "python_abi": TARGET[2],
        "os": TARGET[0],
        "architecture": TARGET[1],
        "wheel_tag": TARGET[3],
        "base_image": None,
        "macos_minimum": "26.0",
    }
    build.update(build_updates or {})
    payload = wheel.read_bytes()
    manifest = {
        "schema_version": 1,
        "package": "agent-flow-core",
        "kernel_version": installer.EXPECTED_KERNEL_VERSION,
        "source_commit": installer.EXPECTED_SOURCE_COMMIT,
        "build": build,
        "wheel": {
            "filename": wheel.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        },
    }
    manifest_path = root / f"{wheel.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return wheel, manifest_path, manifest


def test_verified_target_wheel_is_selected(tmp_path, monkeypatch):
    _target(monkeypatch)
    wheel, manifest_path, manifest = _artifact(tmp_path)

    assert installer.select_artifact(tmp_path) == (wheel, manifest_path, manifest)


def test_damaged_wheel_is_rejected(tmp_path, monkeypatch):
    _target(monkeypatch)
    wheel, _, _ = _artifact(tmp_path)
    wheel.write_bytes(wheel.read_bytes() + b"tampered")

    with pytest.raises(installer.ArtifactError, match="sha256 mismatch"):
        installer.select_artifact(tmp_path)


def test_wrong_abi_is_rejected(tmp_path, monkeypatch):
    _target(monkeypatch)
    _artifact(
        tmp_path,
        build_updates={
            "python_abi": "cp312",
            "wheel_tag": "cp312-cp312-linux_x86_64",
        },
    )

    with pytest.raises(installer.ArtifactError, match="unsupported release target"):
        installer.select_artifact(tmp_path)


def test_missing_kernel_is_rejected(tmp_path, monkeypatch):
    _target(monkeypatch)

    with pytest.raises(installer.ArtifactError, match="no kernel wheel manifest"):
        installer.select_artifact(tmp_path)


def test_source_distribution_cannot_replace_the_wheel(tmp_path, monkeypatch):
    _target(monkeypatch)
    (tmp_path / "agent-flow-core-0.1.0.tar.gz").write_bytes(b"source archive")

    with pytest.raises(installer.ArtifactError, match="source distribution"):
        installer.select_artifact(tmp_path)


@pytest.mark.parametrize("suffix", [".py", ".pyc", ".pyi"])
def test_readable_source_inside_wheel_is_rejected(tmp_path, monkeypatch, suffix):
    _target(monkeypatch)
    entries = {
        "agent_flow_core.cpython-313-darwin.so": b"compiled extension",
        f"agent_flow_core/topology{suffix}": b"def topological_sort(): pass\n",
    }
    _artifact(tmp_path, wheel_entries=entries)

    with pytest.raises(installer.ArtifactError, match="contains readable source"):
        installer.select_artifact(tmp_path)


def test_unrelated_extension_cannot_replace_the_kernel(tmp_path, monkeypatch):
    _target(monkeypatch)
    entries = {
        "unrelated.cpython-313-darwin.so": b"wrong extension",
        "agent_flow_core-0.1.0.dist-info/METADATA": b"Name: agent-flow-core\n",
    }
    _artifact(tmp_path, wheel_entries=entries)

    with pytest.raises(installer.ArtifactError, match="no compiled extension"):
        installer.select_artifact(tmp_path)


def test_unapproved_source_commit_is_rejected(tmp_path, monkeypatch):
    _target(monkeypatch)
    _, manifest_path, manifest = _artifact(tmp_path)
    manifest["source_commit"] = "f" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(installer.ArtifactError, match="unapproved source commit"):
        installer.select_artifact(tmp_path)


def test_macos_binary_minimum_is_part_of_the_release_contract(tmp_path, monkeypatch):
    _target(monkeypatch)
    _artifact(tmp_path, build_updates={"macos_minimum": "14.0"})

    with pytest.raises(installer.ArtifactError, match="minimum must match"):
        installer.select_artifact(tmp_path)


def test_only_planned_macos_target_is_release_supported():
    assert installer.SUPPORTED_TARGETS == {
        ("macos", "arm64", "cp313", "cp313-cp313-macosx_26_0_arm64"),
    }


def test_product_scan_rejects_kernel_source_and_wheel_intermediates(tmp_path):
    source = tmp_path / "agent_flow_core" / "topology.py"
    source.parent.mkdir()
    source.write_text("def sort(): pass\n", encoding="utf-8")
    with pytest.raises(installer.ArtifactError, match="readable kernel source"):
        installer.scan_product(tmp_path)

    source.unlink()
    (tmp_path / WHEEL_NAME).write_bytes(b"wheel")
    with pytest.raises(installer.ArtifactError, match="build intermediates"):
        installer.scan_product(tmp_path)

    (tmp_path / WHEEL_NAME).unlink()
    (tmp_path / "agent_flow_core.py").write_text("source = True\n", encoding="utf-8")
    with pytest.raises(installer.ArtifactError, match="readable kernel source"):
        installer.scan_product(tmp_path)
