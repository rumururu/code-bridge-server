#!/usr/bin/env python3
"""Build desktop server app bundles with PyInstaller."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import plistlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


APP_NAME = "Code Bridge Server"
BUNDLE_IDENTIFIER = "com.mkideabox.codebridge.server"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SERVER_DIR = REPO_ROOT / "server"
DEFAULT_ENTRY = REPO_ROOT / "desktop_server_app" / "launcher.py"
DEFAULT_DIST_DIR = REPO_ROOT / "dist" / "desktop_server_app"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "desktop_server_app"
KERNEL_ARTIFACT_DIR = SERVER_DIR / "vendor" / "agent-flow-core"
KERNEL_INSTALLER = SCRIPT_DIR / "install_closed_kernel.py"
SERVER_STAGE_DIRNAME = "packaged_server"
SCRCPY_DIST_STAGE_DIRNAME = "packaged_scrcpy_dist"
NODE_RUNTIME_STAGE_DIRNAME = "node_runtime"
NODE_EXTRACT_DIRNAME = "node_extract"
NODE_DOWNLOAD_DIRNAME = "downloads"
DEFAULT_NODE_VERSION = "v24.15.0"
PLATFORM_TOOLS_STAGE_DIRNAME = "platform_tools"
PLATFORM_TOOLS_EXTRACT_DIRNAME = "platform_tools_extract"
# Google's repository2-3.xml keeps only the current platform-tools release;
# 37.0.0 vanished from it when 37.0.1 shipped, and the 2.1.0 build failed on
# the pin. Bump this to whatever the index lists when a build says so.
DEFAULT_PLATFORM_TOOLS_VERSION = "37.0.1"
ANDROID_REPOSITORY_BASE_URL = "https://dl.google.com/android/repository"
ANDROID_REPOSITORY_INDEX_URL = f"{ANDROID_REPOSITORY_BASE_URL}/repository2-3.xml"
SERVER_EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    # The decision layer ships compiled, as an installed wheel that
    # PyInstaller picks up from site-packages (ADR-002). Its source must not
    # travel beside it: `server/` goes on `sys.path` ahead of site-packages,
    # so a staged source copy would shadow the extension and the app would run
    # — correctly, and readably — from the very files this is meant to close.
    # Verified the wrong way round first: with both present, `__loader__` was
    # `SourceFileLoader`.
    "code_bridge_core",
    "scrcpy",
    "tests",
    "venv",
}
SERVER_EXCLUDE_SUFFIXES = {
    ".pyc",
}
PACKAGED_EXCLUDE_PATTERNS = {
    # A wheel left under server/ (the closed-package build writes one there
    # if asked) is a zip with an unsigned .so inside; notarization rejected
    # the 2.1.0 DMG on exactly that file. Wheels are installed, not shipped.
    "*.whl",
    "*.bak*",
    "*.backup*",
    "*.db*",
    "*.env*",
    "*.log*",
    "*.pid*",
    "*.sqlite*",
    "*.sqlite3*",
    "api_keys",
    "api_keys.*json*",
    "api_keys.*toml*",
    "api_keys.*yaml*",
    "api_keys.*yml*",
    "config",
    "config.*conf*",
    "config.*ini*",
    "config.*json*",
    "config.*local*",
    "config.*toml*",
    "config.*yaml*",
    "config.*yml*",
    "device_info",
    "device_info.*json*",
    "device_info.*toml*",
    "device_info.*yaml*",
    "device_info.*yml*",
    "firebase_config",
    "firebase_config.*json*",
    "firebase_config.*toml*",
    "firebase_config.*yaml*",
    "firebase_config.*yml*",
    "firebase*.json*",
    "firebase*.toml*",
    "firebase*.yaml*",
    "firebase*.yml*",
    "paired_accounts",
    "paired_accounts.*json*",
    "paired_accounts.*toml*",
    "paired_accounts.*yaml*",
    "paired_accounts.*yml*",
    "server_info",
    "server_info.*json*",
    "server_info.*toml*",
    "server_info.*yaml*",
    "server_info.*yml*",
}
SERVER_EXCLUDE_FILENAMES = {
    ".initialized",
    ".server.pid",
    "api_keys.json",
    "config.yaml",
    "device_info.json",
    "firebase_config.json",
    "paired_accounts.json",
    "server_info.json",
}
MARKER_SCHEMA_VERSION = 1
WIX_NAMESPACE = "http://wixtoolset.org/schemas/v4/wxs"
MSI_UPGRADE_CODE_NAMESPACE = uuid.UUID("7e1b48b6-7bc2-4f92-a682-6b5f31f0e2d0")


@dataclass(frozen=True)
class DataItem:
    source: Path
    destination: str
    required: bool = True


@dataclass(frozen=True)
class NodeRuntimeSpec:
    target: str
    archive_extension: str
    source_executable: str
    packaged_executable: str


@dataclass(frozen=True)
class PlatformToolsSpec:
    target: str
    repository_host_os: str
    executable: str


@dataclass(frozen=True)
class PlatformToolsArchiveInfo:
    revision: str
    url: str
    size: int
    sha1: str


def host_platform() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    raise SystemExit(f"Unsupported host platform: {platform.system()}")


def data_separator() -> str:
    return ";" if os.name == "nt" else ":"


def add_data_arg(source: Path, destination: str) -> str:
    return f"{source}{data_separator()}{destination}"


def check_required_tools(build_format: str) -> None:
    missing: list[str] = []
    if importlib.util.find_spec("PyInstaller") is None:
        missing.append("PyInstaller")
    if build_format == "dmg" and shutil.which("hdiutil") is None:
        missing.append("hdiutil")
    if build_format == "msi" and shutil.which("wix") is None:
        missing.append("WiX Toolset v4")
    if missing:
        install_hint = (
            "Install PyInstaller in this Python environment with `python -m pip install pyinstaller`."
            if "PyInstaller" in missing
            else ""
        )
        raise SystemExit(
            "Missing required build tool(s): "
            + ", ".join(missing)
            + (f"\n{install_hint}" if install_hint else "")
        )


def verify_kernel_is_compiled() -> None:
    """Refuse to package an app whose flow kernel is readable source.

    PyInstaller collects site-packages from the interpreter running this
    script, so whichever `agent_flow_core` is installed here is the one that
    ends up inside the `.app`. A developer environment installs it from the
    local checkout as plain `.py` — which is right for development and wrong
    for distribution, and the difference is invisible in the build log.

    That is not hypothetical. The `.dmg` built on 2026-06-28 carried 216 plain
    `.py` files, and nothing in the build said so. A `.app` is a folder and
    "Show Package Contents" is one click; a Developer ID signature proves who
    built it, not that it cannot be read.

    So the check runs here, before anything is packaged, and it asks the
    artifact rather than the build steps: can this module's source be
    recovered at runtime? `inspect.getsource` is the same question a reader
    with the `.app` open would ask.

    Build the closed wheel with the kernel repo's own script and install it
    into this environment:

        python scripts/build_closed_kernel.py            # in agent-flow-core
        pip install --force-reinstall dist/agent_flow_core-<...>.whl

    Source installs are supported for development, but never for packaging.
    """

    import inspect

    # Both closed packages, checked the same way. `agent_flow_core` is the
    # shared kernel; `code_bridge_core` is Code Bridge's own decision layer —
    # the step vocabulary, the routing, the authoring gates and the
    # Configurator. They are separate packages for the reason ADR-002 gives
    # (the kernel's G5 invariant), and either one shipping as source defeats
    # the same purpose.
    for package, module, symbol, hint in (
        (
            "agent_flow_core",
            "agent_flow_core.model",
            "Flow",
            "python scripts/build_closed_kernel.py",
        ),
        (
            "code_bridge_core",
            "code_bridge_core.step_cursor",
            "StepCursor",
            "python scripts/build_closed_kernel.py --package code_bridge_core"
            " --source-root <code-bridge>/server"
            " --pyproject <code-bridge>/server/code_bridge_core/pyproject.toml",
        ),
    ):
        try:
            loaded = getattr(__import__(module, fromlist=[symbol]), symbol)
        except (ImportError, AttributeError) as exc:
            raise SystemExit(
                f"{package} is not installed in this environment, so the "
                f"packaged app would have none:\n"
                f"    {exc}\n"
                f"Build and install it: {hint}"
            ) from exc

        try:
            inspect.getsource(loaded)
        except (OSError, TypeError):
            continue  # Compiled: no source to recover. The shipping state.

        raise SystemExit(
            f"{package} installed here is readable source, and packaging it "
            "would put that source inside the app.\n"
            f"    {inspect.getfile(loaded)}\n"
            "Build the compiled wheel and install it into this environment:\n"
            f"    {hint}\n"
            f"    pip install --force-reinstall dist/{package}-<...>.whl"
        )


def verify_kernel_artifact_manifest() -> None:
    """Verify the vendored target and the installed compiled kernel."""
    if not KERNEL_INSTALLER.is_file():
        raise SystemExit(f"Kernel artifact verifier is missing: {KERNEL_INSTALLER}")
    spec = importlib.util.spec_from_file_location(
        "code_bridge_closed_kernel_installer", KERNEL_INSTALLER
    )
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    try:
        _wheel, _manifest_path, manifest = verifier.select_artifact(
            KERNEL_ARTIFACT_DIR
        )
        verifier.verify_installed(manifest["kernel_version"])
    except (verifier.ArtifactError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Closed kernel artifact verification failed: {exc}") from exc


def validate_inputs(entry: Path) -> None:
    if not entry.exists():
        raise SystemExit(
            f"Entry point does not exist: {entry}\n"
            "If another worker has produced a desktop launcher, pass it with --entry."
        )
    if not SERVER_DIR.exists():
        raise SystemExit(f"Server directory does not exist: {SERVER_DIR}")
    requirements = SERVER_DIR / "requirements.txt"
    if not requirements.exists():
        raise SystemExit(f"Server requirements file does not exist: {requirements}")


def host_architecture() -> str:
    machine = platform.machine().lower()
    aliases = {
        "amd64": "x64",
        "x86_64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    return aliases.get(machine, machine)


def node_runtime_spec(host: str) -> NodeRuntimeSpec:
    arch = host_architecture()
    if host == "macos":
        if arch not in {"x64", "arm64"}:
            raise SystemExit(f"Unsupported macOS Node runtime architecture: {platform.machine()}")
        return NodeRuntimeSpec(f"darwin-{arch}", "tar.xz", "bin/node", "bin/node")
    if host == "linux":
        if arch not in {"x64", "arm64"}:
            raise SystemExit(f"Unsupported Linux Node runtime architecture: {platform.machine()}")
        return NodeRuntimeSpec(f"linux-{arch}", "tar.xz", "bin/node", "bin/node")
    if host == "windows":
        if arch not in {"x64", "arm64"}:
            raise SystemExit(f"Unsupported Windows Node runtime architecture: {platform.machine()}")
        return NodeRuntimeSpec(f"win-{arch}", "zip", "node.exe", "bin/node.exe")
    raise SystemExit(f"Unsupported Node runtime host: {host}")


def node_archive_name(version: str, spec: NodeRuntimeSpec) -> str:
    return f"node-{version}-{spec.target}.{spec.archive_extension}"


def node_archive_url(version: str, spec: NodeRuntimeSpec) -> str:
    return f"https://nodejs.org/dist/{version}/{node_archive_name(version, spec)}"


def platform_tools_spec(host: str) -> PlatformToolsSpec:
    if host == "macos":
        return PlatformToolsSpec("darwin", "macosx", "adb")
    if host == "linux":
        return PlatformToolsSpec("linux", "linux", "adb")
    if host == "windows":
        return PlatformToolsSpec("win", "windows", "adb.exe")
    raise SystemExit(f"Unsupported platform-tools host: {host}")


def platform_tools_archive_name(version: str, spec: PlatformToolsSpec) -> str:
    if version == "latest":
        return f"platform-tools-latest-{spec.target}.zip"
    normalized = version if version.startswith("r") else f"r{version}"
    return f"platform-tools_{normalized}-{spec.target}.zip"


def platform_tools_archive_url(version: str, spec: PlatformToolsSpec) -> str:
    return f"{ANDROID_REPOSITORY_BASE_URL}/{platform_tools_archive_name(version, spec)}"


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stage_marker(marker: Path) -> dict[str, object] | None:
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != MARKER_SCHEMA_VERSION:
        return None
    return data


def write_stage_marker(marker: Path, data: dict[str, object]) -> None:
    marker.write_text(
        json.dumps({"schema": MARKER_SCHEMA_VERSION, **data}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_file_hash(path: Path, expected: str, algorithm: str, label: str) -> None:
    actual = hash_file(path, algorithm)
    if actual.lower() != expected.lower():
        raise SystemExit(
            f"{label} checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def node_shasums_url(version: str) -> str:
    return f"https://nodejs.org/dist/{version}/SHASUMS256.txt"


def verify_node_archive_checksum(archive_path: Path, version: str) -> None:
    url = node_shasums_url(version)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            shasums = response.read().decode("utf-8")
    except OSError as exc:
        raise SystemExit(f"Failed to download Node.js checksum manifest from {url}: {exc}") from exc

    expected: str | None = None
    for line in shasums.splitlines():
        checksum, _, filename = line.partition("  ")
        if filename == archive_path.name:
            expected = checksum.strip()
            break
    if not expected:
        raise SystemExit(f"Node.js checksum manifest does not contain {archive_path.name}")
    verify_file_hash(archive_path, expected, "sha256", "Node.js runtime")


def ensure_within_directory(base: Path, target: Path) -> None:
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    if target_resolved != base_resolved and base_resolved not in target_resolved.parents:
        raise SystemExit(f"Archive contains an unsafe path: {target}")


def download_node_archive(version: str, spec: NodeRuntimeSpec, build_dir: Path) -> Path:
    download_dir = build_dir / NODE_DOWNLOAD_DIRNAME
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / node_archive_name(version, spec)
    if archive_path.exists() and archive_path.stat().st_size > 0:
        verify_node_archive_checksum(archive_path, version)
        return archive_path

    url = node_archive_url(version, spec)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    print(f"Downloading Node.js runtime: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except OSError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise SystemExit(f"Failed to download Node.js runtime from {url}: {exc}") from exc
    tmp_path.replace(archive_path)
    verify_node_archive_checksum(archive_path, version)
    return archive_path


def node_runtime_marker_data(version: str, spec: NodeRuntimeSpec, archive_path: Path) -> dict[str, object]:
    return {
        "kind": "node-runtime",
        "version": version,
        "target": spec.target,
        "archive_name": archive_path.name,
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": hash_file(archive_path, "sha256"),
        "source_executable": spec.source_executable,
        "packaged_executable": spec.packaged_executable,
    }


def reusable_node_runtime_stage(stage_root: Path, marker: Path, build_dir: Path, version: str, spec: NodeRuntimeSpec) -> bool:
    executable_path = stage_root / spec.packaged_executable
    if not executable_path.exists():
        return False
    archive_path = build_dir / NODE_DOWNLOAD_DIRNAME / node_archive_name(version, spec)
    if not archive_path.exists() or archive_path.stat().st_size <= 0:
        return False
    return read_stage_marker(marker) == node_runtime_marker_data(version, spec, archive_path)


def extract_node_archive(archive_path: Path, version: str, spec: NodeRuntimeSpec, build_dir: Path) -> Path:
    extract_dir = build_dir / NODE_EXTRACT_DIRNAME
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    if spec.archive_extension == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                ensure_within_directory(extract_dir, extract_dir / member.filename)
            archive.extractall(extract_dir)
    else:
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                ensure_within_directory(extract_dir, extract_dir / member.name)
            try:
                archive.extractall(extract_dir, filter="data")
            except TypeError:
                archive.extractall(extract_dir)

    root = extract_dir / f"node-{version}-{spec.target}"
    if not root.exists():
        raise SystemExit(f"Node.js archive did not contain expected root directory: {root}")
    return root


def stage_node_runtime(build_dir: Path, version: str, host: str) -> Path:
    """Stage the minimal official Node runtime needed by ws-scrcpy."""
    spec = node_runtime_spec(host)
    stage_root = build_dir.resolve() / NODE_RUNTIME_STAGE_DIRNAME
    marker = stage_root / ".codebridge-node-runtime"
    if reusable_node_runtime_stage(stage_root, marker, build_dir, version, spec):
        return stage_root

    archive_path = download_node_archive(version, spec, build_dir)
    extracted_root = extract_node_archive(archive_path, version, spec, build_dir)

    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    source_executable = extracted_root / spec.source_executable
    if not source_executable.exists():
        raise SystemExit(f"Node.js executable not found in archive: {source_executable}")
    destination_executable = stage_root / spec.packaged_executable
    destination_executable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_executable, destination_executable)
    if os.name != "nt":
        destination_executable.chmod(0o755)

    for metadata_name in ("LICENSE", "README.md"):
        source_metadata = extracted_root / metadata_name
        if source_metadata.exists():
            shutil.copy2(source_metadata, stage_root / metadata_name)

    write_stage_marker(marker, node_runtime_marker_data(version, spec, archive_path))
    return stage_root


def stage_node_runtime_into_server(build_dir: Path, staged_server: Path, version: str, host: str) -> Path:
    node_stage = stage_node_runtime(build_dir, version, host)
    destination = staged_server / "vendor" / "node"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(node_stage, destination)
    return destination


def _element_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _element_name(child.tag) == name:
            return child.text
    return None


def _find_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _element_name(child.tag) == name:
            return child
    return None


def parse_revision(element: ET.Element) -> str:
    revision = _find_child(element, "revision")
    if revision is None:
        return "unknown"
    major = _child_text(revision, "major") or "0"
    minor = _child_text(revision, "minor")
    micro = _child_text(revision, "micro")
    parts = [major]
    if minor is not None:
        parts.append(minor)
    if micro is not None:
        parts.append(micro)
    return ".".join(parts)


def resolve_platform_tools_archive_info(version: str, spec: PlatformToolsSpec) -> PlatformToolsArchiveInfo:
    try:
        with urllib.request.urlopen(ANDROID_REPOSITORY_INDEX_URL, timeout=60) as response:
            repository_xml = response.read()
    except OSError as exc:
        raise SystemExit(f"Failed to download Android SDK repository metadata: {exc}") from exc

    root = ET.fromstring(repository_xml)
    packages = [
        package
        for package in root.iter()
        if _element_name(package.tag) == "remotePackage" and package.attrib.get("path") == "platform-tools"
    ]
    if not packages:
        raise SystemExit("Android SDK repository metadata does not contain platform-tools")

    selected: ET.Element | None = None
    for package in packages:
        revision = parse_revision(package)
        if version == "latest" or revision == version:
            selected = package
            break
    if selected is None:
        raise SystemExit(f"Android SDK repository metadata does not contain platform-tools {version}")

    revision = parse_revision(selected)
    archives = _find_child(selected, "archives")
    if archives is None:
        raise SystemExit(f"Android platform-tools {revision} has no archives in repository metadata")

    for archive in archives:
        host_os = _child_text(archive, "host-os")
        if host_os != spec.repository_host_os:
            continue
        complete = _find_child(archive, "complete")
        if complete is None:
            continue
        url = _child_text(complete, "url")
        size_raw = _child_text(complete, "size")
        checksum = None
        for child in complete:
            if _element_name(child.tag) == "checksum" and child.attrib.get("type") == "sha1":
                checksum = (child.text or "").strip()
                break
        if not url or not size_raw or not checksum:
            raise SystemExit(f"Android platform-tools {revision} archive metadata is incomplete")
        return PlatformToolsArchiveInfo(
            revision=revision,
            url=f"{ANDROID_REPOSITORY_BASE_URL}/{url}",
            size=int(size_raw),
            sha1=checksum,
        )

    raise SystemExit(f"Android platform-tools {revision} has no {spec.repository_host_os} archive")


def verify_platform_tools_archive(archive_path: Path, info: PlatformToolsArchiveInfo) -> None:
    actual_size = archive_path.stat().st_size
    if actual_size != info.size:
        raise SystemExit(
            f"Android platform-tools size mismatch for {archive_path.name}: expected {info.size}, got {actual_size}"
        )
    verify_file_hash(archive_path, info.sha1, "sha1", "Android platform-tools")


def platform_tools_marker_data(
    version: str,
    spec: PlatformToolsSpec,
    archive_path: Path,
    archive_info: PlatformToolsArchiveInfo,
    staged_revision: str,
) -> dict[str, object]:
    return {
        "kind": "platform-tools",
        "requested_version": version,
        "target": spec.target,
        "repository_host_os": spec.repository_host_os,
        "archive_revision": archive_info.revision,
        "staged_revision": staged_revision,
        "archive_name": archive_path.name,
        "archive_size": archive_path.stat().st_size,
        "archive_sha1": hash_file(archive_path, "sha1"),
        "metadata_sha1": archive_info.sha1,
        "metadata_size": archive_info.size,
        "executable": spec.executable,
    }


def reusable_platform_tools_stage(stage_root: Path, marker: Path, build_dir: Path, version: str, spec: PlatformToolsSpec) -> bool:
    data = read_stage_marker(marker)
    executable_path = stage_root / spec.executable
    if not data or not executable_path.exists():
        return False
    if data.get("kind") != "platform-tools" or data.get("requested_version") != version or data.get("target") != spec.target:
        return False
    archive_name = data.get("archive_name")
    archive_sha1 = data.get("archive_sha1")
    archive_size = data.get("archive_size")
    if not isinstance(archive_name, str) or not isinstance(archive_sha1, str) or not isinstance(archive_size, int):
        return False
    archive_path = build_dir / NODE_DOWNLOAD_DIRNAME / archive_name
    if not archive_path.exists() or archive_path.stat().st_size != archive_size:
        return False
    return hash_file(archive_path, "sha1").lower() == archive_sha1.lower()


def download_platform_tools_archive(version: str, spec: PlatformToolsSpec, build_dir: Path) -> tuple[Path, PlatformToolsArchiveInfo]:
    info = resolve_platform_tools_archive_info(version, spec)
    download_dir = build_dir / NODE_DOWNLOAD_DIRNAME
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / Path(info.url).name
    if archive_path.exists() and archive_path.stat().st_size > 0:
        verify_platform_tools_archive(archive_path, info)
        return archive_path, info

    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    print(f"Downloading Android platform-tools: {info.url}")
    try:
        with urllib.request.urlopen(info.url, timeout=60) as response, tmp_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except OSError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        if archive_path.exists() and archive_path.stat().st_size > 0:
            print(f"Using cached Android platform-tools archive after download failure: {archive_path}")
            verify_platform_tools_archive(archive_path, info)
            return archive_path, info
        raise SystemExit(f"Failed to download Android platform-tools from {info.url}: {exc}") from exc
    tmp_path.replace(archive_path)
    verify_platform_tools_archive(archive_path, info)
    return archive_path, info


def extract_platform_tools_archive(archive_path: Path, build_dir: Path) -> Path:
    extract_dir = build_dir / PLATFORM_TOOLS_EXTRACT_DIRNAME
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            ensure_within_directory(extract_dir, extract_dir / member.filename)
        archive.extractall(extract_dir)

    root = extract_dir / "platform-tools"
    if not root.exists():
        raise SystemExit(f"Android platform-tools archive did not contain expected directory: {root}")
    return root


def read_platform_tools_revision(platform_tools_root: Path) -> str:
    source_properties = platform_tools_root / "source.properties"
    if not source_properties.exists():
        return "unknown"
    for line in source_properties.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "Pkg.Revision":
            return value.strip() or "unknown"
    return "unknown"


def stage_platform_tools(build_dir: Path, version: str, host: str) -> Path:
    """Stage official Android platform-tools so adb is available offline."""
    spec = platform_tools_spec(host)
    stage_root = build_dir.resolve() / PLATFORM_TOOLS_STAGE_DIRNAME
    marker = stage_root / ".codebridge-platform-tools"
    if reusable_platform_tools_stage(stage_root, marker, build_dir, version, spec):
        return stage_root

    archive_path, archive_info = download_platform_tools_archive(version, spec, build_dir)
    extracted_root = extract_platform_tools_archive(archive_path, build_dir)

    if stage_root.exists():
        shutil.rmtree(stage_root)
    shutil.copytree(extracted_root, stage_root)
    executable_path = stage_root / spec.executable
    if not executable_path.exists():
        raise SystemExit(f"adb executable not found in Android platform-tools archive: {executable_path}")
    if os.name != "nt":
        executable_path.chmod(0o755)
    revision = read_platform_tools_revision(stage_root)
    write_stage_marker(marker, platform_tools_marker_data(version, spec, archive_path, archive_info, revision))
    return stage_root


def stage_platform_tools_into_server(build_dir: Path, staged_server: Path, version: str, host: str) -> Path:
    platform_tools_stage = stage_platform_tools(build_dir, version, host)
    destination = staged_server / "vendor" / "platform-tools"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(platform_tools_stage, destination)
    return destination


def matches_packaged_exclude_pattern(name: str) -> bool:
    normalized = name.lower()
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in PACKAGED_EXCLUDE_PATTERNS)


def should_copy_packaged_path(path: Path, *, excluded_dirs: set[str]) -> bool:
    if path.name in SERVER_EXCLUDE_FILENAMES:
        return False
    if path.name in excluded_dirs:
        return False
    if matches_packaged_exclude_pattern(path.name):
        return False
    if path.suffix in SERVER_EXCLUDE_SUFFIXES:
        return False
    return True


def stage_server_tree(build_dir: Path) -> Path:
    """Copy server source/assets to a packaging stage without runtime state."""
    stage_root = build_dir.resolve() / SERVER_STAGE_DIRNAME
    if stage_root.exists():
        shutil.rmtree(stage_root)

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        base = Path(directory)
        for name in names:
            if not should_copy_packaged_path(base / name, excluded_dirs=SERVER_EXCLUDE_DIRS):
                ignored.add(name)
        return ignored

    shutil.copytree(SERVER_DIR, stage_root, ignore=ignore)
    return stage_root


def stage_scrcpy_dist(build_dir: Path) -> Path:
    """Copy scrcpy dist runtime assets through the same package hardening filter."""
    source_root = SERVER_DIR / "scrcpy" / "dist"
    stage_root = build_dir.resolve() / SCRCPY_DIST_STAGE_DIRNAME
    if stage_root.exists():
        shutil.rmtree(stage_root)
    if not source_root.exists():
        return stage_root

    excluded_dirs = SERVER_EXCLUDE_DIRS - {"scrcpy"}

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        base = Path(directory)
        for name in names:
            if not should_copy_packaged_path(base / name, excluded_dirs=excluded_dirs):
                ignored.add(name)
        return ignored

    shutil.copytree(source_root, stage_root, ignore=ignore)
    guard_root = SERVER_DIR / "devices" / "tango"
    for name, target in (
        ("tango-server.mjs", stage_root / "tango-server.mjs"),
        ("TangoScrcpyService.mjs", stage_root / "src/server/goog-device/tango/TangoScrcpyService.mjs"),
        ("control_decoder.mjs", stage_root / "src/server/goog-device/tango/control_decoder.mjs"),
    ):
        if not (guard_root / name).is_file():
            raise FileNotFoundError(f"Tango device guard source missing: {guard_root / name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(guard_root / name, target)
    return stage_root


def collect_data_items(include_scrcpy: bool, staged_server: Path) -> list[DataItem]:
    items = [
        DataItem(staged_server, "server"),
    ]
    if include_scrcpy:
        items.append(DataItem(staged_server.parent / SCRCPY_DIST_STAGE_DIRNAME, "server/scrcpy/dist", required=False))
    items.extend(
        [
            DataItem(REPO_ROOT / "assets" / "app_icon_512.png", "assets", required=False),
            DataItem(REPO_ROOT / "assets" / "app_icon.png", "assets", required=False),
        ]
    )
    return items


def existing_data_args(items: list[DataItem]) -> list[str]:
    args: list[str] = []
    missing_required: list[Path] = []
    for item in items:
        if item.source.exists():
            args.append(add_data_arg(item.source, item.destination))
        elif item.required:
            missing_required.append(item.source)
    if missing_required:
        joined = "\n".join(f"  - {path}" for path in missing_required)
        raise SystemExit(f"Required packaged asset path(s) are missing:\n{joined}")
    return args


def generated_icon_from_png(host: str, build_dir: Path) -> Optional[Path]:
    source = next(
        (
            path
            for path in [
                REPO_ROOT / "assets" / "app_icon_512.png",
                REPO_ROOT / "assets" / "app_icon.png",
                REPO_ROOT / "web" / "icons" / "Icon-512.png",
            ]
            if path.exists()
        ),
        None,
    )
    if source is None:
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    icon_dir = build_dir.resolve() / "generated_icons"
    icon_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGBA")

    if host == "macos":
        icon_path = icon_dir / "app_icon.icns"
        image.save(
            icon_path,
            format="ICNS",
            sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)],
        )
        return icon_path
    if host == "windows":
        icon_path = icon_dir / "app_icon.ico"
        image.save(
            icon_path,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        return icon_path
    if host == "linux":
        icon_path = icon_dir / "app_icon.png"
        image.resize((512, 512)).save(icon_path)
        return icon_path
    return None


def resolve_icon(host: str, explicit_icon: Optional[Path], build_dir: Path) -> Optional[Path]:
    if explicit_icon:
        if not explicit_icon.exists():
            raise SystemExit(f"Icon file does not exist: {explicit_icon}")
        return explicit_icon
    candidates = {
        "macos": [REPO_ROOT / "assets" / "app_icon.icns"],
        "windows": [REPO_ROOT / "assets" / "app_icon.ico"],
        "linux": [REPO_ROOT / "assets" / "app_icon.png", REPO_ROOT / "assets" / "app_icon_512.png"],
    }
    existing = next((path for path in candidates.get(host, []) if path.exists()), None)
    return existing or generated_icon_from_png(host, build_dir)


def pyinstaller_args(args: argparse.Namespace, build_format: str, host: str) -> list[str]:
    dist_dir = args.dist_dir.resolve()
    build_dir = args.build_dir.resolve()
    work_path = build_dir / "work"
    spec_path = build_dir / "spec"
    entry = args.entry.resolve()
    icon = resolve_icon(host, args.icon.resolve() if args.icon else None, build_dir)
    staged_server = stage_server_tree(build_dir)
    if args.include_node_runtime:
        if args.dry_run:
            cached_node_runtime = build_dir.resolve() / NODE_RUNTIME_STAGE_DIRNAME
            node_spec = node_runtime_spec(host)
            node_marker = cached_node_runtime / ".codebridge-node-runtime"
            if reusable_node_runtime_stage(cached_node_runtime, node_marker, build_dir, args.node_version, node_spec):
                destination = staged_server / "vendor" / "node"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(cached_node_runtime, destination)
            else:
                print("Skipping Node.js runtime download during --dry-run; no validated cache is available.")
        else:
            stage_node_runtime_into_server(build_dir, staged_server, args.node_version, host)
    if args.include_platform_tools:
        if args.dry_run:
            cached_platform_tools = build_dir.resolve() / PLATFORM_TOOLS_STAGE_DIRNAME
            platform_tools_spec_for_host = platform_tools_spec(host)
            platform_tools_marker = cached_platform_tools / ".codebridge-platform-tools"
            if reusable_platform_tools_stage(
                cached_platform_tools,
                platform_tools_marker,
                build_dir,
                args.platform_tools_version,
                platform_tools_spec_for_host,
            ):
                destination = staged_server / "vendor" / "platform-tools"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(cached_platform_tools, destination)
            else:
                print("Skipping Android platform-tools download during --dry-run; no validated cache is available.")
        else:
            stage_platform_tools_into_server(build_dir, staged_server, args.platform_tools_version, host)
    if args.include_scrcpy:
        stage_scrcpy_dist(build_dir)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        args.name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
        # Analyse the *staged* tree, not the repo's server/. The repo tree still
        # holds code_bridge_core as source, and pointing the analysis there put
        # every one of its modules into the PYZ as bytecode — the 2.1.0 DMG
        # shipped that way, readable, while the compiled wheel was left out.
        "--paths",
        str(staged_server),
        # Both closed packages are single extension modules in site-packages
        # (Nuitka `--module`); nothing in the analysed tree names them by a
        # static import PyInstaller can follow, so they must be asked for.
        "--hidden-import",
        "agent_flow_core",
        "--hidden-import",
        "code_bridge_core",
        "--collect-submodules",
        "fastapi",
        "--collect-submodules",
        "uvicorn",
        "--collect-submodules",
        "websockets",
        "--collect-submodules",
        "starlette",
        "--collect-submodules",
        "pydantic",
        "--collect-submodules",
        "qrcode",
        "--collect-submodules",
        "PIL",
        "--collect-submodules",
        "pystray",
        "--hidden-import",
        "yaml",
        "--hidden-import",
        "jwt",
        "--hidden-import",
        "httpx",
        "--hidden-import",
        "multipart",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.protocols.websockets.auto",
    ]
    if host == "macos":
        command.extend(
            [
                "--hidden-import",
                "pystray._darwin",
                "--hidden-import",
                "AppKit",
                "--hidden-import",
                "Foundation",
                "--hidden-import",
                "objc",
            ]
        )
    elif host == "windows":
        command.extend(["--hidden-import", "pystray._win32"])

    if build_format in {"exe", "msi"} and args.onefile:
        command.append("--onefile")
    else:
        command.append("--onedir")

    if build_format in {"app", "dmg"}:
        command.extend(["--windowed", "--osx-bundle-identifier", args.bundle_identifier])
    elif build_format in {"exe", "msi"}:
        command.append("--windowed")
    elif args.windowed:
        command.append("--windowed")
    else:
        command.append("--console")

    if icon:
        command.extend(["--icon", str(icon)])

    if not args.include_scrcpy:
        print("Skipping server/scrcpy/dist. Use --include-scrcpy to package device mirror assets.")

    for data_arg in existing_data_args(collect_data_items(args.include_scrcpy, staged_server)):
        command.extend(["--add-data", data_arg])

    command.append(str(entry))
    return command


def run(command: list[str], *, cwd: Path) -> None:
    printable = shlex.join(command)
    print(f"\n$ {printable}\n")
    subprocess.run(command, cwd=cwd, check=True)


CLOSED_PACKAGES = ("agent_flow_core", "code_bridge_core")


def frozen_executable(dist_dir: Path, app_name: str, build_format: str) -> Path:
    if build_format in {"app", "dmg"}:
        return dist_dir / f"{app_name}.app" / "Contents" / "MacOS" / app_name
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    return dist_dir / app_name / f"{app_name}{suffix}"


def verify_frozen_artifact_closed(dist_dir: Path, app_name: str, build_format: str) -> None:
    """Ask the frozen artifact, not the build environment, whether it is closed.

    `verify_kernel_is_compiled` checks the interpreter the build runs from.
    That was not enough: the 2.1.0 DMG was built from an environment that
    passed it, and still carried every `code_bridge_core` module as bytecode
    in the PYZ (the analysis path pointed at the repo's source tree) while the
    compiled extensions were absent (no hidden import named them). The app
    then failed on `import agent_flow_core` at first launch. Two checks, both
    against what PyInstaller actually produced:

    1. No pure-Python entry for either closed package in the PYZ archive.
    2. One extension module file per closed package in the frozen tree.
    """
    exe = frozen_executable(dist_dir, app_name, build_format)
    if not exe.exists():
        raise SystemExit(f"Frozen executable not found, cannot verify the artifact: {exe}")
    listing = subprocess.run(
        [sys.executable, "-m", "PyInstaller.utils.cliutils.archive_viewer", "-l", "-r", str(exe)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # archive_viewer prints PYZ entries as "<type>, <offset>, <length>, '<name>'";
    # type 0 is a module, 1 a package — both are bytecode a reader can recover.
    leaked = sorted(
        {
            match.group(1)
            for match in re.finditer(r"^\s*[01],\s*\d+,\s*\d+,\s*'((?:%s)(?:\.[\w.]+)?)'" % "|".join(CLOSED_PACKAGES), listing, re.M)
        }
    )
    if leaked:
        raise SystemExit(
            "The frozen app contains closed-package modules as bytecode:\n"
            + "".join(f"    {name}\n" for name in leaked[:12])
            + ("    ...\n" if len(leaked) > 12 else "")
            + "The analysis found their source. Check `--paths` and the staged tree."
        )
    root = exe.parent.parent if build_format in {"app", "dmg"} else exe.parent
    for package in CLOSED_PACKAGES:
        found = list(root.rglob(f"{package}.*.so")) + list(root.rglob(f"{package}.*.pyd"))
        if not found:
            raise SystemExit(
                f"The frozen app has no compiled extension for {package}. "
                "It is installed in the build environment but PyInstaller did not collect it; "
                "the hidden import for it is missing or the wheel does not match this interpreter."
            )
    print(f"Closed packages verified in the frozen artifact: {', '.join(CLOSED_PACKAGES)}")


def create_dmg(dist_dir: Path, app_name: str) -> None:
    app_path = dist_dir / f"{app_name}.app"
    if not app_path.exists():
        raise SystemExit(f"Cannot create DMG because app bundle was not produced: {app_path}")
    dmg_path = dist_dir / f"{app_name}.dmg"
    if dmg_path.exists():
        dmg_path.unlink()
    run(
        [
            "hdiutil",
            "create",
            "-volname",
            app_name,
            "-srcfolder",
            str(app_path),
            "-ov",
            "-format",
            "UDZO",
            str(dmg_path),
        ],
        cwd=REPO_ROOT,
    )


ENTITLEMENTS = REPO_ROOT / "desktop_server_app" / "macos_entitlements.plist"


def sign_macos_app(app_path: Path, identity: str | None) -> None:
    """Sign the bundle — ad-hoc for development, Developer ID for release.

    The ad-hoc signature (`--sign -`) this used to hardcode is not a
    distribution signature. It satisfies the loader on the machine that built
    it and nothing else: Gatekeeper refuses it everywhere, and notarization
    will not accept it. So a real identity has to be asked for, and when one
    is given three other things change with it.

    **Hardened runtime** (`--options runtime`) is not optional — notarization
    rejects a bundle without it. It is also what makes the entitlements file
    matter, which is why `desktop_server_app/macos_entitlements.plist` had sat
    in this repo unreferenced: nothing signed with a runtime to apply it to.

    **Inside-out, not `--deep`.** `--deep` is Apple's own "for emergency use
    only" flag and it silently skips nested code in ways that surface as a
    notarization rejection long after the build. A PyInstaller bundle is full
    of nested Mach-O — every `.so`, every `.dylib`, the vendored Node runtime,
    `adb` — and each has to carry its own signature, innermost first, because
    signing the outer bundle seals whatever is inside it at that moment.

    **Existing signatures are replaced.** Vendored binaries arrive signed by
    whoever shipped them (Node, platform-tools), and a Developer ID bundle
    cannot contain foreign signatures it did not re-sign, so `--force` is
    deliberate rather than convenient.
    """

    if not shutil.which("codesign"):
        print("codesign is unavailable; the bundle is left unsigned.")
        return

    if identity is None:
        run(["codesign", "--force", "--sign", "-", "--deep", str(app_path)], cwd=REPO_ROOT)
        print(
            "Signed ad-hoc. This artifact is NOT distributable — Gatekeeper "
            "refuses it on any other machine. Pass --sign-identity for a "
            "release build."
        )
        return

    if not ENTITLEMENTS.exists():
        raise SystemExit(
            f"Hardened runtime needs an entitlements file and it is missing: "
            f"{ENTITLEMENTS}\nA Python bundle signed without "
            "`allow-unsigned-executable-memory` crashes on launch."
        )

    base = [
        "codesign",
        "--force",
        "--timestamp",
        "--options",
        "runtime",
        "--entitlements",
        str(ENTITLEMENTS),
        "--sign",
        identity,
    ]

    inner = sorted(macho_files(app_path), key=lambda path: len(path.parts), reverse=True)
    print(f"Signing {len(inner)} nested binaries, then the bundle.")
    for path in inner:
        run([*base, str(path)], cwd=REPO_ROOT)
    run([*base, str(app_path)], cwd=REPO_ROOT)

    # Asked of the artifact, not assumed from the commands: `--verify --strict`
    # walks what was actually produced and checks every nested signature.
    #
    # Gatekeeper's own verdict (`spctl --assess`) is deliberately NOT asked
    # here. A Developer ID bundle that has not been notarized yet is *supposed*
    # to be rejected — "source=Unnotarized Developer ID" — so asking at this
    # point fails every correct release build at the step before the one that
    # fixes it. It is asked after stapling instead, where a rejection means
    # something.
    run(["codesign", "--verify", "--strict", "--verbose=2", str(app_path)], cwd=REPO_ROOT)


def macho_files(root: Path) -> list[Path]:
    """Every Mach-O binary inside the bundle.

    Found by reading each file's magic number rather than by extension. A
    PyInstaller bundle carries executables with no suffix at all (the vendored
    `node`, `adb`), and an extension allow-list would leave exactly those
    unsigned — which notarization reports as a rejected nested binary, not as
    a missing file.
    """
    magics = {
        b"\xcf\xfa\xed\xfe",  # 64-bit little-endian
        b"\xce\xfa\xed\xfe",  # 32-bit little-endian
        b"\xca\xfe\xba\xbe",  # universal (fat)
        b"\xbe\xba\xfe\xca",  # universal, byte-swapped
    }
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                if handle.read(4) in magics:
                    found.append(path)
        except OSError:
            continue
    return found


def notarize_macos_artifact(artifact: Path, profile: str) -> None:
    """Submit to Apple, wait, and staple the ticket onto the artifact.

    Stapling is the step that is easy to skip and expensive to miss: without
    it the artifact only opens on a machine that can reach Apple to check, so
    it works on the build machine and fails for the first person who opens it
    offline.

    The credentials live in a keychain profile rather than being passed here.
    That is Apple's own mechanism (`notarytool store-credentials`) and it
    keeps the App Store Connect key out of this script, out of its arguments,
    and out of any shell history.
    """

    if not shutil.which("xcrun"):
        raise SystemExit("xcrun is unavailable, so the artifact cannot be notarized.")

    probe = subprocess.run(
        ["xcrun", "notarytool", "history", "--keychain-profile", profile],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"No notarization credentials are stored under the profile "
            f"'{profile}'.\n"
            "Store them once, on this machine:\n"
            "    xcrun notarytool store-credentials \"" + profile + "\" \\\n"
            "        --key <AuthKey_XXXXXXXX.p8> \\\n"
            "        --key-id <the key id from the file name> \\\n"
            "        --issuer <issuer UUID from App Store Connect > Keys>\n"
            "The key and issuer belong to your Apple Developer account; this "
            "script never handles them."
        )

    print(f"Submitting {artifact.name} to Apple. This takes a few minutes.")
    run(
        [
            "xcrun",
            "notarytool",
            "submit",
            str(artifact),
            "--keychain-profile",
            profile,
            "--wait",
        ],
        cwd=REPO_ROOT,
    )
    run(["xcrun", "stapler", "staple", str(artifact)], cwd=REPO_ROOT)
    run(["xcrun", "stapler", "validate", str(artifact)], cwd=REPO_ROOT)
    # Now Gatekeeper's answer means something: this is the same check the
    # first person to download the artifact triggers, and the only point in
    # the build where a pass is evidence rather than luck.
    if shutil.which("spctl"):
        run(["spctl", "--assess", "--type", "open", "--context", "context:primary-signature",
             "--verbose=2", str(artifact)], cwd=REPO_ROOT)


def default_signing_identity() -> str | None:
    """The Developer ID Application identity on this machine, if there is one.

    Reported rather than used automatically. Signing for distribution is a
    decision, and a build that silently picked up whichever certificate
    happened to be installed would make release and development builds
    indistinguishable from the command line.
    """

    if not shutil.which("security"):
        return None
    found = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        capture_output=True,
        text=True,
    )
    for line in found.stdout.splitlines():
        if "Developer ID Application" in line:
            start = line.find('"')
            end = line.rfind('"')
            if start != -1 and end > start:
                return line[start + 1 : end]
    return None


def patch_macos_menu_bar_bundle(
    dist_dir: Path, app_name: str, build_format: str, identity: str | None
) -> None:
    if build_format not in {"app", "dmg"} or platform.system().lower() != "darwin":
        return

    app_path = dist_dir / f"{app_name}.app"
    plist_path = app_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        raise SystemExit(f"Cannot patch macOS app Info.plist because it was not produced: {plist_path}")

    with plist_path.open("rb") as fh:
        info = plistlib.load(fh)
    info["LSUIElement"] = True
    # Same env var the MSI already uses for its ProductVersion. Without it
    # PyInstaller leaves 0.0.0 in the bundle, which is what the 2026-08-30
    # build shipped in "About" and what the notarization log then named.
    desktop_version = os.environ.get("CODEBRIDGE_DESKTOP_VERSION")
    if desktop_version:
        info["CFBundleShortVersionString"] = desktop_version
        info["CFBundleVersion"] = desktop_version
    with plist_path.open("wb") as fh:
        plistlib.dump(info, fh)

    sign_macos_app(app_path, identity)


def restore_macos_scrcpy_runtime_files(dist_dir: Path, app_name: str, include_scrcpy: bool, build_dir: Path) -> None:
    """Keep Node-loadable native addons in their original Resources paths.

    PyInstaller may reclassify Mach-O files from --add-data into
    Contents/Frameworks. That is good for its own loader, but ws-scrcpy's Node
    module resolution expects native addons such as pty.node at the original
    node_modules path under server/scrcpy/dist.
    """
    if not include_scrcpy or platform.system().lower() != "darwin":
        return
    app_path = dist_dir / f"{app_name}.app"
    resources_dist = app_path / "Contents" / "Resources" / "server" / "scrcpy" / "dist"
    source_dist = build_dir.resolve() / SCRCPY_DIST_STAGE_DIRNAME
    if not resources_dist.exists() or not source_dist.exists():
        return

    native_suffixes = {".node", ".dylib", ".so"}
    for source in source_dist.rglob("*"):
        if not source.is_file() or source.suffix not in native_suffixes:
            continue
        destination = resources_dist / source.relative_to(source_dist)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def restore_macos_node_runtime_files(dist_dir: Path, app_name: str, include_node_runtime: bool, build_dir: Path) -> None:
    """Keep the bundled Node executable at the runtime Resources path."""
    if not include_node_runtime or platform.system().lower() != "darwin":
        return

    app_path = dist_dir / f"{app_name}.app"
    resources_node = app_path / "Contents" / "Resources" / "server" / "vendor" / "node"
    source_node = build_dir.resolve() / SERVER_STAGE_DIRNAME / "vendor" / "node"
    if not resources_node.exists() or not source_node.exists():
        return

    executable_rel = Path("bin") / "node"
    source_executable = source_node / executable_rel
    destination_executable = resources_node / executable_rel
    if source_executable.exists():
        destination_executable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_executable, destination_executable)
        destination_executable.chmod(0o755)


def restore_macos_platform_tools_files(
    dist_dir: Path,
    app_name: str,
    include_platform_tools: bool,
    build_dir: Path,
) -> None:
    """Keep adb available at the runtime Resources path after PyInstaller reclassification."""
    if not include_platform_tools or platform.system().lower() != "darwin":
        return

    app_path = dist_dir / f"{app_name}.app"
    resources_platform_tools = app_path / "Contents" / "Resources" / "server" / "vendor" / "platform-tools"
    source_platform_tools = build_dir.resolve() / SERVER_STAGE_DIRNAME / "vendor" / "platform-tools"
    if not resources_platform_tools.exists() or not source_platform_tools.exists():
        return

    for source in source_platform_tools.rglob("*"):
        if not source.is_file():
            continue
        destination = resources_platform_tools / source.relative_to(source_platform_tools)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    adb_path = resources_platform_tools / "adb"
    if adb_path.exists():
        adb_path.chmod(0o755)


def wix_id(prefix: str, value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.]", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = uuid.uuid5(MSI_UPGRADE_CODE_NAMESPACE, value).hex[:12]
    if not re.match(r"[A-Za-z_]", cleaned[0]):
        cleaned = f"_{cleaned}"
    return f"{prefix}_{cleaned}"[:72]


def wix_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def package_version(raw: str) -> str:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    return match.group(0) if match else "1.0.0"


def generate_wix_source(source_dir: Path, wxs_path: Path, app_name: str, version: str) -> None:
    """Generate a WiX v4 source file from the PyInstaller one-folder output."""
    ET.register_namespace("", WIX_NAMESPACE)
    ns = f"{{{WIX_NAMESPACE}}}"

    root = ET.Element(f"{ns}Wix")
    package = ET.SubElement(
        root,
        f"{ns}Package",
        {
            "Name": app_name,
            "Manufacturer": "Code Bridge",
            "Version": package_version(version),
            "UpgradeCode": str(uuid.uuid5(MSI_UPGRADE_CODE_NAMESPACE, app_name)),
            "Scope": "perMachine",
        },
    )
    ET.SubElement(
        package,
        f"{ns}MajorUpgrade",
        {"DowngradeErrorMessage": f"A newer version of {app_name} is already installed."},
    )
    ET.SubElement(package, f"{ns}MediaTemplate", {"EmbedCab": "yes"})

    program_files = ET.SubElement(package, f"{ns}StandardDirectory", {"Id": "ProgramFiles64Folder"})
    install_dir = ET.SubElement(program_files, f"{ns}Directory", {"Id": "INSTALLFOLDER", "Name": app_name})
    program_menu = ET.SubElement(package, f"{ns}StandardDirectory", {"Id": "ProgramMenuFolder"})
    shortcut_dir = ET.SubElement(program_menu, f"{ns}Directory", {"Id": "ApplicationProgramsFolder", "Name": "Code Bridge"})
    component_group = ET.SubElement(package, f"{ns}ComponentGroup", {"Id": "AppComponents"})

    directories: dict[Path, ET.Element] = {Path("."): install_dir}
    used_ids: set[str] = {"INSTALLFOLDER", "ApplicationProgramsFolder", "AppComponents"}

    def unique_id(prefix: str, value: str) -> str:
        base = wix_id(prefix, value)
        candidate = base
        index = 2
        while candidate in used_ids:
            suffix = f"_{index}"
            candidate = f"{base[: 72 - len(suffix)]}{suffix}"
            index += 1
        used_ids.add(candidate)
        return candidate

    shortcut_component_id = unique_id("CMP", "start-menu-shortcut")
    shortcut_component = ET.SubElement(
        shortcut_dir,
        f"{ns}Component",
        {"Id": shortcut_component_id, "Guid": str(uuid.uuid5(MSI_UPGRADE_CODE_NAMESPACE, "start-menu-shortcut"))},
    )
    ET.SubElement(
        shortcut_component,
        f"{ns}Shortcut",
        {
            "Id": unique_id("SC", "start-menu-shortcut"),
            "Name": app_name,
            "Description": f"Launch {app_name}",
            "Target": f"[INSTALLFOLDER]{app_name}.exe",
            "WorkingDirectory": "INSTALLFOLDER",
        },
    )
    ET.SubElement(shortcut_component, f"{ns}RemoveFolder", {"Id": "ApplicationProgramsFolder", "On": "uninstall"})
    ET.SubElement(
        shortcut_component,
        f"{ns}RegistryValue",
        {
            "Root": "HKCU",
            "Key": rf"Software\Code Bridge\{app_name}",
            "Name": "installed",
            "Type": "integer",
            "Value": "1",
            "KeyPath": "yes",
        },
    )
    ET.SubElement(component_group, f"{ns}ComponentRef", {"Id": shortcut_component_id})

    for directory in sorted((path for path in source_dir.rglob("*") if path.is_dir()), key=lambda path: path.parts):
        rel = directory.relative_to(source_dir)
        parent_rel = rel.parent if rel.parent != Path("") else Path(".")
        parent_element = directories[parent_rel]
        directory_id = unique_id("DIR", rel.as_posix())
        directories[rel] = ET.SubElement(parent_element, f"{ns}Directory", {"Id": directory_id, "Name": directory.name})

    files = sorted((path for path in source_dir.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    if not files:
        raise SystemExit(f"Cannot create MSI because app folder has no files: {source_dir}")

    for file_path in files:
        rel = file_path.relative_to(source_dir)
        directory_element = directories.get(rel.parent if rel.parent != Path("") else Path("."))
        if directory_element is None:
            raise SystemExit(f"Internal MSI packaging error. Missing directory for {file_path}")
        component_id = unique_id("CMP", rel.as_posix())
        file_id = unique_id("FIL", rel.as_posix())
        component = ET.SubElement(
            directory_element,
            f"{ns}Component",
            {"Id": component_id, "Guid": str(uuid.uuid5(MSI_UPGRADE_CODE_NAMESPACE, rel.as_posix()))},
        )
        ET.SubElement(
            component,
            f"{ns}File",
            {"Id": file_id, "Source": wix_path(file_path.resolve()), "KeyPath": "yes"},
        )
        ET.SubElement(component_group, f"{ns}ComponentRef", {"Id": component_id})

    feature = ET.SubElement(package, f"{ns}Feature", {"Id": "DefaultFeature", "Title": app_name, "Level": "1"})
    ET.SubElement(feature, f"{ns}ComponentGroupRef", {"Id": "AppComponents"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    wxs_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(wxs_path, encoding="utf-8", xml_declaration=True)


def create_msi(dist_dir: Path, app_name: str) -> None:
    """Create an MSI from the PyInstaller one-folder output when WiX is installed."""
    if platform.system().lower() != "windows":
        raise SystemExit("MSI builds must be run on Windows.")
    source_dir = dist_dir / app_name
    if not source_dir.exists():
        raise SystemExit(f"Cannot create MSI because app folder was not produced: {source_dir}")
    wix = os.environ.get("CODEBRIDGE_WIX_PATH") or shutil.which("wix")
    if not wix:
        raise SystemExit("WiX v4 `wix` command was not found. Install WiX Toolset to build MSI.")
    if not Path(wix).exists():
        raise SystemExit(f"CODEBRIDGE_WIX_PATH does not exist: {wix}")
    wxs_path = REPO_ROOT / "build" / "desktop_server_app" / "installer.generated.wxs"
    generate_wix_source(source_dir, wxs_path, app_name, os.environ.get("CODEBRIDGE_DESKTOP_VERSION", "1.0.0"))
    run(
        [
            wix,
            "build",
            "-arch",
            "x64",
            "-out",
            str(dist_dir / f"{app_name}.msi"),
            str(wxs_path),
        ],
        cwd=REPO_ROOT,
    )


def default_format(host: str) -> str:
    if host == "macos":
        return "app"
    if host == "windows":
        return "exe"
    return "onedir"


def parse_args() -> argparse.Namespace:
    host = host_platform()
    parser = argparse.ArgumentParser(
        description="Build Code Bridge desktop server packages with PyInstaller.",
    )
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY, help="Python entry point to package.")
    parser.add_argument("--name", default=APP_NAME, help="Application/executable name.")
    parser.add_argument(
        "--format",
        choices=["auto", "app", "dmg", "exe", "msi", "onedir"],
        default="auto",
        help="Output format. Cross-compilation is not supported by PyInstaller.",
    )
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST_DIR, help="PyInstaller dist output.")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="PyInstaller build/spec output.")
    parser.add_argument("--icon", type=Path, default=None, help="Optional .icns/.ico/.png icon path.")
    parser.add_argument("--bundle-identifier", default=BUNDLE_IDENTIFIER, help="macOS bundle identifier.")
    parser.add_argument("--include-scrcpy", action="store_true", help="Package server/scrcpy/dist assets if present.")
    parser.add_argument(
        "--sign-identity",
        default=None,
        metavar="IDENTITY",
        help=(
            "Developer ID Application identity to sign with, enabling the "
            "hardened runtime. Pass --sign-identity auto to use the one "
            "installed on this machine. Omitted: ad-hoc signature, which is "
            "fine for development and not distributable."
        ),
    )
    parser.add_argument(
        "--notarize",
        default=None,
        metavar="PROFILE",
        help=(
            "notarytool keychain profile to submit and staple with. Requires "
            "--sign-identity: Apple rejects an unsigned or ad-hoc bundle."
        ),
    )
    parser.add_argument(
        "--include-node-runtime",
        dest="include_node_runtime",
        action="store_true",
        default=None,
        help="Package the official Node.js runtime used by server/scrcpy.",
    )
    parser.add_argument(
        "--skip-node-runtime",
        dest="include_node_runtime",
        action="store_false",
        help="Do not package Node.js even when --include-scrcpy is enabled.",
    )
    parser.add_argument("--node-version", default=DEFAULT_NODE_VERSION, help="Official Node.js version to package.")
    parser.add_argument(
        "--include-platform-tools",
        dest="include_platform_tools",
        action="store_true",
        default=None,
        help="Package official Android platform-tools so adb is available.",
    )
    parser.add_argument(
        "--skip-platform-tools",
        dest="include_platform_tools",
        action="store_false",
        help="Do not package Android platform-tools even when --include-scrcpy is enabled.",
    )
    parser.add_argument(
        "--platform-tools-version",
        default=DEFAULT_PLATFORM_TOOLS_VERSION,
        help="Android platform-tools revision to package, or 'latest'.",
    )
    parser.add_argument("--onefile", action="store_true", help="Use one-file mode for Windows exe builds.")
    parser.add_argument("--windowed", action="store_true", help="Hide console for Windows/Linux builds.")
    parser.add_argument("--dry-run", action="store_true", help="Print the PyInstaller command without running it.")
    parser.set_defaults(host=host)
    args = parser.parse_args()
    if args.include_node_runtime is None:
        args.include_node_runtime = args.include_scrcpy and host in {"macos", "windows", "linux"}
    if args.include_platform_tools is None:
        args.include_platform_tools = args.include_scrcpy and host in {"macos", "windows", "linux"}
    return args


def main() -> None:
    args = parse_args()
    host = args.host
    build_format = default_format(host) if args.format == "auto" else args.format

    if build_format in {"app", "dmg"} and host != "macos":
        raise SystemExit("macOS .app/.dmg builds must be run on macOS.")
    if build_format == "exe" and host != "windows":
        raise SystemExit("Windows exe builds must be run on Windows.")
    if build_format == "msi" and host != "windows":
        raise SystemExit("Windows MSI builds must be run on Windows.")
    if build_format == "onedir" and host not in {"linux", "macos", "windows"}:
        raise SystemExit(f"Unsupported one-folder build host: {host}")

    if args.sign_identity == "auto":
        args.sign_identity = default_signing_identity()
        if args.sign_identity is None:
            raise SystemExit(
                "--sign-identity auto found no Developer ID Application "
                "certificate in this keychain. Install one from your Apple "
                "Developer account, or name an identity explicitly."
            )
        print(f"Signing with: {args.sign_identity}")
    # Refused here rather than after the build: notarization is the last step
    # and Apple would reject the submission, minutes later, for a reason that
    # was knowable before anything was compiled.
    if args.notarize and not args.sign_identity:
        raise SystemExit(
            "--notarize needs --sign-identity. Apple rejects an unsigned or "
            "ad-hoc bundle, so submitting one only wastes the round trip."
        )

    validate_inputs(args.entry.resolve())
    if not args.dry_run:
        check_required_tools(build_format)
        # Before anything is packaged, not after: a build that discovers this
        # at the end has already spent several minutes producing an artifact
        # nobody should ship. A dry run skips it on purpose — printing the
        # command is not producing an app.
        verify_kernel_artifact_manifest()
        verify_kernel_is_compiled()
    command = pyinstaller_args(args, build_format, host)

    if args.dry_run:
        print(shlex.join(command))
        return

    run(command, cwd=REPO_ROOT)
    verify_frozen_artifact_closed(args.dist_dir.resolve(), args.name, build_format)
    restore_macos_scrcpy_runtime_files(args.dist_dir.resolve(), args.name, args.include_scrcpy, args.build_dir)
    restore_macos_node_runtime_files(
        args.dist_dir.resolve(),
        args.name,
        args.include_node_runtime,
        args.build_dir,
    )
    restore_macos_platform_tools_files(
        args.dist_dir.resolve(),
        args.name,
        args.include_platform_tools,
        args.build_dir,
    )
    patch_macos_menu_bar_bundle(
        args.dist_dir.resolve(), args.name, build_format, args.sign_identity
    )
    if build_format == "dmg":
        create_dmg(args.dist_dir.resolve(), args.name)
        dmg_path = args.dist_dir.resolve() / f"{args.name}.dmg"
        # The disk image carries its own signature and its own ticket. A
        # notarized `.app` inside an unsigned `.dmg` still warns on download,
        # because what the user opens first is the image.
        if args.sign_identity:
            run(
                [
                    "codesign",
                    "--force",
                    "--timestamp",
                    "--sign",
                    args.sign_identity,
                    str(dmg_path),
                ],
                cwd=REPO_ROOT,
            )
        if args.notarize:
            notarize_macos_artifact(dmg_path, args.notarize)
    if build_format == "msi":
        create_msi(args.dist_dir.resolve(), args.name)

    print("\nBuild output:")
    print(f"  {args.dist_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
