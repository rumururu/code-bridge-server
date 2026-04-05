"""Helpers for launching and summarizing Flutter device runs."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .project_device_logs import read_log_tail


@dataclass(frozen=True)
class FlutterRunStartResult:
    """Result of attempting to start a flutter run process."""

    success: bool
    command: list[str]
    process: subprocess.Popen | None = None
    error_message: str | None = None


def build_flutter_run_command(device_id: str) -> list[str]:
    """Build canonical flutter run command for one device."""
    return [
        "flutter",
        "run",
        "-d",
        device_id,
        "--machine",
        "--target",
        "lib/main.dart",
    ]


def prepare_device_run_log(log_path: Path) -> None:
    """Create/clear the device run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_path.write_text("", encoding="utf-8")
    except OSError:
        pass


def start_flutter_run_process(
    project_path: str,
    *,
    device_id: str,
    log_path: Path,
) -> FlutterRunStartResult:
    """Start flutter run and redirect output to a log file."""
    command = build_flutter_run_command(device_id)
    prepare_device_run_log(log_path)

    try:
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                command,
                cwd=project_path,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        return FlutterRunStartResult(success=True, command=command, process=process)
    except FileNotFoundError:
        return FlutterRunStartResult(
            success=False,
            command=command,
            error_message="Flutter CLI not found on server",
        )
    except OSError as exc:
        return FlutterRunStartResult(
            success=False,
            command=command,
            error_message=f"Failed to start flutter run: {exc}",
        )


def summarize_flutter_run_exit(log_path: str | Path) -> tuple[str, str]:
    """Build concise failure summary and log tail for exited flutter run."""
    tail = read_log_tail(log_path, max_lines=80, max_chars=4000)
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    summary = lines[-1] if lines else "flutter run exited immediately"
    return summary, tail


def extract_vm_service_uri_from_log(log_path: str | Path) -> str | None:
    """Extract VM Service URI from flutter run --machine log.

    The log contains JSON lines, and we look for the app.debugPort event
    which contains the wsUri field.

    Example JSON line:
    [{"event":"app.debugPort","params":{"port":50139,"wsUri":"ws://127.0.0.1:50139/xxx=/ws"}}]
    """
    path = Path(log_path)
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Look for wsUri in the log
    # Pattern 1: JSON format from --machine flag
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try to parse as JSON array (flutter run --machine format)
        if line.startswith("["):
            try:
                events = json.loads(line)
                if isinstance(events, list):
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        if event.get("event") == "app.debugPort":
                            params = event.get("params", {})
                            if isinstance(params, dict):
                                ws_uri = params.get("wsUri")
                                if isinstance(ws_uri, str) and ws_uri.startswith("ws://"):
                                    return ws_uri
            except json.JSONDecodeError:
                pass

        # Pattern 2: Direct wsUri in JSON object
        if '"wsUri"' in line or '"vmServiceUri"' in line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    for key in ("wsUri", "vmServiceUri", "ws_uri", "vm_service_uri"):
                        uri = obj.get(key)
                        if isinstance(uri, str) and uri.startswith("ws://"):
                            return uri
                    # Check nested params
                    params = obj.get("params", {})
                    if isinstance(params, dict):
                        for key in ("wsUri", "vmServiceUri"):
                            uri = params.get(key)
                            if isinstance(uri, str) and uri.startswith("ws://"):
                                return uri
            except json.JSONDecodeError:
                pass

    # Pattern 3: Regex fallback for ws:// URLs
    ws_pattern = re.compile(r"ws://127\.0\.0\.1:\d+/[^/]+/ws")
    match = ws_pattern.search(text)
    if match:
        return match.group(0)

    return None
