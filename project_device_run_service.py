"""Helpers for launching and summarizing Flutter device runs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from project_device_logs import read_log_tail


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
    except Exception:
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
    except Exception as exc:
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
