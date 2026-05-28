"""Record direct tool action outputs into Agent Cockpit runs."""

import re
from pathlib import Path
from typing import Any

from core.base_result import BaseRouteResult

from .agent_store import get_agent_store

ARTIFACT_ROOT = Path("/tmp") / "code_bridge_agent_artifacts"


def record_tool_action_result(
    *,
    run_id: str | None,
    operation: str,
    project_name: str,
    details: dict[str, Any] | None,
    result: BaseRouteResult,
) -> dict[str, Any] | None:
    """Attach a tool result event/artifact to a run when one is supplied."""
    if not run_id:
        return None

    store = get_agent_store()
    if store.get_run(run_id) is None:
        return None

    payload = result.as_response_fields()
    app_event = {
        "operation": operation,
        "project_name": project_name,
        "details": details or {},
        "success": result.success,
        "status_code": result.status_code,
    }
    event = store.append_event(
        run_id=run_id,
        event_type="tool.result",
        app_event={**app_event, "result": payload},
    )
    artifact = store.add_artifact(
        run_id=run_id,
        kind=_artifact_kind_for_operation(operation),
        mime_type="application/json",
        metadata={
            **app_event,
            "result": payload,
            "event_id": event["id"] if event else None,
        },
    )
    extra_artifacts = _record_output_artifacts(
        store=store,
        run_id=run_id,
        operation=operation,
        project_name=project_name,
        details=details or {},
        payload=payload,
        event_id=event["id"] if event else None,
    )
    return {"event": event, "artifact": artifact, "artifacts": [artifact, *extra_artifacts]}


def _artifact_kind_for_operation(operation: str) -> str:
    return operation.replace(".", "_").replace("-", "_")


def _record_output_artifacts(
    *,
    store: Any,
    run_id: str,
    operation: str,
    project_name: str,
    details: dict[str, Any],
    payload: dict[str, Any],
    event_id: str | None,
) -> list[dict[str, Any]]:
    if operation != "process.terminal":
        return []

    command = str(details.get("command") or "")
    artifacts: list[dict[str, Any]] = []
    log_text = _terminal_log_text(command, payload)
    if log_text:
        log_path = _write_text_artifact(
            run_id=run_id,
            stem="terminal",
            suffix=".log",
            text=log_text,
        )
        if log_path is not None:
            artifact = store.add_artifact(
                run_id=run_id,
                kind="terminal_log",
                path=str(log_path),
                mime_type="text/plain",
                metadata={
                    "operation": operation,
                    "project_name": project_name,
                    "command": command,
                    "event_id": event_id,
                    "bytes": log_path.stat().st_size,
                },
            )
            if artifact is not None:
                artifacts.append(artifact)

    stdout = payload.get("stdout")
    if _looks_like_source_diff_command(command) and isinstance(stdout, str) and stdout:
        diff_path = _write_text_artifact(
            run_id=run_id,
            stem="source_diff",
            suffix=".diff",
            text=stdout,
        )
        if diff_path is not None:
            artifact = store.add_artifact(
                run_id=run_id,
                kind="source_diff",
                path=str(diff_path),
                mime_type="text/x-diff",
                metadata={
                    "operation": operation,
                    "project_name": project_name,
                    "command": command,
                    "event_id": event_id,
                    "bytes": diff_path.stat().st_size,
                },
            )
            if artifact is not None:
                artifacts.append(artifact)

    if _looks_like_build_command(command) and _command_succeeded(payload):
        for output_path in _find_build_outputs(project_name):
            artifact = store.add_artifact(
                run_id=run_id,
                kind="build_output",
                path=str(output_path),
                mime_type=_mime_for_output(output_path),
                metadata={
                    "operation": operation,
                    "project_name": project_name,
                    "command": command,
                    "event_id": event_id,
                    "is_directory": output_path.is_dir(),
                    **_build_output_metadata(output_path),
                },
            )
            if artifact is not None:
                artifacts.append(artifact)

    return artifacts


def _terminal_log_text(command: str, payload: dict[str, Any]) -> str:
    parts = [
        f"$ {command}".rstrip(),
        f"exit_code: {payload.get('exit_code')}",
        f"timed_out: {payload.get('timed_out')}",
    ]
    error = payload.get("error")
    if error:
        parts.extend(["", "[error]", str(error)])
    stdout = payload.get("stdout")
    if isinstance(stdout, str) and stdout:
        parts.extend(["", "[stdout]", stdout.rstrip()])
    stderr = payload.get("stderr")
    if isinstance(stderr, str) and stderr:
        parts.extend(["", "[stderr]", stderr.rstrip()])
    return "\n".join(parts).strip() + "\n"


def _write_text_artifact(
    *,
    run_id: str,
    stem: str,
    suffix: str,
    text: str,
) -> Path | None:
    try:
        directory = ARTIFACT_ROOT / _safe_filename(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        index = len(list(directory.glob(f"{stem}_*{suffix}"))) + 1
        path = directory / f"{stem}_{index:03d}{suffix}"
        path.write_text(text, encoding="utf-8")
        return path
    except OSError:
        return None


def _safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "artifact"


def _looks_like_source_diff_command(command: str) -> bool:
    normalized = command.strip().lower()
    return normalized == "git diff" or normalized.startswith("git diff ")


def _looks_like_build_command(command: str) -> bool:
    tokens = command.strip().lower().split()
    return "build" in tokens or any(token.startswith("build:") for token in tokens)


def _command_succeeded(payload: dict[str, Any]) -> bool:
    return (
        int(payload.get("exit_code", 1) or 0) == 0
        and not bool(payload.get("timed_out"))
        and not payload.get("error")
    )


def _find_build_outputs(project_name: str) -> list[Path]:
    try:
        from core.database import get_project_db

        project = get_project_db().get(project_name)
    except Exception:
        return []
    if not project:
        return []
    raw_path = project.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return []

    root = Path(raw_path)
    candidates = [
        root / "dist",
        root / "out",
        root / ".next",
        root / "build" / "web",
        root / "build" / "app" / "outputs" / "flutter-apk" / "app-release.apk",
        root / "build" / "app" / "outputs" / "flutter-apk" / "app-debug.apk",
        root / "build" / "app" / "outputs" / "flutter-apk",
        root / "build" / "ios" / "ipa",
    ]
    found: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        found.append(candidate)
        if len(found) >= 12:
            break
    return found


def _mime_for_output(path: Path) -> str:
    if path.is_dir():
        return "application/vnd.codebridge.directory"
    suffix = path.suffix.lower()
    if suffix == ".apk":
        return "application/vnd.android.package-archive"
    if suffix == ".ipa":
        return "application/octet-stream"
    if suffix == ".zip":
        return "application/zip"
    return "application/octet-stream"


def _build_output_metadata(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        metadata: dict[str, Any] = {
            "size_bytes": size_bytes,
            "suffix": path.suffix.lower(),
        }
        # APK / IPA get their manifest unpacked so the Cockpit shows package
        # id, version, signing block, provisioning profile, etc. instead of
        # just file size. inspect_build_artifact returns None for everything
        # else and never raises on a malformed archive.
        try:
            from build_inspection import inspect_build_artifact

            inspection = inspect_build_artifact(path)
        except Exception:  # noqa: BLE001 — must never break artifact recording
            inspection = None
        if inspection:
            metadata.update(inspection)
        return metadata
    if path.is_dir():
        try:
            entries = sorted(item.name for item in path.iterdir())[:50]
        except OSError:
            entries = []
        return {
            "entry_count_sampled": len(entries),
            "entry_sample": entries,
        }
    return {}
