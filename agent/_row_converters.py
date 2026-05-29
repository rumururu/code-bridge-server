"""SQLite row -> dict converters for Agent Cockpit entities.

Extracted from ``agent_store.py`` so the AgentStore class can focus on
query/mutation logic. Each converter mirrors one table's row shape.

These are intentionally module-private (`_row_to_*`) — call sites should
go through the AgentStore methods that wrap them, not the converters
directly.
"""

from __future__ import annotations

import json
from typing import Any


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _row_to_run(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "project_name": row["project_name"],
        "task_id": row["task_id"] if "task_id" in row.keys() else None,
        "provider_id": row["provider_id"],
        "model": row["model"],
        "status": row["status"],
        "title": row["title"],
        "goal": row["goal"],
        "cwd": row["cwd"],
        "native_session_id": row["native_session_id"],
        "parent_run_id": row["parent_run_id"],
        "started_at": row["started_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "ended_at": row["ended_at"],
    }


def _row_to_task(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "run_id": row["run_id"],
        "project_name": row["project_name"],
        "kind": row["kind"],
        "source": row["source"],
        "title": row["title"],
        "description": row["description"],
        "goal": row["goal"],
        "status": row["status"],
        "priority": row["priority"],
        "due_at": row["due_at"],
        "labels": _json_loads(row["labels_json"], []),
        "assignee": row["assignee"],
        "requester": row["requester"],
        "acceptance": _json_loads(row["acceptance_json"], []),
        "metadata": _json_loads(row["metadata_json"], {}),
        "result": _json_loads(row["result_json"], {}),
        "error": _json_loads(row["error_json"], {}),
        "started_at": row["started_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "ended_at": row["ended_at"],
    }


def _row_to_task_run(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "run_id": row["run_id"],
        "role": row["role"],
        "sequence": row["sequence"],
        "metadata": _json_loads(row["metadata_json"], {}),
        "created_at": row["created_at"],
    }


def _row_to_task_step(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "run_id": row["run_id"],
        "capability_id": row["capability_id"],
        "sequence": row["sequence"],
        "title": row["title"],
        "status": row["status"],
        "input": _json_loads(row["input_json"], {}),
        "output": _json_loads(row["output_json"], {}),
        "approval_id": row["approval_id"],
        "artifact_id": row["artifact_id"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_capability(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "type": row["type"],
        "name": row["name"],
        "provider_id": row["provider_id"],
        "status": row["status"],
        "scope": row["scope"],
        "source": row["source"],
        "description": row["description"],
        "permission_level": row["permission_level"],
        "desktop_only": bool(row["desktop_only"]),
        "local_only": bool(row["local_only"]),
        "metadata": _json_loads(row["metadata_json"], {}),
        "discovered_at": row["discovered_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_task_capability(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "capability_id": row["capability_id"],
        "mode": row["mode"],
        "created_at": row["created_at"],
    }


def _row_to_connector_request(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "step_id": row["step_id"],
        "run_id": row["run_id"],
        "connector_type": row["connector_type"],
        "name": row["name"],
        "status": row["status"],
        "adapter": _json_loads(row["adapter_json"], {}),
        "parameters": _json_loads(row["parameters_json"], {}),
        "result": _json_loads(row["result_json"], {}),
        "error": _json_loads(row["error_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }


def _row_to_event(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "sequence": row["sequence"],
        "event_type": row["event_type"],
        "provider_id": row["provider_id"],
        "provider_event": _json_loads(row["provider_event_json"], None),
        "app_event": _json_loads(row["app_event_json"], None),
        "created_at": row["created_at"],
    }


def _row_to_message(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "role": row["role"],
        "content": row["content"],
        "attachments": _json_loads(row["attachments_json"], []),
        "created_at": row["created_at"],
    }


def _row_to_artifact(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "kind": row["kind"],
        "path": row["path"],
        "mime_type": row["mime_type"],
        "metadata": _json_loads(row["metadata_json"], {}),
        "created_at": row["created_at"],
    }
