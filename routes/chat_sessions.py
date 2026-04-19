"""Browse and resume past Claude Code conversations.

Claude CLI stores every session as a JSON-Lines file under
`~/.claude/projects/<slugified-path>/<session-id>.jsonl`. This router reads
those files so the iPad client can show a history list and resume an older
conversation without the user having to type anything.
"""

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from llm.claude_session import get_session_manager
from projects.project_manager import get_project_manager

from .deps import verify_api_key

router = APIRouter(tags=["chat-sessions"])


def _slugify_path(project_path: str) -> str:
    """Turn `/Users/x/Foo` into `-Users-x-Foo` the way Claude CLI does."""
    return project_path.replace("/", "-")


def _sessions_dir(project_path: str) -> Path:
    home = Path(os.path.expanduser("~"))
    return home / ".claude" / "projects" / _slugify_path(project_path)


def _resolve_project_path(project_name: str) -> str:
    pm = get_project_manager()
    project = pm.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
    path = project.get("path")
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="Project has no path configured")
    return path


def _read_first_user_message(file: Path) -> str | None:
    """Scan a jsonl file for the first human-typed user message."""
    try:
        with file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "user":
                    continue
                message = event.get("message") or {}
                content = message.get("content")
                # Claude stores content either as a plain string or a list of
                # content blocks. Only the first string-ish block counts.
                if isinstance(content, str):
                    return content.strip() or None
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text.strip():
                                return text.strip()
                        elif isinstance(block, str) and block.strip():
                            return block.strip()
    except OSError:
        return None
    return None


@router.get(
    "/api/chat/sessions/{project_name}",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def list_sessions(project_name: str) -> dict[str, Any]:
    """Return all past Claude sessions for a project, newest first."""
    project_path = _resolve_project_path(project_name)
    directory = _sessions_dir(project_path)
    if not directory.is_dir():
        return {"sessions": []}

    entries: list[dict[str, Any]] = []
    for file in directory.glob("*.jsonl"):
        try:
            stat = file.stat()
        except OSError:
            continue
        preview = _read_first_user_message(file)
        if not preview:
            # Empty/orphaned session files aren't worth surfacing.
            continue
        entries.append({
            "session_id": file.stem,
            "preview": preview[:160],
            "updated_at": stat.st_mtime,
            "size_bytes": stat.st_size,
        })

    entries.sort(key=lambda e: e["updated_at"], reverse=True)
    return {"sessions": entries}


def _extract_text_content(content: Any) -> str:
    """Flatten Claude's message.content into plain display text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            elif isinstance(block, str):
                pieces.append(block)
        return "".join(pieces)
    return ""


@router.get(
    "/api/chat/sessions/{project_name}/{session_id}/messages",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def get_session_messages(project_name: str, session_id: str) -> dict[str, Any]:
    """Return the user/assistant messages from one past session."""
    project_path = _resolve_project_path(project_name)
    file = _sessions_dir(project_path) / f"{session_id}.jsonl"
    if not file.is_file():
        raise HTTPException(status_code=404, detail="Session not found")

    messages: list[dict[str, Any]] = []
    try:
        with file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type not in ("user", "assistant"):
                    continue
                message = event.get("message") or {}
                text = _extract_text_content(message.get("content"))
                if not text.strip():
                    continue
                messages.append({
                    "type": event_type,
                    "content": text,
                    "timestamp": event.get("timestamp"),
                })
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read session: {exc}")

    return {"session_id": session_id, "messages": messages}


@router.post(
    "/api/chat/sessions/{project_name}/{session_id}/resume",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def resume_session(project_name: str, session_id: str) -> dict[str, Any]:
    """Point the live Claude session at this past session_id.

    The next send_message will spawn `claude --resume <session_id>`, so Claude
    continues with full memory of that prior conversation.
    """
    project_path = _resolve_project_path(project_name)
    file = _sessions_dir(project_path) / f"{session_id}.jsonl"
    if not file.is_file():
        raise HTTPException(status_code=404, detail="Session not found")

    manager = get_session_manager()
    session = manager.get_session_if_exists(project_name)
    if session is None:
        # Lazily create so the caller doesn't have to send a turn first just
        # to pin the session id.
        session = await manager.get_or_create_session(
            project_name=project_name,
            project_path=project_path,
        )

    if hasattr(session, "resume_session"):
        await session.resume_session(session_id)
    return {"ok": True, "session_id": session_id}
