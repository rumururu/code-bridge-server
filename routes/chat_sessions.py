"""Browse and resume past provider-native CLI conversations."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from chat.chat_session_service import (
    ChatProviderSelection,
    ChatSessionInitError,
    create_chat_session,
    get_chat_provider_selection,
)
from llm.history import (
    HistoryNotFoundError,
    HistoryScopeError,
    get_history_adapter,
)
from projects.project_manager import get_project_manager

from .deps import verify_api_key

router = APIRouter(tags=["chat-sessions"])


def _resolve_project_path(project_name: str) -> str:
    pm = get_project_manager()
    project = pm.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
    path = project.get("path")
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="Project has no path configured")
    return path


def _resolve_provider_selection() -> ChatProviderSelection:
    try:
        return get_chat_provider_selection()
    except ChatSessionInitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _map_history_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HistoryNotFoundError):
        return HTTPException(status_code=404, detail=str(exc) or "Session not found")
    if isinstance(exc, HistoryScopeError):
        return HTTPException(status_code=403, detail=str(exc) or "Session is not in this project")
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail="Failed to read session history")


@router.get(
    "/api/chat/sessions/{project_name}",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def list_sessions(project_name: str) -> dict[str, Any]:
    """Return past native CLI sessions for the active provider."""
    project_path = _resolve_project_path(project_name)
    selection = _resolve_provider_selection()
    adapter = get_history_adapter(selection.provider_id)
    try:
        return adapter.list_sessions(project_path).to_dict()
    except Exception as exc:
        raise _map_history_error(exc) from exc


@router.get(
    "/api/chat/sessions/{project_name}/{session_id}/messages",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def get_session_messages(project_name: str, session_id: str) -> dict[str, Any]:
    """Return the user/assistant messages from one past native session."""
    project_path = _resolve_project_path(project_name)
    selection = _resolve_provider_selection()
    adapter = get_history_adapter(selection.provider_id)
    try:
        return adapter.get_messages(project_path, session_id).to_dict()
    except Exception as exc:
        raise _map_history_error(exc) from exc


@router.post(
    "/api/chat/sessions/{project_name}/{session_id}/resume",
    dependencies=[Depends(verify_api_key)],
    response_model=None,
)
async def resume_session(project_name: str, session_id: str) -> dict[str, Any]:
    """Point the live provider session at this past native CLI session_id."""
    project_path = _resolve_project_path(project_name)
    selection = _resolve_provider_selection()
    adapter = get_history_adapter(selection.provider_id)

    if not adapter.supports_resume:
        return (
            await adapter.resume_session(
                project_path=project_path,
                session_id=session_id,
                live_session=None,
            )
        ).to_dict()

    try:
        adapter.require_resumable(project_path, session_id)
    except Exception as exc:
        raise _map_history_error(exc) from exc

    session = await create_chat_session(
        project_name=project_name,
        project_path=project_path,
        selection=selection,
    )

    try:
        return (
            await adapter.resume_session(
                project_path=project_path,
                session_id=session_id,
                live_session=session,
            )
        ).to_dict()
    except Exception as exc:
        raise _map_history_error(exc) from exc
