"""Usage dashboard APIs for Work Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, Query

from core.config import get_config
from core.database import get_usage_db
from llm.claude_usage import (
    fetch_claude_cli_stats,
    fetch_claude_usage_snapshot,
    merge_usage_for_display,
)

from .deps import verify_api_key

router = APIRouter(prefix="/api/usage", tags=["usage"])


def _summary_payload(
    *,
    window_days: int | None,
    project_name: str | None,
    workspace_id: str | None,
    task_id: str | None,
    run_id: str | None,
    provider_id: str | None,
) -> dict[str, Any]:
    config = get_config()
    days = window_days or config.usage_window_days
    usage_db = get_usage_db()
    summary = usage_db.get_weekly_summary(
        budget_usd=config.weekly_budget_usd,
        window_days=days,
        project_name=project_name,
        workspace_id=workspace_id,
        task_id=task_id,
        run_id=run_id,
        provider_id=provider_id,
    )
    return {
        **summary,
        "filters": {
            "window_days": days,
            "project_name": project_name,
            "workspace_id": workspace_id,
            "task_id": task_id,
            "run_id": run_id,
            "provider_id": provider_id,
        },
    }


@router.get("/summary", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_usage_summary(
    window_days: int | None = Query(default=None, ge=1, le=365),
    project_name: str | None = None,
    workspace_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    provider_id: str | None = None,
) -> dict[str, Any]:
    """Return filtered usage summary for Work Cockpit.

    The unfiltered summary — what the phone's settings screen asks for — also
    carries Claude's own weekly reading as ``display_usage_percent``. It never
    used to: only the chat socket merged it in, so the settings "usage" row
    had no real number to show and the app filled the slot with cost divided
    by a default budget. A filtered summary (one task, one run, one project)
    is a slice of cost, and the account-wide limit does not belong on it.
    """
    payload = _summary_payload(
        window_days=window_days,
        project_name=project_name,
        workspace_id=workspace_id,
        task_id=task_id,
        run_id=run_id,
        provider_id=provider_id,
    )
    if any((project_name, workspace_id, task_id, run_id, provider_id)):
        return payload
    snapshot = await fetch_claude_usage_snapshot()
    return merge_usage_for_display(payload, snapshot)


@router.get("/turns", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_usage_turns(
    window_days: int | None = Query(default=None, ge=1, le=365),
    project_name: str | None = None,
    workspace_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    provider_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List filtered usage events."""
    config = get_config()
    days = window_days or config.usage_window_days
    return {
        "turns": get_usage_db().list_events(
            window_days=days,
            project_name=project_name,
            workspace_id=workspace_id,
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            limit=limit,
        )
    }


@router.get("/cli", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_claude_cli_usage(
    force_refresh: bool = Query(default=False),
) -> dict[str, Any]:
    """Return Claude CLI /stats snapshot (current week, all models + sonnet only)."""
    return await fetch_claude_cli_stats(force_refresh=force_refresh)


@router.get("/breakdown", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_usage_breakdown(
    group_by: str = "provider_id",
    window_days: int | None = Query(default=None, ge=1, le=365),
    project_name: str | None = None,
    workspace_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    provider_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Return grouped usage totals."""
    config = get_config()
    days = window_days or config.usage_window_days
    return {
        "group_by": group_by,
        "items": get_usage_db().breakdown(
            group_by,
            window_days=days,
            project_name=project_name,
            workspace_id=workspace_id,
            task_id=task_id,
            run_id=run_id,
            provider_id=provider_id,
            limit=limit,
        ),
    }
