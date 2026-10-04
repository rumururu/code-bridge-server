"""Authenticated agent experience read models and review record."""

from typing import Any
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from agent import experience_service
from agent.agent_store import get_agent_store
from .deps import verify_api_key

router = APIRouter(prefix="/api/agent", tags=["agent-experience"], dependencies=[Depends(verify_api_key)])


class ReviewUpdate(BaseModel):
    reviewed: bool


@router.get("/overview")
def overview() -> dict[str, Any]:
    return experience_service.overview()


@router.get("/action-items")
def action_items(limit: int = Query(50, ge=1, le=200), cursor: str | None = None) -> dict[str, Any]:
    try:
        return experience_service.action_items(limit=limit, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/history")
def history(project_name: str | None = None, status: str | None = None,
            since: date | None = None, until: date | None = None,
            limit: int = Query(50, ge=1, le=200), cursor: str | None = None) -> dict[str, Any]:
    try:
        if since and until and since > until:
            raise ValueError("since must be on or before until")
        return experience_service.history(project_name=project_name, status=status, limit=limit, cursor=cursor,
                                          since=since.isoformat() if since else None,
                                          until=until.isoformat() if until else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}/summary")
def summary(run_id: str) -> dict[str, Any]:
    try:
        return experience_service.summary(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@router.put("/runs/{run_id}/review")
def review(run_id: str, body: ReviewUpdate) -> dict[str, Any]:
    if get_agent_store().get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"review": experience_service.set_review(run_id, body.reviewed, "paired_user")}
