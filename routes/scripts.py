"""Registered-script APIs.

A ``shell`` workflow step names a script from this registry; it can never
carry a command line of its own. Registering is therefore the security
boundary, and it is deliberately a separate, audited act from running.

Registration is dashboard-only (localhost). The phone can list what is
registered — so a run's steps are readable — but cannot add one: pairing a
phone should not hand out the ability to register arbitrary executables.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.script_store import ScriptRegistrationError, get_script_store
from agent.script_models import ScriptRegister, ScriptUpdate

from .deps import verify_api_key

router = APIRouter(prefix="/api/agent/scripts", tags=["scripts"])


@router.get("", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_scripts(limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
    return {"scripts": get_script_store().list_scripts(limit=limit)}


@router.get("/{script_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_script(script_id: str) -> dict[str, Any]:
    script = get_script_store().get(script_id)
    if script is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return {"script": script}


async def register_script(body: ScriptRegister) -> dict[str, Any]:
    """Register a script. Dashboard-only — see module docstring."""
    try:
        script = get_script_store().register(
            name=body.name,
            path=body.path,
            description=body.description,
            interpreter=body.interpreter,
            default_args=body.default_args,
            timeout_seconds=body.timeout_seconds,
            created_by=body.created_by,
        )
    except ScriptRegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"script": script}


async def update_script(script_id: str, body: ScriptUpdate) -> dict[str, Any]:
    try:
        script = get_script_store().update(
            script_id, body.model_dump(exclude_unset=True)
        )
    except ScriptRegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if script is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return {"script": script}


async def delete_script(script_id: str) -> dict[str, Any]:
    if not get_script_store().delete(script_id):
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return {"deleted": True, "script_id": script_id}


__all__ = ["router", "register_script", "update_script", "delete_script"]
