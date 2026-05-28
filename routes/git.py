"""Project Git API routes."""

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response

from agent.tool_artifacts import record_tool_action_result
from audit.route_audit import record_api_action
from files.git_action_service import (
    checkout_git_for_current_server,
    commit_git_changes_for_current_server,
    discard_git_changes_for_current_server,
    get_git_branches_for_current_server,
    get_git_diff_for_current_server,
    get_git_log_for_current_server,
    get_git_status_for_current_server,
    pull_git_for_current_server,
    push_git_for_current_server,
    stage_git_file_for_current_server,
    unstage_git_file_for_current_server,
)
from models import GitCommit, GitPull, GitPush
from policy.policy_gate import evaluate_direct_action_gate

from .deps import verify_api_key
from .result_response import as_route_response

router = APIRouter(tags=["git"])


def _gate_response(gate: dict[str, Any]) -> JSONResponse | None:
    if gate["allowed"]:
        return None
    return JSONResponse(status_code=int(gate["status_code"]), content=gate["payload"])


def _record_git_result(
    *,
    run_id: str | None,
    operation: str,
    project_name: str,
    details: dict[str, Any],
    result: Any,
) -> None:
    record_api_action(
        operation=operation,
        project_name=project_name,
        run_id=run_id,
        details=details,
        success=result.success,
        status_code=result.status_code,
    )
    record_tool_action_result(
        run_id=run_id,
        operation=operation,
        project_name=project_name,
        details=details,
        result=result,
    )


@router.get("/api/projects/{name}/git/status", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_git_status(name: str) -> dict[str, Any] | Response:
    """Return Git status for a project."""
    return as_route_response(await get_git_status_for_current_server(name))


@router.get("/api/projects/{name}/git/diff", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_git_diff(
    name: str,
    staged: bool = False,
    file: str | None = None,
) -> dict[str, Any] | Response:
    """Return Git diff for a project."""
    return as_route_response(
        await get_git_diff_for_current_server(name, staged=staged, file=file)
    )


@router.get("/api/projects/{name}/git/log", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_git_log(
    name: str,
    limit: int = Query(default=20, ge=1, le=100),
    file: str | None = None,
) -> dict[str, Any] | Response:
    """Return Git commit log for a project."""
    return as_route_response(
        await get_git_log_for_current_server(name, limit=limit, file=file)
    )


@router.get("/api/projects/{name}/git/branches", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_git_branches(name: str) -> dict[str, Any] | Response:
    """Return Git branches for a project."""
    return as_route_response(await get_git_branches_for_current_server(name))


@router.post("/api/projects/{name}/git/stage", dependencies=[Depends(verify_api_key)], response_model=None)
async def stage_git_file(
    name: str,
    path: str,
    run_id: str | None = None,
) -> dict[str, Any] | Response:
    """Stage one file."""
    result = await stage_git_file_for_current_server(name, path)
    _record_git_result(
        run_id=run_id,
        operation="git.stage",
        project_name=name,
        details={"path": path},
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/git/unstage", dependencies=[Depends(verify_api_key)], response_model=None)
async def unstage_git_file(
    name: str,
    path: str,
    run_id: str | None = None,
) -> dict[str, Any] | Response:
    """Unstage one file."""
    result = await unstage_git_file_for_current_server(name, path)
    _record_git_result(
        run_id=run_id,
        operation="git.unstage",
        project_name=name,
        details={"path": path},
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/git/discard", dependencies=[Depends(verify_api_key)], response_model=None)
async def discard_git_changes(
    name: str,
    path: str,
    run_id: str | None = None,
    approval_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Discard local changes for one file."""
    details = {"path": path}
    gate = evaluate_direct_action_gate(
        operation="file.delete",
        project_name=name,
        run_id=run_id,
        details=details,
        require_approval=require_approval,
        approval_id=approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await discard_git_changes_for_current_server(name, path)
    _record_git_result(
        run_id=run_id,
        operation="git.discard",
        project_name=name,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/git/commit", dependencies=[Depends(verify_api_key)], response_model=None)
async def commit_git_changes(
    name: str,
    body: GitCommit,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Commit staged changes."""
    details = {"message": body.message}
    gate = evaluate_direct_action_gate(
        operation="git.commit",
        project_name=name,
        run_id=body.run_id,
        details=details,
        require_approval=require_approval,
        approval_id=body.approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await commit_git_changes_for_current_server(name, body.message)
    _record_git_result(
        run_id=body.run_id,
        operation="git.commit",
        project_name=name,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/git/push", dependencies=[Depends(verify_api_key)], response_model=None)
async def push_git(
    name: str,
    body: GitPush,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Push Git changes."""
    details = {"remote": body.remote, "branch": body.branch}
    gate = evaluate_direct_action_gate(
        operation="git.push",
        project_name=name,
        run_id=body.run_id,
        details=details,
        require_approval=require_approval,
        approval_id=body.approval_id,
    )
    gate_response = _gate_response(gate)
    if gate_response is not None:
        return gate_response

    result = await push_git_for_current_server(name, remote=body.remote, branch=body.branch)
    _record_git_result(
        run_id=body.run_id,
        operation="git.push",
        project_name=name,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/git/pull", dependencies=[Depends(verify_api_key)], response_model=None)
async def pull_git(name: str, body: GitPull) -> dict[str, Any] | Response:
    """Pull Git changes."""
    result = await pull_git_for_current_server(name, remote=body.remote, branch=body.branch)
    _record_git_result(
        run_id=body.run_id,
        operation="git.pull",
        project_name=name,
        details={"remote": body.remote, "branch": body.branch},
        result=result,
    )
    return as_route_response(result)


@router.post("/api/projects/{name}/git/checkout", dependencies=[Depends(verify_api_key)], response_model=None)
async def checkout_git(
    name: str,
    branch: str,
    create: bool = False,
    run_id: str | None = None,
) -> dict[str, Any] | Response:
    """Checkout or create a branch."""
    result = await checkout_git_for_current_server(name, branch=branch, create=create)
    _record_git_result(
        run_id=run_id,
        operation="git.checkout",
        project_name=name,
        details={"branch": branch, "create": create},
        result=result,
    )
    return as_route_response(result)
