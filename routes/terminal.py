"""Project terminal API routes."""

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from agent.tool_artifacts import record_tool_action_result
from audit.route_audit import record_api_action
from models import TerminalCommand
from policy.policy_gate import evaluate_direct_action_gate
from terminal_action_service import (
    cancel_terminal_command_for_current_server,
    execute_terminal_command_for_current_server,
    get_terminal_history_for_current_server,
)

from .deps import verify_api_key
from .result_response import as_route_response

router = APIRouter(tags=["terminal"])


@router.post("/api/projects/{name}/terminal/execute", dependencies=[Depends(verify_api_key)], response_model=None)
async def execute_terminal_command(
    name: str,
    body: TerminalCommand,
    require_approval: bool = False,
) -> dict[str, Any] | Response:
    """Execute a command in the project terminal session."""
    details = {"command": body.command, "timeout": body.timeout}
    gate = evaluate_direct_action_gate(
        operation="process.terminal",
        project_name=name,
        run_id=body.run_id,
        details=details,
        require_approval=require_approval,
        approval_id=body.approval_id,
    )
    if not gate["allowed"]:
        return JSONResponse(
            status_code=int(gate["status_code"]),
            content=gate["payload"],
        )

    result = await execute_terminal_command_for_current_server(
        name,
        command=body.command,
        timeout=body.timeout,
    )
    record_api_action(
        operation="process.terminal",
        project_name=name,
        run_id=body.run_id,
        details=details,
        success=result.success,
        status_code=result.status_code,
    )
    record_tool_action_result(
        run_id=body.run_id,
        operation="process.terminal",
        project_name=name,
        details=details,
        result=result,
    )
    return as_route_response(result)


@router.get("/api/projects/{name}/terminal/history", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_terminal_history(name: str) -> dict[str, Any] | Response:
    """Return terminal command history for a project."""
    return as_route_response(get_terminal_history_for_current_server(name))


@router.post("/api/projects/{name}/terminal/cancel", dependencies=[Depends(verify_api_key)], response_model=None)
async def cancel_terminal_command(name: str) -> dict[str, Any] | Response:
    """Cancel the current terminal command for a project."""
    result = await cancel_terminal_command_for_current_server(name)
    record_api_action(
        operation="process.terminal.cancel",
        project_name=name,
        success=result.success,
        status_code=result.status_code,
    )
    return as_route_response(result)
