"""Persistent policy rule APIs for Agent Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from chat.chat_stream_service import LLM_TOOL_APPROVAL_OPERATIONS
from policy.policy_models import PolicyRuleCreate
from policy.policy_store import get_policy_rule_store
from policy.required_operations import step_type_operation_catalog

from .deps import verify_api_key

router = APIRouter(prefix="/api/policies", tags=["policies"])


# Operations meaningful for a dashboard standing-permission rule, each tagged
# with the request surface that actually consults it. `LLM_TOOL_APPROVAL_OPERATIONS`
# covers every operation an LLM tool-permission prompt can carry; `device.control`
# is a separate, real surface — direct device-action API routes (see
# routes/projects.py) call `evaluate_direct_action_gate(operation="device.control", ...)`
# directly, never through a chat/LLM tool call. Two operations the dashboard
# used to also offer, `file.read` and `browser.control`, are deliberately not
# here: no call site anywhere (LLM tool path or direct-action gate) ever asks
# for approval under either name, so a rule for them can never be consulted —
# offering them told the user they had authorized something that does not
# exist.
_DIRECT_ACTION_OPERATIONS: tuple[dict[str, str], ...] = (
    {
        "value": "device.control",
        "surface": "direct_action",
        "surface_detail": (
            "Connected-phone actions requested directly through the API — "
            "not through a chat/LLM tool call."
        ),
    },
)


def list_policy_operations() -> dict[str, Any]:
    """Operations a standing policy rule can actually match, with their surface.

    This is the dashboard permissions form's data source. It exists so that
    list is generated from `LLM_TOOL_APPROVAL_OPERATIONS` instead of being
    hand-copied into HTML a second time — that hand-copy is exactly how the
    form's option list drifted from reality before.
    """
    operations = [
        {
            "value": operation,
            "surface": "llm_tool_call",
            "surface_detail": (
                "Any other tool the model asks to use during a chat/agent turn."
                if operation == "provider.tool"
                else "Requested when the model calls this tool during a chat/agent turn."
            ),
        }
        for operation in LLM_TOOL_APPROVAL_OPERATIONS
    ]
    operations.extend(dict(item) for item in _DIRECT_ACTION_OPERATIONS)
    return {"operations": operations}


def list_step_operations() -> dict[str, Any]:
    """Per-step-type required-operation map, for the readiness rail.

    This is the dashboard's replacement for the hand-written
    ``OPERATION_FOR_STEP`` object it used to keep in ``agents.html``: instead
    of guessing which step types a scheduled run's approval gate cares about,
    the rail fetches this once and applies it locally to each agent's own
    ``flow_json``. See ``policy/required_operations.py`` for what "required"
    means for an ``llm`` step (a documented superset, not a precise
    prediction) versus every other step type (currently ungated at runtime).
    """
    return {"operations_by_step_type": step_type_operation_catalog()}


@router.get("/rules", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_policy_rules(
    scope: str | None = None,
    operation: str | None = None,
    include_expired: bool = False,
) -> dict[str, Any]:
    """List persistent policy rules."""
    return {
        "rules": get_policy_rule_store().list_rules(
            scope=scope,
            operation=operation,
            include_expired=include_expired,
        )
    }


@router.post("/rules", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_policy_rule(body: PolicyRuleCreate) -> dict[str, Any]:
    """Create a persistent policy rule."""
    return {
        "rule": get_policy_rule_store().create_rule(
            scope=body.scope,
            operation=body.operation,
            effect=body.effect,
            constraints=body.constraints,
            created_by=body.created_by,
            expires_at=body.expires_at,
        )
    }


@router.delete("/rules/{rule_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def delete_policy_rule(rule_id: str) -> dict[str, Any]:
    """Delete a persistent policy rule."""
    deleted = get_policy_rule_store().delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Policy rule '{rule_id}' not found")
    return {"deleted": True}
