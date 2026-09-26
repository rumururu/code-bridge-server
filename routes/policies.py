"""Persistent policy rule APIs for Agent Cockpit."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from agent.device_permissions import OPERATION as DEVICE_PERMISSION_OPERATION
from audit.audit_store import get_audit_store
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
# directly, never through a chat/LLM tool call. `browser.control`, which the
# dashboard used to also offer, is deliberately not here: no call site
# anywhere (LLM tool path or direct-action gate) ever asks for approval under
# that name, so a rule for it can never be consulted — offering it told the
# user they had authorized something that does not exist. `file.read` was in
# that same category and is no longer: the read-only tools now request
# approval under it (see `_approval_operation_for_tool`), so it arrives here
# through `LLM_TOOL_APPROVAL_OPERATIONS` like any other real operation.
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


# Every operation name some code path actually evaluates through the policy
# engine — and therefore the only names a standing rule can ever match.
#
# Two surfaces feed it. `LLM_TOOL_APPROVAL_OPERATIONS` is what a tool-permission
# prompt from the model can carry. `_DIRECT_ACTION_GATED_OPERATIONS` is every
# literal passed to `evaluate_direct_action_gate(operation=...)` or
# `request_approval_for_operation(operation=...)` by a route or agent module;
# `tests/test_gated_operations_catalog.py` reads those call sites and fails if
# this tuple drifts from them.
#
# This static set is not the whole story: `POST /api/approvals/request` (and
# its dashboard mirror) gates whatever operation name the caller sends, so an
# external agent can put a name here that no server module knows. Those show
# up in the audit log the first time they pass the gate, and
# `annotate_rules` unions the static set with what the log has seen.
#
# A rule whose operation is in neither is *inert*: nothing has ever asked the
# policy engine about it, so it sits in the rules table looking like an
# authorization and grants nothing. That has already happened twice — the
# dashboard once offered `browser.control`, and the feedback agent used to
# file approval requests by writing rows straight into `approval_requests`,
# leaving "always allow" rules for `feedback.reply.send` behind that its own
# request path never consulted. `list_policy_rules` flags such rules as
# `consulted: false` so the dashboard can say so instead of showing a green
# `allow` badge.
_DIRECT_ACTION_GATED_OPERATIONS: tuple[str, ...] = (
    "device.control",
    "file.copy",
    "file.delete",
    "file.move",
    "file.upload",
    "file.write",
    "git.commit",
    "git.push",
    "process.devserver",
    "process.terminal",
)


def gated_operations() -> frozenset[str]:
    """Operation names the runtime consults standing rules for."""
    return frozenset(
        (*LLM_TOOL_APPROVAL_OPERATIONS, *_DIRECT_ACTION_GATED_OPERATIONS, DEVICE_PERMISSION_OPERATION)
    )


def annotate_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach what a reader needs to judge a rule: is it live, has it fired.

    ``consulted`` — whether the policy engine is ever asked about this
    operation: it is in the static catalog (:func:`gated_operations`) or the
    audit log shows the gate has evaluated it at least once (an external
    caller of ``POST /approvals/request``). ``matched_count`` /
    ``last_matched_at`` — the audit trail's record of the rule deciding an
    actual request. All derived, never stored: the rules table is the grant,
    the audit log is the use, and joining them at read time is what keeps the
    two from disagreeing.
    """
    audit = get_audit_store()
    gated = gated_operations() | audit.gated_operations_seen()
    stats = audit.rule_match_stats()
    annotated: list[dict[str, Any]] = []
    for rule in rules:
        rule_stats = stats.get(str(rule.get("id") or ""), {})
        annotated.append(
            {
                **rule,
                "consulted": str(rule.get("operation") or "") in gated,
                "matched_count": int(rule_stats.get("matched_count") or 0),
                "last_matched_at": rule_stats.get("last_matched_at"),
            }
        )
    return annotated


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


# The catalogs below are readable by the phone, not just the dashboard.
#
# They were dashboard-only on the reasoning that they are "UI metadata for
# building the permissions form". That stopped being true once the phone could
# write standing rules through `POST /rules` below: a client that can grant
# `provider.tool` has to be able to find out that it means every tool the model
# asks for beyond the named three. Granting a permission whose meaning you
# cannot look up is the worse position, and these are descriptions of the
# runtime's own gating — reading them grants nothing.


@router.get("/operations", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_policy_operations_shared() -> dict[str, Any]:
    """What each operation a standing rule can name actually governs."""
    return list_policy_operations()


@router.get("/step-operations", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_step_operations_shared() -> dict[str, Any]:
    """Which operations each workflow step type will ask for at run time."""
    return list_step_operations()


@router.get("/rules", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_policy_rules(
    scope: str | None = None,
    operation: str | None = None,
    include_expired: bool = False,
) -> dict[str, Any]:
    """List persistent policy rules, each annotated with whether it is live and how often it fired."""
    return {
        "rules": annotate_rules(
            get_policy_rule_store().list_rules(
                scope=scope,
                operation=operation,
                include_expired=include_expired,
            )
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
