"""Tell the phone about an approval request that no run is parked on.

A run that parks for approval already rings the phone
(``task_orchestrator._notify_waiting_for_user_best_effort``). A request filed
by something that is not a run — the feedback agent asking to send a reply —
had no such path: the card appeared in the pending list and waited for someone
to happen to open the app. Nine of them expired that way. This is the ring for
those: a durable inbox notification plus a best-effort push.

Never raises. The request is already stored by the time this runs; failing to
ring must not turn into failing to ask.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_BODY = 300


def notify_runless_approval_best_effort(request: dict[str, Any] | None) -> dict[str, Any] | None:
    """Store a notification and push it, for a pending request with no run."""
    try:
        if not isinstance(request, dict) or not request.get("id"):
            return None
        if request.get("run_id"):
            # A run is parked on this one and its own notifier rings for it.
            return None
        details = request.get("details") if isinstance(request.get("details"), dict) else {}
        actor = request.get("actor") if isinstance(request.get("actor"), dict) else {}
        display = details.get("display") if isinstance(details.get("display"), dict) else {}
        who = str(actor.get("name") or details.get("agent_name") or "An agent")
        target = str(display.get("target") or details.get("subject") or request.get("operation") or "")
        summary = str(details.get("summary") or "")
        # Same template and label the orchestrator uses for a parked run
        # (`_WAIT_REASON_LABELS["approval_required"]`), so the app's
        # `ServerNotificationText` localizes this title with no app change.
        title = f"{who} needs you: needs your approval"
        body = " — ".join(part for part in (target, summary) if part)[:_MAX_BODY] or None

        from agent.notification_store import get_notification_store
        from agent.task_orchestrator import _push_notification_best_effort

        agent_id = actor.get("id") or details.get("agent_id")
        notification = get_notification_store().create(
            title=title,
            body=body,
            level="warning",
            agent_id=str(agent_id) if agent_id else None,
            reason="approval_request",
        )
        _push_notification_best_effort(
            notification=notification,
            title=title,
            body=body,
            level="warning",
            data_extra={
                "kind": "approval_request",
                "approval_id": str(request["id"]),
                "operation": str(request.get("operation") or ""),
                "agent_id": str(agent_id) if agent_id else None,
            },
        )
        return notification
    except Exception:
        logger.exception("could not notify for approval %s", (request or {}).get("id"))
        return None
