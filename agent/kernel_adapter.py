"""Code Bridge product adapter for the agent-flow-core kernel (T-C-06).

Assembles the kernel pieces that already speak Code Bridge into one
:class:`agent_flow_core.conformance.ProductAdapter` implementation:

* flow view       — :func:`agent.flow_graph.to_graph` / ``from_graph``
                    (the linear flow_json stays the canon; the kernel
                    :class:`~agent_flow_core.model.Flow` is a derived view);
* run statuses    — :data:`agent_flow_core.lifecycle.CODE_BRIDGE_RUN_STATUS_MAP`
                    imported as-is (never copied: the kernel map *is* the
                    contract, and a local clone would drift silently);
* human gates     — :func:`agent_flow_core.gate.gate_from_code_bridge_checkpoint`
                    / ``gate_to_code_bridge_checkpoint`` over the checkpoint
                    dicts ``_wait_for_user_step`` writes
                    (agent/task_orchestrator.py:3492-3508);
* notifications   — a thin wrapper expressing Code Bridge's *decision*
                    (send / suppress-with-reason) as a kernel
                    :class:`~agent_flow_core.gate.NotifyOutcome`. It never
                    sends FCM itself; the real transport is injected and
                    defaults to a no-op so tests can verify the policy.

Native shapes
-------------

* **flow native** — the *normalized* linear step list, i.e. the output of
  :func:`code_bridge_core.workflow_v2.normalize_workflow`. Raw stored flow_json must be
  normalized before it reaches :meth:`CodeBridgeAdapter.to_flow`, exactly as
  ``flow_graph.to_graph`` requires.
* **gate native** — a park record ``{"run_id": str, "checkpoint": dict}``.
  The checkpoint dict itself never carries the run id (it lives on the
  surrounding run/task metadata — the ``active_checkpoint`` marker and the
  ``task.step.waiting_for_user`` event both pair the checkpoint with
  ``run_id``, task_orchestrator.py:3521-3539), so the adapter's native park
  record is that same pairing: ``run_id`` is read from the wrapper dict, and
  :meth:`CodeBridgeAdapter.gate_to_native` writes the wrapper back so the
  gate round-trip is lossless.

.. warning::
   **Do not import this module from any runtime path** (task orchestrator,
   scheduler, routes, dashboard). It imports ``agent_flow_core`` at module
   top, and the *deployed* server venv does not have the kernel installed —
   an import from a live code path would crash the server at startup. This
   module is for the conformance suite and graph API layers only — the same
   contract ``agent/flow_graph.py`` carries.
"""

from __future__ import annotations

from typing import Any, Callable

from agent_flow_core.gate import (
    HumanGate,
    NotifyOutcome,
    gate_from_code_bridge_checkpoint,
    gate_to_code_bridge_checkpoint,
)
from agent_flow_core.lifecycle import CODE_BRIDGE_RUN_STATUS_MAP
from agent_flow_core.model import Flow

from .flow_graph import from_graph, to_graph

__all__ = [
    "CodeBridgeAdapter",
    "CodeBridgeGateNotifier",
    "repeat_park_decision",
]


# ---------------------------------------------------------------------------
# Notification policy wrapper
# ---------------------------------------------------------------------------

#: ``decide(gate, context) -> (should_notify, why)`` — the same return
#: contract as ``_wait_notification_gate`` (task_orchestrator.py:3774-3856).
DecideFn = Callable[[HumanGate, dict[str, Any]], tuple[bool, str]]

#: ``send(gate, context) -> None`` — the real transport (FCM push +
#: notification-store row). Injected; the default is a no-op.
SendFn = Callable[[HumanGate, dict[str, Any]], None]


def repeat_park_decision(
    gate: HumanGate, context: dict[str, Any]
) -> tuple[bool, str]:
    """Code Bridge's in-process dedup decision, store-free.

    Mirrors the repeat-park suppression at the ``_wait_for_user_step`` call
    site (task_orchestrator.py:3546-3560): a park of the same run, same
    step, same reason as the previously recorded ``active_checkpoint``
    marker is a resume-then-immediately-re-park loop the user has already
    been told about, so it does not notify again. The marker is read from
    ``context["previous_checkpoint"]`` — the caller passes what task
    metadata held before this park, just as the orchestrator does.

    The store-backed agent×reason 24h throttle (``_wait_notification_gate``)
    needs run and notification history and therefore cannot live here; a
    caller with a store injects a ``decide`` that consults it. Anything this
    function cannot evaluate notifies — the same fail-open posture as both
    orchestrator gates (a duplicate push costs less than a swallowed first
    park).
    """
    previous = context.get("previous_checkpoint")
    cb_extension = gate.extensions.get("code_bridge")
    reason = (
        cb_extension.get("reason") if isinstance(cb_extension, dict) else None
    )
    if (
        isinstance(previous, dict)
        and previous.get("run_id") == gate.run_id
        and previous.get("step_id") == gate.step_id
        and previous.get("reason") == reason
    ):
        return False, (
            "repeat park of the same run/step/reason — a "
            "resume-then-re-park loop the user was already notified about "
            "(task_orchestrator.py repeat-park dedup)"
        )
    return True, "not a repeat park; notifying"


def _noop_send(gate: HumanGate, context: dict[str, Any]) -> None:
    """Default transport: deliver nowhere. Real FCM is injected by callers."""


class CodeBridgeGateNotifier:
    """Code Bridge's notification policy as a kernel ``HumanGateNotifier``.

    Wraps two injected callables:

    * ``decide`` — should this gate notify, and why/why not
      (default :func:`repeat_park_decision`);
    * ``send`` — the actual transport (default no-op; production would
      inject the notification-store + FCM path,
      ``_notify_waiting_for_user_best_effort``).

    Contract posture (kernel ``HumanGateNotifier``):

    * hooks never raise — a broken decision fails **open** (notify) like
      the orchestrator's gates, and a broken transport becomes
      ``NotifyOutcome(status="failed", reason=...)``;
    * every suppression carries its reason, mirroring CB's
      ``notification.suppressed`` audit event
      (task_orchestrator.py:3859-3898).
    """

    def __init__(
        self,
        *,
        decide: DecideFn | None = None,
        send: SendFn | None = None,
    ) -> None:
        self._decide: DecideFn = decide or repeat_park_decision
        self._send: SendFn = send or _noop_send

    def on_gate_opened(
        self, gate: HumanGate, context: dict
    ) -> NotifyOutcome:
        try:
            should_notify, why = self._decide(gate, dict(context))
        except Exception as exc:  # noqa: BLE001 - best-effort hook
            # Fail open, like _wait_notification_gate: a throttle that
            # cannot be evaluated notifies rather than going quiet.
            should_notify = True
            why = f"throttle decision failed ({exc!r}); notifying (fail open)"
        if not should_notify:
            return NotifyOutcome(status="suppressed", reason=why)
        try:
            self._send(gate, dict(context))
        except Exception as exc:  # noqa: BLE001 - best-effort hook
            return NotifyOutcome(
                status="failed",
                reason=f"notification transport failed: {exc!r}",
            )
        return NotifyOutcome(status="delivered")

    def on_gate_settled(
        self, gate: HumanGate, settlement: Any, context: dict
    ) -> NotifyOutcome:
        """Code Bridge sends no settlement push today.

        A resume/settlement simply proceeds (or re-parks); there is no
        "your gate was answered" notification anywhere in the product. The
        kernel contract explicitly allows a no-op settled hook expressed as
        a reasoned suppression (gate.py HumanGateNotifier docstring — the
        IG inbox precedent).
        """
        return NotifyOutcome(
            status="suppressed",
            reason=(
                "Code Bridge sends no gate-settlement notification; the "
                "resume path itself is the user-visible outcome"
            ),
        )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class CodeBridgeAdapter:
    """Code Bridge's :class:`~agent_flow_core.conformance.ProductAdapter`."""

    product_name = "code_bridge"

    #: The kernel map, shared by reference on purpose — the kernel owns the
    #: status vocabulary contract (lifecycle.py evidence inventory) and a
    #: plain dict already refuses unknown statuses with ``KeyError``, which
    #: is the explicit-refusal posture the conformance probe demands.
    run_status_map = CODE_BRIDGE_RUN_STATUS_MAP

    def __init__(
        self,
        *,
        decide: DecideFn | None = None,
        send: SendFn | None = None,
    ) -> None:
        self._decide = decide
        self._send = send

    # -- flow view ---------------------------------------------------------

    def to_flow(self, native: list[dict[str, Any]]) -> Flow:
        """Normalized linear step list → kernel :class:`Flow` (derived view)."""
        return to_graph(native)

    def from_flow(self, flow: Flow) -> list[dict[str, Any]]:
        """Kernel :class:`Flow` → normalized linear step list (the canon)."""
        return from_graph(flow)

    # -- human gates -------------------------------------------------------

    def gate_from_native(self, native_checkpoint: dict[str, Any]) -> HumanGate:
        """Park record ``{"run_id", "checkpoint"}`` → :class:`HumanGate`.

        ``run_id`` comes from the wrapper dict, not the checkpoint: the
        checkpoint dict ``_wait_for_user_step`` writes has no run id of its
        own — the surrounding metadata (``active_checkpoint`` marker /
        run event) supplies it, and this native shape preserves that
        pairing (see module docstring, "Native shapes").
        """
        run_id = native_checkpoint.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(
                "Code Bridge park record has no 'run_id' — the native gate "
                "shape is {'run_id': str, 'checkpoint': dict}"
            )
        checkpoint = native_checkpoint.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Code Bridge park record has no 'checkpoint' dict — the "
                "native gate shape is {'run_id': str, 'checkpoint': dict}"
            )
        return gate_from_code_bridge_checkpoint(checkpoint, run_id=run_id)

    def gate_to_native(self, gate: HumanGate) -> dict[str, Any]:
        """Inverse of :meth:`gate_from_native` (gate round-trip proof)."""
        return {
            "run_id": gate.run_id,
            "checkpoint": gate_to_code_bridge_checkpoint(gate),
        }

    # -- notifications -----------------------------------------------------

    def notifier(self) -> CodeBridgeGateNotifier:
        return CodeBridgeGateNotifier(decide=self._decide, send=self._send)
