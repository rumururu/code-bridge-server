"""Code Bridge adapter conformance proof (agent-flow-core T-C-06).

Runs the kernel's product-adapter conformance suite
(``agent_flow_core.conformance.check_adapter``) over
``agent.kernel_adapter.CodeBridgeAdapter`` with **real** samples:

* flow samples      — the full stored-shape snapshot net of
  ``test_flow_json_snapshot_regression.py`` (imported, not copied),
  normalized exactly as the orchestrator normalizes stored flow_json;
* checkpoint samples — park records in the exact field shape
  ``_wait_for_user_step`` writes (task_orchestrator.py:3492-3510), with
  the reason-derived fields produced by the *same* orchestrator helpers
  (``_required_user_action`` / ``_checkpoint_resume_behavior`` /
  ``_checkpoint_resume_label`` / ``_checkpoint_allows_memory``) so the
  samples cannot drift from the writer;
* status sequences  — run-status histories Code Bridge actually produces,
  per the write-path evidence inventory in the kernel's lifecycle.py
  docstring (parking, restart reconciliation, approval blocking,
  stall abandonment, cancellation).

Acceptance criterion: ``check_adapter`` returns **zero failures**. A
failure here is either an adapter bug (fix the adapter) or a genuine
contract violation between the kernel and Code Bridge's real behavior
(report it — do not paper over it in the adapter).
"""

from __future__ import annotations

import copy
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for path in (str(SERVER_DIR), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent.kernel_adapter import (  # noqa: E402
    CodeBridgeAdapter,
    CodeBridgeGateNotifier,
)
from agent.task_orchestrator import (  # noqa: E402
    _checkpoint_allows_memory,
    _checkpoint_resume_behavior,
    _checkpoint_resume_label,
    _required_user_action,
)
from code_bridge_core.workflow_v2 import normalize_workflow  # noqa: E402
from agent_flow_core.conformance import (  # noqa: E402
    ConformanceSamples,
    check_adapter,
)
from agent_flow_core.gate import (  # noqa: E402
    GATE_EXPIRY_ACTOR,
    GateSettlement,
)
from agent_flow_core.lifecycle import CODE_BRIDGE_RUN_STATUS_MAP  # noqa: E402
from test_flow_json_snapshot_regression import SNAPSHOTS  # noqa: E402


# ---------------------------------------------------------------------------
# Real checkpoint samples — built with the orchestrator's own field helpers
# so every reason-derived value is exactly what the writer would stamp.
# ---------------------------------------------------------------------------

_CREATED_AT = "2026-08-17T02:04:05.678901+00:00"


def _checkpoint(
    reason: str,
    *,
    prompt: str,
    workflow_step_id: str,
    step_id: str,
    step_title: str,
    workflow_type: str,
    success_criteria: str | None,
    on_failure: dict,
    extra: dict | None = None,
) -> dict:
    """A checkpoint dict in ``_wait_for_user_step``'s literal shape
    (task_orchestrator.py:3492-3510): same keys, same key order, same
    derivation helpers, plus ``checkpoint_extra`` merged last."""
    checkpoint = {
        "status": "waiting_for_user",
        "reason": reason,
        "prompt": prompt,
        "workflow_step_id": workflow_step_id,
        "step_id": step_id,
        "step_title": step_title,
        "workflow_type": workflow_type,
        "success_criteria": success_criteria,
        "resume": on_failure.get("resume") or "same_step",
        "resume_step_id": on_failure.get("resume_step_id"),
        "resume_behavior": _checkpoint_resume_behavior(on_failure),
        "resume_label": _checkpoint_resume_label(on_failure),
        "required_user_action": _required_user_action(reason),
        "allow_memory": _checkpoint_allows_memory(
            reason, {"workflow_type": workflow_type}
        ),
        "created_at": _CREATED_AT,
    }
    if extra:
        checkpoint.update(extra)
    return checkpoint


def _native_parks() -> list[dict]:
    """Three real park records: ``{"run_id", "checkpoint"}`` pairs.

    The wrapper mirrors how the product pairs a checkpoint with its run —
    the ``active_checkpoint`` marker and the ``task.step.waiting_for_user``
    event (task_orchestrator.py:3521-3539); the checkpoint dict itself
    never carries the run id.
    """
    approval = _checkpoint(
        "approval_required",
        prompt="Bash needs your approval: rm -rf build/",
        workflow_step_id="wf-step-3",
        step_id="step-uuid-77",
        step_title="Clean build artifacts",
        workflow_type="shell",
        success_criteria="build/ directory removed",
        on_failure={"type": "ask_user", "resume": "same_step"},
        # checkpoint_extra: the approval identity
        # (_approval_checkpoint_extra, task_orchestrator.py:1775-1781).
        extra={
            "approval_id": "approval-42",
            "tool_name": "Bash",
            "tool_target": "rm -rf build/",
        },
    )
    manual_handoff = _checkpoint(
        "manual_handoff",
        prompt="Complete captcha, then continue.",
        workflow_step_id="captcha",
        step_id="step-uuid-12",
        step_title="Resolve captcha",
        workflow_type="manual_handoff",
        success_criteria=None,
        on_failure={
            "type": "manual_handoff",
            "prompt": "Complete captcha, then continue.",
            "resume": "same_step",
        },
    )
    login_required = _checkpoint(
        "login_required",
        prompt="The site is showing a login form. Sign in, then resume.",
        workflow_step_id="open_page",
        step_id="step-uuid-31",
        step_title="Open page",
        workflow_type="browser_action",
        success_criteria="Page opens.",
        on_failure={"type": "ask_user", "resume": "same_step"},
        # Adapter wait detail carried as checkpoint_extra — the open-set
        # overflow the extensions namespace must preserve verbatim.
        extra={"wait_details": {"login_url": "https://example.com/login"}},
    )
    return [
        {"run_id": "run-appr-1", "checkpoint": approval},
        {"run_id": "run-hand-2", "checkpoint": manual_handoff},
        {"run_id": "run-login-3", "checkpoint": login_required},
    ]


#: Run-status histories Code Bridge actually produces (native strings, in
#: chronological order). Sources: the kernel lifecycle.py evidence inventory
#: (write paths verified 2026-08-17) —
#: * queued at creation (agent_store.py:533), running
#:   (task_orchestrator.py:961), completed (:4740)
#: * waiting_for_user parking write (:3529) and resume back to running
#: * queued→failed: restart reconciliation fails stale runs
#:   (run_reconciliation.py:59)
#: * running→blocked: approval parking via _finish_execution("blocked")
#:   (:1029 → update_run_status :4754)
#: * starting: progressing alias between queued and running
#:   (scheduler.py:46)
#: * waiting_for_user→failed: stalled-park abandonment
#:   (task_orchestrator.py:1236-1243, approval_resume.py:364-437)
#: * cancellation from running or parked (routes/agents.py:2490)
_NATIVE_STATUS_SEQUENCES: list[list[str]] = [
    ["queued", "running", "completed"],
    ["queued", "running", "waiting_for_user", "running", "completed"],
    ["queued", "failed"],
    ["queued", "running", "blocked"],
    ["queued", "starting", "running", "completed"],
    ["queued", "running", "waiting_for_user", "failed"],
    ["queued", "running", "cancelled"],
    ["queued", "running", "waiting_for_user", "cancelled"],
]


def _samples() -> ConformanceSamples:
    parks = _native_parks()
    approval_park = parks[0]
    return ConformanceSamples(
        # The full stored-shape snapshot net, normalized the way the
        # orchestrator normalizes stored flow_json before every run.
        native_flows=[
            normalize_workflow(copy.deepcopy(snapshot))
            for snapshot in SNAPSHOTS.values()
        ],
        native_checkpoints=parks,
        native_status_sequences=[
            list(sequence) for sequence in _NATIVE_STATUS_SEQUENCES
        ],
        notify_contexts=[
            # First park — no marker recorded yet.
            {},
            # A different run's marker: not a repeat, must notify.
            {
                "previous_checkpoint": {
                    "run_id": "some-other-run",
                    "step_id": "step-uuid-99",
                    "reason": "ask_user",
                }
            },
            # Repeat park of the exact gate the suite probes with
            # (gates[0] = the approval park): exercises the suppressed
            # path, which must carry a reason.
            {
                "previous_checkpoint": {
                    "run_id": approval_park["run_id"],
                    "step_id": approval_park["checkpoint"]["step_id"],
                    "reason": approval_park["checkpoint"]["reason"],
                }
            },
            # Corrupt marker: the decision cannot be evaluated from it —
            # fail-open (notify), never raise.
            {"previous_checkpoint": "corrupt-marker"},
        ],
    )


# ---------------------------------------------------------------------------
# The conformance proof
# ---------------------------------------------------------------------------


class CodeBridgeAdapterConformanceTest(unittest.TestCase):
    """``check_adapter`` must return zero failures for the CB adapter."""

    def test_check_adapter_reports_zero_failures(self) -> None:
        failures = check_adapter(CodeBridgeAdapter(), _samples())
        self.assertEqual(
            failures,
            [],
            msg="adapter conformance failures:\n"
            + "\n".join(f"  - {failure}" for failure in failures),
        )

    def test_run_status_map_is_the_kernel_map_not_a_copy(self) -> None:
        # 복제 금지: the kernel map is the contract; a clone would drift.
        self.assertIs(
            CodeBridgeAdapter.run_status_map, CODE_BRIDGE_RUN_STATUS_MAP
        )

    def test_gate_native_round_trip_is_lossless_for_every_park(self) -> None:
        adapter = CodeBridgeAdapter()
        for park in _native_parks():
            with self.subTest(reason=park["checkpoint"]["reason"]):
                gate = adapter.gate_from_native(copy.deepcopy(park))
                self.assertEqual(adapter.gate_to_native(gate), park)

    def test_gate_from_native_requires_the_run_id_pairing(self) -> None:
        adapter = CodeBridgeAdapter()
        checkpoint_only = _native_parks()[0]["checkpoint"]
        with self.assertRaises(ValueError):
            adapter.gate_from_native({"checkpoint": checkpoint_only})
        with self.assertRaises(ValueError):
            adapter.gate_from_native(checkpoint_only)


class CodeBridgeGateNotifierPolicyTest(unittest.TestCase):
    """The notifier expresses CB's decisions; it never sends or raises."""

    def _gate(self):
        return CodeBridgeAdapter().gate_from_native(_native_parks()[0])

    def test_first_park_delivers_through_the_injected_transport(self) -> None:
        sent = []
        notifier = CodeBridgeGateNotifier(
            send=lambda gate, context: sent.append((gate.gate_id, context))
        )
        outcome = notifier.on_gate_opened(self._gate(), {})
        self.assertEqual(outcome.status, "delivered")
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], "run-appr-1:step-uuid-77")

    def test_repeat_park_is_suppressed_with_a_reason_and_no_send(self) -> None:
        sent = []
        notifier = CodeBridgeGateNotifier(
            send=lambda gate, context: sent.append(gate)
        )
        gate = self._gate()
        outcome = notifier.on_gate_opened(
            gate,
            {
                "previous_checkpoint": {
                    "run_id": gate.run_id,
                    "step_id": gate.step_id,
                    "reason": "approval_required",
                }
            },
        )
        self.assertEqual(outcome.status, "suppressed")
        self.assertTrue(outcome.reason)
        self.assertEqual(sent, [])

    def test_transport_failure_becomes_a_failed_outcome_not_a_raise(
        self,
    ) -> None:
        def broken_send(gate, context):
            raise RuntimeError("FCM is down")

        notifier = CodeBridgeGateNotifier(send=broken_send)
        outcome = notifier.on_gate_opened(self._gate(), {})
        self.assertEqual(outcome.status, "failed")
        self.assertIn("FCM is down", outcome.reason)

    def test_broken_throttle_decision_fails_open(self) -> None:
        # Same posture as _wait_notification_gate: a throttle that cannot
        # be evaluated notifies rather than swallowing a first park.
        def broken_decide(gate, context):
            raise RuntimeError("store unreachable")

        sent = []
        notifier = CodeBridgeGateNotifier(
            decide=broken_decide,
            send=lambda gate, context: sent.append(gate),
        )
        outcome = notifier.on_gate_opened(self._gate(), {})
        self.assertEqual(outcome.status, "delivered")
        self.assertEqual(len(sent), 1)

    def test_settled_hook_is_a_reasoned_suppression(self) -> None:
        gate = self._gate()
        settlement = GateSettlement(
            gate_id=gate.gate_id,
            action="park_forever",
            outcome="still_parked",
            reason="test settlement",
            decided_by=GATE_EXPIRY_ACTOR,
            settled_at=datetime.now(UTC),
        )
        outcome = CodeBridgeGateNotifier().on_gate_settled(
            gate, settlement, {}
        )
        self.assertEqual(outcome.status, "suppressed")
        self.assertTrue(outcome.reason)


if __name__ == "__main__":
    unittest.main()
