"""Stored flow_json snapshot regression net (agent-flow-core T-B-08).

Why this file exists
--------------------

Code Bridge's safety model for agent workflows is deliberately migration-free:
`agents.flow_json` is stored exactly as the author saved it, and
`normalize_workflow` (agent/workflow_v2.py) re-normalizes that stored shape
from scratch on every task run — nothing ever writes the normalized shape back
(see the `_UNRESTRICTED_LEGACY_FIELDS` comment, workflow_v2.py:106-124, and
`_workflow_steps_for_task` in task_orchestrator.py). That means the *stored*
shapes in the wild are frozen history, while the normalizer is live code that
Phase 2 (graph representation of flows) and Phase 3 (runner delegation to the
kernel worker) are both about to touch.

The top acceptance criterion for those phases is: **not one already-saved
flow_json may break.** Because normalization failure is not a visible error —
the orchestrator catches `WorkflowNormalizationError` and treats the workflow
as zero steps (task_orchestrator.py) — a regression here silently turns a
working scheduled agent into a no-op the next time its schedule fires.

So this file pins, as literals, a set of representative stored flow_json
shapes (each sourced from the repo's own tests or from the real-agent example
documented in workflow_v2.py) together with the *exact* normalized output the
current normalizer produces for them (goldens captured by executing
`normalize_workflow` at the time of writing). For every snapshot it asserts:

  (a) `normalize_workflow` raises no exception,
  (b) it is deterministic — two runs over the same input agree,
  (c) the result equals the recorded golden, byte-for-value,
  (d) the stored input is not mutated (the store hands the same list out
      again on the next run, so mutation would corrupt the "stored" shape).

If Phase 2/3 work changes any of this — a renamed field, a reshaped policy
object, a newly rejected legacy shape, an alias key dropped — this test rings
first, before a live agent quietly stops running.

Goldens are full literal step lists, not structural digests: each golden is
small (1-3 steps) and the whole point is that *no* key of the normalized
shape may drift, because downstream consumers (orchestrator dispatch, policy
application, dashboard serialization, the Flutter step cards) read them all.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.workflow_v2 import normalize_workflow  # noqa: E402


# ---------------------------------------------------------------------------
# Stored-shape snapshots.
#
# Each entry reproduces a flow_json shape that demonstrably exists (or existed)
# in this repo's own usage — the source of every snapshot is noted next to it.
# Do NOT "modernize" these inputs: their point is to stay old.
# ---------------------------------------------------------------------------

SNAPSHOTS: dict[str, list[dict]] = {
    # Source: tests/test_agents_crud.py:47 — the default agent body used by
    # every CRUD test. No type, no policies: the maximally bare stored step.
    "minimal_bare_step": [{"id": "step_one", "name": "Plan"}],
    # Source: tests/test_agents_crud.py:330-345 (builder-commit draft) — no
    # id (id gets generated), legacy string policy "retry_once", explicit
    # tool_hint: null. The shape Configurator drafts were saved in.
    "legacy_string_policies_and_nulls": [
        {
            "name": "Retry once",
            "description": "Try once more before aborting.",
            "tool_hint": None,
            "success_criteria": "Done",
            "on_failure": "retry_once",
        }
    ],
    # Source: real scheduled agent agent_9f6af9fdeed24da8804ca8eb5d3368cc as
    # documented in workflow_v2.py:99-104 (_UNRESTRICTED_LEGACY_FIELDS): a
    # shell step carrying success_criteria / tool_hint: null / actions: []
    # that the per-type schema does not list for shell — stamped there by the
    # pre-scoping normalizer. Escalation pattern (goto llm on failure) and
    # legacy string success policy "end" from tests/test_shell_workflow_step.py
    # :112-126 and tests/test_workflow_success_policy.py:56-66. Also pins the
    # non-string script_args coercion (test_shell_workflow_step.py:105-110).
    "legacy_shell_with_offschema_fields": [
        {
            "id": "run_cycle",
            "type": "shell",
            "name": "사이클 실행",
            "script_id": "script_1",
            "script_args": ["--full", 2],
            "success_criteria": "사이클 정상 종료",
            "tool_hint": None,
            "actions": [],
            "on_failure": "goto_step:diagnose",
            "on_success": "end",
        },
        {"id": "diagnose", "type": "llm", "name": "Diagnose"},
    ],
    # Source: tests/test_workflow_runtime.py:60-71 — the parking flow: a
    # manual_handoff step with an object manual_handoff failure policy.
    "manual_handoff_park": [
        {
            "id": "captcha",
            "type": "manual_handoff",
            "name": "Resolve captcha",
            "on_failure": {
                "type": "manual_handoff",
                "prompt": "Complete captcha, then continue.",
                "resume": "same_step",
            },
        }
    ],
    # Source: tests/test_workflow_runtime.py:386-404 — goto_step failure
    # routing that jumps over a middle step. The first shape Phase 2's graph
    # representation must reproduce as an edge.
    "goto_failure_routing": [
        {
            "id": "check",
            "type": "llm",
            "name": "Check",
            "on_failure": {"type": "goto_step", "target_step_id": "handoff"},
        },
        {"id": "skip_me", "type": "llm", "name": "Skip me"},
        {
            "id": "handoff",
            "type": "manual_handoff",
            "name": "Handoff",
            "on_failure": {
                "type": "manual_handoff",
                "prompt": "Handle this manually.",
            },
        },
    ],
    # Source: tests/test_workflow_runtime.py:343-356 — object retry policy
    # with an explicit `then` chain (retry ownership is open decision #2 of
    # the initiative, so its stored shape must not move underneath us).
    "retry_object_policy": [
        {
            "id": "flaky",
            "type": "llm",
            "name": "Flaky",
            "on_failure": {
                "type": "retry",
                "max_attempts": 1,
                "then": {"type": "abort"},
            },
        }
    ],
    # Source: tests/test_workflow_runtime.py:220-238 — an llm step using every
    # extended authoring field at once (instruction/observation/memory pair/
    # tool_hint/actions/success_criteria + ask_user prompt).
    "rich_llm_step": [
        {
            "id": "inspect",
            "type": "llm",
            "name": "Inspect",
            "description": "Legacy fallback.",
            "instruction": "Inspect the failed build.",
            "observation": "Review the latest task output first.",
            "memory_read": "Load prior build caveats.",
            "memory_write": "Remember newly discovered build caveats.",
            "tool_hint": "openai",
            "actions": [{"type": "extract", "target": "build_log"}],
            "success_criteria": "Root cause is identified",
            "on_failure": {"type": "ask_user", "prompt": "Need more context."},
        }
    ],
    # Source: tests/test_agent_checkpoint_routes.py:164-185 — the route-level
    # E2E flow: browser_action with a navigate action, then a manual gate.
    "browser_then_handoff": [
        {
            "id": "open_page",
            "type": "browser_action",
            "name": "Open page",
            "actions": [
                {"type": "navigate", "url": "data:text/html,<h1>Route E2E</h1>"}
            ],
            "success_criteria": "Page opens.",
        },
        {
            "id": "human_gate",
            "type": "manual_handoff",
            "name": "Manual confirmation",
            "on_failure": {
                "type": "manual_handoff",
                "prompt": "Confirm the route E2E handoff.",
                "resume": "same_step",
            },
        },
    ],
    # Source: tests/test_workflow_runtime.py:620-631 — app_action step with a
    # device binding (android_device_id) followed by an llm report step.
    "app_action_device_flow": [
        {
            "id": "read_requests",
            "type": "app_action",
            "name": "Read requests",
            "android_device_id": "emulator-5554",
            "actions": [{"type": "read_screen"}],
        },
        {"id": "report", "type": "llm", "name": "Report"},
    ],
    # Sources: tests/test_mcp_injection_need.py:132-136 (legacy `step_type`
    # alias key instead of `type`), tests/test_notify_step.py:108-117 (notify
    # payload block), tests/test_workflow_runtime.py:154-163 (mcp_tool with
    # tool_hint + instruction).
    "step_type_alias_notify_mcp": [
        {"id": "open", "step_type": "browser_action", "name": "Open"},
        {
            "id": "tell_user",
            "type": "notify",
            "name": "Warn about disk",
            "notify": {
                "title": "Disk almost full",
                "body": "8% left",
                "level": "warning",
            },
        },
        {
            "id": "call_it",
            "type": "mcp_tool",
            "name": "Run the workflow",
            "tool_hint": "n8n",
            "instruction": "Trigger the daily digest.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Goldens: the exact output normalize_workflow produced for each snapshot at
# the time this file was written (captured by executing the current code, not
# hand-derived). If a change makes one of these differ, that change alters
# what every already-saved agent executes as — prove it is intended and
# behavior-preserving for stored agents before updating the literal.
# ---------------------------------------------------------------------------

GOLDENS: dict[str, list[dict]] = {
    "minimal_bare_step": [
        {
            "id": "step_one",
            "type": "llm",
            "name": "Plan",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [],
        }
    ],
    "legacy_string_policies_and_nulls": [
        {
            "id": "step_1",
            "type": "llm",
            "name": "Retry once",
            "description": "Try once more before aborting.",
            "tool_hint": None,
            "success_criteria": "Done",
            "on_failure": {
                "type": "retry",
                "max_attempts": 1,
                "then": {"type": "abort"},
            },
            "on_success": {"type": "continue"},
            "actions": [],
        }
    ],
    "legacy_shell_with_offschema_fields": [
        {
            "id": "run_cycle",
            "type": "shell",
            "name": "사이클 실행",
            "description": "",
            # The carve-out at work: these three stay on a shell step even
            # though WORKFLOW_STEP_SCHEMA does not list them for shell.
            "tool_hint": None,
            "success_criteria": "사이클 정상 종료",
            "actions": [],
            "on_failure": {"type": "goto_step", "target_step_id": "diagnose"},
            "on_success": {"type": "end"},
            "script_id": "script_1",
            # Non-string args are coerced to strings.
            "script_args": ["--full", "2"],
        },
        {
            "id": "diagnose",
            "type": "llm",
            "name": "Diagnose",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [],
        },
    ],
    "manual_handoff_park": [
        {
            "id": "captcha",
            "type": "manual_handoff",
            "name": "Resolve captcha",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "Complete captcha, then continue.",
            },
            "on_success": {"type": "continue"},
            "actions": [],
        }
    ],
    "goto_failure_routing": [
        {
            "id": "check",
            "type": "llm",
            "name": "Check",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "goto_step", "target_step_id": "handoff"},
            "on_success": {"type": "continue"},
            "actions": [],
        },
        {
            "id": "skip_me",
            "type": "llm",
            "name": "Skip me",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [],
        },
        {
            "id": "handoff",
            "type": "manual_handoff",
            "name": "Handoff",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "Handle this manually.",
            },
            "on_success": {"type": "continue"},
            "actions": [],
        },
    ],
    "retry_object_policy": [
        {
            "id": "flaky",
            "type": "llm",
            "name": "Flaky",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {
                "type": "retry",
                "max_attempts": 1,
                "then": {"type": "abort"},
            },
            "on_success": {"type": "continue"},
            "actions": [],
        }
    ],
    "rich_llm_step": [
        {
            "id": "inspect",
            "type": "llm",
            "name": "Inspect",
            "description": "Legacy fallback.",
            # Type-specific authoring fields are carried through unchanged.
            "instruction": "Inspect the failed build.",
            "observation": "Review the latest task output first.",
            "memory_read": "Load prior build caveats.",
            "memory_write": "Remember newly discovered build caveats.",
            "tool_hint": "openai",
            "success_criteria": "Root cause is identified",
            "on_failure": {
                "type": "ask_user",
                "resume": "same_step",
                "prompt": "Need more context.",
            },
            "on_success": {"type": "continue"},
            "actions": [{"type": "extract", "target": "build_log"}],
        }
    ],
    "browser_then_handoff": [
        {
            "id": "open_page",
            "type": "browser_action",
            "name": "Open page",
            "description": "",
            "tool_hint": None,
            "success_criteria": "Page opens.",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [
                {"type": "navigate", "url": "data:text/html,<h1>Route E2E</h1>"}
            ],
        },
        {
            "id": "human_gate",
            "type": "manual_handoff",
            "name": "Manual confirmation",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {
                "type": "manual_handoff",
                "resume": "same_step",
                "prompt": "Confirm the route E2E handoff.",
            },
            "on_success": {"type": "continue"},
            "actions": [],
        },
    ],
    "app_action_device_flow": [
        {
            "id": "read_requests",
            "type": "app_action",
            "name": "Read requests",
            "description": "",
            "android_device_id": "emulator-5554",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [{"type": "read_screen"}],
        },
        {
            "id": "report",
            "type": "llm",
            "name": "Report",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [],
        },
    ],
    "step_type_alias_notify_mcp": [
        {
            "id": "open",
            # The legacy alias key is *kept* alongside the normalized `type`
            # (normalize deep-copies the raw step and only overwrites known
            # keys). Dropping `step_type` from the output would be a silent
            # contract change for consumers that round-trip raw steps.
            "step_type": "browser_action",
            "type": "browser_action",
            "name": "Open",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [],
        },
        {
            "id": "tell_user",
            "type": "notify",
            "name": "Warn about disk",
            "description": "",
            "tool_hint": None,
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [],
            "notify": {
                "title": "Disk almost full",
                "body": "8% left",
                "level": "warning",
            },
        },
        {
            "id": "call_it",
            "type": "mcp_tool",
            "name": "Run the workflow",
            "description": "",
            "instruction": "Trigger the daily digest.",
            "tool_hint": "n8n",
            "success_criteria": "",
            "on_failure": {"type": "ask_user", "resume": "same_step"},
            "on_success": {"type": "continue"},
            "actions": [],
        },
    ],
}


class FlowJsonSnapshotRegressionTest(unittest.TestCase):
    """Every stored-shape snapshot must keep normalizing exactly as today."""

    def test_snapshots_and_goldens_cover_the_same_cases(self) -> None:
        self.assertEqual(set(SNAPSHOTS), set(GOLDENS))
        # The ticket asks for 6-10 representative shapes; keep the net wide.
        self.assertGreaterEqual(len(SNAPSHOTS), 6)

    def test_every_snapshot_normalizes_without_error(self) -> None:
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                normalize_workflow(copy.deepcopy(snapshot))

    def test_normalization_is_deterministic(self) -> None:
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                first = normalize_workflow(snapshot)
                second = normalize_workflow(snapshot)
                self.assertEqual(first, second)

    def test_normalized_output_matches_the_recorded_golden(self) -> None:
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                self.assertEqual(
                    normalize_workflow(snapshot),
                    GOLDENS[name],
                    msg=(
                        f"normalize_workflow drifted for stored shape "
                        f"'{name}'. This is what an already-saved agent "
                        f"would now execute as — if the change is intended, "
                        f"prove it preserves stored-agent behavior, then "
                        f"update the golden."
                    ),
                )

    def test_the_stored_input_is_not_mutated(self) -> None:
        # The store hands the same flow_json out on the next run; an in-place
        # edit during normalization would corrupt the stored contract.
        for name, snapshot in SNAPSHOTS.items():
            with self.subTest(snapshot=name):
                frozen = copy.deepcopy(snapshot)
                normalize_workflow(snapshot)
                self.assertEqual(snapshot, frozen)


if __name__ == "__main__":
    unittest.main()
