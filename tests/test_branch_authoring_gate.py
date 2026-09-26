"""The authoring gate's three branch judgements (T-H-12).

A condition step is the one step type that can make the rest of a workflow
unreachable, and the two ways it goes wrong are both silent until 3am: an arm
that names a step nobody wrote (the run aborts when that arm is taken), and a
table of predicates with nothing to catch a value none of them matched (the
step *fails* — RUNNER_BRANCHING_SPEC 4.2 is explicit that an unmatched run is
not quietly carried on). The third judgement is the one only a graph can make:
a condition's arms replace its sequential successor (spec 7.2), so the step
someone wrote directly underneath it is reached only if an arm says so.

Two lines this file exists to hold:

* **The predicate's meaning is never judged.** Whether
  ``{{steps.build.exit_code}}`` will hold a value is knowable only once the run
  has produced one. ``_check_placeholder_targets`` learned this the expensive
  way — it read every ``{{name}}`` as unresolved until it was taught that an
  earlier ``extract`` supplies some of them, and until then the one pattern the
  browser vocabulary existed for could not be saved. So the tests below assert
  that a predicate whose operands cannot possibly be checked at authoring time
  produces *nothing*.

* **A workflow with no branching condition gets the verdict it always got.**
  The check returns early, so nothing about the existing corpus of saved
  workflows changes — including the unreachable steps that a condition-free
  flow can already have (a step written after ``on_success: end``). Widening
  that is a separate decision about a separate population, and this is not it.
"""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core.workflow_contract import (  # noqa: E402
    CODE_BRANCH_NO_DEFAULT,
    CODE_BRANCH_TARGET_UNKNOWN,
    CODE_BRANCH_UNREACHABLE_STEP,
    SEVERITY_BLOCKING,
    SEVERITY_WARNING,
    _check_condition_branches,
    analyze_workflow,
)
from code_bridge_core.workflow_v2 import normalize_workflow  # noqa: E402


def _codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def _report(flow: list[dict]):
    return analyze_workflow(flow, browser_readiness=None)


def _notify(step_id: str, **extra) -> dict:
    step = {
        "id": step_id,
        "type": "notify",
        "name": step_id,
        "notify": {"title": step_id, "body": "x", "level": "info"},
    }
    step.update(extra)
    return step


def _condition(step_id: str, branches: list[dict]) -> dict:
    return {"id": step_id, "type": "condition", "name": step_id, "branches": branches}


def _when(left: str, op: str, right: str | None = None) -> dict:
    predicate = {"left": left, "op": op}
    if right is not None:
        predicate["right"] = right
    return predicate


# Spec 10.1's shape: a shell step, a condition on its exit code, and one
# report step per outcome. Two predicates that between them cover every value,
# so there is deliberately no default arm.
def _example_a() -> list[dict]:
    return [
        {
            "id": "run_sync",
            "type": "shell",
            "name": "run_sync",
            "script_id": "sync_devices",
            "on_failure": {"type": "continue"},
        },
        _condition(
            "check_exit",
            [
                {
                    "label": "성공",
                    "when": _when("{{steps.run_sync.exit_code}}", "equals", "0"),
                    "target_step_id": "report_ok",
                },
                {
                    "label": "실패",
                    "when": _when("{{steps.run_sync.exit_code}}", "not_equals", "0"),
                    "target_step_id": "report_fail",
                },
            ],
        ),
        _notify("report_ok", on_success={"type": "end"}),
        _notify("report_fail"),
    ]


class NoDefaultBranchTest(unittest.TestCase):
    def test_a_table_with_no_fallback_warns_once(self) -> None:
        report = _report(normalize_workflow(_example_a()))

        self.assertEqual(_codes(report), [CODE_BRANCH_NO_DEFAULT])
        finding = report.findings[0]
        self.assertEqual(finding.severity, SEVERITY_WARNING)
        self.assertEqual(finding.step_id, "check_exit")
        self.assertEqual(finding.detail["branch_count"], 2)
        self.assertEqual(finding.detail["labels"], ["성공", "실패"])

    def test_it_is_a_warning_because_covering_every_case_is_legitimate(self) -> None:
        """Spec 1.3: a default arm is not required, and spec 10.1 — the first
        use case this feature was built for — has none. Blocking here would
        refuse the shape the spec holds up as correct."""

        report = _report(normalize_workflow(_example_a()))

        self.assertFalse(report.has_blocking)

    def test_a_fallback_arm_silences_it(self) -> None:
        flow = _example_a()
        flow[1]["branches"][1] = {"label": "그 외", "target_step_id": "report_fail"}

        report = _report(normalize_workflow(flow))

        self.assertEqual(_codes(report), [])

    def test_a_condition_without_branches_is_not_a_table_at_all(self) -> None:
        """Every workflow saved before track H. It still honours `on_success`,
        so it neither needs a fallback nor breaks the step after it."""

        flow = [
            {"id": "look", "type": "condition", "name": "look"},
            _notify("tell"),
        ]

        self.assertEqual(_codes(_report(normalize_workflow(flow))), [])

    def test_an_empty_branch_list_is_left_to_the_topology_gate(self) -> None:
        """Spec 1.5 gives `branches: []` to the topology gate as
        `branch.empty`. It really does fail at runtime
        (`condition_eval.evaluate_branches` raises `empty_branches`), but
        reporting it here as well would say one problem twice in two
        vocabularies and leave a client guessing which to render."""

        flow = [_condition("look", []), _notify("tell")]
        normalized = normalize_workflow(flow)
        self.assertEqual(normalized[0].get("branches"), [])

        self.assertEqual(_codes(_report(normalized)), [])


class UnknownBranchTargetTest(unittest.TestCase):
    """`normalize_workflow` refuses an unknown target outright, so these flows
    are handed to the gate raw — which is exactly what the module's contract
    says it accepts ("normalized or raw"). The finding is the gate's answer for
    every caller that has not normalized first; the commit routes have, and
    there the normalizer speaks first with a 400 of its own."""

    def test_an_arm_naming_a_step_that_does_not_exist_blocks(self) -> None:
        flow = _example_a()
        flow[1]["branches"][0]["target_step_id"] = "report_okay"

        report = _report(flow)

        blocking = report.by_code(CODE_BRANCH_TARGET_UNKNOWN)
        self.assertEqual(len(blocking), 1)
        finding = blocking[0]
        self.assertEqual(finding.severity, SEVERITY_BLOCKING)
        self.assertEqual(finding.step_id, "check_exit")
        self.assertEqual(finding.detail["branch_index"], 0)
        self.assertEqual(finding.detail["label"], "성공")
        self.assertEqual(finding.detail["target_step_id"], "report_okay")
        self.assertIn("report_ok", finding.detail["known_step_ids"])
        self.assertIn("report_okay", finding.ask)

    def test_an_arm_with_no_target_at_all_blocks(self) -> None:
        flow = _example_a()
        flow[1]["branches"][0].pop("target_step_id")

        report = _report(flow)

        blocking = report.by_code(CODE_BRANCH_TARGET_UNKNOWN)
        self.assertEqual(len(blocking), 1)
        self.assertIsNone(blocking[0].detail["target_step_id"])

    def test_the_normalizer_still_refuses_it_first(self) -> None:
        """Pinned so the gate's defence is understood as a second opinion for
        raw input, not as the only thing standing between an unknown target and
        the database."""

        from code_bridge_core.workflow_v2 import WorkflowNormalizationError  # noqa: PLC0415

        flow = _example_a()
        flow[1]["branches"][0]["target_step_id"] = "report_okay"

        with self.assertRaises(WorkflowNormalizationError):
            normalize_workflow(flow)

    def test_an_arm_pointing_at_a_real_step_is_silent(self) -> None:
        report = _report(normalize_workflow(_example_a()))

        self.assertEqual(report.by_code(CODE_BRANCH_TARGET_UNKNOWN), [])


class UnreachableStepTest(unittest.TestCase):
    def test_a_step_no_arm_names_is_reported(self) -> None:
        """The mistake the rule exists for: a step written under the condition,
        which before branching would simply have run next."""

        flow = _example_a()
        flow.insert(2, _notify("draft_note"))

        report = _report(normalize_workflow(flow))

        unreachable = report.by_code(CODE_BRANCH_UNREACHABLE_STEP)
        self.assertEqual([f.step_id for f in unreachable], ["draft_note"])
        self.assertEqual(unreachable[0].severity, SEVERITY_WARNING)
        self.assertEqual(unreachable[0].detail["step_index"], 2)
        self.assertEqual(unreachable[0].detail["branching_step_ids"], ["check_exit"])

    def test_a_step_reached_by_falling_out_of_an_arm_is_not_reported(self) -> None:
        """`report_fail` continues into the step after it, so that step is
        reachable even though no arm names it. Reporting it would be the
        reachability answer a naive "is this step named anywhere" scan gives,
        and it would be wrong."""

        flow = _example_a()
        flow.append(_notify("cleanup"))

        report = _report(normalize_workflow(flow))

        self.assertEqual(report.by_code(CODE_BRANCH_UNREACHABLE_STEP), [])

    def test_a_backward_arm_is_a_loop_not_an_unreachable_step(self) -> None:
        """Spec 10.3's polling loop. The arm points at a step *earlier* in the
        list, which a forward-only reading would call a dead end."""

        flow = [
            {
                "id": "poll",
                "type": "shell",
                "name": "poll",
                "script_id": "poll_status",
                "on_failure": {"type": "continue"},
            },
            _condition(
                "gate",
                [
                    {
                        "label": "완료",
                        "when": _when("{{steps.poll.stdout}}", "contains", "DONE"),
                        "target_step_id": "done",
                    },
                    {"label": "대기", "target_step_id": "poll"},
                ],
            ),
            _notify("done"),
        ]

        self.assertEqual(_codes(_report(normalize_workflow(flow))), [])

    def test_a_step_reached_only_by_a_failure_goto_is_not_reported(self) -> None:
        """`on_failure: goto_step` is a real way in — the whole point of a
        diagnosis step placed after an `end`."""

        flow = _example_a()
        flow[0]["on_failure"] = {"type": "goto_step", "target_step_id": "escalate"}
        flow[2]["on_success"] = {"type": "end"}
        flow[3]["on_success"] = {"type": "end"}
        flow.append(_notify("escalate"))

        report = _report(normalize_workflow(flow))

        self.assertEqual(report.by_code(CODE_BRANCH_UNREACHABLE_STEP), [])

    def test_a_step_reached_only_by_a_retry_escalation_goto_is_not_reported(
        self,
    ) -> None:
        """The failure policy's jump can sit inside a retry chain
        (`{"type": "retry", "then": {"type": "goto_step"}}`). Reading only the
        outer type would miss it — which is why the gate borrows the runner's
        own `_failure_goto_target` rather than reading the dict itself."""

        flow = _example_a()
        flow[0]["on_failure"] = {
            "type": "retry",
            "max_attempts": 2,
            "then": {"type": "goto_step", "target_step_id": "escalate"},
        }
        flow[2]["on_success"] = {"type": "end"}
        flow[3]["on_success"] = {"type": "end"}
        flow.append(_notify("escalate"))

        report = _report(normalize_workflow(flow))

        self.assertEqual(report.by_code(CODE_BRANCH_UNREACHABLE_STEP), [])


class ThePredicateIsNotJudgedTest(unittest.TestCase):
    """The exclusion the ticket names, and the reason the gate stays credible.

    Every operand below is unknowable at authoring time. A gate that guessed
    would refuse workflows that run.
    """

    def test_a_reference_to_a_fact_no_step_has_produced_yet_is_fine(self) -> None:
        report = _report(normalize_workflow(_example_a()))

        self.assertEqual(report.by_code(CODE_BRANCH_TARGET_UNKNOWN), [])
        self.assertFalse(report.has_blocking)

    def test_operands_that_can_only_ever_be_wrong_at_runtime_are_fine(self) -> None:
        """`gt` against a word, and a `matches` pattern that will match
        nothing. Both fail *the run* if they are ever reached with those
        values, and neither is knowable now: the left operand is a reference,
        and the author may well know something the gate does not."""

        flow = _example_a()
        flow[1]["branches"] = [
            {
                "label": "큼",
                "when": _when("{{steps.run_sync.stdout}}", "gt", "많음"),
                "target_step_id": "report_ok",
            },
            {
                "label": "패턴",
                "when": _when("{{steps.run_sync.stdout}}", "matches", "^NEVER$"),
                "target_step_id": "report_fail",
            },
            {"label": "그 외", "target_step_id": "report_fail"},
        ]

        report = _report(normalize_workflow(flow))

        self.assertEqual(_codes(report), [])

    def test_a_literal_left_operand_is_not_second_guessed(self) -> None:
        """`equals ""` is expressible on purpose (spec 1.2) and a constant
        predicate is a legitimate, if odd, thing to write."""

        flow = _example_a()
        flow[1]["branches"] = [
            {
                "label": "빈 값",
                "when": _when("{{steps.run_sync.stdout}}", "equals", ""),
                "target_step_id": "report_ok",
            },
            {"label": "그 외", "target_step_id": "report_fail"},
        ]

        report = _report(normalize_workflow(flow))

        self.assertEqual(_codes(report), [])


class AConditionFreeWorkflowIsUntouchedTest(unittest.TestCase):
    """The acceptance criterion, asserted as a property rather than by
    inspecting one verdict: the check returns nothing at all unless the flow
    declares an arm, so nothing it could say can reach a workflow without one.
    """

    def _flows(self) -> dict[str, list[dict]]:
        return {
            "empty": [],
            "single_llm": [
                {
                    "id": "check_disk",
                    "name": "Check disk",
                    "description": "df -h 로 여유 공간을 읽는다.",
                    "success_criteria": "여유 공간 비율을 얻었다",
                }
            ],
            "browser_placeholder": [
                {
                    "id": "open_cafe",
                    "type": "browser_action",
                    "name": "카페 열기",
                    "actions": [{"type": "navigate", "url": "configured_cafe_url"}],
                }
            ],
            "goto_chain": [
                {
                    "id": "build",
                    "type": "shell",
                    "name": "build",
                    "script_id": "build_app",
                    "on_failure": {"type": "goto_step", "target_step_id": "escalate"},
                    "on_success": {"type": "end"},
                },
                _notify("escalate"),
            ],
            # A condition-free flow that *does* have an unreachable step: the
            # notify after an `end` is reached by nothing. It stays unreported,
            # which is the point — the existing corpus does not change verdict.
            "unreachable_after_end": [
                {
                    "id": "work",
                    "type": "llm",
                    "name": "work",
                    "on_success": {"type": "end"},
                },
                _notify("never_runs"),
            ],
            "condition_without_branches": [
                {"id": "look", "type": "condition", "name": "look"},
                _notify("tell"),
            ],
        }

    def test_the_branch_check_contributes_nothing(self) -> None:
        for name, flow in self._flows().items():
            with self.subTest(flow=name):
                normalized = normalize_workflow(flow)
                self.assertEqual(_check_condition_branches(normalized), [])

    def test_the_findings_carry_no_branch_code(self) -> None:
        branch_codes = {
            CODE_BRANCH_NO_DEFAULT,
            CODE_BRANCH_TARGET_UNKNOWN,
            CODE_BRANCH_UNREACHABLE_STEP,
        }
        for name, flow in self._flows().items():
            with self.subTest(flow=name):
                report = _report(normalize_workflow(flow))
                self.assertEqual(set(_codes(report)) & branch_codes, set())


if __name__ == "__main__":
    unittest.main()


class FailureContinueReachabilityTest(unittest.TestCase):
    """E6 (LINEAR_FLOW_MAPPING 3.0.1) — a failure-continue is a way in.

    ``run_tests`` succeeds into a goto that jumps over the diagnosis step and
    fails into ``continue``, which walks straight into it. The reachability
    walk drew only the goto, so with a condition anywhere in the workflow the
    diagnosis step was reported unreachable — while the runner reached it on
    every failed run.
    """

    def _flow(self) -> list[dict]:
        return [
            _condition(
                "gate",
                [
                    {"label": "go", "when": _when("{{x}}", "equals", "1"),
                     "target_step_id": "run_tests"},
                    {"label": "else", "when": None, "target_step_id": "run_tests"},
                ],
            ),
            {
                "id": "run_tests",
                "type": "shell",
                "name": "run_tests",
                "script_id": "tests",
                "on_success": {"type": "goto_step", "target_step_id": "record_pass"},
                "on_failure": {"type": "continue"},
            },
            _notify("analyze_failure", on_success={"type": "end"}),
            _notify("record_pass", on_success={"type": "end"}),
        ]

    def test_the_step_after_a_goto_plus_failure_continue_is_reachable(self) -> None:
        report = _report(normalize_workflow(self._flow()))
        self.assertEqual(report.by_code(CODE_BRANCH_UNREACHABLE_STEP), [])

    def test_without_the_failure_continue_it_is_still_unreachable(self) -> None:
        # The control: same shape, failure aborts instead — now nothing walks
        # into analyze_failure, and the warning is right to fire.
        flow = self._flow()
        flow[1]["on_failure"] = {"type": "abort"}
        report = _report(normalize_workflow(flow))
        self.assertEqual(
            [f.step_id for f in report.by_code(CODE_BRANCH_UNREACHABLE_STEP)],
            ["analyze_failure"],
        )
