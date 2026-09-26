"""Contract analysis is pure: no DB, no network, no Playwright."""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.app_action_executor import app_action_gap  # noqa: E402
from agent.browser_action_adapter import _is_placeholder  # noqa: E402
from code_bridge_core.workflow_contract import (  # noqa: E402
    CODE_APP_ACTIONS_MISSING,
    CODE_APPROVED_SCRIPT_UNUSED,
    CODE_BRANCH_PREDICATE_UNRESOLVABLE,
    CODE_BROWSER_RUNTIME_UNAVAILABLE,
    CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE,
    CODE_SCRIPT_PARAMETERS_UNFILLED,
    CODE_UNKNOWN_STEP_REFERENCE,
    CODE_UNRESOLVED_APP_TARGET,
    CODE_UNRESOLVED_BROWSER_TARGET,
    CODE_UNSUPPORTED_APP_ACTION,
    SEVERITY_BLOCKING,
    SEVERITY_WARNING,
    analyze_workflow,
    referenced_script_ids,
)


def _browser_step(step_id="open_cafe", url="configured_cafe_url"):
    return {
        "id": step_id,
        "type": "browser_action",
        "name": "카페 열기",
        "actions": [{"type": "navigate", "url": url}],
    }


class PlaceholderTargetTest(unittest.TestCase):
    def test_configured_target_blocks(self):
        report = analyze_workflow([_browser_step()], browser_readiness=None)

        blocking = report.by_code(CODE_UNRESOLVED_BROWSER_TARGET)
        self.assertEqual(len(blocking), 1)
        finding = blocking[0]
        self.assertEqual(finding.severity, SEVERITY_BLOCKING)
        self.assertEqual(finding.step_id, "open_cafe")
        self.assertEqual(finding.detail["action_index"], 0)
        self.assertEqual(finding.detail["field"], "url")
        self.assertEqual(finding.detail["value"], "configured_cafe_url")
        self.assertTrue(finding.ask)
        self.assertTrue(report.has_blocking)

    def test_concrete_target_produces_no_finding(self):
        report = analyze_workflow(
            [_browser_step(url="https://cafe.naver.com/example")],
            browser_readiness=None,
        )

        self.assertEqual(report.findings, [])

    def test_placeholder_rule_is_the_runtime_rule(self):
        # The commit gate must not drift from the adapter's own judgement.
        for value in ("configured_cafe_url", "{{approved_note_body}}", "recipient_required"):
            with self.subTest(value=value):
                self.assertTrue(_is_placeholder(value))
                report = analyze_workflow(
                    [_browser_step(url=value)], browser_readiness=None
                )
                self.assertEqual(len(report.by_code(CODE_UNRESOLVED_BROWSER_TARGET)), 1)

    def test_placeholder_in_a_step_with_no_runtime_target_is_ignored(self):
        # A shell step has no target the adapter resolves, so a word that looks
        # like a placeholder in it is just a word.
        flow = [
            {"id": "run_tests", "type": "shell", "script_id": "configured_script"}
        ]

        self.assertEqual(analyze_workflow(flow, browser_readiness=None).findings, [])

    def test_analysis_does_not_mutate_the_flow(self):
        flow = [_browser_step()]
        before = repr(flow)

        analyze_workflow(flow, browser_readiness={"ready": False})

        self.assertEqual(repr(flow), before)


class AppActionTargetTest(unittest.TestCase):
    """The device half of the gate. Each payload here is one the builder writes."""

    @staticmethod
    def _app_step(actions, step_id="launch_app", step_type="app_action"):
        return {
            "id": step_id,
            "type": step_type,
            "name": "앱 실행",
            "actions": actions,
        }

    def test_builder_written_verify_launch_blocks(self):
        # `configurator._verify_launch_actions` with no package extracted.
        flow = [
            self._app_step([{"type": "verify_launch", "app": "installed_app_from_previous_step"}])
        ]

        report = analyze_workflow(flow, browser_readiness=None)

        findings = report.by_code(CODE_UNRESOLVED_APP_TARGET)
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.severity, SEVERITY_BLOCKING)
        self.assertEqual(finding.step_id, "launch_app")
        self.assertEqual(finding.detail["action_index"], 0)
        self.assertEqual(finding.detail["action_type"], "verify_launch")
        # The field the author actually wrote, so a UI patches that key rather
        # than adding a second spelling of it.
        self.assertEqual(finding.detail["field"], "app")
        self.assertEqual(finding.detail["value"], "installed_app_from_previous_step")
        self.assertEqual(finding.detail["reason"], "app_action_needs_package_name")
        self.assertTrue(finding.ask)

    def test_builder_written_install_and_play_store_block(self):
        flow = [
            self._app_step(
                [
                    {"type": "install_app", "source": "user_provided_store_or_package"},
                    {"type": "open_play_store", "source": "user_provided_store_or_package"},
                ]
            )
        ]

        report = analyze_workflow(flow, browser_readiness=None)

        reasons = [f.detail["reason"] for f in report.by_code(CODE_UNRESOLVED_APP_TARGET)]
        self.assertEqual(
            reasons,
            ["app_install_needs_package_or_apk", "app_action_needs_package_name"],
        )

    def test_builder_written_join_tap_blocks(self):
        flow = [
            self._app_step(
                [{"type": "tap_text", "text": "join_or_apply_control_from_current_screen"}]
            )
        ]

        report = analyze_workflow(flow, browser_readiness=None)

        self.assertEqual(len(report.by_code(CODE_UNRESOLVED_APP_TARGET)), 1)

    def test_literal_app_step_commits(self):
        # The evidence actions the builder bundles alongside a launch carry
        # placeholder-looking targets the adapter never reads. Flagging them
        # would refuse a workflow the device runs happily.
        flow = [
            self._app_step(
                [
                    {"type": "verify_launch", "package": "com.example.app"},
                    {"type": "wait", "seconds": 1},
                    {"type": "read_screen", "target": "launched_app_screen"},
                    {"type": "screenshot", "label": "app_launch_result"},
                    {"type": "tap_text", "text": "확인"},
                    {"type": "install_app", "apk_path": "/tmp/app.apk"},
                ]
            )
        ]

        self.assertEqual(analyze_workflow(flow, browser_readiness=None).findings, [])

    def test_every_app_step_type_is_inspected(self):
        for step_type in ("app_action", "android_action", "mobile_action", "device_action"):
            with self.subTest(step_type=step_type):
                flow = [
                    self._app_step(
                        [{"type": "verify_launch", "app": "installed_app_from_previous_step"}],
                        step_type=step_type,
                    )
                ]

                report = analyze_workflow(flow, browser_readiness=None)

                self.assertEqual(len(report.by_code(CODE_UNRESOLVED_APP_TARGET)), 1)

    def test_browser_action_type_in_an_app_step_blocks(self):
        # `workflow_v2.ALLOWED_ACTION_TYPES` is one set shared with browser
        # steps, so `navigate` normalizes into an app step and then parks.
        flow = [self._app_step([{"type": "navigate", "url": "https://example.com"}])]

        report = analyze_workflow(flow, browser_readiness=None)

        findings = report.by_code(CODE_UNSUPPORTED_APP_ACTION)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SEVERITY_BLOCKING)
        self.assertEqual(findings[0].detail["action_type"], "navigate")

    def test_app_step_with_no_actions_blocks(self):
        for actions in ([], None):
            with self.subTest(actions=actions):
                step = self._app_step(actions)
                if actions is None:
                    step.pop("actions")

                report = analyze_workflow([step], browser_readiness=None)

                findings = report.by_code(CODE_APP_ACTIONS_MISSING)
                self.assertEqual(len(findings), 1)
                self.assertNotIn("action_index", findings[0].detail)

    def test_gate_judgement_is_the_adapter_judgement(self):
        # No second opinion: every finding here is one `app_action_gap` made.
        actions = [
            {"type": "verify_launch", "app": "installed_app_from_previous_step"},
            {"type": "read_screen", "target": "current_screen"},
            {"type": "press_key", "key": "sudo_make_me_a_sandwich"},
            {"type": "input_text", "text": "hello"},
        ]

        report = analyze_workflow([self._app_step(actions)], browser_readiness=None)

        self.assertEqual(
            [f.detail["reason"] for f in report.findings],
            [gap.reason for gap in (app_action_gap(a) for a in actions) if gap],
        )

    def test_analysis_does_not_mutate_an_app_flow(self):
        flow = [self._app_step([{"type": "verify_launch", "app": "installed_app_from_previous_step"}])]
        before = repr(flow)

        analyze_workflow(flow, browser_readiness=None)

        self.assertEqual(repr(flow), before)


class BrowserRuntimeTest(unittest.TestCase):
    def test_unavailable_runtime_warns_with_install_command(self):
        readiness = {
            "ready": False,
            "playwright_python": False,
            "chromium_executable": False,
            "install_command": "/opt/venv/bin/python -m playwright install chromium",
            "message": "Python Playwright package is not available.",
        }

        report = analyze_workflow(
            [_browser_step(url="https://example.com")], browser_readiness=readiness
        )

        findings = report.by_code(CODE_BROWSER_RUNTIME_UNAVAILABLE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SEVERITY_WARNING)
        self.assertEqual(
            findings[0].detail["install_command"],
            "/opt/venv/bin/python -m playwright install chromium",
        )
        self.assertIn("/opt/venv/bin/python", findings[0].ask)
        self.assertEqual(findings[0].detail["step_ids"], ["open_cafe"])
        # A missing runtime never blocks: the author cannot fix the server.
        self.assertFalse(report.has_blocking)

    def test_ready_runtime_produces_no_finding(self):
        report = analyze_workflow(
            [_browser_step(url="https://example.com")],
            browser_readiness={"ready": True, "install_command": "irrelevant"},
        )

        self.assertEqual(report.findings, [])

    def test_no_browser_step_means_no_runtime_finding(self):
        flow = [{"id": "run_tests", "type": "shell", "script_id": "s1"}]

        report = analyze_workflow(flow, browser_readiness={"ready": False})

        self.assertEqual(report.by_code(CODE_BROWSER_RUNTIME_UNAVAILABLE), [])

    def test_unknown_readiness_makes_no_claim(self):
        report = analyze_workflow(
            [_browser_step(url="https://example.com")], browser_readiness=None
        )

        self.assertEqual(report.findings, [])


class StepReferenceTest(unittest.TestCase):
    def _flow(self, description):
        return [
            {"id": "collect", "type": "llm", "instruction": "수집한다"},
            {"id": "report", "type": "llm", "description": description},
        ]

    def test_template_reference_to_missing_step_blocks(self):
        report = analyze_workflow(
            self._flow("{{steps.run_flutter_test}} 결과를 요약한다"),
            browser_readiness=None,
        )

        findings = report.by_code(CODE_UNKNOWN_STEP_REFERENCE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SEVERITY_BLOCKING)
        self.assertEqual(findings[0].detail["reference"], "run_flutter_test")
        self.assertEqual(findings[0].step_id, "report")

    def test_output_template_and_goto_forms_block(self):
        for text in ("{{run_flutter_test.output}} 확인", "goto:run_flutter_test"):
            with self.subTest(text=text):
                report = analyze_workflow(self._flow(text), browser_readiness=None)
                findings = report.by_code(CODE_UNKNOWN_STEP_REFERENCE)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].detail["reference"], "run_flutter_test")

    def test_template_reference_to_existing_step_is_clean(self):
        report = analyze_workflow(
            self._flow("{{steps.collect}} 결과를 요약한다"), browser_readiness=None
        )

        self.assertEqual(report.findings, [])

    @staticmethod
    def _notify_flow(body: str) -> list[dict]:
        return [
            {"id": "collect", "type": "llm", "instruction": "수집한다"},
            {
                "id": "report",
                "type": "notify",
                "name": "결과 알림",
                "notify": {"title": "완료", "body": body},
            },
        ]

    def test_a_notify_body_naming_a_missing_step_blocks(self):
        # The message is nested in `notify`, so a scan that only read
        # top-level fields never saw it. That gap let a typo be saved and
        # then delivered to a phone verbatim, because the runtime leaves an
        # unanswerable reference written as it is rather than guessing.
        report = analyze_workflow(
            self._notify_flow("오늘 등록한 제목: {{steps.pick_titel.text}}"),
            browser_readiness=None,
        )

        findings = report.by_code(CODE_UNKNOWN_STEP_REFERENCE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SEVERITY_BLOCKING)
        self.assertEqual(findings[0].detail["reference"], "pick_titel")
        self.assertEqual(findings[0].step_id, "report")

    def test_a_notify_title_is_judged_too(self):
        report = analyze_workflow(
            self._notify_flow("본문"), browser_readiness=None
        )
        self.assertEqual(report.findings, [])

        flow = self._notify_flow("본문")
        flow[1]["notify"]["title"] = "완료: {{steps.nowhere.text}}"
        report = analyze_workflow(flow, browser_readiness=None)

        findings = report.by_code(CODE_UNKNOWN_STEP_REFERENCE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].detail["reference"], "nowhere")

    def test_a_notify_body_naming_a_real_step_is_clean(self):
        report = analyze_workflow(
            self._notify_flow("오늘 등록한 제목: {{steps.collect.text}}"),
            browser_readiness=None,
        )

        self.assertEqual(report.findings, [])

    def test_quoted_snake_token_warns_only(self):
        report = analyze_workflow(
            self._flow("`run_flutter_test` 결과를 요약한다"), browser_readiness=None
        )

        findings = report.by_code(CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SEVERITY_WARNING)
        self.assertEqual(findings[0].detail["reference"], "run_flutter_test")
        self.assertFalse(report.has_blocking)

    def test_step_keyword_sentence_warns_only(self):
        report = analyze_workflow(
            self._flow("앞 단계 run_flutter_test 의 출력을 요약한다"),
            browser_readiness=None,
        )

        findings = report.by_code(CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SEVERITY_WARNING)
        self.assertFalse(report.has_blocking)

    def test_plain_snake_case_filename_mention_is_not_a_finding(self):
        report = analyze_workflow(
            self._flow("pubspec_yaml 파일을 읽어 버전을 확인한다"), browser_readiness=None
        )

        self.assertEqual(report.findings, [])

    def test_quoted_file_path_is_not_a_reference(self):
        for text in ("`pubspec.yaml` 확인", "`lib/foo_bar.dart` 확인"):
            with self.subTest(text=text):
                report = analyze_workflow(self._flow(text), browser_readiness=None)
                self.assertEqual(report.findings, [])


class ReportShapeTest(unittest.TestCase):
    def test_report_partitions_and_serializes(self):
        flow = [_browser_step(), {"id": "b", "type": "llm", "description": "`ghost_step`"}]
        readiness = {"ready": False, "install_command": "python -m playwright install chromium"}

        report = analyze_workflow(flow, browser_readiness=readiness)
        payload = report.to_dict()

        codes = {f.code for f in report.findings}
        self.assertEqual(
            codes,
            {
                CODE_UNRESOLVED_BROWSER_TARGET,
                CODE_BROWSER_RUNTIME_UNAVAILABLE,
                CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE,
            },
        )
        self.assertEqual(len(payload["blocking"]), 1)
        self.assertEqual(len(payload["warnings"]), 2)
        self.assertEqual(len(payload["findings"]), 3)
        self.assertEqual(
            set(payload["findings"][0]),
            {"severity", "code", "step_id", "detail", "ask"},
        )

    def test_non_list_flow_is_empty_report(self):
        for flow in (None, {}, "flow", [None, 3]):
            with self.subTest(flow=flow):
                self.assertEqual(analyze_workflow(flow, browser_readiness=None).findings, [])


def _shell_step(*, script_id="script_health", args=None, step_id="check_health"):
    step = {
        "id": step_id,
        "type": "shell",
        "name": "헬스 체크",
        "script_id": script_id,
    }
    if args is not None:
        step["script_args"] = args
    return step


def _registry(**overrides):
    script = {
        "id": "script_health",
        "name": "disk and health check",
        "default_args": [],
        "parameters": [
            {"name": "CHECK_DIR", "required": True, "description": "검사할 디렉터리"},
            {"name": "HEALTH_URL", "required": True, "description": "헬스 엔드포인트"},
            {"name": "BEARER_TOKEN", "required": False, "description": "토큰"},
        ],
    }
    script.update(overrides)
    return {script["id"]: script}


class ScriptParameterTest(unittest.TestCase):
    """A shell step that will stop at 3am asking for an argument.

    The reproduced defect: the script's `usage()` demanded CHECK_DIR and
    HEALTH_URL, the step was bound to it with `script_args` empty, and the
    runtime honestly refused to guess — so the run parked on step 1 with
    `waiting_for_user`. Nothing was wrong with the runtime. What was wrong is
    that this was knowable at save time and nothing looked.
    """

    def test_a_required_parameter_with_no_arguments_blocks(self):
        report = analyze_workflow(
            [_shell_step()], browser_readiness=None, scripts=_registry()
        )

        blocking = report.by_code(CODE_SCRIPT_PARAMETERS_UNFILLED)
        self.assertEqual(len(blocking), 1)
        finding = blocking[0]
        self.assertEqual(finding.severity, SEVERITY_BLOCKING)
        self.assertEqual(finding.step_id, "check_health")
        self.assertEqual(finding.detail["script_id"], "script_health")
        self.assertEqual(finding.detail["script_name"], "disk and health check")
        # The field a client puts input boxes on.
        self.assertEqual(finding.detail["field"], "script_args")
        # Only the required ones, in declaration order.
        self.assertEqual(finding.detail["parameter_names"], ["CHECK_DIR", "HEALTH_URL"])
        self.assertEqual(
            finding.detail["missing_parameters"][0],
            {"name": "CHECK_DIR", "description": "검사할 디렉터리"},
        )
        # The refusal has to name the script and the parameters, because it is
        # read back to the Configurator as the question to ask the user.
        self.assertIn("disk and health check", finding.ask)
        self.assertIn("CHECK_DIR", finding.ask)
        self.assertIn("HEALTH_URL", finding.ask)
        self.assertNotIn("BEARER_TOKEN", finding.ask)

    def test_arguments_supplied_by_the_step_produce_no_finding(self):
        report = analyze_workflow(
            [_shell_step(args=["/Users/me/project", "https://example.com/health"])],
            browser_readiness=None,
            scripts=_registry(),
        )
        self.assertEqual(report.findings, [])

    def test_arguments_baked_into_the_registration_produce_no_finding(self):
        # `build_command` concatenates the registry's default_args and the
        # step's script_args, so a script registered with its directory
        # already bound is a script the step need not pass one to.
        report = analyze_workflow(
            [_shell_step()],
            browser_readiness=None,
            scripts=_registry(default_args=["/srv", "https://example.com/health"]),
        )
        self.assertEqual(report.findings, [])

    def test_a_partially_filled_step_is_not_second_guessed(self):
        # One argument against two required parameters. Whether that is enough
        # depends on what the script does with `$@` — flags, defaults, `shift`
        # — none of which is readable from here. Counting and refusing would
        # be this module inventing a calling convention.
        report = analyze_workflow(
            [_shell_step(args=["/Users/me/project"])],
            browser_readiness=None,
            scripts=_registry(),
        )
        self.assertEqual(report.findings, [])

    def test_a_script_with_only_optional_parameters_produces_no_finding(self):
        report = analyze_workflow(
            [_shell_step()],
            browser_readiness=None,
            scripts=_registry(
                parameters=[{"name": "BEARER_TOKEN", "required": False}]
            ),
        )
        self.assertEqual(report.findings, [])

    def test_an_unknown_interface_is_not_grounds_to_refuse(self):
        # Every row registered before the interface was captured. Unknown is
        # not "proven safe" — it is simply not something to refuse work over.
        for parameters in (None, []):
            with self.subTest(parameters=parameters):
                report = analyze_workflow(
                    [_shell_step()],
                    browser_readiness=None,
                    scripts=_registry(parameters=parameters),
                )
                self.assertEqual(report.findings, [])

    def test_no_registry_lookup_makes_no_claim(self):
        # `scripts=None` is "the caller did not look", which must read exactly
        # as it did before this check existed.
        self.assertEqual(
            analyze_workflow([_shell_step()], browser_readiness=None).findings, []
        )

    def test_a_script_id_absent_from_the_lookup_is_left_to_the_runtime(self):
        report = analyze_workflow(
            [_shell_step(script_id="script_gone")],
            browser_readiness=None,
            scripts=_registry(),
        )
        self.assertEqual(report.findings, [])

    def test_only_shell_steps_are_judged(self):
        # A `script_id` on an llm step is not a script this runs.
        report = analyze_workflow(
            [{"id": "think", "type": "llm", "name": "판단", "script_id": "script_health"}],
            browser_readiness=None,
            scripts=_registry(),
        )
        self.assertEqual(report.findings, [])

    def test_referenced_ids_are_read_from_the_shell_steps_in_order(self):
        flow = [
            _shell_step(script_id="script_a", step_id="a"),
            {"id": "think", "type": "llm", "name": "판단", "script_id": "script_x"},
            _shell_step(script_id="script_b", step_id="b"),
            _shell_step(script_id="script_a", step_id="c"),
        ]
        self.assertEqual(referenced_script_ids(flow), ["script_a", "script_b"])


def _condition(when, *, step_id="usage_gate", target="diagnose", label="넘음"):
    return {
        "id": step_id,
        "type": "condition",
        "name": "판정",
        "branches": [
            {"label": label, "when": when, "target_step_id": target},
            {"label": "정상", "when": None, "target_step_id": target},
        ],
    }


def _reported_flow():
    """The four steps that were actually committed, with no shell step at all.

    `usage_gate` compares `{{disk_check.USED_PCT}}`; no step named `disk_check`
    exists, so `condition_eval._resolve_operand` raises `unbound_reference` on
    every run and the step takes neither arm. `analyze_workflow` reported this
    as one *warning* — the prose heuristic noticing the word in an llm step's
    observation — and the commit was accepted.
    """
    return [
        _condition({"left": "{{disk_check.USED_PCT}}", "op": "gt", "right": "90"}),
        {
            "id": "diagnose",
            "type": "llm",
            "name": "AI 디스크 진단",
            "observation": "{{disk_check}} 단계의 표준 출력과 종료 코드",
        },
    ]


class ConditionPredicateTest(unittest.TestCase):
    """A predicate naming a value nothing in the flow produces.

    Certain, not heuristic: the operand's entire content is reference syntax
    and the set of names a run scope can hold is closed
    (`run_scope.build_run_scope`). That is the line between this check and
    `possible_unknown_step_reference`, which reads free text, guesses, and
    therefore only ever warns.
    """

    def test_the_reported_flow_is_refused_and_names_the_reference(self):
        report = analyze_workflow(_reported_flow(), browser_readiness=None)
        blocking = report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE)
        self.assertEqual(len(blocking), 1)
        finding = blocking[0]
        self.assertEqual(finding.severity, SEVERITY_BLOCKING)
        self.assertEqual(finding.step_id, "usage_gate")
        self.assertEqual(finding.detail["reference"], "disk_check.USED_PCT")
        self.assertEqual(finding.detail["operand"], "left")
        self.assertIn("{{disk_check.USED_PCT}}", finding.ask)

    def test_the_prose_heuristic_is_untouched_and_still_only_warns(self):
        report = analyze_workflow(_reported_flow(), browser_readiness=None)
        heuristic = report.by_code(CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE)
        self.assertEqual(len(heuristic), 1)
        self.assertEqual(heuristic[0].severity, SEVERITY_WARNING)
        self.assertEqual(heuristic[0].detail["reference"], "disk_check")

    def test_a_step_fact_of_an_existing_step_is_satisfiable(self):
        flow = [
            {"id": "disk_check", "type": "shell", "name": "확인", "script_id": "s"},
            _condition(
                {"left": "{{steps.disk_check.exit_code}}", "op": "not_equals", "right": "0"}
            ),
            {"id": "diagnose", "type": "llm", "name": "진단"},
        ]
        report = analyze_workflow(flow, browser_readiness=None)
        self.assertEqual(report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE), [])

    def test_a_step_fact_the_run_scope_never_publishes_is_refused(self):
        # `run_scope.STEP_FACTS` is status/exit_code/stdout and nothing else.
        flow = [
            {"id": "disk_check", "type": "shell", "name": "확인", "script_id": "s"},
            _condition({"left": "{{steps.disk_check.stderr}}", "op": "is_not_empty"}),
            {"id": "diagnose", "type": "llm", "name": "진단"},
        ]
        report = analyze_workflow(flow, browser_readiness=None)
        blocking = report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE)
        self.assertEqual(len(blocking), 1)
        self.assertEqual(blocking[0].detail["reference"], "steps.disk_check.stderr")

    def test_a_fact_of_a_step_that_does_not_exist_is_refused(self):
        report = analyze_workflow(
            [_condition({"left": "{{steps.nowhere.status}}", "op": "equals", "right": "ok"})],
            browser_readiness=None,
        )
        blocking = report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE)
        self.assertEqual(len(blocking), 1)
        self.assertEqual(blocking[0].detail["reference"], "steps.nowhere.status")

    def test_a_browser_extract_supplies_the_name(self):
        flow = [
            {
                "id": "read_page",
                "type": "browser_action",
                "name": "읽기",
                "actions": [
                    {"type": "navigate", "url": "https://example.com"},
                    {"type": "extract", "name": "USED_PCT", "selector": "#pct"},
                ],
            },
            _condition({"left": "{{USED_PCT}}", "op": "gt", "right": "90"}),
            {"id": "diagnose", "type": "llm", "name": "진단"},
        ]
        report = analyze_workflow(flow, browser_readiness=None)
        self.assertEqual(report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE), [])

    def test_an_extract_declared_later_still_counts(self):
        """List order is not run order once an arm or a goto can go backwards.

        Refusing on position would be the guess this check exists to avoid.
        """
        flow = [
            _condition({"left": "{{cafe_id}}", "op": "is_not_empty"}),
            {"id": "diagnose", "type": "llm", "name": "진단"},
            {
                "id": "read_page",
                "type": "browser_action",
                "name": "읽기",
                "actions": [{"type": "extract", "name": "cafe_id", "selector": "#id"}],
            },
        ]
        report = analyze_workflow(flow, browser_readiness=None)
        self.assertEqual(report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE), [])

    def test_the_matches_pattern_is_literal_and_never_checked(self):
        # `condition_eval._matches` never substitutes `right`.
        flow = [
            {
                "id": "read_page",
                "type": "browser_action",
                "name": "읽기",
                "actions": [{"type": "extract", "name": "body", "selector": "#b"}],
            },
            _condition({"left": "{{body}}", "op": "matches", "right": "{{d+}}"}),
            {"id": "diagnose", "type": "llm", "name": "진단"},
        ]
        report = analyze_workflow(flow, browser_readiness=None)
        self.assertEqual(report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE), [])

    def test_a_unary_operator_right_operand_is_never_read(self):
        flow = [
            {"id": "run", "type": "shell", "name": "실행", "script_id": "s"},
            _condition(
                {
                    "left": "{{steps.run.stdout}}",
                    "op": "is_empty",
                    "right": "{{never_read}}",
                }
            ),
            {"id": "diagnose", "type": "llm", "name": "진단"},
        ]
        report = analyze_workflow(flow, browser_readiness=None)
        self.assertEqual(report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE), [])

    def test_both_operands_of_a_binary_operator_are_judged(self):
        report = analyze_workflow(
            [_condition({"left": "{{a}}", "op": "equals", "right": "{{b}}"})],
            browser_readiness=None,
        )
        blocking = report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE)
        self.assertEqual(
            [(f.detail["operand"], f.detail["reference"]) for f in blocking],
            [("left", "a"), ("right", "b")],
        )

    def test_the_default_arm_judges_nothing(self):
        report = analyze_workflow(
            [
                {
                    "id": "gate",
                    "type": "condition",
                    "name": "판정",
                    "branches": [{"label": "그 외", "when": None, "target_step_id": "gate"}],
                }
            ],
            browser_readiness=None,
        )
        self.assertEqual(report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE), [])

    def test_a_literal_predicate_produces_no_finding(self):
        report = analyze_workflow(
            [_condition({"left": "ok", "op": "equals", "right": "ok"})],
            browser_readiness=None,
        )
        self.assertEqual(report.by_code(CODE_BRANCH_PREDICATE_UNRESOLVABLE), [])

    def test_a_flow_with_no_condition_step_is_unchanged(self):
        flow = [{"id": "note", "type": "notify", "name": "알림"}]
        self.assertEqual(analyze_workflow(flow, browser_readiness=None).findings, [])

    def test_analysis_does_not_mutate_a_condition_flow(self):
        flow = _reported_flow()
        import copy

        before = copy.deepcopy(flow)
        analyze_workflow(flow, browser_readiness=None)
        self.assertEqual(flow, before)


class ApprovedScriptTest(unittest.TestCase):
    """A script the user approved that no step in the committed flow runs.

    `apply_registered_script` wires the approved script into a step and the
    reply the user read says so. The next turn's draft — the model re-emits
    the whole thing every turn — can drop that step again, and did: the
    observed session went flow 4 → 5 on approval and 5 → 4 on the following
    turn, then committed.
    """

    APPROVED = {"script_disk": {"id": "script_disk", "name": "Check disk usage"}}

    def test_an_approved_script_no_step_runs_is_refused(self):
        report = analyze_workflow(
            _reported_flow(), browser_readiness=None, approved_scripts=self.APPROVED
        )
        blocking = report.by_code(CODE_APPROVED_SCRIPT_UNUSED)
        self.assertEqual(len(blocking), 1)
        self.assertEqual(blocking[0].severity, SEVERITY_BLOCKING)
        self.assertEqual(blocking[0].detail["script_id"], "script_disk")
        self.assertIn("Check disk usage", blocking[0].ask)

    def test_a_step_naming_it_clears_the_finding(self):
        flow = [
            {"id": "disk_check", "type": "shell", "name": "확인", "script_id": "script_disk"},
            *_reported_flow(),
        ]
        report = analyze_workflow(
            flow, browser_readiness=None, approved_scripts=self.APPROVED
        )
        self.assertEqual(report.by_code(CODE_APPROVED_SCRIPT_UNUSED), [])

    def test_the_step_id_does_not_have_to_be_the_one_approval_chose(self):
        flow = [
            {"id": "anything", "type": "shell", "name": "확인", "script_id": "script_disk"},
        ]
        report = analyze_workflow(
            flow, browser_readiness=None, approved_scripts=self.APPROVED
        )
        self.assertEqual(report.by_code(CODE_APPROVED_SCRIPT_UNUSED), [])

    def test_no_approved_scripts_makes_no_claim(self):
        # Every caller but the builder commit. A registered script the flow
        # does not use is otherwise entirely ordinary.
        report = analyze_workflow(_reported_flow(), browser_readiness=None)
        self.assertEqual(report.by_code(CODE_APPROVED_SCRIPT_UNUSED), [])

    def test_only_a_shell_step_counts_as_running_it(self):
        flow = [{"id": "think", "type": "llm", "name": "판단", "script_id": "script_disk"}]
        report = analyze_workflow(
            flow, browser_readiness=None, approved_scripts=self.APPROVED
        )
        self.assertEqual(len(report.by_code(CODE_APPROVED_SCRIPT_UNUSED)), 1)


if __name__ == "__main__":
    unittest.main()
