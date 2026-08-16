"""Contract analysis is pure: no DB, no network, no Playwright."""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.app_action_executor import app_action_gap  # noqa: E402
from agent.browser_action_adapter import _is_placeholder  # noqa: E402
from agent.workflow_contract import (  # noqa: E402
    CODE_APP_ACTIONS_MISSING,
    CODE_BROWSER_RUNTIME_UNAVAILABLE,
    CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE,
    CODE_UNKNOWN_STEP_REFERENCE,
    CODE_UNRESOLVED_APP_TARGET,
    CODE_UNRESOLVED_BROWSER_TARGET,
    CODE_UNSUPPORTED_APP_ACTION,
    SEVERITY_BLOCKING,
    SEVERITY_WARNING,
    analyze_workflow,
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


if __name__ == "__main__":
    unittest.main()
