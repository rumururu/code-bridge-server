"""Contract analysis is pure: no DB, no network, no Playwright."""

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import _is_placeholder  # noqa: E402
from agent.workflow_contract import (  # noqa: E402
    CODE_BROWSER_RUNTIME_UNAVAILABLE,
    CODE_POSSIBLE_UNKNOWN_STEP_REFERENCE,
    CODE_UNKNOWN_STEP_REFERENCE,
    CODE_UNRESOLVED_BROWSER_TARGET,
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

    def test_placeholder_in_non_browser_step_is_ignored(self):
        flow = [
            {
                "id": "tap_thing",
                "type": "app_action",
                "actions": [{"type": "tap", "target": "configured_button"}],
            }
        ]

        self.assertEqual(analyze_workflow(flow, browser_readiness=None).findings, [])

    def test_analysis_does_not_mutate_the_flow(self):
        flow = [_browser_step()]
        before = repr(flow)

        analyze_workflow(flow, browser_readiness={"ready": False})

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
