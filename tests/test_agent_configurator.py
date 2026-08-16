"""Agent Builder Configurator post-processing tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.configurator import (  # noqa: E402
    build_configurator_system_prompt,
    create_builder_session,
)


class AgentConfiguratorTest(unittest.TestCase):
    def test_llm_refinement_replaces_auto_generated_task_goal(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
First fallback-shaped draft.

```draft
{
  "name": "Web Automation Agent",
  "description": "Runs a browser workflow with target confirmation, safe interaction, and result reporting.",
  "system_prompt": "Run the requested browser workflow safely.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {
      "id": "open_page",
      "type": "browser_action",
      "name": "Open page",
      "description": "Open the configured URL.",
      "actions": [{"type": "navigate", "target": "configured_url"}],
      "success_criteria": "Page loads"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="매시간 웹사이트 상태를 확인하고 문제가 있으면 요약 보고하는 Agent 초안을 만들어줘.",
        )

        self.assertIsNotNone(session.task_draft)
        self.assertEqual(
            session.task_draft.goal,
            "Runs a browser workflow with target confirmation, safe interaction, and result reporting.",
        )

        session.apply_llm_response(
            """
Refined draft.

```draft
{
  "name": "사이트 상태 모니터링 봇",
  "description": "매시간 지정한 웹사이트를 열어 정상 동작 여부를 확인하고, 오류가 발견되면 요약해서 보고한다.",
  "system_prompt": "지정된 웹사이트의 가용성과 정상 동작을 정기적으로 점검한다.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {
      "id": "load_target_site",
      "type": "browser_action",
      "name": "대상 사이트 열기",
      "description": "지정된 URL을 열고 페이지 로드 결과를 캡처한다.",
      "actions": [{"type": "navigate", "target": "configured_url"}],
      "success_criteria": "페이지가 로드됨"
    }
  ],
  "memory_seeds": []
}
```
READY_TO_COMMIT
""",
            user_message="매시간 웹사이트 상태를 확인하고 문제가 있으면 요약 보고하는 Agent 초안을 만들어줘.",
        )

        self.assertIsNotNone(session.task_draft)
        self.assertEqual(
            session.task_draft.goal,
            "매시간 지정한 웹사이트를 열어 정상 동작 여부를 확인하고, 오류가 발견되면 요약해서 보고한다.",
        )

    def test_sparse_android_review_exchange_request_gets_no_invented_prose(self) -> None:
        """A blank description/system_prompt stays blank.

        This used to assert the opposite -- that the sparse draft came back
        "saveable", with a description and a full system prompt written for it
        by `_fallback_agent_description` / `_fallback_system_prompt`. The
        system prompt is the agent's runtime instruction set: writing one from
        a keyword match means the agent runs on rules the user never read and
        the model never wrote. `provider_id` still defaults, because that
        selects a configured backend rather than authoring text.
        """

        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
I'll create a simple agent draft.

```draft
{
  "name": "Android Review Exchange",
  "description": "",
  "system_prompt": "",
  "provider_id": null,
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
""",
            user_message="Make an Android review exchange agent.",
        )

        draft = session.current_draft
        self.assertEqual(draft.provider_id, "openai")
        self.assertEqual(draft.system_prompt, "")
        self.assertEqual(draft.description, "")
        self.assertEqual(draft.name, "Android Review Exchange")
        self.assertTrue(any(tool.mcp_id == "app_action" for tool in draft.tools))
        step_ids = [step.id for step in draft.flow]
        self.assertIn("collect_review_exchange_requests", step_ids)
        self.assertIn("select_review_exchange_candidate", step_ids)
        self.assertIn("approve_join_request_submission", step_ids)
        self.assertIn("submit_join_request", step_ids)
        self.assertIn("open_app_store_listing", step_ids)
        self.assertIn("approve_app_install", step_ids)
        self.assertIn("install_app", step_ids)
        self.assertIn("verify_app_launch", step_ids)
        self.assertIn("record_execution_result", step_ids)
        self.assertTrue(all(step.type != "browser_action" for step in draft.flow))

    def test_missing_provider_reuses_existing_provider(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
Initial draft.

```draft
{
  "name": "Gemini Helper",
  "description": "Runs a generic workflow.",
  "system_prompt": "Ask for missing details and report results.",
  "provider_id": "google",
  "tools": [],
  "flow": [
    {
      "id": "plan",
      "type": "llm",
      "name": "Plan",
      "description": "Plan the work.",
      "success_criteria": "Plan is ready"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="Create a helper with Gemini.",
        )

        session.apply_llm_response(
            """
Updated the description.

```draft
{
  "name": "Gemini Helper",
  "description": "Runs a generic workflow and reports status.",
  "system_prompt": "",
  "provider_id": null,
  "tools": [],
  "flow": [
    {
      "id": "plan",
      "type": "llm",
      "name": "Plan",
      "description": "Plan the work.",
      "success_criteria": "Plan is ready"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="Keep it simple and make it saveable.",
        )

        self.assertEqual(session.current_draft.provider_id, "google")
        self.assertTrue(session.current_draft.system_prompt.strip())

    def test_short_korean_android_review_exchange_request_becomes_saveable_app_draft(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
초안입니다.

```draft
{
  "name": "Android Review Exchange Agent",
  "description": "Android 리뷰 품앗이 요청을 확인하고 참여한 뒤 앱 설치와 실행을 확인한다.",
  "system_prompt": "Android 리뷰 품앗이 요청을 안전하게 처리하고 앱 설치/실행 결과를 보고한다.",
  "provider_id": "openai",
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Android 리뷰 품앗이 요청을 매시간 확인해서 참여하고 "
                "앱 설치/실행 확인하는 에이전트 만들어줘"
            ),
        )

        draft = session.current_draft
        self.assertTrue((draft.system_prompt or "").strip())
        self.assertEqual(draft.provider_id, "openai")
        self.assertTrue(draft.flow)
        self.assertTrue(any(tool.mcp_id == "app_action" for tool in draft.tools))
        self.assertTrue(all(step.type != "browser_action" for step in draft.flow))

        step_ids = [step.id for step in draft.flow]
        self.assertIn("prepare_join_request_context", step_ids)
        self.assertIn("submit_join_request", step_ids)
        self.assertIn("install_app", step_ids)
        self.assertIn("verify_app_launch", step_ids)

        submit_step = _step_by_id(draft.flow, "submit_join_request")
        self.assertEqual(submit_step.type, "app_action")
        self.assertEqual(submit_step.tool_hint, "android_adb")
        self.assertEqual(submit_step.actions[0]["type"], "tap_text")

        install_step = _step_by_id(draft.flow, "install_app")
        self.assertEqual(install_step.type, "app_action")
        self.assertEqual(install_step.tool_hint, "android_adb")
        self.assertEqual(install_step.actions[0]["type"], "install_app")

        verify_step = _step_by_id(draft.flow, "verify_app_launch")
        self.assertEqual(verify_step.type, "app_action")
        self.assertEqual(verify_step.tool_hint, "android_adb")
        self.assertEqual(verify_step.actions[0]["type"], "verify_launch")

        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "every 1h")
        self.assertTrue((session.task_draft.goal or "").strip())

    def test_empty_browser_actions_in_android_review_exchange_are_repaired(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
초안입니다.

```draft
{
  "name": "Android 리뷰 품앗이 자동 참여 봇",
  "description": "Android 앱 리뷰 품앗이 요청을 주기적으로 확인해 처리합니다.",
  "system_prompt": "리뷰 품앗이 요청을 안전하게 처리하고 결과를 기록합니다.",
  "provider_id": "anthropic",
  "tools": [
    {"mcp_id": "playwright", "tool_names": ["browser_navigate"]}
  ],
  "flow": [
    {
      "id": "check_requests",
      "name": "품앗이 요청 확인",
      "type": "browser_action",
      "description": "지정 커뮤니티에서 신규 리뷰 품앗이 요청 목록을 수집한다.",
      "tool_hint": "playwright",
      "actions": [],
      "success_criteria": "신규 요청이 추출됨"
    },
    {
      "id": "install_app",
      "name": "앱 설치",
      "type": "browser_action",
      "description": "Play Store에서 요청 앱을 설치한다.",
      "tool_hint": "playwright",
      "actions": [],
      "success_criteria": "앱 설치 완료"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Android 리뷰 품앗이 요청을 매시간 확인해서 참여하고 "
                "앱 설치/실행 확인하는 에이전트 만들어줘"
            ),
        )

        draft = session.current_draft
        check_step = _step_by_id(draft.flow, "check_requests")
        self.assertEqual(check_step.type, "app_action")
        self.assertEqual(check_step.tool_hint, "android_adb")
        self.assertEqual(check_step.actions[0]["type"], "read_screen")

        install_step = _step_by_id(draft.flow, "install_app")
        self.assertEqual(install_step.type, "app_action")
        self.assertEqual(install_step.tool_hint, "android_adb")
        self.assertEqual(install_step.actions[0]["type"], "install_app")

        runtime_steps = [
            step
            for step in draft.flow
            if step.type in {"browser_action", "app_action", "android_action", "mobile_action", "device_action"}
        ]
        self.assertTrue(runtime_steps)
        self.assertTrue(all(step.actions for step in runtime_steps))

    def test_generic_install_capability_request_gets_guarded_workflow_steps(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
I'll keep this generic and fill in the workflow.

```draft
{
  "name": "Mobile Install Verifier",
  "description": "Opens a store listing, installs an app, verifies launch, and records the result.",
  "system_prompt": "Track the request ID and ask before sensitive external actions.",
  "provider_id": "openai",
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Create a generic agent that remembers the request id, opens the Play Store, "
                "installs the app, verifies launch, and saves the result."
            ),
        )

        draft = session.current_draft
        step_ids = [step.id for step in draft.flow]
        self.assertIn("remember_request_id", step_ids)
        self.assertIn("open_app_store_listing", step_ids)
        self.assertIn("approve_app_install", step_ids)
        self.assertIn("install_app", step_ids)
        self.assertIn("verify_app_launch", step_ids)
        self.assertIn("record_execution_result", step_ids)
        self.assertEqual(_step_by_id(draft.flow, "approve_app_install").type, "approval_gate")
        install_step = _step_by_id(draft.flow, "install_app")
        self.assertEqual(install_step.type, "app_action")
        self.assertEqual(install_step.actions[0]["type"], "install_app")
        self.assertEqual(install_step.on_failure["type"], "manual_handoff")
        verify_step = _step_by_id(draft.flow, "verify_app_launch")
        self.assertEqual(verify_step.type, "app_action")
        self.assertEqual(verify_step.actions[0]["type"], "verify_launch")
        open_store_step = _step_by_id(draft.flow, "open_app_store_listing")
        self.assertEqual(open_store_step.type, "app_action")
        self.assertEqual(open_store_step.tool_hint, "android_adb")
        self.assertEqual(open_store_step.actions[0]["type"], "open_play_store")
        self.assertTrue(any(tool.mcp_id == "app_action" for tool in draft.tools))

    def test_empty_app_mcp_steps_are_normalized_to_app_actions(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
I'll model this after the existing runner.

```draft
{
  "name": "Android Review Exchange Worker",
  "description": "Handles review exchange requests.",
  "system_prompt": "Use app_action tools only, never a browser.",
  "provider_id": "openai",
  "tools": [{"mcp_id": "app_action", "tool_names": ["install_package", "launch_app"]}],
      "flow": [
    {
      "id": "collect_requests",
      "type": "mcp_tool",
      "name": "Collect requests",
      "description": "Collect current review exchange requests.",
      "tool_hint": "app_action",
      "actions": [],
      "success_criteria": "Requests collected"
    },
    {
      "id": "join_exchange",
      "type": "mcp_tool",
      "name": "Join exchange",
      "description": "Join the selected review exchange request.",
      "tool_hint": "app_action",
      "actions": [],
      "success_criteria": "Joined"
    },
    {
      "id": "install_app",
      "type": "mcp_tool",
      "name": "Install app",
      "description": "Install the target app.",
      "tool_hint": "app_action",
      "actions": [],
      "success_criteria": "Installed"
    },
    {
      "id": "verify_launch",
      "type": "mcp_tool",
      "name": "Verify launch",
      "description": "Launch and verify the app.",
      "tool_hint": "app_action",
      "actions": [],
      "success_criteria": "Launched"
    },
    {
      "id": "save_result",
      "type": "mcp_tool",
      "name": "Save result",
      "description": "Save the result to memory.",
      "tool_hint": "app_action",
      "actions": [],
      "success_criteria": "Saved"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Create a generic Android review exchange agent. join request, "
                "install app, verify launch, save result. Use app_action not browser."
            ),
        )

        flow = session.current_draft.flow
        collect_step = _step_by_id(flow, "collect_requests")
        join_step = _step_by_id(flow, "join_exchange")
        install_step = _step_by_id(flow, "install_app")
        verify_step = _step_by_id(flow, "verify_launch")
        save_step = _step_by_id(flow, "save_result")
        self.assertEqual(collect_step.type, "app_action")
        self.assertEqual(collect_step.actions[0]["type"], "read_screen")
        self.assertEqual(join_step.type, "app_action")
        self.assertEqual(join_step.actions[0]["type"], "tap_text")
        self.assertEqual(install_step.type, "app_action")
        self.assertEqual(install_step.actions[0]["type"], "install_app")
        self.assertEqual(verify_step.type, "app_action")
        self.assertEqual(verify_step.actions[0]["type"], "verify_launch")
        self.assertEqual(save_step.type, "llm")
        self.assertEqual(save_step.actions, [])
        self.assertTrue(all(step.type != "browser_action" for step in flow))

    def test_concrete_android_package_launch_request_gets_executable_actions(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "",
  "description": "",
  "system_prompt": "",
  "provider_id": null,
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
""",
            user_message=(
                "매 30분마다 Android 설정 앱 com.android.settings 을 열고 "
                "화면 상태를 읽고 스크린샷을 남겨줘. Use app_action not browser."
            ),
        )

        draft = session.current_draft
        self.assertTrue(any(tool.mcp_id == "app_action" for tool in draft.tools))
        verify_step = _step_by_id(draft.flow, "verify_app_launch")
        self.assertEqual(verify_step.type, "app_action")
        self.assertEqual(verify_step.actions[0]["type"], "verify_launch")
        self.assertEqual(verify_step.actions[0]["package"], "com.android.settings")
        self.assertEqual(verify_step.actions[1]["type"], "wait")
        self.assertEqual(verify_step.actions[2]["type"], "read_screen")
        self.assertEqual(verify_step.actions[3]["type"], "screenshot")
        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "every 30m")

    def test_english_android_settings_launch_request_gets_app_action_flow(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "",
  "description": "",
  "system_prompt": "",
  "provider_id": null,
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Create a login-free Android app automation. Every 30 minutes "
                "open Android Settings app package com.android.settings, read "
                "the visible screen state, and save a screenshot. Use app_action "
                "not browser."
            ),
        )

        draft = session.current_draft
        # This draft block does not even parse: `AgentDraft.name` has
        # min_length=1, so `"name": ""` fails validation and the block is
        # dropped whole (`builder_configurator_invalid_draft_block`). The
        # identity fields therefore stay at their session defaults. That the
        # old assertions here read "Android App Agent" and a paragraph of
        # operating instructions shows what the fabrication was doing: a draft
        # the model never successfully produced was presented to the user as
        # a named, described, fully-instructed agent, assembled entirely from
        # keyword matches on their own sentence. Commit now refuses (422)
        # naming the missing fields.
        self.assertIsNone(draft.name)
        self.assertIsNone(draft.description)
        self.assertEqual(draft.system_prompt, "")
        self.assertTrue(any(tool.mcp_id == "app_action" for tool in draft.tools))
        self.assertTrue(all(step.type != "browser_action" for step in draft.flow))
        verify_step = _step_by_id(draft.flow, "verify_app_launch")
        self.assertEqual(verify_step.type, "app_action")
        self.assertEqual(verify_step.actions[0]["type"], "verify_launch")
        self.assertEqual(verify_step.actions[0]["package"], "com.android.settings")
        self.assertEqual([action["type"] for action in verify_step.actions[1:]], ["wait", "read_screen", "screenshot"])
        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "every 30m")

    def test_android_device_mcp_tool_settings_step_is_normalized_to_app_action(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "Settings Screen Watcher",
  "description": "Every 30 minutes, opens the device Settings app, reads the visible screen state, and saves a screenshot for later review.",
  "system_prompt": "Monitor the Android Settings screen.",
  "provider_id": "openai",
  "tools": [
    {"mcp_id": "android-device", "tool_names": ["launch_app", "dump_screen", "screenshot"]}
  ],
  "flow": [
    {
      "id": "open_settings",
      "name": "Open Settings",
      "type": "mcp_tool",
      "description": "Launch the Settings app on the target device.",
      "tool_hint": "android-device",
      "actions": [],
      "success_criteria": "Settings app is visible in foreground.",
      "instruction": "Launch the system Settings app on the connected device."
    },
    {
      "id": "read_screen",
      "name": "Read screen state",
      "type": "mcp_tool",
      "description": "Dump the visible UI hierarchy as text for diffing.",
      "tool_hint": "android-device",
      "actions": [],
      "success_criteria": "Non-empty screen dump returned."
    },
    {
      "id": "screenshot",
      "name": "Save screenshot",
      "type": "mcp_tool",
      "description": "Capture a PNG of the current Settings screen.",
      "tool_hint": "android-device",
      "actions": [],
      "success_criteria": "Screenshot file written to disk."
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="Open Settings every 30m. Read screen. Screenshot.",
        )

        draft = session.current_draft
        self.assertTrue(any(tool.mcp_id == "app_action" for tool in draft.tools))
        open_step = _step_by_id(draft.flow, "open_settings")
        self.assertEqual(open_step.type, "app_action")
        self.assertEqual(open_step.tool_hint, "android_adb")
        self.assertEqual(open_step.actions, [{"type": "verify_launch", "package": "com.android.settings"}])
        self.assertEqual(_step_by_id(draft.flow, "read_screen").type, "app_action")
        self.assertEqual(_step_by_id(draft.flow, "screenshot").type, "app_action")
        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "every 30m")

    def test_browser_action_settings_steps_are_normalized_to_app_action(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "Settings Screen Watcher",
  "description": "Every 30 minutes, verifies Android Settings, reads the screen, and saves a screenshot.",
  "system_prompt": "Use the connected Android device. Do not use browser automation.",
  "provider_id": "openai",
  "tools": [
    {"mcp_id": "app_action", "tool_names": ["verify_launch", "read_screen", "screenshot"]}
  ],
  "flow": [
    {
      "id": "verify_settings_launch",
      "name": "Verify Settings launch",
      "type": "browser_action",
      "description": "Verify launch of Android Settings package com.android.settings.",
      "tool_hint": "app_action",
      "actions": [{"type": "assert"}],
      "success_criteria": "Settings app launches."
    },
    {
      "id": "wait_after_launch",
      "name": "Wait after launch",
      "type": "browser_action",
      "description": "Wait briefly after the Android app launches.",
      "tool_hint": "app_action",
      "actions": [{"type": "wait"}],
      "success_criteria": "Screen is stable."
    },
    {
      "id": "read_visible_screen",
      "name": "Read visible screen",
      "type": "browser_action",
      "description": "Read the visible screen state from the device UI hierarchy.",
      "tool_hint": "app_action",
      "actions": [{"type": "extract"}],
      "success_criteria": "Screen state was read."
    },
    {
      "id": "capture_settings_screen",
      "name": "Capture Settings screen",
      "type": "browser_action",
      "description": "Take a screenshot of the Android Settings screen.",
      "tool_hint": "app_action",
      "actions": [{"type": "screenshot"}],
      "success_criteria": "Screenshot file is saved."
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Create app_action agent. Every 30m verify_launch "
                "com.android.settings, wait, read_screen, screenshot."
            ),
        )

        draft = session.current_draft
        self.assertTrue(all(step.type == "app_action" for step in draft.flow))
        verify_step = _step_by_id(draft.flow, "verify_settings_launch")
        self.assertEqual(verify_step.tool_hint, "android_adb")
        self.assertEqual(verify_step.actions, [{"type": "verify_launch", "package": "com.android.settings"}])
        wait_step = _step_by_id(draft.flow, "wait_after_launch")
        self.assertEqual(wait_step.actions, [{"type": "wait", "seconds": 1}])
        read_step = _step_by_id(draft.flow, "read_visible_screen")
        self.assertEqual(read_step.actions, [{"type": "read_screen", "target": "current_screen"}])
        screenshot_step = _step_by_id(draft.flow, "capture_settings_screen")
        self.assertEqual(screenshot_step.actions[0]["type"], "screenshot")
        self.assertTrue(screenshot_step.actions[0]["label"])
        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "every 30m")

    def test_split_settings_workflow_keeps_step_specific_app_actions(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "App Action Agent",
  "description": "Every 30 minutes, launches com.android.settings, waits for it to load, reads the on-screen UI, and captures a screenshot as evidence.",
  "system_prompt": "Use app_action on the connected Android device.",
  "provider_id": "openai",
  "tools": [
    {"mcp_id": "android-adb", "tool_names": ["shell", "screencap", "uiautomator_dump", "pull"]}
  ],
  "flow": [
    {
      "id": "verify_launch",
      "name": "Launch Settings",
      "type": "app_action",
      "description": "Launch com.android.settings via ADB",
      "tool_hint": "android_adb",
      "actions": [{"type": "verify_launch", "package": "com.android.settings"}],
      "success_criteria": "com.android.settings is in the foreground."
    },
    {
      "id": "wait_for_load",
      "name": "Wait",
      "type": "llm",
      "description": "Allow the UI to render before inspection",
      "actions": [],
      "success_criteria": "2 seconds elapsed.",
      "instruction": "Sleep ~2 seconds so the Settings activity has time to draw."
    },
    {
      "id": "read_screen",
      "name": "Read Screen",
      "type": "app_action",
      "description": "Dump the current UI hierarchy for verification",
      "tool_hint": "android_adb",
      "actions": [{"type": "read_screen", "target": "current_screen"}],
      "success_criteria": "UI dump contains Settings nodes."
    },
    {
      "id": "capture_screenshot",
      "name": "Screenshot",
      "type": "app_action",
      "description": "Capture device screenshot as run evidence",
      "tool_hint": "android_adb",
      "actions": [{"type": "screenshot"}],
      "success_criteria": "PNG file is non-empty."
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Create app_action agent. Every 30m verify_launch "
                "com.android.settings, wait, read_screen, screenshot."
            ),
        )

        flow = session.current_draft.flow
        self.assertEqual(_step_by_id(flow, "verify_launch").actions, [{"type": "verify_launch", "package": "com.android.settings"}])
        self.assertEqual(_step_by_id(flow, "wait_for_load").type, "app_action")
        self.assertEqual(_step_by_id(flow, "wait_for_load").actions, [{"type": "wait", "seconds": 2}])
        self.assertEqual(_step_by_id(flow, "read_screen").actions, [{"type": "read_screen", "target": "current_screen"}])
        self.assertEqual(_step_by_id(flow, "capture_screenshot").actions, [{"type": "screenshot", "label": "capture_screenshot_screenshot_1"}])

    def test_split_settings_workflow_does_not_expand_empty_launch_step(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "Android Settings Monitor",
  "description": "Every 30 minutes, opens the Android Settings app, captures the current screen state, and reports it.",
  "system_prompt": "Use Android app actions exactly as configured.",
  "provider_id": "openai",
  "tools": [{"mcp_id": "app_action", "tool_names": ["verify_launch", "wait", "read_screen", "screenshot"]}],
  "flow": [
    {
      "id": "launch_settings",
      "name": "Launch Settings",
      "type": "app_action",
      "description": "Open the Android Settings app on the connected device.",
      "tool_hint": "android_adb",
      "actions": [],
      "success_criteria": "Settings app is in the foreground."
    },
    {
      "id": "wait_render",
      "name": "Wait 2 seconds",
      "type": "llm",
      "description": "Give the Settings UI time to render before reading.",
      "actions": [],
      "instruction": "Wait approximately 2000ms so the Settings UI finishes rendering.",
      "success_criteria": "2 seconds have elapsed."
    },
    {
      "id": "read_screen",
      "name": "Read screen",
      "type": "app_action",
      "description": "Dump the current UI hierarchy to extract visible text.",
      "tool_hint": "android_adb",
      "actions": [{"type": "read_screen", "target": "current_screen"}],
      "success_criteria": "Visible text is captured."
    },
    {
      "id": "screenshot",
      "name": "Screenshot",
      "type": "app_action",
      "description": "Capture a PNG of the current Settings screen.",
      "tool_hint": "android_adb",
      "actions": [{"type": "screenshot", "label": "screenshot_screen"}],
      "success_criteria": "A PNG file is saved."
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Make an app agent: 4 steps. Launch Android Settings, "
                "wait 2 seconds, read screen, screenshot. Run every 30 minutes."
            ),
        )

        flow = session.current_draft.flow
        self.assertEqual(_step_by_id(flow, "launch_settings").actions, [{"type": "verify_launch", "package": "com.android.settings"}])
        self.assertEqual(_step_by_id(flow, "wait_render").actions, [{"type": "wait", "seconds": 2}])
        self.assertEqual(_step_by_id(flow, "read_screen").actions, [{"type": "read_screen", "target": "current_screen"}])
        self.assertEqual(_step_by_id(flow, "screenshot").actions, [{"type": "screenshot", "label": "screenshot_screen"}])

    def test_app_screenshot_actions_get_default_labels(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "Screenshot fixer",
  "description": "Verifies app state.",
  "system_prompt": "Use app_action not browser.",
  "provider_id": "openai",
  "tools": [{"mcp_id": "app_action", "tool_names": ["read_screen"]}],
  "flow": [
    {
      "id": "verify_state",
      "type": "mcp_tool",
      "name": "Verify state",
      "description": "Read screen and capture proof.",
      "tool_hint": "app_action",
      "actions": [{"type": "read_screen", "target": "current_screen"}, {"type": "screenshot"}],
      "success_criteria": "Proof captured"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="Create an Android app_action verifier, not browser.",
        )

        step = _step_by_id(session.current_draft.flow, "verify_state")
        self.assertEqual(step.type, "app_action")
        self.assertEqual(step.actions[1]["label"], "verify_state_screenshot_2")

    def test_install_step_with_play_store_text_still_installs(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
```draft
{
  "name": "Installer",
  "description": "Installs from Play Store.",
  "system_prompt": "Use app_action not browser.",
  "provider_id": "openai",
  "tools": [{"mcp_id": "app_action", "tool_names": ["install_app"]}],
  "flow": [
    {
      "id": "install_app",
      "type": "mcp_tool",
      "name": "Install app",
      "description": "Open Play Store and install app.",
      "tool_hint": "app_action",
      "actions": [{"type": "open_play_store", "source": "user_provided_store_or_package"}],
      "success_criteria": "Installed"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="Install app from Play Store using app_action not browser.",
        )

        step = _step_by_id(session.current_draft.flow, "install_app")
        self.assertEqual(step.type, "app_action")
        self.assertTrue(any(action["type"] == "install_app" for action in step.actions))

    def test_unhinted_app_mcp_context_step_becomes_llm_step(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
Drafted from the chat request.

```draft
{
  "name": "Android Review Exchange Worker",
  "description": "Handles Android review exchange requests.",
  "system_prompt": "Use app_action tools only, never a browser.",
  "provider_id": "openai",
  "tools": [{"mcp_id": "app_action", "tool_names": ["read_screen", "install_app"]}],
  "flow": [
    {
      "id": "collect_requests",
      "type": "app_action",
      "name": "Collect requests",
      "description": "Collect current review exchange requests.",
      "tool_hint": "android_adb",
      "actions": [{"type": "read_screen", "target": "current_review_exchange_request_list"}],
      "success_criteria": "Requests collected"
    },
    {
      "id": "select_candidate",
      "type": "mcp_tool",
      "name": "Select candidate",
      "description": "Select the safest request to process.",
      "actions": [],
      "success_criteria": "Candidate selected"
    },
    {
      "id": "install_app",
      "type": "mcp_tool",
      "name": "Install app",
      "description": "Install the target app.",
      "tool_hint": "app_action",
      "actions": [],
      "success_criteria": "Installed"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Android review exchange agent. Every hour. openai. Use app_action not browser. "
                "Steps collect requests join exchange open Play Store install app verify launch save result."
            ),
        )

        select_step = _step_by_id(session.current_draft.flow, "select_candidate")
        self.assertEqual(select_step.type, "llm")
        self.assertIsNone(select_step.tool_hint)
        self.assertEqual(select_step.actions, [])

    def test_existing_app_actions_prevent_duplicate_template_steps(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
Draft with model-generated app action steps.

```draft
{
  "name": "Android Review Exchange Bot",
  "description": "Handles Android review exchange requests.",
  "system_prompt": "Use app_action and android_adb only.",
  "provider_id": "openai",
  "tools": [{"mcp_id": "app_action", "tool_names": ["open_play_store", "install_app", "verify_launch"]}],
  "flow": [
    {
      "id": "collect_requests",
      "type": "llm",
      "name": "Collect requests",
      "description": "Collect and select a request.",
      "success_criteria": "Request selected"
    },
    {
      "id": "open_play_store",
      "type": "app_action",
      "name": "Open Play Store",
      "description": "Open the Play Store listing.",
      "tool_hint": "android_adb",
      "actions": [{"type": "open_play_store", "source": "user_provided_store_or_package"}],
      "success_criteria": "Listing visible"
    },
    {
      "id": "install_app",
      "type": "app_action",
      "name": "Install app",
      "description": "Install app.",
      "tool_hint": "android_adb",
      "actions": [{"type": "install_app", "source": "user_provided_store_or_package"}],
      "success_criteria": "Installed"
    },
    {
      "id": "verify_launch",
      "type": "app_action",
      "name": "Verify launch",
      "description": "Verify launch.",
      "tool_hint": "android_adb",
      "actions": [{"type": "verify_launch", "app": "installed_app_from_previous_step"}],
      "success_criteria": "Launched"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Run every hour. Use app_action and android_adb only. "
                "Steps collect requests join exchange open Play Store install app verify launch save result."
            ),
        )

        step_ids = [step.id for step in session.current_draft.flow]
        self.assertIn("open_play_store", step_ids)
        self.assertNotIn("open_app_store_listing", step_ids)
        self.assertIn("verify_launch", step_ids)
        self.assertNotIn("verify_app_launch", step_ids)

    def test_add_install_step_preserves_existing_workflow(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
Initial workflow.

```draft
{
  "name": "Release Helper",
  "description": "Checks release readiness and reports status.",
  "system_prompt": "Preserve the release workflow.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {
      "id": "collect_release_context",
      "type": "llm",
      "name": "Collect release context",
      "description": "Read the release request and constraints.",
      "success_criteria": "Release context is understood",
      "on_failure": "ask_user"
    },
    {
      "id": "report_release_status",
      "type": "llm",
      "name": "Report release status",
      "description": "Summarize readiness and blockers.",
      "success_criteria": "Status is reported",
      "on_failure": "ask_user"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="Create a release helper with a simple readiness workflow.",
        )

        session.apply_llm_response(
            """
Added the install step.

```draft
{
  "name": "Release Helper",
  "description": "Checks release readiness and reports status.",
  "system_prompt": "Preserve the release workflow.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {
      "id": "install_app",
      "type": "app_action",
      "name": "Install app",
      "description": "Install the app after approval.",
      "actions": [{"type": "install_app", "source": "user_provided_store_or_package"}],
      "success_criteria": "Install result is known",
      "on_failure": "ask_user"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message="Add an install app step that opens the Play Store before installing.",
        )

        step_ids = [step.id for step in session.current_draft.flow]
        self.assertLess(
            step_ids.index("collect_release_context"),
            step_ids.index("install_app"),
        )
        self.assertIn("report_release_status", step_ids)
        self.assertIn("open_app_store_listing", step_ids)
        self.assertIn("approve_app_install", step_ids)

    def test_generic_capability_detection_does_not_require_named_template(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
            """
Generic mobile QA workflow.

```draft
{
  "name": "Independent Mobile QA",
  "description": "Handles a generic app store install and launch verification.",
  "system_prompt": "Use generic steps and request approval for external actions.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {
      "id": "qa_context",
      "type": "llm",
      "name": "QA context",
      "description": "Collect target app and request details.",
      "success_criteria": "The target app is known",
      "on_failure": "ask_user"
    }
  ],
  "memory_seeds": []
}
```
""",
            user_message=(
                "For a completely generic mobile QA helper, add request id memory, "
                "Play Store open, app install, launch verification, and result recording."
            ),
        )

        step_ids = [step.id for step in session.current_draft.flow]
        self.assertIn("qa_context", step_ids)
        self.assertIn("remember_request_id", step_ids)
        self.assertIn("open_app_store_listing", step_ids)
        self.assertIn("install_app", step_ids)
        self.assertIn("verify_app_launch", step_ids)
        self.assertIn("record_execution_result", step_ids)

    def test_naver_hourly_web_agent_gets_tools_and_schedule_but_no_invented_flow(
        self,
    ) -> None:
        """An empty flow stays empty, and the reply says so.

        This test used to assert the opposite: that the server answered an
        empty `flow` with a six-step `_naver_cafe_flow()` template, complete
        with cafe URLs, captcha handling and duplicate checks. The user had
        asked for none of it and could not tell it from the model's own work.
        Under the project's "no fabricated fallback" rule the server may
        declare the browser tool the request needs, but it may not author the
        workflow -- so the flow stays as the model left it and the turn says
        out loud that there are no steps yet.
        """

        session = create_builder_session(system_prompt="test")

        parsed = session.apply_llm_response(
            """
네이버 카페 품앗이용 에이전트로 시작할게요.

```draft
{
  "name": "UI F966N Naver Cafe 20260608",
  "description": "네이버 카페 품앗이를 매 1시간마다 자동 수행하는 에이전트",
  "system_prompt": "캡차 우회 금지. 중복 신청 금지. 결과를 요약 보고.",
  "provider_id": "openai",
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
""",
            user_message=(
                "Create an hourly Naver Cafe pumasi agent name UI F966N "
                "Naver Cafe 20260608 provider openai safe no captcha no duplicates"
            ),
        )

        draft = session.current_draft
        self.assertEqual(parsed.draft.name, "UI F966N Naver Cafe 20260608")
        self.assertEqual(draft.tools[0].mcp_id, "playwright")
        self.assertEqual(draft.tools[0].category.value, "browser")

        self.assertEqual(
            draft.flow,
            [],
            "the model returned no steps; the server must not write them",
        )
        # The user is told, in the reply they actually read, rather than
        # finding out at save time when `_validate_commit_draft` 422s.
        self.assertIn("워크플로 단계가 하나도 없습니다", parsed.assistant_message)
        self.assertIn("서버가 대신 단계를 만들어 넣지도 않습니다", parsed.assistant_message)
        # Nothing from the deleted cafe template may appear anywhere.
        self.assertNotIn("open_cafe_and_check_login", parsed.assistant_message)

        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "every 1h")
        self.assertIn("네이버 카페", session.task_draft.goal)

    def test_placeholder_flow_is_preserved_not_replaced_for_web_automation(self) -> None:
        """A bare "Step 1" is the model's placeholder, not an invitation.

        The old contract here was that a placeholder-only flow got thrown away
        and `_generic_web_automation_flow()` written in its place. A
        placeholder is a weak draft, but it is the draft the model produced and
        the user is looking at; replacing it with invented steps is the same
        defect as inventing them from nothing. It is kept, and the turn says
        the steps have no content yet.
        """

        session = create_builder_session(system_prompt="test")

        parsed = session.apply_llm_response(
            """
초안입니다.

```draft
{
  "name": "Browser bot",
  "description": "Open a website and submit a form",
  "system_prompt": "Use browser automation safely.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {"name": "Step 1", "description": "", "tool_hint": null, "success_criteria": "", "on_failure": "ask_user"}
  ],
  "memory_seeds": []
}
```
""",
            user_message="Create a browser automation agent for a website form",
        )

        draft = session.current_draft
        self.assertEqual(draft.tools[0].mcp_id, "playwright")
        self.assertEqual(len(draft.flow), 1)
        self.assertEqual(draft.flow[0].name, "Step 1")
        self.assertEqual(draft.flow[0].description, "")
        self.assertNotIn(
            "browser_action",
            [step.type for step in draft.flow],
            "no step may appear that the model did not write",
        )
        self.assertIn("이름만 있고", parsed.assistant_message)
        self.assertIn("임의로 채우지 않으므로", parsed.assistant_message)

    def test_naver_note_agent_gets_no_invented_workflow_or_approval_gate(self) -> None:
        """The note-sending template was the most convincing fabrication.

        `_naver_note_flow()` read the recipient address out of the user's
        sentence and built six steps around it -- a login step pointed at
        note.naver.com, a compose step pre-filled with that address, an
        approval gate, a send step and a memory write. It looked exactly like
        careful work by the model, and none of it was. It is gone: the empty
        flow the model returned stays empty and the reply says so.
        """

        session = create_builder_session(system_prompt="test")

        parsed = session.apply_llm_response(
            """
네이버 쪽지 발송 에이전트 초안을 만들게요.

```draft
{
  "name": "Naver Note Sender",
  "description": "네이버 쪽지로 지정된 대상에게 메시지를 보내는 에이전트",
  "system_prompt": "로그인과 캡차는 우회하지 않고 사용자에게 요청한다.",
  "provider_id": "openai",
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
""",
            user_message=(
                "mktoolbox로 로그인해서 rumururu@naver.com으로 네이버 쪽지 보내는 "
                "워크플로우를 만들어줘"
            ),
        )

        draft = session.current_draft
        self.assertEqual(draft.tools[0].mcp_id, "playwright")
        self.assertEqual(draft.flow, [])
        self.assertIn("워크플로 단계가 하나도 없습니다", parsed.assistant_message)
        # The address the user typed must not come back as a pre-filled form
        # value in a step nobody wrote.
        self.assertNotIn("rumururu@naver.com", parsed.assistant_message)

    # The gate these two tests need must be one *this file* inserts, not one
    # the model wrote. `_deterministic_gate_disclosure` compares the flow
    # before enrichment with the flow after it, and a gate that arrived inside
    # the model's own ```draft``` block is present in both -- so it produces no
    # disclosure, by design (the model saw its own gate and rule 7 tells it to
    # mention the cost itself). Measured, not assumed: an `approval_gate` in
    # the draft block plus a "매일 09:00" schedule yields an assistant message
    # with no notice in it at all.
    #
    # These tests used to get their gate from `_naver_note_flow`, which A6
    # deleted. `_install_app_steps` still inserts `approve_app_install`
    # deterministically -- correctly, per rules 13/14 -- so it is the same
    # situation the disclosure exists for: a park-for-human step the model
    # never wrote, landing in a flow that is being scheduled.
    _INSTALL_AGENT_RESPONSE = """
설치 확인 에이전트 초안입니다.

```draft
{
  "name": "Mobile Install Verifier",
  "description": "스토어 페이지를 열고 앱을 설치한 뒤 실행을 확인한다.",
  "system_prompt": "민감한 외부 동작 전에는 사용자에게 확인받는다.",
  "provider_id": "openai",
  "tools": [],
  "flow": [],
  "memory_seeds": []
}
```
"""

    def test_scheduled_flow_that_gains_approval_gate_discloses_stall_risk(self) -> None:
        """A deterministically-added gate must warn the user in the same turn.

        `_install_app_steps` inserts an `approval_gate` step
        (`approve_app_install`) ahead of the install per rules 13/14 --
        correctly. But that step parks every single run for a human
        (task_orchestrator._wait_for_user_step dispatches it unconditionally),
        and here the schedule is set in the very same turn the gate is added.
        If nothing discloses that, the user walks away believing they built an
        unattended daily agent when they actually built a chore that stalls the
        first morning nobody answers it -- the defect this initiative exists to
        close.
        """

        session = create_builder_session(system_prompt="test")

        parsed = session.apply_llm_response(
            self._INSTALL_AGENT_RESPONSE,
            user_message=(
                "Create an agent that opens the Play Store, installs the app and "
                "verifies launch. 매일 09:00에 자동으로 실행해줘."
            ),
        )

        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "daily 09:00")
        gate_ids = [
            step.id for step in session.current_draft.flow if step.type == "approval_gate"
        ]
        self.assertIn(
            "approve_app_install",
            gate_ids,
            "the gate under test must be one the server inserted, not one the model wrote",
        )

        # The disclosure must be in the reply actually handed back to the
        # caller (routes/agents.py surfaces `parsed.assistant_message`
        # verbatim as the user-visible reply), not just logged somewhere.
        self.assertIn("멈추", parsed.assistant_message)
        self.assertIn("무인", parsed.assistant_message)
        self.assertIn(
            "어떤 권한(permission) 설정도 이 승인 단계 자체를 건너뛰게 하지는 못합니다.",
            parsed.assistant_message,
        )
        # Never let the disclosure imply a permission grant is a fix.
        self.assertNotIn("권한을 허용하면", parsed.assistant_message)
        self.assertNotIn("권한 규칙을 추가하면", parsed.assistant_message)

    def test_unscheduled_flow_with_approval_gate_does_not_disclose(self) -> None:
        """The same gate in a manually-run agent must not trigger the warning.

        A gate the user calls by hand is expected and unremarkable; repeating
        the stall warning on every manually-run agent would be noise, not
        safety, so it must only fire when a schedule is actually present.
        """

        session = create_builder_session(system_prompt="test")

        parsed = session.apply_llm_response(
            self._INSTALL_AGENT_RESPONSE,
            user_message=(
                "Create an agent that opens the Play Store, installs the app and "
                "verifies launch."
            ),
        )

        self.assertIsNone(session.task_draft)
        self.assertTrue(
            any(step.type == "approval_gate" for step in session.current_draft.flow)
        )
        self.assertNotIn("완전한 무인 실행", parsed.assistant_message)
        self.assertNotIn("skip_if_active", parsed.assistant_message.casefold())

    def test_system_prompt_never_implies_permission_bypasses_gate(self) -> None:
        """The static prompt rule must state the true runtime contract.

        `_wait_for_user_step` dispatches `approval_gate`/`manual_handoff`
        unconditionally (task_orchestrator.py); no policy rule is consulted.
        The prompt must tell the model that plainly instead of leaving room
        for it to imply a permission grant can pass the gate -- a confident
        false statement is worse than the silence this initiative found.
        """

        prompt = build_configurator_system_prompt()

        self.assertIn("approval_gate", prompt)
        self.assertIn("manual_handoff", prompt)
        self.assertIn(
            "Do NOT say a permission rule, standing approval, or policy setting can\n"
            "   make an `approval_gate` or `manual_handoff` step skip itself",
            prompt,
        )
        self.assertIn("false statement", prompt)


class SpokenDailyScheduleTest(unittest.TestCase):
    """"매일 오전 9시" is how the time is actually written, and it was unread.

    The turn's own schedule extractor needed the hour to follow "매일"
    immediately, so a period word in between ("오전", "아침") made the whole
    phrase invisible and `session.task_draft` stayed None for the rest of the
    conversation. The commit path reads the same phrasing correctly
    (`routes/agents.py::_schedule_expression_from_draft`), which is why the
    schedule still got created and the gap went unnoticed: the two disagreed,
    and only the quieter one was wrong.
    """

    def test_a_spoken_period_before_the_hour_is_read(self) -> None:
        from agent.configurator import _extract_schedule  # noqa: PLC0415

        cases = {
            "매일 오전 9시": "daily 09:00",
            "매일 아침 9시": "daily 09:00",
            "매일 9시": "daily 09:00",
            "매일 오후 3시": "daily 15:00",
            "매일 오후 3시 30분": "daily 15:30",
            # Midday and midnight are the two the naive +12 gets backwards.
            "매일 오후 12시": "daily 12:00",
            "매일 오전 12시": "daily 00:00",
            "매일 09:00": "daily 09:00",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(_extract_schedule(phrase), expected)

    def test_a_weekly_phrase_is_still_not_a_daily_one(self) -> None:
        # "매주 월요일 오전 9시" has no daily reading, and inventing one would
        # silently run a weekly job seven times a week.
        from agent.configurator import _extract_schedule  # noqa: PLC0415

        self.assertIsNone(_extract_schedule("매주 월요일 오전 9시"))

    def test_the_turn_now_carries_the_schedule_it_was_told(self) -> None:
        session = create_builder_session(system_prompt="test")
        session.apply_llm_response(
            """
초안입니다.

```draft
{
  "name": "Disk Watch",
  "description": "디스크 여유 공간을 확인한다.",
  "system_prompt": "디스크를 확인하고 보고한다.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {"id": "check", "type": "llm", "name": "확인", "description": "디스크를 확인한다."}
  ],
  "memory_seeds": []
}
```
""",
            user_message="매일 오전 9시에 디스크 확인해줘",
        )

        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "daily 09:00")


def _step_by_id(flow, step_id):
    for step in flow:
        if step.id == step_id:
            return step
    raise AssertionError(f"missing workflow step {step_id}")


if __name__ == "__main__":
    unittest.main()
