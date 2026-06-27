"""Agent Builder Configurator post-processing tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.configurator import create_builder_session  # noqa: E402


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

    def test_sparse_android_review_exchange_request_gets_saveable_defaults(self) -> None:
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
        self.assertTrue(draft.system_prompt.strip())
        self.assertTrue(draft.description)
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
        self.assertEqual(draft.name, "Android App Agent")
        self.assertIn("Android app automation", draft.description)
        self.assertIn("Android app workflow", draft.system_prompt)
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

    def test_naver_hourly_web_agent_gets_tools_flow_and_schedule(self) -> None:
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
        self.assertGreaterEqual(len(draft.flow), 5)
        self.assertIn("네이버 카페", draft.flow[1].description)
        self.assertEqual(draft.flow[1].id, "open_cafe_and_check_login")
        self.assertEqual(draft.flow[1].type, "browser_action")
        self.assertEqual(draft.flow[1].actions[0]["type"], "navigate")
        self.assertEqual(draft.flow[1].on_failure["type"], "manual_handoff")
        self.assertEqual(draft.flow[1].on_failure["resume"], "same_step")
        self.assertTrue(
            any(step.tool_hint == "playwright" for step in draft.flow),
            "browser automation steps should declare Playwright",
        )
        self.assertIsNotNone(session.task_draft)
        self.assertEqual(session.task_draft.schedule, "every 1h")
        self.assertIn("네이버 카페", session.task_draft.goal)

    def test_placeholder_flow_is_replaced_for_web_automation(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
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
        self.assertGreaterEqual(len(draft.flow), 3)
        self.assertNotEqual(draft.flow[0].name, "Step 1")
        self.assertEqual(draft.flow[1].type, "browser_action")
        self.assertTrue(draft.flow[1].actions)
        self.assertTrue(any(step.tool_hint == "playwright" for step in draft.flow))

    def test_naver_note_agent_gets_concrete_workflow_and_send_approval(self) -> None:
        session = create_builder_session(system_prompt="test")

        session.apply_llm_response(
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
        self.assertGreaterEqual(len(draft.flow), 6)
        self.assertEqual(draft.flow[0].id, "prepare_note_context")
        self.assertEqual(draft.flow[1].id, "open_naver_note_and_login")
        self.assertEqual(draft.flow[1].type, "browser_action")
        self.assertEqual(draft.flow[1].actions[0]["url"], "https://note.naver.com/")
        self.assertEqual(draft.flow[1].on_failure["type"], "manual_handoff")
        compose = draft.flow[2]
        self.assertEqual(compose.id, "compose_note_draft")
        self.assertEqual(compose.actions[1]["value"], "rumururu@naver.com")
        self.assertEqual(compose.actions[2]["value"], "{{approved_note_body}}")
        self.assertEqual(draft.flow[3].type, "approval_gate")
        self.assertEqual(draft.flow[3].id, "approve_note_send")
        self.assertEqual(draft.flow[4].id, "submit_note_and_capture_result")
        self.assertTrue(
            any(
                getattr(step, "memory_write", None)
                for step in draft.flow
            ),
            "note workflow should persist send-result memory",
        )


def _step_by_id(flow, step_id):
    for step in flow:
        if step.id == step_id:
            return step
    raise AssertionError(f"missing workflow step {step_id}")


if __name__ == "__main__":
    unittest.main()
