import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, browser_session_store  # noqa: E402
from agent.app_action_adapter import AppActionAdapterResult  # noqa: E402
from agent.browser_action_adapter import BrowserActionAdapterResult  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    prepare_task_orchestration,
    resume_task_orchestration,
)
from core import database  # noqa: E402


class WorkflowRuntimeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "workflow_runtime.db"
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

    def tearDown(self):
        agent_store._agent_store = None
        browser_session_store._browser_session_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _prepare(self, flow_json):
        agent = self.store.create_agent(
            name="workflow bot",
            system_prompt="Run workflow steps.",
            provider_id="openai",
            flow_json=flow_json,
        )
        task = self.store.create_task(
            title="Run workflow",
            assigned_agent_id=agent["id"],
            goal="Finish the workflow.",
        )
        result = prepare_task_orchestration(
            task["id"],
            provider_id="openai",
            auto_start=False,
        )
        assert result is not None
        return agent, task, result

    def test_manual_handoff_step_waits_for_user_without_provider_call(self):
        _agent, task, result = self._prepare(
            [
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
            ]
        )

        asyncio.run(execute_task_orchestration(result["execution"]))

        run = self.store.get_run(result["run"]["id"])
        updated_task = self.store.get_task(task["id"])
        steps = self.store.list_task_steps(task["id"])
        checkpoint = self.store.get_task_checkpoint(task["id"])

        assert run is not None
        assert updated_task is not None
        assert checkpoint is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(updated_task["status"], "waiting_for_user")
        self.assertEqual(steps[0]["status"], "waiting_for_user")
        self.assertEqual(
            checkpoint["checkpoint"]["prompt"],
            "Complete captcha, then continue.",
        )
        self.assertEqual(checkpoint["checkpoint"]["step_title"], "Resolve captcha")
        self.assertEqual(checkpoint["checkpoint"]["workflow_type"], "manual_handoff")
        self.assertEqual(
            checkpoint["checkpoint"]["resume_behavior"],
            "complete_waiting_step_then_continue",
        )
        self.assertIn("완료 처리", checkpoint["checkpoint"]["resume_label"])
        self.assertFalse(checkpoint["checkpoint"]["allow_memory"])
        self.assertIn("created_at", checkpoint["checkpoint"])
        self.assertEqual(
            updated_task["metadata"]["active_checkpoint"]["workflow_step_id"],
            "captcha",
        )

    def test_llm_step_completes_before_manual_handoff_waits(self):
        _agent, task, result = self._prepare(
            [
                {"id": "inspect", "type": "llm", "name": "Inspect"},
                {
                    "id": "login",
                    "type": "manual_handoff",
                    "name": "Log in",
                    "on_failure": {
                        "type": "manual_handoff",
                        "prompt": "Log in, then continue.",
                    },
                },
            ]
        )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            return True

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])

        assert run is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(steps[0]["status"], "completed")
        self.assertEqual(steps[1]["status"], "waiting_for_user")
        events = self.store.list_events(run["id"])
        self.assertIn(
            "task.step.waiting_for_user",
            [event["event_type"] for event in events],
        )

    def test_mcp_tool_step_runs_when_its_server_is_configured(self):
        """`mcp_tool` used to park unconditionally, so an agent containing one
        never ran unattended. It executes as a turn scoped to that server —
        which is where its MCP servers are already injected."""
        _agent, task, result = self._prepare(
            [
                {
                    "id": "call_it",
                    "type": "mcp_tool",
                    "name": "Run the workflow",
                    "tool_hint": "n8n",
                    "instruction": "Trigger the daily digest.",
                }
            ]
        )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            return True

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ), patch(
            "agent.task_orchestrator.detected_mcp_server_configs",
            return_value={"n8n": {"type": "stdio"}},
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        run = self.store.get_run(result["run"]["id"])
        steps = self.store.list_task_steps(task["id"])
        assert run is not None
        self.assertEqual(steps[0]["status"], "completed")
        self.assertEqual(run["status"], "completed")

    def test_mcp_tool_step_asks_when_its_server_is_not_installed(self):
        """Running would mean reporting success on a server that is not there.
        The prompt has to name it, or the person cannot tell what to install."""
        _agent, task, result = self._prepare(
            [
                {
                    "id": "call_it",
                    "type": "mcp_tool",
                    "name": "Run the workflow",
                    "tool_hint": "n8n",
                    "instruction": "Trigger the daily digest.",
                }
            ]
        )

        with patch(
            "agent.task_orchestrator.detected_mcp_server_configs",
            return_value={},
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        run = self.store.get_run(result["run"]["id"])
        steps = self.store.list_task_steps(task["id"])
        checkpoint = self.store.get_task_checkpoint(task["id"])
        assert run is not None and checkpoint is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(steps[0]["status"], "waiting_for_user")
        self.assertIn("n8n", checkpoint["checkpoint"]["prompt"])

    def test_llm_step_message_includes_extended_workflow_fields(self):
        _agent, task, result = self._prepare(
            [
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
            ]
        )

        seen_messages = []

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **kwargs):
            seen_messages.append(kwargs["user_message"])
            return True

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        self.assertEqual(self.store.get_task(task["id"])["status"], "completed")
        self.assertEqual(len(seen_messages), 1)
        message = seen_messages[0]
        self.assertIn("- instruction: Inspect the failed build.", message)
        self.assertIn("- observation: Review the latest task output first.", message)
        # memory_read is a selector resolved against the agent's memories, not
        # prose relayed verbatim; this agent has no memories, so nothing
        # matches and neither the query text nor an empty section appears.
        self.assertNotIn("- memory read:", message)
        self.assertNotIn("- relevant memories:", message)
        self.assertIn(
            "- memory write: Remember newly discovered build caveats.",
            message,
        )
        self.assertIn("- tool_hint: openai", message)
        self.assertIn('"target": "build_log"', message)
        self.assertIn("- success_criteria: Root cause is identified", message)
        self.assertIn('"prompt": "Need more context."', message)

    def test_user_response_resumes_waiting_step_and_continues_next_step(self):
        _agent, task, result = self._prepare(
            [
                {"id": "inspect", "type": "llm", "name": "Inspect"},
                {
                    "id": "login",
                    "type": "manual_handoff",
                    "name": "Log in",
                    "on_failure": {
                        "type": "manual_handoff",
                        "prompt": "Log in, then continue.",
                    },
                },
                {"id": "report", "type": "llm", "name": "Report"},
            ]
        )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            return True

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "completed")
        self.assertEqual(steps[1]["status"], "waiting_for_user")
        self.assertEqual(steps[2]["status"], "queued")

        response = self.store.append_step_user_response(
            task_id=task["id"],
            step_id=steps[1]["id"],
            message="Login completed.",
            remember=True,
        )
        self.assertIsNotNone(response)
        resume = resume_task_orchestration(task["id"])
        assert resume is not None

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(resume["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])
        updated_task = self.store.get_task(task["id"])
        assert run is not None
        assert updated_task is not None
        self.assertEqual([step["status"] for step in steps], ["completed", "completed", "completed"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(updated_task["status"], "completed")
        self.assertIsNone(self.store.get_task_checkpoint(task["id"])["checkpoint"])

    def test_retry_policy_requeues_step_until_success(self):
        _agent, task, result = self._prepare(
            [
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
            ]
        )

        async def fake_create_chat_session(**_kwargs):
            return object()

        outcomes = [False, True]

        async def fake_stream(_sink, _session, **_kwargs):
            return outcomes.pop(0)

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        step = self.store.list_task_steps(task["id"])[0]
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(run["status"], "completed")
        self.assertEqual(step["status"], "completed")
        self.assertEqual(step["input"]["retry_state"], {"attempts": 1})
        event_types = [
            event["event_type"] for event in self.store.list_events(run["id"])
        ]
        self.assertIn("task.step.retry_scheduled", event_types)

    def test_goto_policy_jumps_to_target_step(self):
        _agent, task, result = self._prepare(
            [
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
            ]
        )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            return False

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(steps[0]["status"], "failed")
        self.assertEqual(steps[1]["status"], "queued")
        self.assertEqual(steps[2]["status"], "waiting_for_user")
        event_types = [
            event["event_type"] for event in self.store.list_events(run["id"])
        ]
        self.assertIn("task.step.goto", event_types)

    def test_ask_user_failure_policy_waits_for_user(self):
        _agent, task, result = self._prepare(
            [
                {
                    "id": "ask",
                    "type": "llm",
                    "name": "Ask on failure",
                    "on_failure": {
                        "type": "ask_user",
                        "prompt": "What should I do next?",
                    },
                }
            ]
        )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            return False

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        step = self.store.list_task_steps(task["id"])[0]
        checkpoint = self.store.get_task_checkpoint(task["id"])
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        assert checkpoint is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(step["status"], "waiting_for_user")
        self.assertEqual(checkpoint["checkpoint"]["prompt"], "What should I do next?")

    def test_browser_action_step_executes_and_continues_workflow(self):
        _agent, task, result = self._prepare(
            [
                {
                    "id": "open_site",
                    "type": "browser_action",
                    "name": "Open site",
                    "actions": [{"type": "navigate", "url": "https://example.com"}],
                },
                {"id": "report", "type": "llm", "name": "Report"},
            ]
        )

        async def fake_execute_browser_actions(actions, *, context):
            self.assertEqual(actions[0]["type"], "navigate")
            self.assertEqual(context["workflow_step_id"], "open_site")
            return BrowserActionAdapterResult(
                status="completed",
                message="ok",
                observations=[{"url": "https://example.com"}],
            )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **kwargs):
            user_message = kwargs["user_message"]
            self.assertIn("Previous workflow evidence:", user_message)
            self.assertIn("workflow_step_id: open_site", user_message)
            self.assertIn("url=https://example.com", user_message)
            self.assertIn("Do not re-open URLs", user_message)
            return True

        with patch(
            "agent.task_orchestrator.execute_browser_actions",
            fake_execute_browser_actions,
        ), patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(run["status"], "completed")
        self.assertEqual([step["status"] for step in steps], ["completed", "completed"])
        self.assertEqual(
            steps[0]["output"]["browser_action"]["observations"],
            [{"url": "https://example.com"}],
        )
        event_types = [
            event["event_type"] for event in self.store.list_events(run["id"])
        ]
        self.assertIn("task.step.browser_action.started", event_types)
        self.assertIn("task.step.browser_action.completed", event_types)

    def test_browser_action_waiting_result_creates_checkpoint(self):
        _agent, task, result = self._prepare(
            [
                {
                    "id": "login",
                    "type": "browser_action",
                    "name": "Login",
                    "actions": [{"type": "navigate", "url": "https://example.com"}],
                    "on_failure": {
                        "type": "ask_user",
                        "prompt": "Login manually.",
                    },
                },
                {"id": "report", "type": "llm", "name": "Report"},
            ]
        )

        async def fake_execute_browser_actions(_actions, *, context):
            self.assertEqual(context["workflow_step_id"], "login")
            return BrowserActionAdapterResult(
                status="waiting_for_user",
                wait_reason="captcha_or_bot_challenge",
                prompt="Complete captcha, then resume.",
                observations=[{"title": "Login"}],
            )

        with patch(
            "agent.task_orchestrator.execute_browser_actions",
            fake_execute_browser_actions,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])
        checkpoint = self.store.get_task_checkpoint(task["id"])
        assert run is not None
        assert checkpoint is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual([step["status"] for step in steps], ["waiting_for_user", "queued"])
        self.assertEqual(checkpoint["checkpoint"]["reason"], "captcha_or_bot_challenge")
        self.assertTrue(checkpoint["checkpoint"]["browser_session_id"].startswith("bs_"))
        sessions = browser_session_store.get_browser_session_store().list_for_run(
            result["run"]["id"]
        )
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["status"], "waiting_for_user")
        self.assertEqual(sessions[0]["workflow_step_id"], "login")
        self.assertEqual(steps[0]["output"]["browser_session_id"], sessions[0]["id"])
        self.assertEqual(
            steps[0]["output"]["browser_action"]["wait_reason"],
            "captcha_or_bot_challenge",
        )

    def test_browser_action_screenshots_are_registered_as_artifacts(self):
        screenshot = Path(self._tmp.name) / "browser.png"
        screenshot.write_bytes(b"\x89PNG\r\n\x1a\n")
        _agent, task, result = self._prepare(
            [
                {
                    "id": "capture",
                    "type": "browser_action",
                    "name": "Capture",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )

        async def fake_execute_browser_actions(_actions, *, context):
            self.assertEqual(context["workflow_step_id"], "capture")
            return BrowserActionAdapterResult(
                status="completed",
                screenshots=[str(screenshot)],
            )

        with patch(
            "agent.task_orchestrator.execute_browser_actions",
            fake_execute_browser_actions,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        step = self.store.list_task_steps(task["id"])[0]
        artifacts = self.store.list_artifacts(result["run"]["id"])
        self.assertEqual(step["status"], "completed")
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["kind"], "browser_screenshot")

    def test_app_action_step_executes_and_continues_workflow(self):
        _agent, task, result = self._prepare(
            [
                {
                    "id": "read_requests",
                    "type": "app_action",
                    "name": "Read requests",
                    "android_device_id": "emulator-5554",
                    "actions": [{"type": "read_screen"}],
                },
                {"id": "report", "type": "llm", "name": "Report"},
            ]
        )

        async def fake_execute_app_actions(actions, *, context):
            self.assertEqual(actions[0]["type"], "read_screen")
            self.assertEqual(context["workflow_step_id"], "read_requests")
            self.assertEqual(context["android_device_id"], "emulator-5554")
            return AppActionAdapterResult(
                status="completed",
                message="ok",
                observations=[{"type": "ui_dump", "text": "request"}],
            )

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(_sink, _session, **_kwargs):
            return True

        with patch(
            "agent.task_orchestrator.execute_app_actions",
            fake_execute_app_actions,
        ), patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])
        assert run is not None
        self.assertEqual(run["status"], "completed")
        self.assertEqual([step["status"] for step in steps], ["completed", "completed"])
        self.assertEqual(
            steps[0]["output"]["app_action"]["observations"],
            [{"type": "ui_dump", "text": "request"}],
        )
        event_types = [
            event["event_type"] for event in self.store.list_events(run["id"])
        ]
        self.assertIn("task.step.app_action.started", event_types)
        self.assertIn("task.step.app_action.completed", event_types)

    def test_app_action_waiting_result_creates_checkpoint(self):
        _agent, task, result = self._prepare(
            [
                {
                    "id": "install_app",
                    "type": "app_action",
                    "name": "Install app",
                    "actions": [{"type": "install_app", "source": "app_from_request"}],
                    "on_failure": {
                        "type": "ask_user",
                        "prompt": "Select app target.",
                    },
                },
                {"id": "report", "type": "llm", "name": "Report"},
            ]
        )

        async def fake_execute_app_actions(_actions, *, context):
            self.assertEqual(context["workflow_step_id"], "install_app")
            return AppActionAdapterResult(
                status="waiting_for_user",
                wait_reason="app_install_needs_package_or_apk",
                prompt="Package or APK is required.",
            )

        with patch(
            "agent.task_orchestrator.execute_app_actions",
            fake_execute_app_actions,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])
        checkpoint = self.store.get_task_checkpoint(task["id"])
        assert run is not None
        assert checkpoint is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual([step["status"] for step in steps], ["waiting_for_user", "queued"])
        self.assertEqual(
            checkpoint["checkpoint"]["reason"],
            "app_install_needs_package_or_apk",
        )
        self.assertEqual(
            steps[0]["output"]["app_action"]["wait_reason"],
            "app_install_needs_package_or_apk",
        )

    def test_app_action_exception_uses_failure_policy(self):
        _agent, task, result = self._prepare(
            [
                {
                    "id": "verify_launch",
                    "type": "app_action",
                    "name": "Verify launch",
                    "actions": [{"type": "verify_launch", "package": "com.android.settings"}],
                    "on_failure": {
                        "type": "manual_handoff",
                        "prompt": "Launch failed; check the device.",
                    },
                },
            ]
        )

        async def fake_execute_app_actions(_actions, *, context):
            self.assertEqual(context["workflow_step_id"], "verify_launch")
            raise RuntimeError("adb command timed out")

        with patch(
            "agent.task_orchestrator.execute_app_actions",
            fake_execute_app_actions,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        run = self.store.get_run(result["run"]["id"])
        checkpoint = self.store.get_task_checkpoint(task["id"])
        assert run is not None
        assert checkpoint is not None
        self.assertEqual(run["status"], "waiting_for_user")
        self.assertEqual(steps[0]["status"], "waiting_for_user")
        self.assertEqual(checkpoint["checkpoint"]["prompt"], "Launch failed; check the device.")
        event_types = [
            event["event_type"] for event in self.store.list_events(run["id"])
        ]
        self.assertIn("task.step.app_action.failed", event_types)
        self.assertIn("task.step.waiting_for_user", event_types)

    def test_browser_action_reuses_previous_storage_state(self):
        _agent, task, result = self._prepare(
            [
                {
                    "id": "login",
                    "type": "browser_action",
                    "name": "Login",
                    "actions": [{"type": "navigate", "url": "https://example.com"}],
                },
                {
                    "id": "after_login",
                    "type": "browser_action",
                    "name": "After login",
                    "actions": [{"type": "assert", "kind": "page_state_readable"}],
                },
            ]
        )

        seen_contexts = []

        async def fake_execute_browser_actions(_actions, *, context):
            seen_contexts.append(dict(context))
            storage_path = Path(context["browser_storage_state_path"])
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_text("{}", encoding="utf-8")
            return BrowserActionAdapterResult(
                status="completed",
                observations=[{"title": context["workflow_step_id"]}],
                storage_state_path=str(storage_path),
            )

        with patch(
            "agent.task_orchestrator.execute_browser_actions",
            fake_execute_browser_actions,
        ):
            asyncio.run(execute_task_orchestration(result["execution"]))

        steps = self.store.list_task_steps(task["id"])
        sessions = browser_session_store.get_browser_session_store().list_for_run(
            result["run"]["id"]
        )
        self.assertEqual([step["status"] for step in steps], ["completed", "completed"])
        self.assertEqual(len(seen_contexts), 2)
        self.assertIn("browser_storage_state_path", seen_contexts[0])
        self.assertNotIn("browser_input_storage_state_path", seen_contexts[0])
        self.assertEqual(
            seen_contexts[1]["browser_input_storage_state_path"],
            seen_contexts[0]["browser_storage_state_path"],
        )
        self.assertEqual(len(sessions), 2)
        self.assertEqual([session["status"] for session in sessions], ["resumed", "resumed"])
        self.assertEqual(steps[0]["output"]["browser_session_id"], sessions[0]["id"])
        self.assertEqual(steps[1]["output"]["browser_session_id"], sessions[1]["id"])


if __name__ == "__main__":
    unittest.main()
