import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from agent.prompt_composer import compose_system_prompt  # noqa: E402
from agent.task_orchestrator import prepare_task_orchestration  # noqa: E402
from core import database  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402


class PromptComposerTest(unittest.TestCase):
    def test_empty_memory_and_empty_workflow_returns_system_prompt_only(self):
        prompt = compose_system_prompt(
            {
                "system_prompt": "Base prompt.",
                "memories": [],
                "flow_json": [],
            }
        )

        self.assertEqual(prompt, "Base prompt.")

    def test_pinned_memories_are_rendered_before_recent_unpinned(self):
        prompt = compose_system_prompt(
            {
                "system_prompt": "Base prompt.",
                "memories": [
                    {
                        "content": "plain newest",
                        "pinned": False,
                        "created_at": "2026-05-31T00:05:00",
                    },
                    {
                        "content": "pinned older",
                        "pinned": True,
                        "created_at": "2026-05-31T00:01:00",
                    },
                    {
                        "content": "plain middle",
                        "pinned": False,
                        "created_at": "2026-05-31T00:03:00",
                    },
                    {
                        "content": "pinned newer",
                        "pinned": True,
                        "created_at": "2026-05-31T00:04:00",
                    },
                    {
                        "content": "plain oldest",
                        "pinned": False,
                        "created_at": "2026-05-31T00:02:00",
                    },
                ],
                "flow_json": [],
            }
        )

        memory_lines = [line for line in prompt.splitlines() if line.startswith("- ")]
        self.assertEqual(
            memory_lines,
            [
                "- pinned newer",
                "- pinned older",
                "- plain newest",
                "- plain middle",
                "- plain oldest",
            ],
        )

    def test_many_memories_are_truncated_within_max_chars(self):
        memories = [
            {
                "content": f"memory {index:02d} " + ("x" * 48),
                "pinned": False,
                "created_at": f"2026-05-31T00:{index:02d}:00",
            }
            for index in range(50)
        ]

        prompt = compose_system_prompt(
            {
                "system_prompt": "Base prompt.",
                "memories": memories,
                "flow_json": [],
            },
            max_memory_items=50,
            max_chars=450,
        )

        self.assertLessEqual(len(prompt), 450)
        self.assertIn("memory 49", prompt)
        self.assertNotIn("memory 00", prompt)

    def test_workflow_steps_are_rendered_as_ordered_prose(self):
        prompt = compose_system_prompt(
            {
                "system_prompt": "Base prompt.",
                "memories": [],
                "flow_json": [
                    {
                        "name": "Inspect",
                        "description": "Read the relevant files",
                        "success_criteria": "scope is clear",
                        "on_failure": "ask_user",
                    },
                    {
                        "name": "Implement",
                        "description": "Make the scoped change",
                        "success_criteria": "tests cover the behavior",
                        "on_failure": "abort",
                    },
                    {
                        "name": "Verify",
                        "description": "Run focused checks",
                        "success_criteria": "checks pass",
                        "on_failure": "retry_once",
                    },
                ],
            }
        )

        self.assertEqual(
            prompt,
            "Base prompt.\n\n"
            "---\n"
            "Your workflow (run this in order):\n"
            "1. Inspect "
            "(type: llm; instruction: Read the relevant files. "
            "success_criteria: scope is clear. on_failure: ask_user.)\n"
            "2. Implement "
            "(type: llm; instruction: Make the scoped change. "
            "success_criteria: tests cover the behavior. on_failure: abort.)\n"
            "3. Verify "
            "(type: llm; instruction: Run focused checks. "
            "success_criteria: checks pass. on_failure: retry_once.)",
        )

    def test_workflow_steps_render_extended_schema_fields(self):
        prompt = compose_system_prompt(
            {
                "system_prompt": "Base prompt.",
                "memories": [],
                "flow_json": [
                    {
                        "name": "Inspect",
                        "description": "Legacy fallback should not appear.",
                        "instruction": "Inspect the code path.",
                        "observation": "Check current task status first.",
                        "memory_read": ["release rules", "test policy"],
                        "memory_write": "Save any new release caveats.",
                        "tool_hint": "filesystem",
                        "actions": [{"type": "extract", "target": "logs"}],
                        "success_criteria": "Relevant files are understood",
                        "on_failure": {"type": "ask_user", "resume": "same_step"},
                    }
                ],
            }
        )

        self.assertIn("instruction: Inspect the code path.", prompt)
        self.assertIn("observation: Check current task status first.", prompt)
        self.assertIn('memory read: ["release rules", "test policy"]', prompt)
        self.assertIn("memory write: Save any new release caveats.", prompt)
        self.assertIn("tool_hint: filesystem", prompt)
        self.assertIn('"type": "extract"', prompt)
        self.assertIn("success_criteria: Relevant files are understood", prompt)
        self.assertIn('"type": "ask_user"', prompt)
        self.assertNotIn("Legacy fallback should not appear.", prompt)

    def test_workflow_browser_actions_are_rendered(self):
        prompt = compose_system_prompt(
            {
                "system_prompt": "Base prompt.",
                "memories": [],
                "flow_json": [
                    {
                        "type": "browser_action",
                        "name": "Open cafe",
                        "description": "Open the configured cafe page.",
                        "tool_hint": "playwright",
                        "actions": [
                            {"type": "navigate", "target": "configured_url"},
                            {"type": "assert", "kind": "page_state_readable"},
                        ],
                        "success_criteria": "Page is readable",
                        "on_failure": {
                            "type": "manual_handoff",
                            "resume": "same_step",
                        },
                    }
                ],
            }
        )

        self.assertIn("type: browser_action", prompt)
        self.assertIn("tool_hint: playwright", prompt)
        self.assertIn('"type": "navigate"', prompt)
        self.assertIn('"type": "manual_handoff"', prompt)

    def test_task_goal_none_omits_current_task_section(self):
        prompt = compose_system_prompt(
            {
                "system_prompt": "Base prompt.",
                "memories": [{"content": "Keep diffs focused.", "created_at": "2026-05-31"}],
                "flow_json": [],
            },
            task_goal=None,
        )

        self.assertNotIn("Current task:", prompt)
        self.assertIn("Keep diffs focused.", prompt)


class PromptComposerRoutesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_prompt_composer_test.db"
        agent_store._agent_store = None

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _create_agent(self, **overrides):
        body = {
            "name": "maintenance",
            "system_prompt": "You maintain the app.",
            "provider_id": "openai",
            "flow_json": [
                {
                    "name": "Check",
                    "description": "Inspect current state",
                    "success_criteria": "state is understood",
                    "on_failure": "ask_user",
                }
            ],
        }
        body.update(overrides)
        response = self.client.post("/api/agent/agents", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_preview_prompt_returns_composed_text_and_counts(self):
        agent = self._create_agent()
        self.client.post(
            f"/api/agent/agents/{agent['id']}/memories",
            json={"content": "Avoid touching billing queues.", "pinned": True},
        )
        self.client.post(
            f"/api/agent/agents/{agent['id']}/memories",
            json={"content": "Run tests before reporting done."},
        )

        response = self.client.get(
            f"/api/agent/agents/{agent['id']}/preview-prompt?task_goal=Ship%20build%2038"
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["memory_count"], 2)
        self.assertEqual(payload["workflow_steps"], 1)
        self.assertIn("Your accumulated learnings", payload["composed_prompt"])
        self.assertIn("- Avoid touching billing queues.", payload["composed_prompt"])
        self.assertIn("Your workflow (run this in order):", payload["composed_prompt"])
        self.assertIn("Current task: Ship build 38", payload["composed_prompt"])

    def test_run_feedback_creates_agent_memory(self):
        agent = self._create_agent()
        run = self.client.post(
            "/api/agent/runs",
            json={"agent_id": agent["id"], "title": "Review", "goal": "Review app"},
        ).json()["run"]

        response = self.client.post(
            f"/api/agent/runs/{run['id']}/feedback",
            json={"content": "Prefer smoke tests first.", "pinned": True},
        )

        self.assertEqual(response.status_code, 200, response.text)
        memory = response.json()["memory"]
        self.assertEqual(memory["content"], "Prefer smoke tests first.")
        self.assertEqual(memory["source_run_id"], run["id"])
        self.assertEqual(memory["source_event_type"], "user_annotation")
        self.assertTrue(memory["pinned"])

        memories_response = self.client.get(f"/api/agent/agents/{agent['id']}/memories")
        self.assertEqual(memories_response.status_code, 200)
        self.assertEqual(memories_response.json()["total"], 1)
        self.assertEqual(memories_response.json()["memories"][0]["id"], memory["id"])

    def test_task_orchestration_propagates_assigned_agent_id_to_run(self):
        store = agent_store.get_agent_store()
        agent = store.create_agent(
            name="ops bot",
            system_prompt="Run operations tasks.",
            provider_id="google",
            flow_json=[
                {
                    "name": "Verify",
                    "description": "Run the requested checks",
                    "success_criteria": "checks pass",
                    "on_failure": "ask_user",
                }
            ],
        )
        store.add_memory(
            agent_id=agent["id"],
            content="Always include the smoke test result.",
        )
        task = store.create_task(
            title="Run smoke checks",
            assigned_agent_id=agent["id"],
            goal="Run focused smoke checks.",
        )

        result = prepare_task_orchestration(
            task["id"],
            provider_id="google",
            auto_start=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["run"]["agent_id"], agent["id"])
        self.assertIn("Always include the smoke test result.", result["system_prompt"])
        self.assertIn("Your workflow (run this in order):", result["system_prompt"])
        self.assertIn("Current task: Run focused smoke checks.", result["system_prompt"])
        self.assertTrue(result["launch_message"].startswith(result["system_prompt"]))

    def test_task_orchestration_uses_assigned_agent_workflow_steps(self):
        store = agent_store.get_agent_store()
        agent = store.create_agent(
            name="ops bot",
            system_prompt="Run operations tasks.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "inspect",
                    "name": "Inspect current state",
                    "type": "llm",
                    "description": "Read current context.",
                    "instruction": "Inspect the runtime state.",
                    "observation": "Note recent failures.",
                    "memory_read": "Read prior smoke-check notes.",
                    "memory_write": "Remember new smoke-check caveats.",
                    "tool_hint": "openai",
                    "success_criteria": "state is known",
                    "on_failure": {
                        "type": "retry",
                        "max_attempts": 2,
                        "then": {"type": "abort"},
                    },
                },
                {
                    "id": "open_browser",
                    "name": "Open browser",
                    "type": "browser_action",
                    "tool_hint": "browser",
                    "actions": [{"type": "navigate", "url": "https://example.test"}],
                    "success_criteria": "page is open",
                    "on_failure": {"type": "manual_handoff", "prompt": "Open it manually."},
                },
            ],
        )
        task = store.create_task(
            title="Run smoke checks",
            assigned_agent_id=agent["id"],
            goal="Run focused smoke checks.",
        )

        result = prepare_task_orchestration(
            task["id"],
            provider_id="openai",
            auto_start=False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            [step["title"] for step in result["steps"]],
            ["Inspect current state", "Open browser"],
        )
        first_input = result["steps"][0]["input"]
        self.assertEqual(first_input["workflow_step_id"], "inspect")
        self.assertEqual(first_input["workflow_type"], "llm")
        self.assertEqual(first_input["description"], "Read current context.")
        self.assertEqual(first_input["instruction"], "Inspect the runtime state.")
        self.assertEqual(first_input["observation"], "Note recent failures.")
        self.assertEqual(first_input["memory_read"], "Read prior smoke-check notes.")
        self.assertEqual(first_input["memory_write"], "Remember new smoke-check caveats.")
        self.assertEqual(first_input["tool_hint"], "openai")
        self.assertEqual(first_input["actions"], [])
        self.assertEqual(first_input["success_criteria"], "state is known")
        self.assertEqual(
            first_input["on_failure"],
            {"type": "retry", "max_attempts": 2, "then": {"type": "abort"}},
        )
        self.assertEqual(first_input["retry_state"], {"attempts": 0})

        second_input = result["steps"][1]["input"]
        self.assertEqual(second_input["workflow_step_id"], "open_browser")
        self.assertEqual(second_input["workflow_type"], "browser_action")
        self.assertEqual(second_input["actions"], [{"type": "navigate", "url": "https://example.test"}])
        self.assertEqual(second_input["on_failure"]["type"], "manual_handoff")
        capability_names = {item["name"] for item in result["capabilities"]}
        self.assertIn("browser", capability_names)
        self.assertIn("mcp_server:browser", result["launch_message"])

    def test_workflow_capability_summary_includes_app_action_runtime(self):
        store = agent_store.get_agent_store()
        agent = store.create_agent(
            name="device bot",
            system_prompt="Run device tasks.",
            provider_id="openai",
            flow_json=[
                {
                    "id": "open_settings",
                    "name": "Open Settings",
                    "type": "app_action",
                    "tool_hint": "android_adb",
                    "actions": [{"type": "verify_launch", "package": "com.android.settings"}],
                    "success_criteria": "Settings is foreground",
                },
            ],
        )
        task = store.create_task(
            title="Check settings",
            assigned_agent_id=agent["id"],
            goal="Check settings.",
        )

        result = prepare_task_orchestration(
            task["id"],
            provider_id="openai",
            auto_start=False,
        )

        self.assertIsNotNone(result)
        capability_names = {item["name"] for item in result["capabilities"]}
        self.assertIn("device.control", capability_names)
        self.assertIn("builtin:device.control", result["launch_message"])


if __name__ == "__main__":
    unittest.main()
