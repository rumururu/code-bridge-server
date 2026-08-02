"""A workflow step's ``memory_read`` and ``memory_write`` fields are a
promise: the author is telling the agent "look this up before you start" and
"remember this once you're done". Before this file existed, neither half of
that promise was kept. ``memory_read`` was pasted into the prompt as literal
text — the model was told a query existed, with no mechanism to run it, so it
saw its own unanswered instruction and nothing else. ``memory_write`` was the
same: the words "remember the request id" reached the model, and then nothing
happened. The agent's memories *are* used elsewhere (the whole memory bank
loads into the system prompt every run), but that path ignores these fields
entirely, so an author who scopes a step to "just this" gets either the
model's blind guess or an unrelated flood of every memory the agent has ever
had — never the one relevant note.

If this regresses: a step author writes `memory_read: "prior install
failures"` expecting the step to see prior install failures, and instead the
model either invents an answer to the query itself or drowns in irrelevant
context; or a step author writes `memory_write: "save the outcome"` expecting
the next run to know what happened, and the next run starts exactly as blind
as this one did. Two failure modes are worse than doing nothing: inventing a
memory for a step that produced no result (a fabricated note is worse than a
missing one — nothing here should manufacture content the model never said),
and letting a memory-store hiccup fail a step whose actual work already
succeeded.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store  # noqa: E402
from agent.task_orchestrator import (  # noqa: E402
    execute_task_orchestration,
    prepare_task_orchestration,
)
from core import database  # noqa: E402


class StepMemoryWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "step_memory_wiring.db"
        agent_store._agent_store = None
        database.init_db()
        self.store = agent_store.get_agent_store()

    def tearDown(self) -> None:
        agent_store._agent_store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _prepare(self, agent: dict, flow_json):
        updated_agent = self.store.update_agent(agent["id"], {"flow_json": flow_json})
        assert updated_agent is not None
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

    def _run_with_fake_stream(self, execution, *, result_text: str | None):
        seen_messages: list[str] = []

        async def fake_create_chat_session(**_kwargs):
            return object()

        async def fake_stream(sink, _session, **kwargs):
            seen_messages.append(kwargs["user_message"])
            if result_text is not None:
                sink.result_text = result_text
            return True

        with patch(
            "agent.task_orchestrator.create_chat_session",
            fake_create_chat_session,
        ), patch(
            "agent.task_orchestrator.stream_claude_turn",
            fake_stream,
        ):
            asyncio.run(execute_task_orchestration(execution))

        return seen_messages

    @staticmethod
    def _current_step_section(message: str) -> str:
        # The full user message also carries the agent's whole-workflow
        # overview (built elsewhere, out of scope here — it previews every
        # step's fields as authored, same as it always has). What this change
        # governs is only the "Current workflow step" block built fresh for
        # the step about to run.
        return message.split("Current workflow step:", 1)[1]

    # -- memory_read -----------------------------------------------------

    def test_matching_memory_read_puts_memories_in_prompt_not_the_query(self):
        agent = self.store.create_agent(
            name="memory bot", system_prompt="Run workflow steps.", provider_id="openai"
        )
        self.store.add_memory(
            agent_id=agent["id"],
            content="The settings screen blocked payment sync last run.",
        )
        _agent, _task, result = self._prepare(
            agent,
            [
                {
                    "id": "diagnose",
                    "type": "llm",
                    "name": "Diagnose",
                    "instruction": "Work out what went wrong.",
                    "memory_read": "which screen blocked us before",
                }
            ],
        )

        messages = self._run_with_fake_stream(result["execution"], result_text="Found it.")

        self.assertEqual(len(messages), 1)
        step_section = self._current_step_section(messages[0])
        self.assertIn("- relevant memories:", step_section)
        self.assertIn(
            "The settings screen blocked payment sync last run.", step_section
        )
        # The selector itself must never reach the model as if it were content
        # in the current-step block this change controls.
        self.assertNotIn("which screen blocked us before", step_section)
        self.assertNotIn("- memory read:", step_section)

    def test_non_matching_memory_read_adds_nothing(self):
        agent = self.store.create_agent(
            name="memory bot", system_prompt="Run workflow steps.", provider_id="openai"
        )
        self.store.add_memory(
            agent_id=agent["id"],
            content="The settings screen blocked payment sync last run.",
        )
        _agent, _task, result = self._prepare(
            agent,
            [
                {
                    "id": "diagnose",
                    "type": "llm",
                    "name": "Diagnose",
                    "instruction": "Work out what went wrong.",
                    "memory_read": "database credentials rotation history",
                }
            ],
        )

        messages = self._run_with_fake_stream(result["execution"], result_text="Found it.")

        self.assertEqual(len(messages), 1)
        step_section = self._current_step_section(messages[0])
        self.assertNotIn("- relevant memories:", step_section)
        self.assertNotIn("- memory read:", step_section)
        self.assertNotIn("database credentials rotation history", step_section)

    # -- memory_write ------------------------------------------------------

    def test_completed_step_with_memory_write_creates_memory_with_run_id(self):
        agent = self.store.create_agent(
            name="memory bot", system_prompt="Run workflow steps.", provider_id="openai"
        )
        _agent, task, result = self._prepare(
            agent,
            [
                {
                    "id": "diagnose",
                    "type": "llm",
                    "name": "Diagnose",
                    "instruction": "Work out what went wrong.",
                    "memory_write": "Remember which screen blocked us.",
                }
            ],
        )
        run_id = result["run"]["id"]

        self._run_with_fake_stream(
            result["execution"], result_text="The settings screen blocked us."
        )

        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "completed")
        memories = self.store.list_memories(agent["id"]) or []
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "The settings screen blocked us.")
        self.assertEqual(memories[0]["source_run_id"], run_id)
        # Distinguishable from a hand-written memory (source_event_type "manual").
        self.assertNotEqual(memories[0]["source_event_type"], "manual")

    def test_step_with_no_result_writes_no_memory(self):
        agent = self.store.create_agent(
            name="memory bot", system_prompt="Run workflow steps.", provider_id="openai"
        )
        _agent, task, result = self._prepare(
            agent,
            [
                {
                    "id": "diagnose",
                    "type": "llm",
                    "name": "Diagnose",
                    "instruction": "Work out what went wrong.",
                    "memory_write": "Remember which screen blocked us.",
                }
            ],
        )

        # The provider turn completes but the model said nothing worth
        # recording (sink.result_text stays None) — nothing should be invented
        # to fill the memory this step promised.
        self._run_with_fake_stream(result["execution"], result_text=None)

        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "completed")
        memories = self.store.list_memories(agent["id"]) or []
        self.assertEqual(memories, [])

    def test_failing_memory_write_does_not_fail_the_step(self):
        agent = self.store.create_agent(
            name="memory bot", system_prompt="Run workflow steps.", provider_id="openai"
        )
        _agent, task, result = self._prepare(
            agent,
            [
                {
                    "id": "diagnose",
                    "type": "llm",
                    "name": "Diagnose",
                    "instruction": "Work out what went wrong.",
                    "memory_write": "Remember which screen blocked us.",
                }
            ],
        )

        def broken_add_memory(*_args, **_kwargs):
            raise RuntimeError("memory store unavailable")

        self.store.add_memory = broken_add_memory

        self._run_with_fake_stream(
            result["execution"], result_text="The settings screen blocked us."
        )

        # The step's real work succeeded; losing the note must not turn that
        # into a failed run.
        steps = self.store.list_task_steps(task["id"])
        self.assertEqual(steps[0]["status"], "completed")
        self.assertEqual(self.store.get_task(task["id"])["status"], "completed")


if __name__ == "__main__":
    unittest.main()
