"""Asking for a script mid-conversation, without the conversation ending there.

A ``shell`` step names a registered script by id, so a Configurator that needed
a script nobody had written could only describe one in prose. The user had to
leave the conversation, open the dashboard, have a script written there,
register it, come back and re-explain what the agent was for — and most did not
come back, which is why "check the disk every 6 hours" kept coming out as an
``llm`` step that wakes a model to read one number.

The fix is a proposal: the Configurator says what it needs, the server has it
written, and the user reads the actual script before it exists anywhere. What
must never regress is the gap between those last two. A proposal that
registered itself would turn "write me a script" into "run whatever you just
wrote" — on a machine where a scheduled agent runs unattended under a standing
approval rule, that is the difference between a suggestion and an installation.
So: drafting writes nothing, approval is a separate deliberate act, approving
twice installs one script, a drafting failure stays a failure instead of
arriving as a plausible server-written stand-in, and a name that is already
taken never quietly replaces what is there.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, get_args
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store, configurator, script_store  # noqa: E402
from agent.agent_models import WorkflowStep  # noqa: E402
from agent.workflow_v2 import ALLOWED_STEP_TYPES, normalize_workflow_step  # noqa: E402
from core import database  # noqa: E402
from routes import agents, dashboard_agents, script_proposals, scripts  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

DRAFTED_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
df -h / | awk 'NR==2 {print "USED_PCT=" $5}'
"""

CONFIGURATOR_TURN_WITH_SCRIPT_REQUEST = """디스크 사용량을 읽는 스크립트가 아직 없어서 초안을 만들고 있어요.

```draft
{
  "name": "Disk Watch",
  "description": "Reports free space and warns when the root filesystem fills up.",
  "system_prompt": "Check the disk and say plainly whether it needs attention.",
  "provider_id": "openai",
  "tools": [],
  "flow": [
    {
      "id": "check_disk",
      "type": "shell",
      "name": "Check disk usage",
      "description": "Read how full the root filesystem is."
    },
    {
      "id": "judge",
      "type": "llm",
      "name": "Judge",
      "description": "Decide whether the number needs attention.",
      "instruction": "Given USED_PCT, say whether it is a problem."
    }
  ],
  "memory_seeds": [],
  "script_requests": [
    {
      "name": "Check disk usage",
      "purpose": "Print how full the root filesystem is",
      "expected_output": "USED_PCT=<percent>",
      "step_id": "check_disk"
    }
  ]
}
```
"""


class _FakeLLMSession:
    async def abort_current_turn(self) -> None:  # pragma: no cover - never hit
        return None


class ScriptProposalTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        self._original_db_path = database.DB_PATH
        database.DB_PATH = root / "script_proposals.db"
        agent_store._agent_store = None
        script_store._script_store = None
        database.init_db()

        self._original_scripts_dir = scripts._MANAGED_SCRIPTS_DIR
        self.managed_dir = root / "generated_scripts"
        scripts._MANAGED_SCRIPTS_DIR = self.managed_dir

        configurator.BUILDER_SESSIONS.clear()
        agents.BUILDER_CONVERSE_JOBS.clear()
        script_proposals.SCRIPT_PROPOSALS.clear()
        self.addCleanup(self._restore)

        app = FastAPI()
        app.include_router(agents.router)
        app.include_router(script_proposals.router)
        app.include_router(dashboard_agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def _restore(self) -> None:
        configurator.BUILDER_SESSIONS.clear()
        agents.BUILDER_CONVERSE_JOBS.clear()
        script_proposals.SCRIPT_PROPOSALS.clear()
        scripts._MANAGED_SCRIPTS_DIR = self._original_scripts_dir
        agent_store._agent_store = None
        script_store._script_store = None
        database.DB_PATH = self._original_db_path

    # -- helpers ---------------------------------------------------------

    def _registered_scripts(self) -> list[dict[str, Any]]:
        return script_store.get_script_store().list_scripts(limit=50)

    def _managed_files(self) -> list[Path]:
        if not self.managed_dir.exists():
            return []
        return sorted(self.managed_dir.iterdir())

    def _converse(
        self,
        message: str,
        *,
        session_id: str | None = None,
        script_writer_returns: str | None = DRAFTED_SCRIPT,
    ):
        """One builder turn with both LLMs faked: the Configurator and the writer.

        They are separate calls on purpose — the script writer runs after this
        turn is answered, so the two are patched independently here exactly as
        they fail independently in production.
        """

        async def fake_configurator_turn(_session, *, timeout=120.0):
            return CONFIGURATOR_TURN_WITH_SCRIPT_REQUEST

        async def fake_script_writer(_llm_session, _prompt):
            if script_writer_returns is None:
                raise RuntimeError("provider returned an error")
            return script_writer_returns

        with (
            patch("routes.agents.run_configurator_turn", fake_configurator_turn),
            patch(
                "chat.chat_session_service.create_chat_session",
                AsyncMock(return_value=_FakeLLMSession()),
            ),
            patch(
                "chat.chat_session_service.get_chat_provider_selection",
                lambda: {"provider_id": "openai"},
            ),
            patch("routes.agents._collect_llm_response_text", fake_script_writer),
        ):
            payload: dict[str, Any] = {"user_message": message}
            if session_id:
                payload["session_id"] = session_id
            response = self.client.post(
                "/api/agent/builder/converse",
                json=payload,
            )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _proposal(self, proposal_id: str) -> dict[str, Any]:
        response = self.client.get(
            f"/api/agent/builder/scripts/proposals/{proposal_id}"
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["proposal"]

    def _approve(self, proposal_id: str):
        return self.client.post(
            f"/api/dashboard/agent/builder/scripts/proposals/{proposal_id}/approve"
        )

    # -- tests -----------------------------------------------------------

    def test_a_turn_asking_for_a_script_drafts_it_and_registers_nothing(self):
        turn = self._converse("디스크 6시간마다 확인하고 위험하면 알려주는 에이전트 만들어줘.")

        proposals = turn["script_proposals"]
        self.assertEqual(len(proposals), 1)
        proposal_id = proposals[0]["proposal_id"]
        self.assertEqual(proposals[0]["name"], "Check disk usage")
        self.assertEqual(proposals[0]["step_id"], "check_disk")

        drafted = self._proposal(proposal_id)
        self.assertEqual(drafted["status"], "ready")
        self.assertIn("df -h", drafted["body"])
        self.assertIsNone(drafted["script_id"])

        # The point of the whole design: the script the user is reading exists
        # nowhere but in this response.
        self.assertEqual(self._registered_scripts(), [])
        self.assertEqual(self._managed_files(), [])

    def test_the_next_turn_carries_the_drafted_text(self):
        """The body lands after the turn that asked for it, so turns must catch up.

        Drafting is a second LLM call with a 180s budget; a turn answers in 20.
        A client that only reads turns would never see the script if the text
        were attached solely to the turn that raised the request.
        """
        first = self._converse("디스크 확인 에이전트 만들어줘.")
        # `body` is absent, not empty: the turn omits null fields.
        self.assertIsNone(first["script_proposals"][0].get("body"))
        self.assertEqual(first["script_proposals"][0]["status"], "drafting")

        second = self._converse("좋아, 그걸로 해줘.", session_id=first["session_id"])
        self.assertEqual(len(second["script_proposals"]), 1, "one request, one proposal")
        self.assertIn("df -h", second["script_proposals"][0]["body"])
        self.assertEqual(self._registered_scripts(), [])

    def test_approving_registers_it_and_the_id_normalizes_on_a_shell_step(self):
        turn = self._converse("디스크 확인 에이전트 만들어줘.")
        proposal_id = turn["script_proposals"][0]["proposal_id"]

        approved = self._approve(proposal_id)
        self.assertEqual(approved.status_code, 200, approved.text)
        payload = approved.json()
        script_id = payload["script_id"]
        self.assertTrue(script_id.startswith("script_"))
        self.assertEqual(payload["proposal"]["status"], "registered")

        registered = self._registered_scripts()
        self.assertEqual(len(registered), 1)
        self.assertEqual(registered[0]["id"], script_id)
        on_disk = Path(registered[0]["path"])
        self.assertEqual(on_disk.parent, self.managed_dir)
        self.assertIn("df -h", on_disk.read_text())

        # The id is only worth anything if a shell step accepts it.
        step = normalize_workflow_step(
            {
                "id": "check_disk",
                "type": "shell",
                "name": "Check disk usage",
                "script_id": script_id,
            },
            index=1,
        )
        self.assertEqual(step["script_id"], script_id)

        # And the step in the live conversation already names it, so nobody
        # has to copy an id from one panel into another.
        filled = next(
            item for item in payload["updated_draft"]["flow"] if item["id"] == "check_disk"
        )
        self.assertEqual(filled["script_id"], script_id)
        self.assertEqual(payload["updated_draft"]["script_requests"], [])

    def test_approving_twice_does_not_register_twice(self):
        turn = self._converse("디스크 확인 에이전트 만들어줘.")
        proposal_id = turn["script_proposals"][0]["proposal_id"]

        first = self._approve(proposal_id)
        second = self._approve(proposal_id)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["script_id"], second.json()["script_id"])
        self.assertEqual(len(self._registered_scripts()), 1)
        self.assertEqual(len(self._managed_files()), 1)

    def test_a_drafting_failure_is_reported_not_papered_over(self):
        turn = self._converse("디스크 확인 에이전트 만들어줘.", script_writer_returns=None)
        proposal_id = turn["script_proposals"][0]["proposal_id"]

        failed = self._proposal(proposal_id)
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["error"])
        # No hand-rolled stand-in: a server-written script would arrive on the
        # same card, with the same approve button, and the one thing the user
        # is judging is whether the script is right.
        self.assertIsNone(failed["body"])
        self.assertEqual(self._registered_scripts(), [])
        self.assertEqual(self._managed_files(), [])

        refused = self._approve(proposal_id)
        self.assertEqual(refused.status_code, 409)
        self.assertEqual(self._registered_scripts(), [])

    def test_a_name_that_already_exists_is_not_overwritten(self):
        existing = self.managed_dir
        existing.mkdir(parents=True, exist_ok=True)
        taken = existing / f"{scripts._slugify('Check disk usage')}.sh"
        taken.write_text("#!/usr/bin/env bash\necho 'the one the user already trusts'\n")
        script_store.get_script_store().register(
            name="Check disk usage",
            path=str(taken),
            description="Already registered by hand",
        )

        turn = self._converse("디스크 확인 에이전트 만들어줘.")
        proposal_id = turn["script_proposals"][0]["proposal_id"]
        rejected = self._approve(proposal_id)

        self.assertEqual(rejected.status_code, 400)
        self.assertIn("already exists", rejected.json()["detail"])
        self.assertIn("already trusts", taken.read_text())
        self.assertEqual(len(self._registered_scripts()), 1)
        self.assertEqual(self._managed_files(), [taken])
        # Still approvable once the collision is resolved — a dead end here is
        # the dead end this whole feature exists to remove.
        self.assertEqual(self._proposal(proposal_id)["status"], "ready")

    def test_every_runtime_step_type_can_appear_in_a_draft(self):
        """A type the runtime runs but the draft model rejects is invisible drift.

        ``shell`` and ``notify`` were missing from ``WorkflowStep`` for as long
        as they existed, so a draft block containing the shell step this
        feature is built around failed validation and the entire turn's draft
        was discarded as malformed.
        """
        draftable = set(get_args(WorkflowStep.model_fields["type"].annotation))
        self.assertEqual(draftable, set(ALLOWED_STEP_TYPES))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
