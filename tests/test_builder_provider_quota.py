"""When the builder's LLM is out of quota, the turn fails *and* says what else there is.

Background, because the shape of this is easy to get wrong. Two different
provider choices live in this codebase and only one of them is involved here:

* ``AgentDraft.provider_id`` (``code_bridge_core/configurator.py``) — which backend the
  *finished agent* will run on later. Not this.
* ``llm.selected_company``, read by ``get_chat_provider_selection()`` and used
  in ``routes.agents.run_configurator_turn`` — which backend *writes* the
  agent. This one.

A user drove the Configurator with Codex selected and got back its quota
message, and stopped there — with a working Claude CLI installed on the same
machine. Failing was correct (a fabricated draft would have been far worse);
being a dead end was not.

What these tests pin down:

* the turn still **fails**, and still invents nothing — unchanged;
* nothing on the server switches provider — a failover the user did not ask
  for would silently change the model writing their agent, and would go on
  doing it for the weeks a quota block lasts while Settings claimed otherwise;
* a *quota* failure additionally reports which provider failed and what else
  is installed, so the client can offer the switch;
* a failure that is not plainly about quota attaches no such offer, because
  switching would only hide it.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import agent_store
from code_bridge_core import configurator  # noqa: E402
from approvals import approval_store  # noqa: E402
from audit import audit_store  # noqa: E402
from core import database  # noqa: E402
from policy import policy_store  # noqa: E402
from routes import agents  # noqa: E402
from routes.deps import verify_api_key  # noqa: E402

CODEX_QUOTA_MESSAGE = (
    "You've hit your usage limit. Upgrade to Plus to continue using Codex "
    "(https://chatgpt.com/explore/plus), or try again at Sep 15th, 2026"
)

CLAUDE_ALTERNATIVE = {
    "company_id": "anthropic",
    "name": "Claude",
    "command": "claude",
    "model": "sonnet",
}


class BuilderProviderQuotaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "code_bridge_quota_test.db"
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        agents.BUILDER_CONVERSE_JOBS.clear()
        configurator.BUILDER_SESSIONS.clear()

        app = FastAPI()
        app.include_router(agents.router)
        app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
        self.client = TestClient(app)

    def tearDown(self):
        agent_store._agent_store = None
        approval_store._approval_store = None
        audit_store._audit_store = None
        policy_store._policy_rule_store = None
        agents.BUILDER_CONVERSE_JOBS.clear()
        configurator.BUILDER_SESSIONS.clear()
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def _converse(self, *, failure: Exception, alternatives: list[dict]):
        async def fail(_session, *, timeout=120.0, job=None):
            raise failure

        with patch("routes.agents.run_configurator_turn", fail):
            with patch(
                "routes.agents.list_alternative_chat_providers",
                return_value=alternatives,
            ) as lookup:
                response = self.client.post(
                    "/api/agent/builder/converse",
                    json={
                        "user_message": (
                            "매일 밤 스크립트를 실행하고, 종료코드가 0이 아니면 "
                            "AI가 원인을 진단해서 알림을 보내는 에이전트를 만들어줘."
                        )
                    },
                )
        return response, lookup

    def test_quota_failure_reports_the_provider_and_what_else_is_installed(self):
        response, lookup = self._converse(
            failure=agents.ProviderTurnError(
                CODEX_QUOTA_MESSAGE,
                provider_id="openai",
                provider_name="Codex",
            ),
            alternatives=[CLAUDE_ALTERNATIVE],
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        # Unchanged, and the point of the original fix: a failure is a failure.
        self.assertEqual(payload["status"], "failed")
        self.assertFalse(payload["fallback"])
        self.assertFalse(payload["is_ready_to_commit"])
        self.assertFalse(payload["updated_draft"].get("name"))
        self.assertFalse(payload["updated_draft"]["flow"])
        # The provider's own words survive verbatim — the date in that sentence
        # is the only place the user learns when the limit lifts.
        self.assertEqual(payload["error"], CODEX_QUOTA_MESSAGE)

        # New: enough for a client to offer a way out.
        self.assertEqual(payload["error_kind"], "quota")
        self.assertEqual(payload["error_provider_id"], "openai")
        self.assertEqual(payload["error_provider_name"], "Codex")
        self.assertEqual(payload["provider_alternatives"], [CLAUDE_ALTERNATIVE])
        # The failing provider is excluded from its own replacement list.
        lookup.assert_called_once_with(exclude_company_id="openai")

        # The message the user reads names the alternative rather than telling
        # them to "try again shortly" — the quota lifts in three weeks.
        self.assertIn("Claude", payload["assistant_message"])

    def test_quota_failure_with_nothing_else_installed_offers_nothing(self):
        response, _ = self._converse(
            failure=agents.ProviderTurnError(
                CODEX_QUOTA_MESSAGE,
                provider_id="openai",
                provider_name="Codex",
            ),
            alternatives=[],
        )

        payload = response.json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_kind"], "quota")
        # Omitted rather than sent as an empty list: there is no offer to make,
        # and a client must not render an empty chooser.
        self.assertIsNone(payload.get("provider_alternatives"))
        self.assertIn("설치", payload["assistant_message"])

    def test_a_non_quota_failure_attaches_no_offer(self):
        """Switching provider would hide this, not fix it."""
        response, lookup = self._converse(
            failure=agents.ProviderTurnError(
                "400 Bad Request: unknown field 'flow_json'",
                provider_id="openai",
                provider_name="Codex",
            ),
            alternatives=[CLAUDE_ALTERNATIVE],
        )

        payload = response.json()
        self.assertEqual(payload["status"], "failed")
        self.assertIsNone(payload.get("error_kind"))
        self.assertIsNone(payload.get("provider_alternatives"))
        # Not even probed: an unrecognised failure is not an occasion to go
        # looking for somewhere else to send the user.
        lookup.assert_not_called()

    def test_the_server_never_switches_provider_by_itself(self):
        """The offer is an offer. Only an explicit call changes the setting."""
        with patch("routes.agents.list_alternative_chat_providers", return_value=[CLAUDE_ALTERNATIVE]):
            with patch("llm.llm_settings.set_selected_llm") as set_selected:
                async def fail(_session, *, timeout=120.0, job=None):
                    raise agents.ProviderTurnError(
                        CODEX_QUOTA_MESSAGE,
                        provider_id="openai",
                        provider_name="Codex",
                    )

                with patch("routes.agents.run_configurator_turn", fail):
                    self.client.post(
                        "/api/agent/builder/converse",
                        json={"user_message": "매일 밤 스크립트를 실행해줘"},
                    )

        set_selected.assert_not_called()

    def test_the_job_path_carries_the_same_fields(self):
        """The dashboard polls a job rather than waiting inline, so the offer
        has to survive `_job_payload`'s flattening — otherwise the feature
        exists only on the route nobody's browser uses."""

        async def fail(_session, *, timeout=120.0, job=None):
            raise agents.ProviderTurnError(
                CODEX_QUOTA_MESSAGE,
                provider_id="openai",
                provider_name="Codex",
            )

        with patch("routes.agents.run_configurator_turn", fail):
            with patch(
                "routes.agents.list_alternative_chat_providers",
                return_value=[CLAUDE_ALTERNATIVE],
            ):
                started = self.client.post(
                    "/api/agent/builder/converse/jobs",
                    json={"user_message": "매일 밤 스크립트를 실행해줘"},
                )
                job_id = started.json()["job_id"]
                polled = self.client.get(
                    f"/api/agent/builder/converse/jobs/{job_id}"
                ).json()

        self.assertEqual(polled["status"], "failed")
        self.assertEqual(polled["error_kind"], "quota")
        self.assertEqual(polled["error_provider_name"], "Codex")
        self.assertEqual(polled["provider_alternatives"], [CLAUDE_ALTERNATIVE])


class ProviderTurnErrorTest(unittest.TestCase):
    def test_it_is_still_a_runtime_error(self):
        """Every existing `except RuntimeError` around the builder keeps
        catching it — the provider is added to the failure, not put in the
        way of the handlers that already deal with it."""
        error = agents.ProviderTurnError(
            "boom", provider_id="openai", provider_name="Codex"
        )
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "boom")
        self.assertEqual(error.provider_id, "openai")


if __name__ == "__main__":
    unittest.main()
