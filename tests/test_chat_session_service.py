import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import chat_session_service


class ChatSessionServiceTest(unittest.IsolatedAsyncioTestCase):
    def test_get_chat_provider_selection_defaults_to_anthropic(self):
        with patch.object(chat_session_service, "get_llm_options_snapshot", return_value={}):
            selection = chat_session_service.get_chat_provider_selection()

        self.assertEqual(selection.provider_id, "anthropic")
        self.assertEqual(selection.provider_name, "Claude")
        self.assertIsNone(selection.model)

    def test_get_chat_provider_selection_raises_for_unsupported_provider(self):
        with patch.object(
            chat_session_service,
            "get_llm_options_snapshot",
            return_value={"selected": {"company_id": "unknown", "model": "x"}},
        ):
            with self.assertRaises(chat_session_service.ChatSessionInitError):
                chat_session_service.get_chat_provider_selection()

    async def test_create_chat_session_passes_selection_to_manager(self):
        selection = chat_session_service.ChatProviderSelection(
            provider_id="openai",
            provider_name="Openai",
            model="gpt-4.1",
        )
        fake_manager = MagicMock()
        fake_manager.get_or_create_session = AsyncMock(return_value="session-object")

        with patch.object(chat_session_service, "get_session_manager", return_value=fake_manager):
            session = await chat_session_service.create_chat_session(
                project_name="demo",
                project_path="/tmp/demo",
                selection=selection,
            )

        self.assertEqual(session, "session-object")
        fake_manager.get_or_create_session.assert_awaited_once_with(
            "demo",
            "/tmp/demo",
            provider_id="openai",
            model="gpt-4.1",
        )
