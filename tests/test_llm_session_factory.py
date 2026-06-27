import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm.codex_session import CodexSession
from llm.gemini_session import GeminiSession
from llm.llm_session import LlmSessionFactory


class LlmSessionFactoryTest(unittest.TestCase):
    def test_create_google_session(self):
        session = LlmSessionFactory.create_session(
            provider_id="google",
            project_path="/tmp/demo",
            model="gemini-2.5-pro",
        )

        self.assertIsInstance(session, GeminiSession)
        self.assertEqual(session.project_path, "/tmp/demo")
        self.assertEqual(session.model, "gemini-2.5-pro")

    def test_create_openai_session_loads_codex_settings_from_package(self):
        with patch(
            "llm.llm_settings.get_codex_sandbox_mode",
            return_value="workspace-write",
        ):
            session = LlmSessionFactory.create_session(
                provider_id="openai",
                project_path="/tmp/demo",
                model="gpt-4.1",
            )

        self.assertIsInstance(session, CodexSession)
        self.assertEqual(session.project_path, "/tmp/demo")
        self.assertEqual(session.model, "gpt-4.1")
        self.assertEqual(session.sandbox_mode, "workspace-write")

    def test_supported_providers_includes_google(self):
        self.assertIn("google", LlmSessionFactory.get_supported_providers())


if __name__ == "__main__":
    unittest.main()
