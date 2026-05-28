import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

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

    def test_supported_providers_includes_google(self):
        self.assertIn("google", LlmSessionFactory.get_supported_providers())


if __name__ == "__main__":
    unittest.main()
