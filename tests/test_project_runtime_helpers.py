import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_runtime_helpers import (
    extract_port,
    is_candidate_command,
    is_local_port_open,
    match_project_path_score,
    select_preferred_port,
)
from project_models import ProjectType


class ProjectRuntimeHelpersTest(unittest.TestCase):
    def test_match_project_path_score(self):
        root = Path("/tmp/demo")
        self.assertEqual(match_project_path_score(root, root), 3)
        self.assertEqual(match_project_path_score(root, root / "web"), 2)
        self.assertEqual(match_project_path_score(root / "web", root), 1)
        self.assertEqual(match_project_path_score(root, Path("/tmp/other")), -1)

    def test_is_candidate_command_by_project_type(self):
        self.assertTrue(is_candidate_command("flutter run -d device", ProjectType.FLUTTER))
        self.assertFalse(is_candidate_command("npm run dev", ProjectType.FLUTTER))

        self.assertTrue(is_candidate_command("pnpm dev", ProjectType.NEXTJS))
        self.assertFalse(is_candidate_command("python -m uvicorn app:app", ProjectType.NEXTJS))

        self.assertTrue(is_candidate_command("python -m uvicorn app:app", ProjectType.UNKNOWN))

    def test_select_preferred_port(self):
        self.assertEqual(select_preferred_port([9000, 3000, 5173]), 3000)
        self.assertEqual(select_preferred_port([5173, 7000]), 5173)
        self.assertEqual(select_preferred_port([7000, 9000]), 7000)

    def test_extract_port(self):
        self.assertEqual(extract_port("*:3000"), 3000)
        self.assertEqual(extract_port("TCP 127.0.0.1:5173->127.0.0.1:52888"), 5173)
        self.assertIsNone(extract_port("not-a-port"))
        self.assertIsNone(extract_port("TCP localhost:http"))

    def test_is_local_port_open_returns_true_when_socket_connects(self):
        fake_socket_factory = MagicMock()
        fake_socket_cm = MagicMock()
        fake_socket_instance = MagicMock()
        fake_socket_factory.return_value = fake_socket_cm
        fake_socket_cm.__enter__.return_value = fake_socket_instance
        fake_socket_instance.connect_ex.return_value = 0

        with patch("project_runtime_helpers.socket.socket", fake_socket_factory):
            self.assertTrue(is_local_port_open(3000))

    def test_is_local_port_open_returns_false_when_connect_fails(self):
        fake_socket_factory = MagicMock()
        fake_socket_cm = MagicMock()
        fake_socket_instance = MagicMock()
        fake_socket_factory.return_value = fake_socket_cm
        fake_socket_cm.__enter__.return_value = fake_socket_instance
        fake_socket_instance.connect_ex.return_value = 1

        with patch("project_runtime_helpers.socket.socket", fake_socket_factory):
            self.assertFalse(is_local_port_open(3000))


if __name__ == "__main__":
    unittest.main()
