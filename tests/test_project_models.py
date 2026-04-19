import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from projects.project_models import ProjectType


class ProjectModelsTest(unittest.TestCase):
    def test_project_type_from_string_aliases(self):
        self.assertEqual(ProjectType.from_string("flutter"), ProjectType.FLUTTER)
        self.assertEqual(ProjectType.from_string("next"), ProjectType.NEXTJS)
        self.assertEqual(ProjectType.from_string("next.js"), ProjectType.NEXTJS)
        self.assertEqual(ProjectType.from_string("unknown"), ProjectType.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
