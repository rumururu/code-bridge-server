import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from project_models import BuildStatus, ProjectType


class ProjectModelsTest(unittest.TestCase):
    def test_project_type_from_string_aliases(self):
        self.assertEqual(ProjectType.from_string("flutter"), ProjectType.FLUTTER)
        self.assertEqual(ProjectType.from_string("next"), ProjectType.NEXTJS)
        self.assertEqual(ProjectType.from_string("next.js"), ProjectType.NEXTJS)
        self.assertEqual(ProjectType.from_string("unknown"), ProjectType.UNKNOWN)

    def test_build_status_values(self):
        self.assertEqual(BuildStatus.NONE.value, "none")
        self.assertEqual(BuildStatus.BUILDING.value, "building")
        self.assertEqual(BuildStatus.READY.value, "ready")
        self.assertEqual(BuildStatus.ERROR.value, "error")


if __name__ == "__main__":
    unittest.main()
