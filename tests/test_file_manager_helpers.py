import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from file_manager_helpers import (
    fuzzy_match,
    is_excluded_path,
    match_score,
    validate_project_relative_path,
)


class FileManagerHelpersTest(unittest.TestCase):
    def test_validate_project_relative_path_accepts_project_child(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir).resolve()
            result = validate_project_relative_path(base, "lib/main.dart")

        assert result is not None
        self.assertTrue(str(result).endswith("lib/main.dart"))

    def test_validate_project_relative_path_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir).resolve()
            result = validate_project_relative_path(base, "../secrets.txt")

        self.assertIsNone(result)

    def test_is_excluded_path_matches_exact_and_glob_suffix(self):
        patterns = {"node_modules", "*.lock"}
        self.assertTrue(is_excluded_path(Path("/tmp/node_modules"), patterns))
        self.assertTrue(is_excluded_path(Path("/tmp/pubspec.lock"), patterns))
        self.assertFalse(is_excluded_path(Path("/tmp/main.dart"), patterns))

    def test_fuzzy_match(self):
        self.assertTrue(fuzzy_match("abc", "a_b_c"))
        self.assertTrue(fuzzy_match("rd", "readme.md"))
        self.assertFalse(fuzzy_match("xyz", "readme.md"))

    def test_match_score_ordering(self):
        exact = match_score("readme.md", "readme.md")
        starts = match_score("read", "readme.md")
        contains = match_score("eadm", "readme.md")
        fuzzy = match_score("rmd", "readme.md")

        self.assertGreater(exact, starts)
        self.assertGreater(starts, contains)
        self.assertGreater(contains, fuzzy)


if __name__ == "__main__":
    unittest.main()
