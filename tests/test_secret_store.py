"""Unit tests for ``secret_store_pkg.secret_store``.

Covers the contract demanded by ``TASK_001_CLAUDE.md`` § 5 and the
acceptance criteria in § Acceptance:
  1. Empty file -> empty list.
  2. Upsert -> on disk + ``os.environ``.
  3. Repeated upsert -> last value wins, no dup lines.
  4. Multi-key upsert -> sorted on disk.
  5. Delete -> removed from disk and ``os.environ``, returns True.
  6. Delete missing key -> returns False.
  7. Invalid keys -> ``SecretStoreError``.
  8. Newline in value -> ``SecretStoreError``.
  9. (bonus) File permission ``0600`` (POSIX only).

Every test redirects ``ENV_FILE`` to a temp dir so the developer's real
``~/.code-bridge/.env`` is never touched.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from secret_store_pkg import secret_store  # noqa: E402


class SecretStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_root = Path(self._tmp.name)
        self._tmp_env = tmp_root / "code-bridge" / ".env"
        # Snapshot patched attributes so tearDown can restore.
        self._original_env_file = secret_store.ENV_FILE
        secret_store.ENV_FILE = self._tmp_env
        # Snapshot os.environ keys we may mutate so a leaky test does not
        # pollute the suite-wide environment.
        self._env_snapshot = dict(os.environ)

    def tearDown(self) -> None:
        secret_store.ENV_FILE = self._original_env_file
        # Restore os.environ.
        for key in ("NAVER_ID", "NAVER_PW", "FOO_KEY"):
            if key in os.environ and key not in self._env_snapshot:
                del os.environ[key]
            elif key in self._env_snapshot:
                os.environ[key] = self._env_snapshot[key]
        self._tmp.cleanup()

    # ------------------------------------------------------------------

    def test_list_keys_when_file_missing_returns_empty(self) -> None:
        self.assertFalse(self._tmp_env.exists())
        self.assertEqual(secret_store.list_keys(), [])

    def test_upsert_writes_file_and_sets_environ(self) -> None:
        secret_store.upsert("NAVER_ID", "foo")

        self.assertTrue(self._tmp_env.is_file())
        contents = self._tmp_env.read_text(encoding="utf-8")
        self.assertEqual(contents, "NAVER_ID=foo\n")
        self.assertEqual(os.environ["NAVER_ID"], "foo")

    def test_upsert_same_key_twice_overwrites_without_duplicate_lines(self) -> None:
        secret_store.upsert("NAVER_ID", "first")
        secret_store.upsert("NAVER_ID", "second")

        contents = self._tmp_env.read_text(encoding="utf-8")
        self.assertEqual(contents, "NAVER_ID=second\n")
        # Make sure we only have one entry, not two.
        self.assertEqual(contents.count("NAVER_ID="), 1)
        keys = secret_store.list_keys()
        self.assertEqual(keys, [{"name": "NAVER_ID", "has_value": True}])
        self.assertEqual(os.environ["NAVER_ID"], "second")

    def test_upsert_multiple_keys_sorted_on_disk(self) -> None:
        secret_store.upsert("NAVER_PW", "pw")
        secret_store.upsert("NAVER_ID", "id")  # added second, but sorts first
        secret_store.upsert("FOO_KEY", "fkv")

        contents = self._tmp_env.read_text(encoding="utf-8")
        self.assertEqual(
            contents,
            "FOO_KEY=fkv\nNAVER_ID=id\nNAVER_PW=pw\n",
        )
        keys = secret_store.list_keys()
        self.assertEqual(
            keys,
            [
                {"name": "FOO_KEY", "has_value": True},
                {"name": "NAVER_ID", "has_value": True},
                {"name": "NAVER_PW", "has_value": True},
            ],
        )

    def test_delete_existing_key_returns_true_and_clears_state(self) -> None:
        secret_store.upsert("NAVER_ID", "foo")
        secret_store.upsert("NAVER_PW", "bar")

        result = secret_store.delete("NAVER_ID")

        self.assertTrue(result)
        contents = self._tmp_env.read_text(encoding="utf-8")
        self.assertEqual(contents, "NAVER_PW=bar\n")
        self.assertNotIn("NAVER_ID", os.environ)

    def test_delete_missing_key_returns_false(self) -> None:
        secret_store.upsert("NAVER_PW", "bar")

        result = secret_store.delete("NAVER_ID")

        self.assertFalse(result)
        # Other entries still intact.
        contents = self._tmp_env.read_text(encoding="utf-8")
        self.assertEqual(contents, "NAVER_PW=bar\n")

    def test_invalid_key_raises_secret_store_error(self) -> None:
        bad_keys = [
            "naver_id",  # lowercase start
            "Naver",  # mixed case
            "1NAVER",  # leading digit
            "NAVER-ID",  # hyphen
            "NAVER ID",  # space
            "",  # empty
            "NAVER\nID",  # embedded newline (carrier for injection)
        ]
        for bad in bad_keys:
            with self.subTest(key=bad):
                with self.assertRaises(secret_store.SecretStoreError):
                    secret_store.upsert(bad, "anything")
                with self.assertRaises(secret_store.SecretStoreError):
                    secret_store.delete(bad)

    def test_value_with_newline_or_carriage_return_rejected(self) -> None:
        for bad_value in ("foo\nbar", "foo\rbar", "line1\r\nline2"):
            with self.subTest(value=repr(bad_value)):
                with self.assertRaises(secret_store.SecretStoreError):
                    secret_store.upsert("NAVER_ID", bad_value)
        # Nothing should have been written to disk.
        self.assertFalse(self._tmp_env.exists())
        self.assertNotIn("NAVER_ID", os.environ)

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits only")
    def test_file_permission_is_0600(self) -> None:
        secret_store.upsert("NAVER_ID", "foo")

        mode = stat.S_IMODE(os.stat(self._tmp_env).st_mode)
        self.assertEqual(
            mode,
            0o600,
            f"expected 0o600, got {oct(mode)} on {sys.platform}",
        )

    def test_parser_ignores_blank_lines_and_comments(self) -> None:
        # Simulate a hand-edited ``.env``: blank lines, comments, and one
        # real entry. Should still parse cleanly.
        self._tmp_env.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_env.write_text(
            "# top comment\n\nNAVER_ID=existing\n# trailing\n",
            encoding="utf-8",
        )

        keys = secret_store.list_keys()
        self.assertEqual(keys, [{"name": "NAVER_ID", "has_value": True}])

    def test_has_value_false_for_empty_value(self) -> None:
        # An entry like ``KEY=`` (empty value, e.g. user deleted secret
        # by hand but left the key) should report ``has_value: False``.
        self._tmp_env.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_env.write_text("NAVER_ID=\n", encoding="utf-8")

        keys = secret_store.list_keys()
        self.assertEqual(keys, [{"name": "NAVER_ID", "has_value": False}])


if __name__ == "__main__":
    unittest.main()
