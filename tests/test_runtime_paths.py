import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import push_notifier
from core import server_logging
from core.runtime_paths import runtime_dir, runtime_path


class RuntimePathsTest(unittest.TestCase):
    def test_runtime_path_uses_legacy_path_without_app_support_env(self):
        legacy = Path("/tmp/code_bridge_legacy_state.json")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_path("state.json", legacy), legacy)

    def test_runtime_path_uses_app_support_env_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CODEBRIDGE_APP_SUPPORT_DIR": tmp}):
                self.assertEqual(runtime_path("state.json", Path("/legacy/state.json")), Path(tmp) / "state.json")

    def test_runtime_dir_creates_named_directory_under_app_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CODEBRIDGE_APP_SUPPORT_DIR": tmp}):
                path = runtime_dir("pairing", Path("/legacy/pairing"))

            self.assertEqual(path, Path(tmp) / "pairing")
            self.assertTrue(path.exists())

    def test_runtime_dir_creates_legacy_directory_without_app_support_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "global_chat"

            with patch.dict(os.environ, {}, clear=True):
                path = runtime_dir("global_chat", legacy)

            self.assertEqual(path, legacy)
            self.assertTrue(path.exists())


class ModuleDefaultsThroughResolverTest(unittest.TestCase):
    """push_notifier / server_logging defaults must go through the resolver.

    Both are resolved lazily (at call time, not import time) so an
    ``CODEBRIDGE_APP_SUPPORT_DIR`` set *after* the modules were imported —
    exactly the situation in this very test process, whose conftest sets the
    env but whose modules may have been imported by any earlier test — still
    redirects them. This is what makes the conftest env guard structurally
    unnecessary (it stays as defence in depth).
    """

    def test_service_account_production_default_unchanged_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                push_notifier.default_service_account_path(),
                Path.home() / ".code-bridge" / "firebase_service_account.json",
            )

    def test_service_account_follows_app_support_dir_set_after_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CODEBRIDGE_APP_SUPPORT_DIR": tmp}, clear=True):
                self.assertEqual(
                    push_notifier._service_account_path(),
                    Path(tmp) / "firebase_service_account.json",
                )

    def test_service_account_env_override_beats_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "CODEBRIDGE_APP_SUPPORT_DIR": tmp,
                push_notifier.SERVICE_ACCOUNT_ENV: "/elsewhere/key.json",
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(
                    push_notifier._service_account_path(), Path("/elsewhere/key.json")
                )

    def test_server_log_production_default_unchanged_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                server_logging.resolve_server_log_path(),
                Path("~/.code-bridge/logs/server.log").expanduser(),
            )

    def test_server_log_follows_app_support_dir_set_after_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CODEBRIDGE_APP_SUPPORT_DIR": tmp}, clear=True):
                self.assertEqual(
                    server_logging.resolve_server_log_path(),
                    Path(tmp) / "logs" / "server.log",
                )

    def test_server_log_env_override_beats_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "CODEBRIDGE_APP_SUPPORT_DIR": tmp,
                "CODE_BRIDGE_SERVER_LOG_PATH": "/elsewhere/server.log",
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(
                    server_logging.resolve_server_log_path(),
                    Path("/elsewhere/server.log"),
                )


if __name__ == "__main__":
    unittest.main()
