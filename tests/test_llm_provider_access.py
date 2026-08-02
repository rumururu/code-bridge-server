"""Switching a provider off has to remove it from every model picker.

The dashboard has had this checkbox all along. It called an endpoint that did
not exist, ignored the 404, and reported success — so unchecking OpenAI
announced "LLM disabled" and the box came back checked on the next render,
because nothing was ever stored.

`selectable` is the one gate every picker consults — the dev chat, the
Configurator, the phone's settings list — so that is where the switch lands.
These pin the two ways it can go wrong: a provider that stays offered after
being switched off, and switching off the last one that could answer.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from llm import llm_settings


class _FakeDb:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def get(self, key: str):
        return self._values.get(key)

    def set(self, key: str, value) -> None:
        self._values[key] = value


class ProviderAccessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _FakeDb()
        db_patch = patch.object(llm_settings, "get_settings_db", return_value=self.db)
        db_patch.start()
        self.addCleanup(db_patch.stop)
        # Every CLI present, so the only thing moving `selectable` is the switch.
        cli_patch = patch.object(
            llm_settings, "_check_cli_available", return_value=(True, None)
        )
        cli_patch.start()
        self.addCleanup(cli_patch.stop)

    def _selectable(self) -> list[str]:
        snapshot = llm_settings.get_llm_options_snapshot()
        return [c["id"] for c in snapshot["companies"] if c["selectable"]]

    def test_a_provider_starts_enabled(self):
        self.assertIn("openai", self._selectable())

    def test_disabling_removes_it_from_every_picker(self):
        llm_settings.set_company_enabled("openai", False)
        self.assertNotIn("openai", self._selectable())

    def test_the_choice_survives_a_reread(self):
        # The bug: nothing was stored, so the next render restored the old box.
        llm_settings.set_company_enabled("openai", False)
        llm_settings.get_llm_options_snapshot()
        self.assertNotIn("openai", self._selectable())
        self.assertEqual(
            json.loads(self.db.get(llm_settings.DISABLED_COMPANIES_KEY)), ["openai"]
        )

    def test_re_enabling_puts_it_back(self):
        llm_settings.set_company_enabled("openai", False)
        llm_settings.set_company_enabled("openai", True)
        self.assertIn("openai", self._selectable())

    def test_enabled_is_reported_separately_from_connected(self):
        """A disabled provider is still installed, and the UI must say so.

        Collapsing the two would make "switched off" look like "not installed"
        and offer to reinstall a CLI that is already there.
        """
        llm_settings.set_company_enabled("openai", False)
        company = next(
            c
            for c in llm_settings.get_llm_options_snapshot()["companies"]
            if c["id"] == "openai"
        )
        self.assertTrue(company["connected"])
        self.assertFalse(company["enabled"])
        self.assertFalse(company["selectable"])

    def test_the_last_provider_cannot_be_switched_off(self):
        # Nothing left to answer with would surface later as an unrelated chat
        # failure, far from the checkbox that caused it.
        for company_id in ("openai", "google", "antigravity"):
            llm_settings.set_company_enabled(company_id, False)
        with self.assertRaises(ValueError):
            llm_settings.set_company_enabled("anthropic", False)
        self.assertIn("anthropic", self._selectable())

    def test_unknown_provider_is_refused(self):
        with self.assertRaises(ValueError):
            llm_settings.set_company_enabled("not-a-provider", False)

    def test_disabling_the_selected_provider_moves_the_selection(self):
        llm_settings.set_company_enabled("anthropic", True)
        snapshot = llm_settings.get_llm_options_snapshot()
        selected = snapshot["selected"]["company_id"]

        llm_settings.set_company_enabled(selected, False)

        after = llm_settings.get_llm_options_snapshot()["selected"]["company_id"]
        self.assertNotEqual(after, selected)
        self.assertIn(after, self._selectable())


class AntigravityTest(unittest.TestCase):
    def test_it_is_a_known_provider(self):
        provider = llm_settings._get_provider("antigravity")
        self.assertIsNotNone(provider)
        self.assertEqual(provider.command, "agy")
        self.assertTrue(provider.chat_supported)
        self.assertTrue(provider.models)

    def test_it_has_an_install_route(self):
        self.assertTrue(llm_settings.LLM_PROVIDER_INSTALL_METHODS.get("antigravity"))


if __name__ == "__main__":
    unittest.main()
