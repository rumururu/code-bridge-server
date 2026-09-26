"""The dashboard's `on_failure` / `on_success` editor uses the server's words.

The PC page used to hardcode its policy lists: three success choices and four
failure ones, with `retry` written out as the legacy string `retry_once` and
no way to say how many attempts. The server accepts six failure policies and
three success ones (`FAILURE_POLICY_TYPES` / `SUCCESS_POLICY_TYPES` in
`code_bridge_core.workflow_v2`), so `manual_handoff` and a `continue`-on-failure were
unauthorable from the PC, and a `retry` carrying `max_attempts` or a nested
`then` could only be written by hand through the API. That is the same drift
the step-type list had, one field set over — and it is invisible, because a
narrower list looks like a working screen.

`build_policy_schema()` (commit 780f06e) publishes the vocabulary, the
defaults, the labels and the per-policy parameters; `dashboard/templates/
agents.html` now derives its editor from `stepSchema.policies` instead of
listing anything. These tests pin what cannot be derived:

1. The mirror the page actually fetches carries `policies` (a page reading
   a payload the dashboard app does not serve would fall back forever).
2. The template really reads it, and no longer carries the old lists.
3. The three things the schema used to leave unsaid — the wire key a
   target-taking policy stores its target in, the slot a `retry` nests its
   follow-up policy in, and the older spellings the server accepts but does
   not offer — are now *published* (`option.target_key`,
   `option.nested_policy`, `slot.aliases`). These tests check the publication
   against the normalizer itself, so a rename in `workflow_v2` fails here
   instead of shipping a step editor that writes a key the server drops.
4. The page reads all three off the publication rather than by name, and the
   drift guards in `build_policy_schema` refuse to publish a shape the
   normalizer does not actually produce.
5. The page's built-in fallback list — used only when a server publishes no
   policy schema at all — still matches the published vocabulary, so the
   "old server" path cannot silently become the narrow list this change
   removed. That now covers the three keys above too.
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import app_factory  # noqa: E402
from agent import agent_store, schedule_store  # noqa: E402
from code_bridge_core import workflow_step_schema  # noqa: E402
from code_bridge_core.workflow_step_schema import (  # noqa: E402
    WorkflowSchemaDriftError,
    get_step_schema,
)
from code_bridge_core.workflow_v2 import (  # noqa: E402
    FAILURE_POLICY_ALIASES,
    FAILURE_POLICY_TYPES,
    POLICY_TYPES_REQUIRING_TARGET,
    SUCCESS_POLICY_ALIASES,
    SUCCESS_POLICY_TYPES,
    normalize_failure_policy,
    normalize_success_policy,
)
from core import database  # noqa: E402
from routes.deps import require_local_access  # noqa: E402

MARKUP = (SERVER_DIR / "dashboard" / "templates" / "agents.html").read_text(
    encoding="utf-8"
)


def _js_const(name: str) -> str:
    """The single-quoted value of a top-level `const NAME = '...';` in the page."""

    match = re.search(rf"const {name} = '([^']*)';", MARKUP)
    if match is None:  # pragma: no cover - the assertion below reports it
        raise AssertionError(f"{name} is not declared in agents.html")
    return match.group(1)


def _js_string_set(name: str) -> set[str]:
    """The members of a top-level `const NAME = new Set(['a', 'b']);`."""

    match = re.search(rf"const {name} = new Set\(\[(.*?)\]\);", MARKUP, re.S)
    if match is None:  # pragma: no cover
        raise AssertionError(f"{name} is not declared in agents.html")
    return set(re.findall(r"'([^']*)'", match.group(1)))


def _js_object(name: str) -> dict[str, str]:
    """A flat `const NAME = { a: 'x', b: 'y' };` from the page, as a dict."""

    anchor = MARKUP.index(f"const {name} = ")
    block = _balanced_block(MARKUP, MARKUP.index("{", anchor))
    return dict(re.findall(r"(\w+): '([^']*)'", block))


def _js_alias_tables() -> dict[str, dict[str, str]]:
    """`FALLBACK_POLICY_ALIASES` from the page, as {slot: {alias: name}}."""

    anchor = MARKUP.index("const FALLBACK_POLICY_ALIASES = ")
    block = _balanced_block(MARKUP, MARKUP.index("{", anchor))
    out: dict[str, dict[str, str]] = {}
    for slot in ("on_failure", "on_success"):
        slot_block = _balanced_block(block, block.index("{", block.index(f"{slot}: ")))
        out[slot] = dict(re.findall(r"(\w+): '([^']*)'", slot_block))
    return out


def _balanced_block(source: str, start: int) -> str:
    """The `{...}` beginning at ``start``, brace-matched."""

    depth = 0
    for position in range(start, len(source)):
        char = source[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1]
    raise AssertionError("unbalanced braces while reading the fallback schema")


def _fallback_schema() -> dict[str, dict]:
    """`FALLBACK_POLICY_SCHEMA` from the template, as {slot: {...}}.

    Only the parts this test pins are read: the slot's `default` and the
    ordered `type` / `requires_target` of each option. The labels are i18n
    keys, checked separately against the two locale tables.
    """

    anchor = MARKUP.index("const FALLBACK_POLICY_SCHEMA = ")
    block = _balanced_block(MARKUP, MARKUP.index("{", anchor))
    out: dict[str, dict] = {}
    for slot in ("on_failure", "on_success"):
        slot_start = block.index(f"{slot}: ")
        slot_block = _balanced_block(block, block.index("{", slot_start))
        options = []
        for option_source in re.findall(r"\{ type: '([^']*)'(.*?)\}", slot_block):
            policy_type, rest = option_source
            options.append(
                {
                    "type": policy_type,
                    "requires_target": "requires_target: true" in rest,
                    "label_key": (re.search(r"labelKey: '([^']*)'", rest) or [None, None])[1],
                }
            )
        default = re.search(r"default: '([^']*)'", slot_block)
        out[slot] = {
            "default": default.group(1) if default else None,
            "options": options,
        }
    return out


class MirrorPublishesPoliciesTest(unittest.TestCase):
    """The no-key dashboard mirror hands the page the same policy payload the
    phone gets. Without it the editor would sit on its fallback forever and
    nobody would notice, because a fallback that renders looks like a page
    that works."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._original_db_path = database.DB_PATH
        database.DB_PATH = Path(self._tmp.name) / "dashboard_policy_editor.db"
        agent_store._agent_store = None
        schedule_store._store = None

        self.app = app_factory.create_dashboard_app()
        self.app.dependency_overrides[require_local_access] = lambda: None
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        agent_store._agent_store = None
        schedule_store._store = None
        database.DB_PATH = self._original_db_path
        self._tmp.cleanup()

    def test_the_mirror_carries_the_whole_policy_vocabulary(self):
        response = self.client.get("/api/dashboard/agent/workflow/step-schema")
        self.assertEqual(response.status_code, 200)
        policies = response.json().get("policies")
        self.assertIsInstance(policies, dict)

        self.assertEqual(
            [option["type"] for option in policies["on_failure"]["options"]],
            list(FAILURE_POLICY_TYPES),
        )
        self.assertEqual(
            [option["type"] for option in policies["on_success"]["options"]],
            list(SUCCESS_POLICY_TYPES),
        )
        self.assertEqual(
            policies["on_failure"]["default"], normalize_failure_policy(None)["type"]
        )
        self.assertEqual(
            policies["on_success"]["default"], normalize_success_policy(None)["type"]
        )

    def test_every_option_the_mirror_sends_is_renderable(self):
        """Each choice arrives with words for both locales and a truthful
        `requires_target` — the page draws a step picker off that flag, and a
        policy offered without one saves a workflow the server refuses."""

        policies = self.client.get(
            "/api/dashboard/agent/workflow/step-schema"
        ).json()["policies"]
        for slot, slot_payload in policies.items():
            for option in slot_payload["options"]:
                with self.subTest(slot=slot, policy=option["type"]):
                    for locale in ("en", "ko"):
                        self.assertTrue((option["label"].get(locale) or "").strip())
                        self.assertTrue((option["help"].get(locale) or "").strip())
                    self.assertEqual(
                        option["requires_target"],
                        option["type"] in POLICY_TYPES_REQUIRING_TARGET,
                    )

    def test_the_mirror_carries_the_three_a_client_used_to_know_by_name(self):
        """A page that has to name the target key, the nesting key and the
        alias table is a page that goes wrong quietly when the server renames
        one. They only stop being names if they arrive over the wire."""

        policies = self.client.get(
            "/api/dashboard/agent/workflow/step-schema"
        ).json()["policies"]
        for slot, slot_payload in policies.items():
            with self.subTest(slot=slot):
                self.assertIsInstance(slot_payload["aliases"], dict)
            for option in slot_payload["options"]:
                with self.subTest(slot=slot, policy=option["type"]):
                    self.assertIn("target_key", option)
                    self.assertIn("nested_policy", option)
                    if option["requires_target"]:
                        self.assertTrue(option["target_key"])
        self.assertEqual(
            policies["on_failure"]["aliases"], dict(FAILURE_POLICY_ALIASES)
        )
        self.assertEqual(
            policies["on_success"]["aliases"], dict(SUCCESS_POLICY_ALIASES)
        )


class TemplateReadsThePublishedVocabularyTest(unittest.TestCase):
    def test_the_editor_is_driven_from_stepschema_policies(self):
        self.assertIn("stepSchema && stepSchema.policies", MARKUP)
        self.assertIn("function renderPolicyEditor(", MARKUP)
        self.assertIn("renderPolicyEditor(step, index, 'on_failure')", MARKUP)
        self.assertIn("renderPolicyEditor(step, index, 'on_success')", MARKUP)

    def test_the_old_hardcoded_lists_are_gone(self):
        """The four-choice failure list and the `retry_once` spelling it wrote
        are what this change removes; a re-appearance is the drift returning."""

        for symbol in (
            "function failureOptions(",
            "function successOptions(",
            "function setFailurePolicy(",
            "function setSuccessPolicy(",
            "'retry_once'",
        ):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, MARKUP)

    def test_a_jump_target_is_picked_from_the_workflows_own_steps(self):
        """Not free text: `_validate_policy_targets` refuses the whole
        workflow over a typo, and that refusal is not a field-level warning
        the user can shrug off."""

        self.assertIn("function policyTargetCandidates(", MARKUP)
        self.assertIn("function renderPolicyTargetPicker(", MARKUP)
        self.assertIn("setPolicyTarget(", MARKUP)

    def test_deep_nesting_is_shown_rather_than_flattened(self):
        self.assertIn("POLICY_MAX_NESTING_DEPTH", MARKUP)
        self.assertIn("p_nested_readonly", MARKUP)

    def test_the_policy_wording_exists_in_both_locale_tables(self):
        for key in (
            "p_ask:",
            "p_manual:",
            "p_retry:",
            "p_continue:",
            "p_goto_step:",
            "p_abort:",
            "p_end:",
            "p_fallback_notice:",
            "p_target_pick:",
            "p_target_required:",
            "p_no_other_steps:",
            "p_unknown_type:",
            "p_then_label:",
            "p_nested_readonly:",
        ):
            with self.subTest(key=key):
                self.assertEqual(MARKUP.count(key), 2, key)


class ThePublicationSaysWhatTheNormalizerDoesTest(unittest.TestCase):
    """The three facts that used to live only in each client, checked against
    the normalizer that actually decides them.

    `requires_target: true` said a policy needs a step id but not the key to
    put it under; nothing said a `retry` carries a whole second policy under
    `then`; and nothing said `retry_once` / `skip` / `stop` are accepted
    spellings of policies published under other names. All three are published
    now, so what is checked here is the *publication* — a rename in
    `workflow_v2` fails this test instead of shipping an editor that writes a
    key the server throws away or shows a stored value as unrecognized."""

    def setUp(self):
        self.policies = get_step_schema()["policies"]
        self.normalizers = {
            "on_failure": normalize_failure_policy,
            "on_success": normalize_success_policy,
        }

    def _minimal(self, slot: str, option: dict) -> dict:
        payload: dict = {"type": option["type"]}
        if option["requires_target"]:
            payload[option["target_key"]] = "some_step"
        return self.normalizers[slot](payload)

    def test_the_published_target_key_is_the_key_the_server_stores(self):
        seen_target_taking = False
        for slot, slot_payload in self.policies.items():
            for option in slot_payload["options"]:
                with self.subTest(slot=slot, policy=option["type"]):
                    if not option["requires_target"]:
                        # Publishing a key for a policy that takes no target
                        # would invite a client to write one the server drops.
                        self.assertIsNone(option["target_key"])
                        continue
                    seen_target_taking = True
                    self.assertTrue(option["target_key"])
                    normalized = self._minimal(slot, option)
                    self.assertEqual(
                        normalized.get(option["target_key"]), "some_step"
                    )
        self.assertTrue(
            seen_target_taking,
            "no policy declares requires_target — the target key would go "
            "untested",
        )

    def test_the_publication_names_exactly_the_policies_that_nest(self):
        for slot, slot_payload in self.policies.items():
            for option in slot_payload["options"]:
                with self.subTest(slot=slot, policy=option["type"]):
                    normalized = self._minimal(slot, option)
                    nested_keys = [
                        key
                        for key, value in normalized.items()
                        if isinstance(value, dict) and "type" in value
                    ]
                    published = option["nested_policy"]
                    if not nested_keys:
                        self.assertIsNone(published)
                        continue
                    self.assertIsNotNone(
                        published,
                        "the normalizer nests a policy here and the schema "
                        "does not say so",
                    )
                    self.assertEqual(nested_keys, [published["key"]])
                    # The slot the nested editor draws from must accept what
                    # the normalizer put there, and every one of its choices.
                    nested_normalize = self.normalizers[published["slot"]]
                    nested_normalize(normalized[published["key"]])
                    for nested_option in self.policies[published["slot"]]["options"]:
                        nested_normalize(self._minimal(published["slot"], nested_option))
                    # ...and the default is what an unset nested policy becomes.
                    self.assertEqual(
                        normalized[published["key"]]["type"], published["default"]
                    )

    def test_a_retry_nests_a_follow_up_and_the_schema_says_where(self):
        """The concrete case, named: `retry`.`then` is a failure policy. A
        silent loss of it is a schema that stops describing the one policy a
        client cannot author without help."""

        retry = next(
            option
            for option in self.policies["on_failure"]["options"]
            if option["type"] == "retry"
        )
        self.assertEqual(
            retry["nested_policy"],
            {"key": "then", "slot": "on_failure", "default": "abort"},
        )

    def test_every_published_alias_normalizes_to_the_name_it_claims(self):
        seen = 0
        for slot, slot_payload in self.policies.items():
            offered = {option["type"] for option in slot_payload["options"]}
            aliases = slot_payload["aliases"]
            self.assertIsInstance(aliases, dict)
            for alias, canonical in aliases.items():
                seen += 1
                with self.subTest(slot=slot, alias=alias):
                    self.assertIn(
                        canonical,
                        offered,
                        "an alias folding into a policy the slot does not "
                        "offer would select nothing",
                    )
                    self.assertNotIn(
                        alias,
                        offered,
                        "an alias that is also offered would be two choices "
                        "meaning the same thing",
                    )
                    option = next(
                        entry
                        for entry in slot_payload["options"]
                        if entry["type"] == canonical
                    )
                    payload: dict = {"type": alias}
                    if option["requires_target"]:
                        payload[option["target_key"]] = "some_step"
                    self.assertEqual(
                        self.normalizers[slot](payload)["type"], canonical
                    )
        self.assertEqual(
            seen,
            len(FAILURE_POLICY_ALIASES) + len(SUCCESS_POLICY_ALIASES),
            "every spelling the normalizer's alias table accepts must be "
            "published, or a client still shows it as unrecognized",
        )


class DriftGuardsFireTest(unittest.TestCase):
    """The publication is derived by interrogating the normalizer, so a shape
    it cannot derive must fail the build rather than ship a plausible guess.
    These prove the guards are reachable, not decorative."""

    def test_an_alias_the_normalizer_refuses_is_not_published(self):
        with mock.patch.dict(
            workflow_step_schema._POLICY_SLOT_ALIASES,
            {"on_failure": {"not_a_policy_at_all": "continue"}},
        ):
            with self.assertRaises(WorkflowSchemaDriftError):
                workflow_step_schema.build_policy_schema()

    def test_a_target_taking_policy_no_probe_reaches_is_not_published(self):
        """`abort` takes no target, so declaring that it does leaves the
        schema with nothing truthful to tell a client to write."""

        with mock.patch.object(
            workflow_step_schema,
            "POLICY_TYPES_REQUIRING_TARGET",
            frozenset({"abort"}),
        ):
            with self.assertRaises(WorkflowSchemaDriftError):
                workflow_step_schema.build_policy_schema()

    def test_a_nested_slot_that_matches_no_vocabulary_is_not_published(self):
        with mock.patch.dict(
            workflow_step_schema._POLICY_SLOT_TYPES,
            {"on_failure": ("retry", "abort"), "on_success": ("continue",)},
        ):
            with self.assertRaises(WorkflowSchemaDriftError):
                workflow_step_schema.build_policy_schema()


class TemplateReadsTheThreeInsteadOfNamingThemTest(unittest.TestCase):
    """The page's own constants are now a fallback for a server that predates
    the publication. Every read goes through an accessor that prefers what the
    server sent, and the old unconditional names are gone."""

    def test_the_page_no_longer_knows_the_three_by_name(self):
        for symbol in (
            "const POLICY_TARGET_KEY",
            "const POLICY_NESTED_KEY",
            "const POLICY_NESTED_SLOT",
            "const POLICY_TYPES_WITH_NESTED_POLICY",
        ):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, MARKUP)

    def test_the_page_reads_them_off_the_publication(self):
        for symbol in (
            "option.target_key",
            "option.nested_policy",
            "published.aliases",
            "function policyTargetKey(",
            "function policyNesting(",
            "function policyAliases(",
            "function policyCanonicalType(",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, MARKUP)

    def test_the_pinned_target_key_matches_what_the_server_publishes(self):
        published = get_step_schema()["policies"]
        pinned = _js_const("FALLBACK_POLICY_TARGET_KEY")
        for slot_payload in published.values():
            for option in slot_payload["options"]:
                if option["requires_target"]:
                    with self.subTest(policy=option["type"]):
                        self.assertEqual(option["target_key"], pinned)

    def test_the_pinned_nesting_matches_what_the_server_publishes(self):
        published = get_step_schema()["policies"]
        pinned = _js_object("FALLBACK_POLICY_NESTING")
        nesting_types = _js_string_set("FALLBACK_POLICY_TYPES_WITH_NESTED_POLICY")

        actual_types = set()
        for slot_payload in published.values():
            for option in slot_payload["options"]:
                nested = option["nested_policy"]
                if nested is None:
                    continue
                actual_types.add(option["type"])
                with self.subTest(policy=option["type"]):
                    self.assertEqual(nested, pinned)
        self.assertEqual(nesting_types, actual_types)

    def test_the_pinned_aliases_match_what_the_server_publishes(self):
        published = get_step_schema()["policies"]
        pinned = _js_alias_tables()
        self.assertEqual(
            pinned,
            {slot: payload["aliases"] for slot, payload in published.items()},
        )


class FallbackIsNeitherSilentNorNarrowTest(unittest.TestCase):
    """A fallback that quietly showed fewer choices than the server accepts is
    the defect being fixed, so the built-in list stays equal to the published
    one and says on screen that it is in use."""

    def test_the_fallback_offers_exactly_what_the_server_offers(self):
        fallback = _fallback_schema()
        published = get_step_schema()["policies"]
        for slot in ("on_failure", "on_success"):
            with self.subTest(slot=slot):
                self.assertEqual(
                    [option["type"] for option in fallback[slot]["options"]],
                    [option["type"] for option in published[slot]["options"]],
                )
                self.assertEqual(fallback[slot]["default"], published[slot]["default"])

    def test_the_fallback_agrees_about_which_policies_need_a_target(self):
        fallback = _fallback_schema()
        for slot, slot_payload in fallback.items():
            for option in slot_payload["options"]:
                with self.subTest(slot=slot, policy=option["type"]):
                    self.assertEqual(
                        option["requires_target"],
                        option["type"] in POLICY_TYPES_REQUIRING_TARGET,
                    )

    def test_every_fallback_option_has_wording_in_both_locales(self):
        for slot_payload in _fallback_schema().values():
            for option in slot_payload["options"]:
                with self.subTest(policy=option["type"]):
                    self.assertIsNotNone(option["label_key"])
                    self.assertEqual(MARKUP.count(f"{option['label_key']}:"), 2)

    def test_using_the_fallback_is_announced_on_screen(self):
        self.assertIn("policySlotIsFallback", MARKUP)
        self.assertIn("p_fallback_notice", MARKUP)


if __name__ == "__main__":
    unittest.main()
