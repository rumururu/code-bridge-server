"""The phone's pinned copies must not drift from what the server accepts.

`lib/models/workflow_step_schema.dart` carries a handful of literals used only
while the published schema has not loaded: the common/legacy field sets, the
failure and success policy lists, and — the reason this file exists — a mirror
of `BROWSER_ACTION_VOCABULARY` reduced to the action types and the exact keys
`browser_action_adapter` reads for each.

Every one of those is a copy, and a copy is free to drift. It already had:
the editor's own list offered `duration_ms` and `state` on a wait, and
`label`/`output` on an extract. The adapter reads `timeout_ms` for the first
and `name`/`selector`/`source`/`pattern`/`attribute`/`max_chars` for the
second, so a wait authored on the phone ran the default timeout and an extract
saved nothing at all — no `{{name}}` was ever bound, which is the same as not
having extracted.

So these tests read the Dart literals out of the source file and compare them
against the server's own definitions. Add a policy or change a parameter key
on the server and this file names the client list that is now behind.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.browser_action_adapter import BROWSER_ACTION_VOCABULARY  # noqa: E402
from code_bridge_core.workflow_step_schema import (  # noqa: E402
    ACTION_VOCABULARY_BROWSER,
    build_action_vocabularies,
    get_step_schema,
)
from code_bridge_core.workflow_v2 import (  # noqa: E402
    COMMON_STEP_FIELDS,
    FAILURE_POLICY_TYPES,
    POLICY_TYPES_REQUIRING_TARGET,
    SUCCESS_POLICY_TYPES,
    _UNRESTRICTED_LEGACY_FIELDS,
)

CLIENT_MODEL = SERVER_DIR.parent / "lib" / "models" / "workflow_step_schema.dart"

#: The Flutter app lives in the same repository as this server, but the server
#: is also distributed on its own. Skipping — loudly, by name — beats a test
#: that fails for everyone who only has the server half.
HAS_CLIENT = CLIENT_MODEL.is_file()

SOURCE = CLIENT_MODEL.read_text(encoding="utf-8") if HAS_CLIENT else ""


def _dart_literal(name: str) -> str:
    """The bracketed body of a top-level `const ... name = [...]` / `{...}`."""
    match = re.search(rf"const\s+[^\n=]*\b{name}\s*=\s*([\[{{])", SOURCE)
    if match is None:  # pragma: no cover - the assertions below name it
        raise AssertionError(f"{name} is not declared in {CLIENT_MODEL.name}")
    opening = match.group(1)
    closing = "]" if opening == "[" else "}"
    start = match.end() - 1
    depth = 0
    for index in range(start, len(SOURCE)):
        char = SOURCE[index]
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                assert char == closing
                return SOURCE[start + 1 : index]
    raise AssertionError(f"{name} literal is not closed in {CLIENT_MODEL.name}")


def _dart_strings(name: str) -> list[str]:
    return re.findall(r"'([^']*)'", _dart_literal(name))


def _dart_string_const(name: str) -> str:
    """The value of a top-level `const String name = '...';`."""
    match = re.search(rf"const\s+String\s+{name}\s*=\s*'([^']*)'", SOURCE)
    if match is None:  # pragma: no cover - the assertions below name it
        raise AssertionError(f"{name} is not declared in {CLIENT_MODEL.name}")
    return match.group(1)


def _dart_named_args(name: str) -> dict[str, str]:
    """The `name: 'value'` arguments of a top-level const constructor call."""
    match = re.search(rf"const\s+[^\n=]*\b{name}\s*=\s*\w+\(", SOURCE)
    if match is None:  # pragma: no cover - the assertions below name it
        raise AssertionError(f"{name} is not declared in {CLIENT_MODEL.name}")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(SOURCE)):
        if SOURCE[index] == "(":
            depth += 1
        elif SOURCE[index] == ")":
            depth -= 1
            if depth == 0:
                body = SOURCE[start + 1 : index]
                return dict(re.findall(r"(\w+):\s*'([^']*)'", body))
    raise AssertionError(f"{name} call is not closed in {CLIENT_MODEL.name}")


def _dart_alias_tables() -> dict[str, dict[str, str]]:
    """`kFallbackPolicyAliases` as {slot: {alias: canonical}}."""
    body = _dart_literal("kFallbackPolicyAliases")
    out: dict[str, dict[str, str]] = {}
    for slot_match in re.finditer(r"'(on_failure|on_success)':\s*\{", body):
        slot = slot_match.group(1)
        depth = 0
        start = slot_match.end() - 1
        for index in range(start, len(body)):
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
                if depth == 0:
                    inner = body[start + 1 : index]
                    break
        out[slot] = dict(re.findall(r"'([^']*)':\s*'([^']*)'", inner))
    return out


def _client_browser_vocabulary() -> dict[str, list[str]]:
    """`kFallbackBrowserActionVocabulary` as {action type: [param keys]}."""
    body = _dart_literal("kFallbackBrowserActionVocabulary")
    result: dict[str, list[str]] = {}
    entries = re.split(r"WorkflowActionTypeSchema\(", body)[1:]
    for entry in entries:
        type_match = re.search(r"type:\s*'([^']+)'", entry)
        assert type_match is not None, f"an entry has no type: {entry[:80]}"
        params = re.findall(r"WorkflowActionParamSchema\(key:\s*'([^']+)'", entry)
        result[type_match.group(1)] = params
    return result


@unittest.skipUnless(HAS_CLIENT, f"no Flutter client at {CLIENT_MODEL}")
class PolicyVocabularyTest(unittest.TestCase):
    def test_failure_policies_match(self) -> None:
        self.assertEqual(
            _dart_strings("kFallbackFailurePolicyTypes"),
            list(FAILURE_POLICY_TYPES),
        )

    def test_success_policies_match(self) -> None:
        self.assertEqual(
            _dart_strings("kFallbackSuccessPolicyTypes"),
            list(SUCCESS_POLICY_TYPES),
        )

    def test_target_taking_policies_match(self) -> None:
        self.assertEqual(
            set(_dart_strings("kFallbackPolicyTypesRequiringTarget")),
            set(POLICY_TYPES_REQUIRING_TARGET),
        )


@unittest.skipUnless(HAS_CLIENT, f"no Flutter client at {CLIENT_MODEL}")
class TheThreeThePhoneUsedToKnowByNameTest(unittest.TestCase):
    """The wire key a jump target goes under, the follow-up policy a `retry`
    nests, and the older spellings the server accepts but does not offer.

    All three are published now (`option.target_key`, `option.nested_policy`,
    `slot.aliases`), so the Dart constants are a fallback for an older server
    and nothing else. They are compared against the *publication* here rather
    than against the raw constants, because the publication is what the phone
    reads when it can — a pin that disagrees with it would only ever surface
    on the offline path, which is exactly where nobody looks."""

    def setUp(self) -> None:
        self.policies = get_step_schema()["policies"]

    def test_the_pinned_target_key_is_the_published_one(self) -> None:
        pinned = _dart_string_const("kFallbackPolicyTargetKey")
        published = {
            option["target_key"]
            for slot in self.policies.values()
            for option in slot["options"]
            if option["requires_target"]
        }
        self.assertEqual(published, {pinned})

    def test_the_pinned_nesting_is_the_published_one(self) -> None:
        pinned = _dart_named_args("kFallbackPolicyNesting")
        pinned_types = set(_dart_strings("kFallbackPolicyTypesWithNestedPolicy"))

        published_types = set()
        for slot in self.policies.values():
            for option in slot["options"]:
                nested = option["nested_policy"]
                if nested is None:
                    continue
                published_types.add(option["type"])
                with self.subTest(policy=option["type"]):
                    self.assertEqual(
                        pinned,
                        {
                            "key": nested["key"],
                            "slot": nested["slot"],
                            "defaultType": nested["default"],
                        },
                    )
        self.assertEqual(pinned_types, published_types)

    def test_the_pinned_aliases_are_the_published_ones(self) -> None:
        self.assertEqual(
            _dart_alias_tables(),
            {slot: payload["aliases"] for slot, payload in self.policies.items()},
        )

    def test_the_publication_carries_all_three(self) -> None:
        """A vocabulary the client cannot fetch is a vocabulary it must pin —
        so the point of the pins shrinking is that the wire carries them."""

        for slot, payload in self.policies.items():
            with self.subTest(slot=slot):
                self.assertIsInstance(payload["aliases"], dict)
            for option in payload["options"]:
                with self.subTest(slot=slot, policy=option["type"]):
                    self.assertIn("target_key", option)
                    self.assertIn("nested_policy", option)


@unittest.skipUnless(HAS_CLIENT, f"no Flutter client at {CLIENT_MODEL}")
class FieldSetTest(unittest.TestCase):
    def test_common_fields_match(self) -> None:
        self.assertEqual(
            set(_dart_strings("kFallbackCommonStepFields")),
            set(COMMON_STEP_FIELDS),
        )

    def test_legacy_fields_match(self) -> None:
        self.assertEqual(
            set(_dart_strings("kFallbackLegacyStepFields")),
            set(_UNRESTRICTED_LEGACY_FIELDS),
        )


@unittest.skipUnless(HAS_CLIENT, f"no Flutter client at {CLIENT_MODEL}")
class BrowserActionVocabularyTest(unittest.TestCase):
    """The one that was actually wrong, and the reason for the rest."""

    def setUp(self) -> None:
        self.client = _client_browser_vocabulary()
        self.server = {
            action.type: list(action.param_keys)
            for action in BROWSER_ACTION_VOCABULARY
        }

    def test_the_same_actions_are_offered(self) -> None:
        self.assertEqual(list(self.client), list(self.server))

    def test_every_action_offers_the_keys_the_adapter_reads(self) -> None:
        for action_type, server_keys in self.server.items():
            with self.subTest(action=action_type):
                self.assertEqual(
                    self.client.get(action_type),
                    server_keys,
                    "the offline editor would offer keys the adapter does not "
                    "read (or miss ones it does)",
                )

    def test_no_client_key_is_one_nothing_reads(self) -> None:
        """The exact failure this closes: `duration_ms`, `state`, `label`,
        `output` were offered and read by nothing."""
        for action_type, keys in self.client.items():
            for key in keys:
                with self.subTest(action=action_type, key=key):
                    self.assertIn(key, self.server.get(action_type, []))


@unittest.skipUnless(HAS_CLIENT, f"no Flutter client at {CLIENT_MODEL}")
class ThePublishedSchemaCarriesItTest(unittest.TestCase):
    """A vocabulary the client cannot fetch is a vocabulary it must pin."""

    def test_the_schema_publishes_the_browser_table(self) -> None:
        schema = get_step_schema()
        published = schema["action_vocabularies"][ACTION_VOCABULARY_BROWSER]
        self.assertEqual(
            [entry["type"] for entry in published],
            [action.type for action in BROWSER_ACTION_VOCABULARY],
        )

    def test_every_type_that_offers_actions_points_at_a_published_table(
        self,
    ) -> None:
        schema = get_step_schema()
        vocabularies = schema["action_vocabularies"]
        for entry in schema["types"]:
            offers_actions = any(
                field["key"] == "actions" for field in entry["fields"]
            )
            with self.subTest(step_type=entry["type"]):
                if offers_actions:
                    self.assertIn(entry["action_vocabulary"], vocabularies)
                else:
                    self.assertIsNone(entry["action_vocabulary"])

    def test_no_published_vocabulary_is_left_unpointed_at(self) -> None:
        """A table nobody points at is one no editor can ever draw."""
        schema = get_step_schema()
        pointed_at = {
            entry["action_vocabulary"]
            for entry in schema["types"]
            if entry["action_vocabulary"] is not None
        }
        self.assertEqual(pointed_at, set(build_action_vocabularies()))


if __name__ == "__main__":
    unittest.main()
