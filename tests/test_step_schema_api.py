"""The published step schema is the only thing standing between a client and
four hand-maintained copies of the same field list.

Before this, every authoring surface (phone, dashboard, Configurator) carried
its own idea of which fields a workflow step type has. They drifted: the
phone never learned about `shell` or `notify` and showed `memory_read` /
`memory_write` / `success_criteria` on step types that ignore them, and the
Configurator's hand-written prompt schema offered fields the normalizer would
silently reject (a `device_id` on an `llm` step, for example) — so a draft
the model wrote in good faith could lose a field between the conversation and
the committed agent, with no error anywhere.

`agent.workflow_step_schema` closes this by deriving everything from
`agent.workflow_v2.WORKFLOW_STEP_SCHEMA`, the same definition
`normalize_workflow_step` enforces. These tests pin the two guarantees that
make that safe to trust:

1. What the schema advertises for a type is exactly what the normalizer
   accepts for that type — nothing more (a field it does not list is
   rejected), nothing less (a field it does list is accepted). A regression
   here means an authoring surface offers a field the server throws away, or
   is missing one the server would gladly take.
2. Every field and step type has real label/help text in both languages the
   app ships in today. A field with no `ko` translation would surface as
   English on an otherwise Korean screen — exactly defect #2 this whole
   effort exists to close — and that must fail a test, not a screenshot
   review.

It also pins that `agent.configurator`'s prompt is generated from the same
definition rather than retyped, by checking every step type actually appears
in the rendered prompt.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent.configurator import build_configurator_system_prompt  # noqa: E402
from agent.workflow_v2 import (  # noqa: E402
    ALLOWED_STEP_TYPES,
    WorkflowNormalizationError,
    _UNRESTRICTED_LEGACY_FIELDS,
    normalize_workflow_step,
)
from agent.workflow_step_schema import (  # noqa: E402
    ALL_KINDS,
    KIND_ACTION_LIST,
    KIND_SELECT,
    KIND_STRING_LIST,
    WorkflowSchemaDriftError,
    base_field_types,
    build_step_schema,
    get_step_schema,
)


def _dummy_value_for(kind: str, key: str) -> object:
    """A value normalize_workflow_step will accept for a field of this kind.

    The normalizer does not validate select options against a live catalog
    (script_id/device_id existence is checked at execution time, not
    authoring time — see workflow_v2.py), so any non-empty string clears it.
    """
    if kind == KIND_STRING_LIST:
        return ["arg1", "arg2"]
    if kind == KIND_ACTION_LIST:
        return [{"type": "wait"}]
    if key == "notify.level":
        return "warning"
    if kind == KIND_SELECT:
        return "dummy_value"
    return f"dummy value for {key}"


def _build_valid_step(step_type: str, type_entry: dict) -> dict:
    """A step of ``step_type`` carrying every field the schema advertises for it."""

    step: dict = {"id": f"step_{step_type}", "type": step_type, "name": "Test step"}
    notify_payload: dict = {}
    for field in type_entry["fields"]:
        key = field["key"]
        value = _dummy_value_for(field["kind"], key)
        if key.startswith("notify."):
            notify_payload[key.split(".", 1)[1]] = value
        else:
            step[key] = value
    if notify_payload:
        step["notify"] = notify_payload
    return step


class SchemaCoversEveryTypeTest(unittest.TestCase):
    def test_every_allowed_step_type_appears(self) -> None:
        schema = get_step_schema()
        published_types = {entry["type"] for entry in schema["types"]}
        self.assertEqual(published_types, set(ALLOWED_STEP_TYPES))

    def test_kinds_are_the_small_closed_set(self) -> None:
        schema = get_step_schema()
        for entry in schema["types"]:
            for field in entry["fields"]:
                self.assertIn(field["kind"], ALL_KINDS)


class AdvertisedFieldsAreAcceptedTest(unittest.TestCase):
    """Everything the schema lists for a type, the normalizer takes."""

    def test_every_type_full_of_advertised_fields_normalizes(self) -> None:
        schema = get_step_schema()
        for entry in schema["types"]:
            step_type = entry["type"]
            with self.subTest(step_type=step_type):
                step = _build_valid_step(step_type, entry)
                normalized = normalize_workflow_step(step, index=1)
                self.assertEqual(normalized["type"], step_type)


class UnadvertisedFieldsAreRejectedTest(unittest.TestCase):
    """The anti-drift half: a field the schema does not list for a type is
    refused by the normalizer for that type.

    The legacy-tolerated carve-out (`tool_hint`, `success_criteria`,
    `actions` — see `_UNRESTRICTED_LEGACY_FIELDS` in workflow_v2.py) is
    excluded: those three are deliberately accepted on every type for
    backward compatibility with agents saved before field-scoping existed,
    so they would not demonstrate a real drift.
    """

    def test_a_field_foreign_to_the_type_is_rejected(self) -> None:
        field_types = base_field_types()
        all_base_fields = set(field_types)
        checked_any = False

        for step_type in ALLOWED_STEP_TYPES:
            own_fields = {
                field
                for field, types_for_field in field_types.items()
                if step_type in types_for_field
            }
            candidates = all_base_fields - own_fields - _UNRESTRICTED_LEGACY_FIELDS - {"notify"}
            # `notify` itself is excluded from the candidate pool only when it
            # IS the type's own field (handled by `own_fields` above); a type
            # that does not own it is still a valid candidate, so add it back
            # when foreign.
            if "notify" not in own_fields:
                candidates.add("notify")

            if not candidates:
                continue
            foreign_field = sorted(candidates)[0]
            checked_any = True

            bad_step = {"id": "s", "type": step_type, "name": "Test"}
            if step_type == "shell":
                bad_step["script_id"] = "script_1"
            if foreign_field == "notify":
                bad_step["notify"] = {"title": "x"}
            else:
                bad_step[foreign_field] = "x"

            with self.subTest(step_type=step_type, foreign_field=foreign_field):
                with self.assertRaisesRegex(
                    WorkflowNormalizationError,
                    rf"'{foreign_field}' is not a field of a {step_type} step",
                ):
                    normalize_workflow_step(bad_step, index=1)

        self.assertTrue(checked_any, "expected at least one type/foreign-field pair to check")


class TranslationsAreCompleteTest(unittest.TestCase):
    """A field or type with an English label but no Korean one would surface
    as English on an otherwise Korean screen — defect #2 this schema exists
    to close. Fail here, not in a screenshot review."""

    def test_every_type_has_en_and_ko_label_and_help(self) -> None:
        schema = get_step_schema()
        for entry in schema["types"]:
            with self.subTest(step_type=entry["type"]):
                for locale in ("en", "ko"):
                    self.assertTrue((entry["label"].get(locale) or "").strip())
                    self.assertTrue((entry["help"].get(locale) or "").strip())

    def test_every_field_has_en_and_ko_label_and_help(self) -> None:
        schema = get_step_schema()
        for entry in schema["types"]:
            for field in entry["fields"]:
                with self.subTest(step_type=entry["type"], field=field["key"]):
                    for locale in ("en", "ko"):
                        self.assertTrue((field["label"].get(locale) or "").strip())
                        self.assertTrue((field["help"].get(locale) or "").strip())


class DriftGuardTest(unittest.TestCase):
    """Prove the guard actually fires, rather than merely asserting it exists."""

    def test_an_unmetadata_d_field_added_to_the_schema_fails_the_build(self) -> None:
        fake_schema = {"shell": frozenset({"script_id", "totally_new_field"})}
        with self.assertRaises(WorkflowSchemaDriftError):
            build_step_schema(step_schema=fake_schema, step_types={"shell"})

    def test_an_unmetadata_d_type_fails_the_build(self) -> None:
        fake_schema = {"a_brand_new_step_type": frozenset()}
        with self.assertRaises(WorkflowSchemaDriftError):
            build_step_schema(step_schema=fake_schema, step_types={"a_brand_new_step_type"})

    def test_the_real_definition_builds_clean(self) -> None:
        # No exception: every field and type currently in WORKFLOW_STEP_SCHEMA
        # / ALLOWED_STEP_TYPES has metadata attached. If this ever raises, a
        # field landed in workflow_v2 without its schema entry here.
        build_step_schema()


class ConfiguratorPromptIsGeneratedTest(unittest.TestCase):
    """Phase 2.3: the Configurator's schema block is generated, not retyped."""

    def test_every_step_type_appears_in_the_generated_prompt(self) -> None:
        prompt = build_configurator_system_prompt()
        for step_type in ALLOWED_STEP_TYPES:
            with self.subTest(step_type=step_type):
                self.assertIn(f'"{step_type}"', prompt)

    def test_the_placeholder_is_not_left_unreplaced(self) -> None:
        prompt = build_configurator_system_prompt()
        self.assertNotIn("WORKFLOW_STEP_SCHEMA_BLOCK", prompt)

    def test_a_type_scoped_field_carries_its_scope_note(self) -> None:
        # script_id is shell-only; the generated line should say so, the same
        # information normalize_workflow_step enforces.
        prompt = build_configurator_system_prompt()
        self.assertIn("script_id", prompt)
        self.assertIn("shell steps only", prompt)


if __name__ == "__main__":
    unittest.main()
