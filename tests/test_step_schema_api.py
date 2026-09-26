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

`code_bridge_core.workflow_step_schema` closes this by deriving everything from
`code_bridge_core.workflow_v2.WORKFLOW_STEP_SCHEMA`, the same definition
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

It also pins that `code_bridge_core.configurator`'s prompt is generated from the same
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

from code_bridge_core.configurator import build_configurator_system_prompt  # noqa: E402
from code_bridge_core.workflow_v2 import (  # noqa: E402
    ALLOWED_STEP_TYPES,
    WORKFLOW_STEP_SCHEMA,
    WorkflowNormalizationError,
    _UNRESTRICTED_LEGACY_FIELDS,
    normalize_workflow_step,
)
from code_bridge_core.workflow_step_schema import (  # noqa: E402
    ALL_KINDS,
    ICON_TOKENS,
    _ACTION_VOCABULARY_BY_STEP_TYPE,
    _TypeSpec,
    build_action_vocabularies,
    KIND_ACTION_LIST,
    KIND_SELECT,
    KIND_BRANCH_LIST,
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
    if kind == KIND_BRANCH_LIST:
        # One default branch (no `when`). Its `target_step_id` is checked for
        # existence by `normalize_workflow`, which needs the whole list;
        # `normalize_workflow_step` alone checks only the shape, which is what
        # this helper feeds.
        return [{"label": "otherwise", "target_step_id": "step_elsewhere"}]
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


class EveryTypeIsDrawableTest(unittest.TestCase):
    """A published type a client cannot draw is a grey unnamed box on screen.

    Label and help say what a step *means*; `icon` and `accent` say how to
    draw it, and they are published for the same reason the field list is:
    the canvas that renders these flows holds no step type names of its own
    (agent-flow-core `frontend/packages/flow-canvas/src/stepTypes.ts`), so a
    type that ships without them draws as an anonymous box until somebody
    releases a new client. These tests are the reason that cannot happen
    quietly.
    """

    def test_every_type_names_an_icon_token_from_the_closed_set(self) -> None:
        schema = get_step_schema()
        for entry in schema["types"]:
            with self.subTest(step_type=entry["type"]):
                self.assertIn(entry["icon"], ICON_TOKENS)

    def test_every_type_carries_a_six_digit_hex_accent(self) -> None:
        # Six digits specifically: clients build the header tint by appending
        # two alpha digits to this string (flow-canvas `StepNode.tsx:148`),
        # which only yields a valid colour from a 6-digit hex.
        schema = get_step_schema()
        for entry in schema["types"]:
            with self.subTest(step_type=entry["type"]):
                self.assertRegex(entry["accent"], r"^#[0-9A-Fa-f]{6}$")

    def test_published_tokens_are_all_in_the_published_vocabulary(self) -> None:
        # The canvas asserts it can draw every entry of `icon_tokens`; that
        # assertion is only worth anything if the types stay inside it.
        schema = get_step_schema()
        published = set(schema["icon_tokens"])
        self.assertEqual(published, set(ICON_TOKENS))
        for entry in schema["types"]:
            with self.subTest(step_type=entry["type"]):
                self.assertIn(entry["icon"], published)

    def test_the_four_device_spellings_look_alike(self) -> None:
        # `app_action` / `android_action` / `mobile_action` / `device_action`
        # have identical field sets and one dispatch
        # (`task_orchestrator._is_app_action_workflow_type`), so drawing them
        # four different ways would invent a distinction the runtime does not
        # make.
        by_type = {entry["type"]: entry for entry in get_step_schema()["types"]}
        looks = {
            (by_type[name]["icon"], by_type[name]["accent"])
            for name in ("app_action", "android_action", "mobile_action", "device_action")
        }
        self.assertEqual(len(looks), 1)

    def test_v1_keys_are_untouched(self) -> None:
        # icon/accent were *added*. The dashboard and the phone read this
        # payload today and must keep reading it unchanged.
        for entry in get_step_schema()["types"]:
            with self.subTest(step_type=entry["type"]):
                self.assertLessEqual(
                    {"type", "label", "help", "action_vocabulary", "fields"},
                    set(entry),
                )


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

    def test_a_type_offering_actions_without_a_vocabulary_fails_the_build(
        self,
    ) -> None:
        # A step type whose `actions` point at no executor table would draw an
        # action editor with nothing to put in it — the state the phone was
        # already in, where it filled the gap with a list of its own that had
        # drifted from what the adapter reads.
        fake_schema = {"llm": frozenset({"actions"})}
        with self.assertRaises(WorkflowSchemaDriftError):
            build_step_schema(step_schema=fake_schema, step_types={"llm"})

    def test_no_step_type_maps_to_a_vocabulary_nothing_publishes(self) -> None:
        published = set(build_action_vocabularies())
        for step_type, name in _ACTION_VOCABULARY_BY_STEP_TYPE.items():
            with self.subTest(step_type=step_type):
                self.assertIn(name, published)
                # And the mapping is not stale: the type still offers actions.
                self.assertIn("actions", WORKFLOW_STEP_SCHEMA.get(step_type, set()))

    def test_a_type_with_no_icon_fails_before_it_can_be_published(self) -> None:
        # The point of making icon/accent required on `_TypeSpec`: a 13th step
        # type cannot be added here without deciding how it is drawn. This
        # fires at construction, so it fires at import — long before a client
        # sees an undrawable type.
        with self.assertRaisesRegex(WorkflowSchemaDriftError, "no icon token"):
            _TypeSpec(
                label={"en": "New thing", "ko": "새 스텝"},
                help={"en": "help", "ko": "도움말"},
                accent="#123456",
            )

    def test_an_icon_token_outside_the_closed_set_fails(self) -> None:
        # Open tokens would mean clients quietly fall back to a neutral dot
        # for tokens they have no drawing for, which costs the same as not
        # publishing an icon at all.
        with self.assertRaisesRegex(WorkflowSchemaDriftError, "unknown icon token"):
            _TypeSpec(
                label={"en": "New thing", "ko": "새 스텝"},
                help={"en": "help", "ko": "도움말"},
                icon="unicorn",
                accent="#123456",
            )

    def test_a_type_with_no_accent_fails(self) -> None:
        with self.assertRaisesRegex(WorkflowSchemaDriftError, "no accent"):
            _TypeSpec(
                label={"en": "New thing", "ko": "새 스텝"},
                help={"en": "help", "ko": "도움말"},
                icon="tool",
            )

    def test_an_accent_that_is_not_six_digit_hex_fails(self) -> None:
        for bad in ("#abc", "red", "rgb(1,2,3)", "#12345", "#1234567"):
            with self.subTest(accent=bad):
                with self.assertRaisesRegex(WorkflowSchemaDriftError, "6-digit hex"):
                    _TypeSpec(
                        label={"en": "New thing", "ko": "새 스텝"},
                        help={"en": "help", "ko": "도움말"},
                        icon="tool",
                        accent=bad,
                    )

    def test_the_real_definition_builds_clean(self) -> None:
        # No exception: every field and type currently in WORKFLOW_STEP_SCHEMA
        # / ALLOWED_STEP_TYPES has metadata attached. If this ever raises, a
        # field landed in workflow_v2 without its schema entry here.
        build_step_schema()


class McpToolServerPickerTest(unittest.TestCase):
    """T-I2-06: `mcp_tool`'s server reference is picked, not typed.

    `tool_hint` on `mcp_tool` is the server the runtime dereferences
    (`task_orchestrator._mcp_step_server_id`) and parks without — the one
    hard reference the schema used to publish as free text. On every other
    type it stays the soft hint it always was.
    """

    def test_mcp_tool_tool_hint_is_a_select_over_the_gate_catalog(self) -> None:
        by_type = {entry["type"]: entry for entry in get_step_schema()["types"]}
        field = {f["key"]: f for f in by_type["mcp_tool"]["fields"]}["tool_hint"]
        self.assertEqual(field["kind"], KIND_SELECT)
        self.assertEqual(field["options_source"], "mcp-servers")
        # The runtime parks a step without one, so the schema says so.
        self.assertTrue(field["required"])

    def test_llm_tool_hint_stays_free_text(self) -> None:
        # On `llm` the field is a hint the model may ignore; builtin runtime
        # ids ("playwright") are legitimate there and are not MCP servers.
        by_type = {entry["type"]: entry for entry in get_step_schema()["types"]}
        field = {f["key"]: f for f in by_type["llm"]["fields"]}["tool_hint"]
        self.assertEqual(field["kind"], "text")
        self.assertIsNone(field["options_source"])

    def test_the_option_source_names_real_endpoints(self) -> None:
        sources = get_step_schema()["option_sources"]
        self.assertIn("mcp-servers", sources)
        self.assertEqual(
            sources["mcp-servers"]["endpoint"], "/api/system/mcp-servers/detected"
        )
        self.assertEqual(
            sources["mcp-servers"]["dashboard_endpoint"],
            "/api/dashboard/agent/mcp-servers",
        )

    def test_the_catalog_is_the_gate_s_own_set(self) -> None:
        # The picker and the runtime gate must consult the same set — a
        # dropdown offering a server the gate then parks on is worse than the
        # free-text box it replaced. Both read `_merged_mcp_servers` through
        # the same launchable filter; this pins the parity
        # function-to-function.
        from agent.capability_registry import (
            detected_mcp_server_configs,
            detected_mcp_server_names,
        )

        self.assertEqual(
            sorted(row["name"] for row in detected_mcp_server_names()),
            sorted(detected_mcp_server_configs()),
        )


class TypeDisplayOrderTest(unittest.TestCase):
    def test_types_are_published_cheapest_first_not_alphabetically(self) -> None:
        # The order is rule 2c as a list: shell (no tokens) first, judgement
        # second, the default way of reporting third. Aliases fall to the
        # tail, where alias-folding pickers never show them anyway.
        published = [entry["type"] for entry in get_step_schema()["types"]]
        self.assertEqual(published[:4], ["shell", "llm", "notify", "condition"])
        # Unlisted types append after everything listed — presence is
        # `ALLOWED_STEP_TYPES`'s decision alone, position is all this tuple
        # ever changes.
        self.assertEqual(set(published), set(ALLOWED_STEP_TYPES))
        self.assertEqual(
            published[-3:], ["android_action", "device_action", "mobile_action"]
        )


class StepTypeAliasTest(unittest.TestCase):
    """The four device spellings are one step, and the payload says so."""

    def test_aliases_point_the_three_spellings_at_app_action(self) -> None:
        by_type = {entry["type"]: entry for entry in get_step_schema()["types"]}
        self.assertEqual(
            {t: e["alias_of"] for t, e in by_type.items() if e["alias_of"]},
            {
                "android_action": "app_action",
                "mobile_action": "app_action",
                "device_action": "app_action",
            },
        )
        self.assertIsNone(by_type["app_action"]["alias_of"])

    def test_alias_set_matches_the_runtime_dispatch(self) -> None:
        # `_is_app_action_workflow_type` is the dispatch the aliases claim to
        # share. If a fifth spelling appears there without an alias entry (or
        # vice versa), the picker and the runtime disagree about what is one
        # step.
        from agent.task_orchestrator import _is_app_action_workflow_type
        from code_bridge_core.workflow_step_schema import STEP_TYPE_ALIASES

        dispatched = {
            t for t in ALLOWED_STEP_TYPES if _is_app_action_workflow_type(t)
        }
        self.assertEqual(
            dispatched, set(STEP_TYPE_ALIASES) | set(STEP_TYPE_ALIASES.values())
        )

    def test_an_alias_whose_fields_drift_fails_the_build(self) -> None:
        # Guard of the guard: give `android_action` a field `app_action` does
        # not have and the alias claim must refuse to publish.
        drifted = {
            t: (fields | {"instruction"} if t == "android_action" else fields)
            for t, fields in WORKFLOW_STEP_SCHEMA.items()
        }
        with self.assertRaisesRegex(WorkflowSchemaDriftError, "field sets differ"):
            build_step_schema(drifted)


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
