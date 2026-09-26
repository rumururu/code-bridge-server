"""The Configurator is told branching exists, in the executor's own words (T-H-13).

The rule this rests on is one this repo has already paid to learn: **the
executor is the source of truth for authoring vocabulary.** When the browser
action table was missing from this prompt, every browser step the model wrote
was a placeholder `navigate` — not because the model was careless, but because
it cannot use what it was never told exists
(`browser_action_adapter.py:629-638`). Branching arrived the same way: the
runner routes on `branches`, the schema publishes them, the canvas draws them,
and the prompt said nothing.

Two properties are pinned here, and they are the ticket's acceptance criteria:

* the block is **generated from `workflow_step_schema`**, so every step type,
  field and operator it names is one the server accepts. A hand-kept list is
  the drift this project has paid for repeatedly, and the generator refuses to
  render at all rather than describe a schema that has moved underneath it;
* the block does not restate the branch **shape**. That line is generated once,
  in `_workflow_step_schema_block` (T-H-04), and a second copy here would be
  the same drift wearing a different hat.

The substance beyond the field list is the judgement no schema can hold: when a
condition step is the *wrong* answer. Eleven operators compare two values; they
cannot read a page and decide. A model shown only the branching construct will
reach for it anyway and invent a value to compare, so the block names the real
alternative — a judgement step with success criteria and a routing failure
policy — and the names in that advice are read out of the same schema.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from code_bridge_core import configurator  # noqa: E402
from code_bridge_core.configurator import (  # noqa: E402
    _CONDITION_BRANCH_VOCABULARY_MARKER,
    _condition_branch_vocabulary_block,
    _workflow_step_schema_block,
    build_configurator_system_prompt,
)
from code_bridge_core.workflow_step_schema import (  # noqa: E402
    KIND_BRANCH_LIST,
    OPERATOR_ARITY_UNARY,
    WorkflowSchemaDriftError,
    get_step_schema,
)
from code_bridge_core.workflow_v2 import (  # noqa: E402
    CONDITION_OPERATORS,
    CONDITION_UNARY_OPERATORS,
    FAILURE_POLICY_TYPES,
)


class TheBlockReachesThePromptTest(unittest.TestCase):
    def test_the_marker_is_filled(self) -> None:
        prompt = build_configurator_system_prompt()

        self.assertNotIn(_CONDITION_BRANCH_VOCABULARY_MARKER, prompt)
        self.assertIn(_condition_branch_vocabulary_block(), prompt)

    def test_no_vocabulary_marker_is_left_unfilled(self) -> None:
        """A marker that survives into the prompt is a block the model never
        reads — the exact failure this ticket is fixing, one substitution
        later."""

        prompt = build_configurator_system_prompt()

        for name, value in vars(configurator).items():
            if name.endswith("_MARKER") and isinstance(value, str):
                with self.subTest(marker=name):
                    self.assertNotIn(value, prompt)

    def test_it_appears_once(self) -> None:
        prompt = build_configurator_system_prompt()
        block = _condition_branch_vocabulary_block()

        self.assertEqual(prompt.count(block), 1)


class EveryNameComesFromTheSchemaTest(unittest.TestCase):
    """Read the block against `workflow_step_schema`, term by term."""

    def setUp(self) -> None:
        self.block = _condition_branch_vocabulary_block()
        self.schema = get_step_schema()
        self.types = {entry["type"]: entry for entry in self.schema["types"]}

    def test_the_branching_step_type_is_the_one_that_carries_the_field(self) -> None:
        carriers = [
            (entry["type"], field["key"])
            for entry in self.schema["types"]
            for field in entry["fields"]
            if field["kind"] == KIND_BRANCH_LIST
        ]

        self.assertEqual(len(carriers), 1)
        step_type, field_key = carriers[0]
        self.assertIn(f"`{step_type}` step", self.block)
        self.assertIn(f"`{field_key}`", self.block)

    def test_the_step_types_help_is_quoted_verbatim(self) -> None:
        carrier = next(
            entry
            for entry in self.schema["types"]
            if any(field["kind"] == KIND_BRANCH_LIST for field in entry["fields"])
        )

        self.assertIn(carrier["help"]["en"], self.block)

    def test_the_fields_published_help_is_quoted_verbatim(self) -> None:
        """That help is where the model learns the consequence of leaving out a
        fallback — the step *fails*, it does not carry on. Quoting it rather
        than paraphrasing means the prompt and the runtime cannot disagree."""

        field = next(
            field
            for entry in self.schema["types"]
            for field in entry["fields"]
            if field["kind"] == KIND_BRANCH_LIST
        )

        self.assertIn(field["help"]["en"], self.block)

    def test_every_operator_is_listed_with_its_published_label(self) -> None:
        for operator in self.schema["condition_operators"]:
            with self.subTest(op=operator["op"]):
                self.assertIn(operator["op"], self.block)
                self.assertIn(operator["label"]["en"], self.block)

    def test_the_operator_count_is_stated_from_the_schema(self) -> None:
        self.assertEqual(
            len(self.schema["condition_operators"]), len(CONDITION_OPERATORS)
        )
        self.assertIn(f"The {len(CONDITION_OPERATORS)} operators", self.block)

    def test_the_unary_operators_are_marked_as_taking_no_right_operand(self) -> None:
        """A model that writes `right` on `is_empty` is refused by the
        normalizer. Being told costs one clause; finding out costs a turn."""

        for operator in self.schema["condition_operators"]:
            line = next(
                line
                for line in self.block.splitlines()
                if line.strip().startswith(operator["op"] + " ")
            )
            unary = operator["arity"] == OPERATOR_ARITY_UNARY
            with self.subTest(op=operator["op"]):
                self.assertEqual(unary, operator["op"] in CONDITION_UNARY_OPERATORS)
                self.assertEqual(unary, "left operand only" in line)

    def test_the_judgement_alternative_names_a_real_step_type_and_field(self) -> None:
        judgement = self.types[configurator._JUDGEMENT_STEP_TYPE]
        criteria = [
            field["key"]
            for field in judgement["fields"]
            if field["key"].endswith("success_criteria")
        ]

        self.assertEqual(len(criteria), 1)
        self.assertIn(f"`{configurator._JUDGEMENT_STEP_TYPE}`", self.block)
        self.assertIn(judgement["label"]["en"], self.block)
        self.assertIn(f"`{criteria[0]}`", self.block)

    def test_the_judgement_alternative_names_a_real_failure_policy(self) -> None:
        slot = self.schema["policies"][configurator._FAILURE_POLICY_SLOT]
        option = next(
            option
            for option in slot["options"]
            if option["type"] == configurator._JUDGEMENT_ESCAPE_POLICY
        )

        self.assertIn(configurator._JUDGEMENT_ESCAPE_POLICY, FAILURE_POLICY_TYPES)
        self.assertTrue(option["requires_target"])
        self.assertIn(f'{slot["key"]}: {{type: "{option["type"]}"', self.block)
        self.assertIn(f'{option["target_key"]}: "..."', self.block)


class TheShapeIsNotRestatedTest(unittest.TestCase):
    """The acceptance criterion: no hand-written field list."""

    def test_the_branch_shape_line_is_generated_elsewhere_and_only_once(self) -> None:
        block = _condition_branch_vocabulary_block()
        schema_block = _workflow_step_schema_block()
        prompt = build_configurator_system_prompt()

        shape = "[ { label, when: { left, op, right }, target_step_id } ]"
        self.assertIn(shape, schema_block)
        self.assertNotIn(shape, block)
        self.assertEqual(prompt.count(shape), 1)

    def test_the_branch_entrys_own_keys_are_not_relisted(self) -> None:
        """`label` / `when` / `left` / `right` belong to the generated shape
        line. Naming them again here is how the two copies start to differ."""

        block = _condition_branch_vocabulary_block()

        for key in ("label", "when", "left", "right"):
            with self.subTest(key=key):
                self.assertNotIn(f"`{key}`", block)


class TheGeneratorRefusesToDescribeAMovedSchemaTest(unittest.TestCase):
    """Each guard, fired. A block that renders anyway would be a confident
    false statement about what the server accepts — worse than no block.
    """

    def _schema_without(self, drop_branch_field: bool = False, **edits):
        schema = get_step_schema()
        if drop_branch_field:
            for entry in schema["types"]:
                entry["fields"] = [
                    field
                    for field in entry["fields"]
                    if field["kind"] != KIND_BRANCH_LIST
                ]
        return schema

    def _render_with(self, schema):
        with mock.patch(
            "code_bridge_core.workflow_step_schema.get_step_schema", return_value=schema
        ):
            return _condition_branch_vocabulary_block()

    def test_no_step_type_carries_a_branch_list(self) -> None:
        with self.assertRaises(WorkflowSchemaDriftError):
            self._render_with(self._schema_without(drop_branch_field=True))

    def test_two_step_types_carry_one(self) -> None:
        schema = get_step_schema()
        branch_field = next(
            field
            for entry in schema["types"]
            for field in entry["fields"]
            if field["kind"] == KIND_BRANCH_LIST
        )
        other = next(
            entry
            for entry in schema["types"]
            if all(field["kind"] != KIND_BRANCH_LIST for field in entry["fields"])
        )
        other["fields"] = [*other["fields"], dict(branch_field)]

        with self.assertRaises(WorkflowSchemaDriftError):
            self._render_with(schema)

    def test_the_judgement_step_type_stops_being_published(self) -> None:
        schema = get_step_schema()
        schema["types"] = [
            entry
            for entry in schema["types"]
            if entry["type"] != configurator._JUDGEMENT_STEP_TYPE
        ]

        with self.assertRaises(WorkflowSchemaDriftError):
            self._render_with(schema)

    def test_the_judgement_step_type_stops_offering_success_criteria(self) -> None:
        schema = get_step_schema()
        for entry in schema["types"]:
            if entry["type"] == configurator._JUDGEMENT_STEP_TYPE:
                entry["fields"] = [
                    field
                    for field in entry["fields"]
                    if not field["key"].endswith("success_criteria")
                ]

        with self.assertRaises(WorkflowSchemaDriftError):
            self._render_with(schema)

    def test_the_routing_failure_policy_stops_being_offered(self) -> None:
        schema = get_step_schema()
        slot = schema["policies"][configurator._FAILURE_POLICY_SLOT]
        slot["options"] = [
            option
            for option in slot["options"]
            if option["type"] != configurator._JUDGEMENT_ESCAPE_POLICY
        ]

        with self.assertRaises(WorkflowSchemaDriftError):
            self._render_with(schema)

    def test_the_real_schema_still_renders(self) -> None:
        """The guards above are only worth having if the live schema passes
        them — otherwise they would be pinning a broken prompt."""

        self.assertTrue(_condition_branch_vocabulary_block().strip())


if __name__ == "__main__":
    unittest.main()
