"""Subagents someone already wrote must be importable, and only those.

A Claude Code subagent is a markdown file with YAML frontmatter. It runs only
when a human invokes it — no schedule, no result history, nobody told when it
fails. Code Bridge has all three, so importing one turns an agent its author
already trusts into one that runs unattended. Two things break that:

Being too strict. A shipped agent (`pr-review-toolkit/silent-failure-hunter`)
writes an unquoted description containing "Context: Daisy has just…". Strict
YAML reads that colon-space as a nested mapping and rejects the file, while
Claude Code loads it fine. Reported as malformed, a working agent silently
cannot be imported — and the reason lives in a YAML detail its author never
had to think about.

Being too loose. `source_path` arrives from a client. If import trusted it,
this endpoint would read whichever file a caller named, next to the user's
real config. It only ever imports a path a fresh sweep just found under the
three known roots.

Re-import must not duplicate either: the same file imported twice is one
agent, or a scheduled agent quietly becomes two.
"""

from __future__ import annotations

import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent import subagent_sources  # noqa: E402


class FrontmatterParsingTest(unittest.TestCase):
    def _parse(self, text: str, tmp: Path, name: str = "a.md"):
        path = tmp / name
        path.write_text(text, encoding="utf-8")
        return subagent_sources._parse_subagent_file(path, location="test")

    def test_a_colon_in_a_description_does_not_make_the_file_malformed(self):
        """The real shipped-agent case: prose containing "Context: …"."""
        with mock.patch.object(subagent_sources, "yaml", subagent_sources.yaml):
            import tempfile

            with tempfile.TemporaryDirectory() as raw:
                tmp = Path(raw)
                result = self._parse(
                    textwrap.dedent(
                        """\
                        ---
                        name: hunter
                        description: Use this when reviewing. Example: Context: Daisy has just finished a chunk of work.
                        model: sonnet
                        ---

                        Body text.
                        """
                    ),
                    tmp,
                )

        self.assertIsInstance(result, subagent_sources.SubagentCandidate)
        self.assertEqual(result.name, "hunter")
        self.assertEqual(result.model, "sonnet")
        self.assertIn("Context: Daisy", result.description)

    def test_a_file_with_no_frontmatter_is_skipped_with_a_reason(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            result = self._parse("Just a markdown file.\n", Path(raw))

        self.assertIsInstance(result, subagent_sources.SkippedSubagentFile)
        self.assertEqual(result.reason, subagent_sources.SKIP_NO_FRONTMATTER)

    def test_genuinely_broken_frontmatter_is_still_reported_broken(self):
        """The lenient path must not rescue everything.

        A block that is not `key: value` lines is not a subagent, and quietly
        reducing it to whatever happened to parse would import a fiction.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            result = self._parse(
                "---\n- name: listy\n-  broken: [unclosed\n---\n\nBody.\n", Path(raw)
            )

        self.assertIsInstance(result, subagent_sources.SkippedSubagentFile)

    def test_a_file_without_a_name_is_skipped(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            result = self._parse("---\ndescription: no name here\n---\n\nBody.\n", Path(raw))

        self.assertIsInstance(result, subagent_sources.SkippedSubagentFile)
        self.assertEqual(result.reason, subagent_sources.SKIP_MISSING_NAME)


class ConversionTest(unittest.TestCase):
    @staticmethod
    def _candidate(**over):
        base = dict(
            source_path="/tmp/explore.md",
            candidate_id="c1",
            location="plugin:x/y",
            name="explore",
            description="Reads code.",
            tools=["Read", "Glob", "Agent(claude-security:explore)"],
            model="sonnet",
            effort=None,
            color=None,
            body="You explore the repository.",
        )
        base.update(over)
        return subagent_sources.SubagentCandidate(**base)

    def test_an_imported_subagent_gets_a_runnable_one_step_flow(self):
        """Empty flow_json reads as "steps: not ready" and cannot run."""
        kwargs = subagent_sources.candidate_to_agent_create_kwargs(self._candidate())

        flow = kwargs["flow_json"]
        self.assertEqual(len(flow), 1)
        self.assertEqual(flow[0]["type"], "llm")
        self.assertTrue(kwargs["system_prompt"].strip())

    def test_the_prompt_is_not_copied_into_the_agent(self):
        """A copy would drift from the file and nothing would say so.

        The file is read on every run, so a copy here is text nobody executes —
        displayed in the agent editor as though editing it changed something.
        What is stored instead points at the file and says as much.
        """
        kwargs = subagent_sources.candidate_to_agent_create_kwargs(self._candidate())

        self.assertNotIn("You explore the repository.", kwargs["system_prompt"])
        self.assertNotIn("You explore the repository.", kwargs["flow_json"][0]["instruction"])
        self.assertIn("/tmp/explore.md", kwargs["system_prompt"])
        self.assertIn("/tmp/explore.md", kwargs["flow_json"][0]["instruction"])

    def test_tool_names_are_carried_through_untranslated(self):
        """They are Claude Code tool names, not Code Bridge capability ids.

        Nothing in the runtime dispatches from tools_json, so inventing a
        mapping would fabricate a capability the server does not have.
        """
        kwargs = subagent_sources.candidate_to_agent_create_kwargs(self._candidate())

        self.assertEqual(
            kwargs["tools_json"],
            ["Read", "Glob", "Agent(claude-security:explore)"],
        )

    def test_inherit_is_not_treated_as_a_model_name(self):
        """`model: inherit` is a Claude Code directive, not an id.

        Storing it verbatim would send "inherit" to a provider as a model.
        """
        kwargs = subagent_sources.candidate_to_agent_create_kwargs(
            self._candidate(model="inherit")
        )

        self.assertIsNone(kwargs["model"])

    def test_a_known_model_resolves_its_provider(self):
        kwargs = subagent_sources.candidate_to_agent_create_kwargs(self._candidate())

        self.assertEqual(kwargs["model"], "sonnet")
        self.assertEqual(kwargs["provider_id"], "anthropic")


class ImportGuardTest(unittest.TestCase):
    def test_a_path_no_sweep_found_is_refused(self):
        """source_path comes from a client and is never read on trust."""
        sweep = subagent_sources.SubagentDiscoverySweep(candidates=[], skipped=[])
        with mock.patch.object(
            subagent_sources, "discover_subagent_candidates", return_value=sweep
        ), mock.patch.object(
            subagent_sources, "_find_import_by_source_path", return_value=None
        ):
            with self.assertRaises(subagent_sources.SubagentImportError):
                subagent_sources.import_subagent("/etc/passwd")


if __name__ == "__main__":
    unittest.main()
