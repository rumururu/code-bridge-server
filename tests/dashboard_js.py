"""Run the dashboard template's own JavaScript, instead of grepping for it.

`dashboard/templates/agents.html` carries several thousand lines of behaviour
in a `<script>` block: which field kind draws which control, how a reply is
parsed, when a run list stops polling. None of that can be checked by asserting
that a string appears in the file — a regex that no longer matches what the
server emits, a `switch` arm that stopped being reached, a poll that never
clears its timer all leave the file looking exactly the same.

So the tests lift the functions out and run them in node with the smallest
stubs that let them execute. Three test modules were each carrying their own
copy of the brace counter below, and the copies had already drifted (one
could not extract an `async function`), which is the usual argument for
putting it in one place.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
TEMPLATE = SERVER_DIR / "dashboard" / "templates" / "agents.html"
MARKUP = TEMPLATE.read_text(encoding="utf-8")

#: `None` when node is not installed. Tests skip on it rather than fail —
#: node is a developer-machine tool here, not a server dependency.
NODE = shutil.which("node")

#: Stubs for the page helpers that touch the DOM or the i18n table. Anything
#: that decides *what is drawn* is lifted from the template instead.
COMMON_STUBS = """
const escapeHtml = (value) => String(value ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');
const t = (key) => key;
"""


def js_function(name: str) -> str:
    """Lift one top-level `function name(...) {...}` out of the template.

    Brace-counted rather than regex-matched: the bodies contain both braces
    and template literals, and a lazy regex silently returns half a function
    that still parses. A leading `async` is kept — dropping it produced
    "await is only valid in async functions", which reads like a bug in the
    page rather than in the extractor.
    """
    start = MARKUP.index(f"function {name}(")
    if MARKUP[max(0, start - 6) : start] == "async ":
        start -= 6
    depth = 0
    for index in range(MARKUP.index("{", start), len(MARKUP)):
        if MARKUP[index] == "{":
            depth += 1
        elif MARKUP[index] == "}":
            depth -= 1
            if depth == 0:
                return MARKUP[start : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def js_const(name: str) -> str:
    """Lift one top-level `const name = …;` line out of the template."""
    for line in MARKUP.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"const {name} ") or stripped.startswith(f"const {name}="):
            return stripped
    raise AssertionError(f"{name} is no longer declared in {TEMPLATE.name}")


def run_js(script: str) -> str:
    """Execute `script` in node, raising its stderr as the failure message."""
    assert NODE is not None, "node is not installed"
    completed = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, timeout=30
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.strip())
    return completed.stdout


def json_literal(value: object) -> str:
    """A JS literal for `value`.

    Arguments are handed over as JSON rather than pasted into the source: the
    strings under test are full of double quotes — the `<ask_user/>` tag is
    made of them — and interpolating those into JS source produces a syntax
    error that looks exactly like a broken parser.
    """
    return json.dumps(value)
