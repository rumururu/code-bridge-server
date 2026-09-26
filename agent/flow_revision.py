"""A short, stable name for *this* version of an agent's workflow.

Two writers can edit one agent's workflow at the same time — the canvas in a
browser (``PATCH /api/canvas/agents/{id}/graph``) and the app's edit screen
(``PATCH /api/agent/agents/{id}``, ``lib/screens/agent/agent_builder_screen.dart``)
— and until this existed the second one to press Save simply won. Nothing
refused it, nothing recorded that a version had been passed over, and the first
writer's work was gone with no event anybody could point at. The revision is
what a client holds so it can say "I am editing *that* version" and the server
can say "no, it moved" instead of overwriting.

## Why a content hash and not ``updated_at``

The agents table has one timestamp and it is
``updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`` (``core/database.py``).
SQLite's ``CURRENT_TIMESTAMP`` is second-resolution — measured, not assumed:
a stored row reads back ``2026-08-20T23:50:11+00:00``. Two writes inside the
same second are therefore indistinguishable by it, and "two writes very close
together" is precisely the race a precondition exists to catch: the blind spot
would sit exactly over the target.

A content hash also declines to invent a conflict. Saving byte-identical
content leaves the revision alone, so a reader holding it is still holding the
truth and their save still goes through — where a timestamp would have moved
and refused them over a write that changed nothing.

## What it covers: the stored ``flow_json``, and nothing else

The revision spans the **stored workflow**, not the whole agent row and not the
derived graph view.

* Not the whole agent. ``name``, ``description``, ``model`` and the rest are
  edited by screens that do not touch the workflow; folding them in would make
  renaming an agent from the phone refuse a canvas graph save that has no
  quarrel with it. The rule is: the revision covers what a *graph writer*
  replaces, which is ``flow_json`` wholesale.
* Not ``flow_graph``. That view is derived from ``flow_json`` on every read
  (``routes/agents.py::_flow_graph_view``) and it is lossy in one direction and
  absent in several situations — a server with no kernel installed publishes
  ``flow_graph_unavailable`` instead. A revision computed from it would be
  unavailable on exactly those servers while ``flow_json`` writes kept working,
  which is a precondition that protects nothing when it is needed most. Two
  different stored workflows can also fold to the same graph, and a reader who
  lost the difference would be told nothing had changed.

The input is the **stored** ``flow_json``, never the copy ``routes/agents.py``
annotates with ``script_name``/``script_path`` for display. Those are looked up
from the scripts table at read time, so hashing them would move the revision
when somebody renamed a script — a write to a different table that no workflow
writer performed and no workflow reader lost anything to. Same reasoning as
``_flow_graph_view``'s, which derives from the stored canon for the same reason.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["FLOW_REVISION_LENGTH", "canonical_flow_text", "compute_flow_revision"]

#: How many hex characters of the digest are published.
#:
#: 16 hex characters is 64 bits. This is a concurrency token, not a security
#: one — the thing it must not do is collide by accident between two versions
#: of one agent's workflow, and 64 bits does not do that. It is short enough to
#: sit in a log line, a URL-free JSON body and a bug report unwrapped.
FLOW_REVISION_LENGTH = 16


def canonical_flow_text(flow_json: Any) -> str:
    """The one serialisation of a workflow that hashing is allowed to see.

    ``sort_keys`` and fixed separators are what make "the same content" and
    "the same bytes" the same statement: a dict that came back from SQLite in
    a different key order, or a client library that spaces its JSON
    differently, must not read as a different version. ``ensure_ascii=False``
    so a Korean step name is hashed as the text it is rather than as its
    escape sequence — the two are the same content and one encoder choice
    should not be able to split them.

    ``default=str`` is a floor, not a feature. Everything reaching here comes
    out of ``json.loads`` in the store, so it is already plain JSON types; the
    fallback exists so that a caller who somehow passes a stray object gets a
    revision instead of a ``TypeError`` inside a read route, which would turn
    an unhashable value into a 500 on an agent that is otherwise perfectly
    readable.
    """
    return json.dumps(
        flow_json,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def compute_flow_revision(flow_json: Any) -> str:
    """The revision of one stored workflow: 16 hex characters of sha256.

    Total: every input has a revision, including ``None``. An agent with no
    workflow is a state a writer can replace, so it needs a name too — and it
    is a *different* name from an agent whose workflow is the empty list,
    because those are different stored values and turning one into the other
    is a change a reader would want to know about.
    """
    digest = hashlib.sha256(canonical_flow_text(flow_json).encode("utf-8"))
    return digest.hexdigest()[:FLOW_REVISION_LENGTH]
