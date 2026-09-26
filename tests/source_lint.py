"""Helpers for checks that read a module's own source text.

A handful of tests assert something about *how a module is written* rather
than what it does — that the cursor imports no store, that predicate
evaluation cannot observe the world. They work by reading the `.py` and
parsing it.

Those checks cannot run against a compiled distribution, because there is no
`.py` to read: `code_bridge_core` ships as a single Nuitka extension with its
source deliberately absent (ADR-002). That is not a failure of the check and
not a defect in the build — it is a check that belongs to the source tree,
asked of an artifact that has no source.

Before this, such a test *failed* against a compiled build, which made a
correct verification run look broken and buried the three real questions
underneath it. Now it skips, and says why.
"""

from __future__ import annotations

import unittest
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]


def read_module_source(package: str, module: str) -> str:
    """The module's own text, or skip the test when it was compiled away."""
    path = SERVER_DIR / package / f"{module}.py"
    if not path.is_file():
        raise unittest.SkipTest(
            f"{package}/{module}.py is not present — this build installs "
            f"{package} compiled, and a source-shape check has no source to "
            "read. Run this against the source tree."
        )
    return path.read_text("utf-8")


def source_of(obj: object) -> str:
    """`inspect.getsource`, or skip when the object was compiled."""
    import inspect

    try:
        return inspect.getsource(obj)  # type: ignore[arg-type]
    except (OSError, TypeError) as exc:
        raise unittest.SkipTest(
            f"source for {getattr(obj, '__name__', obj)!r} is unavailable "
            f"({type(exc).__name__}) — this build installs it compiled. Run "
            "this against the source tree."
        ) from exc
