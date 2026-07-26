"""Dashboard page HTML template for Code Bridge server management."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DASHBOARD_HTML: str | None = None
_AGENTS_HTML: str | None = None


def render_dashboard_html(view: str = "status") -> str:
    """Render the console page for one view.

    ``/dashboard`` and ``/settings`` are the same document with a different
    body view. Splitting them into two templates would mean two copies of the
    same 1500 lines of handlers — and the moment one card moved, the other
    page would start calling getElementById on a node that is not there.
    Here every element stays in the DOM and CSS decides what the page is for.
    """
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        _DASHBOARD_HTML = (_TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
    resolved = "settings" if view == "settings" else "status"
    return _DASHBOARD_HTML.replace("__VIEW__", resolved)


def render_agents_html() -> str:
    """Render the standalone Agents page.

    Agent authoring outgrew the dashboard card it started as — it needs a
    workflow editor, a full-height builder chat and a schedule table, none of
    which fit next to the server status widgets.
    """
    global _AGENTS_HTML
    if _AGENTS_HTML is None:
        _AGENTS_HTML = (_TEMPLATE_DIR / "agents.html").read_text(encoding="utf-8")
    return _AGENTS_HTML
