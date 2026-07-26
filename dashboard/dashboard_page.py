"""Dashboard page HTML template for Code Bridge server management."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DASHBOARD_HTML: str | None = None
_AGENTS_HTML: str | None = None


def render_dashboard_html() -> str:
    """Render the main dashboard HTML page.

    Loads the HTML template from templates/dashboard.html.
    The template is cached in memory after first load.
    """
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        _DASHBOARD_HTML = (_TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
    return _DASHBOARD_HTML


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
