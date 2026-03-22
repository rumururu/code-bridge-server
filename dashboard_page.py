"""Dashboard page HTML template for Code Bridge server management."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DASHBOARD_HTML: str | None = None


def render_dashboard_html() -> str:
    """Render the main dashboard HTML page.

    Loads the HTML template from templates/dashboard.html.
    The template is cached in memory after first load.
    """
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        _DASHBOARD_HTML = (_TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
    return _DASHBOARD_HTML
