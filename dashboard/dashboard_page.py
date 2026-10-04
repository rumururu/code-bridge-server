"""Dashboard page HTML template for Code Bridge server management."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DASHBOARD_HTML: str | None = None
_AGENTS_HTML: str | None = None


def render_dashboard_html(view: str = "status", *, embedded: bool = False) -> str:
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
    html = _DASHBOARD_HTML.replace("__VIEW__", resolved)
    return _embed_legacy(html, "management") if embedded else html


def render_agents_html(*, embedded: bool = False) -> str:
    """Render the standalone Agents page.

    Agent authoring outgrew the dashboard card it started as — it needs a
    workflow editor, a full-height builder chat and a schedule table, none of
    which fit next to the server status widgets.
    """
    global _AGENTS_HTML
    if _AGENTS_HTML is None:
        _AGENTS_HTML = (_TEMPLATE_DIR / "agents.html").read_text(encoding="utf-8").replace(
            "/* APPROVAL_REVIEW_MODULE */",
            (_TEMPLATE_DIR / "approval_review.js").read_text(encoding="utf-8"),
        )
    return _embed_legacy(_AGENTS_HTML, "automation") if embedded else _AGENTS_HTML


def render_experience_html() -> str:
    """Render the local console shell without embedding credentials."""
    return (_TEMPLATE_DIR / "experience.html").read_text(encoding="utf-8").replace(
        "/* APPROVAL_REVIEW_MODULE */",
        (_TEMPLATE_DIR / "approval_review.js").read_text(encoding="utf-8"),
    ).replace(
        "/* BROWSER_HANDOFF_MODULE */",
        (_TEMPLATE_DIR / "browser_handoff.js").read_text(encoding="utf-8"),
    )


def _embed_legacy(html: str, surface: str) -> str:
    """Keep the existing authoring DOM and handlers alive inside the shell."""
    if surface == "management":
        # Disable the old page-level split, without overriding a card's own
        # visibility (for example the pairing banner after a client pairs).
        html = html.replace('<body data-view="settings">', '<body data-view="management">', 1)
        html = html.replace('<body data-view="status">', '<body data-view="management">', 1)
    adapter = (_TEMPLATE_DIR / "experience_embed.js").read_text(encoding="utf-8")
    style = """<style>
    body[data-embedded] {
        color-scheme: light;
        --ink: #f6f7f2; --slate: #fff; --seam: #e2e7e1;
        --seam-soft: #eaf0e7; --wire: #203c3c;
        --wire-dim: #6d7d79; --wire-faint: #718178;
        --link: #265c52; --signal-bg: #eaf1e6;
        --accent-hover: #1d4b43;
        background: var(--ink); color: var(--wire);
    }
    body[data-embedded] .header .brand,
    body[data-embedded] .header .nav { display: none !important; }
    body[data-embedded] .header {
        position: static !important; min-height: 42px; height: auto;
        background: var(--slate); border-color: var(--seam);
    }
    body[data-embedded] .card {
        border-color: var(--seam); border-radius: 13px;
        background: var(--slate);
    }
    body[data-embedded] .card-header {
        background: #f5f7f1; border-color: var(--seam);
    }
    body[data-embedded] .btn { border-radius: 9px; }
    body[data-embedded] .btn-primary,
    body[data-embedded="automation"] .btn:not(.btn-secondary):not(.btn-danger) {
        background: var(--link); border-color: var(--link); color: #fff;
    }
    body[data-embedded] .btn-primary:hover,
    body[data-embedded="automation"] .btn:not(.btn-secondary):not(.btn-danger):hover {
        background: var(--accent-hover); border-color: var(--accent-hover);
    }
    body[data-embedded="management"] .pairing-banner.visible {
        background: #e9eee1; border: 1px solid #dce3d3;
        color: var(--wire); box-shadow: none;
    }
    body[data-embedded="management"] .pairing-banner .btn {
        border: 1px solid #cbd9c8; color: var(--link);
    }
    body[data-embedded="management"] .pairing-banner .btn:hover { background: #f6f7f2; }
    body[data-embedded="management"] .app-store-link {
        background: #e1ebdb; color: var(--link);
    }
    body[data-embedded="management"] [data-management-hidden] { display: none !important; }
    body[data-embedded="management"] .grid { align-items: start; }
    body[data-embedded="management"] #agentOverviewCard { display: none !important; }
    body[data-embedded="management"] .container { max-width: none; }
    @media(max-width: 600px) {
        body[data-embedded="management"] .grid { grid-template-columns: minmax(0, 1fr) !important; }
        body[data-embedded="management"] .pairing-banner.visible { flex-direction: column; align-items: stretch; }
        body[data-embedded="management"] .pairing-banner-content { min-width: 0; }
        body[data-embedded="management"] .pairing-banner-actions { flex-wrap: wrap; }
        body[data-embedded="management"] .card,
        body[data-embedded="management"] .container,
        body[data-embedded="management"] .card-content { min-width: 0; }
        body[data-embedded="management"] .info-value { overflow-wrap: anywhere; }
    }
    </style>"""
    html = html.replace("</head>", style + "</head>")
    return html.replace("</body>", f'<script>document.body.dataset.embedded = "{surface}";\n{adapter}</script></body>')
