"""Static hosting for the workflow canvas bundle under ``/canvas/``.

Three decisions are baked into this file; each one is the answer to a concrete
hazard in this repo.

**A router, not ``app.mount``.** This repo says "which listener holds what" in
exactly two tuples — ``_SHARED_ROUTERS`` and ``_DASHBOARD_ONLY_ROUTERS`` in
:mod:`routes` — and the docstring there explains why: so that changing the bind
address alone cannot leak a surface. A ``StaticFiles`` mount in ``app_factory``
would be a surface that does not appear in either tuple. As a router, the
decision "canvas is dashboard-only after all" is one line moved between two
tuples, and the test for it is a route lookup.

**Everything lives under ``/canvas/``, never ``/assets/``.** The root asset
paths are already owned by the preview proxy: ``/assets/{path}``,
``/_next/{path}``, ``/@{path}``, ``/src/{path}``, ``/node_modules/{path}`` and
the ``/{filename}`` allow-list (see :mod:`routes.preview`). A Vite build with
the default ``base`` would emit ``/assets/index-<hash>.js``, the preview proxy
would answer it, and the canvas would fail with "No active preview session".
The bundle must therefore be built with ``base: '/canvas/'``.

**The bundle carries no credential.** These files are served without
authentication on purpose. They are public code; gating them would break
caching and protect nothing. The authority lives in the canvas *token*
(:mod:`canvas.canvas_access`), which the page is handed at runtime and which
opens only :mod:`routes.canvas_api`.

Two failure modes are made loud rather than silent:

* **No bundle** → ``503`` naming the expected path and how to produce it. A
  blank page or a bare 404 would send the reader looking for a routing bug
  that is not there — the same reason ``routes/agents.py`` answers a missing
  kernel with an install hint instead of an empty graph.
* **A missing asset never falls back to ``index.html``.** SPA fallback is
  right for unknown *routes* and catastrophic for a mistyped ``.js`` URL: the
  browser would receive HTML with a 200 and report ``Unexpected token '<'``,
  which reads like a bundler bug. Assets 404 as assets.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, Response

from core.config import get_config
from core.runtime_paths import SERVER_DIR

router = APIRouter(prefix="/canvas", tags=["canvas"])

# Where the built bundle lives when nothing overrides it. It is committed to
# the repo (T-I1-07) because ``install/sync-local-install.sh`` rsyncs
# ``server/`` verbatim and the machines it installs onto have no node — a
# build step at deploy time would either break the install or skip silently
# and leave a 404 behind.
DEFAULT_CANVAS_BUNDLE_DIR = SERVER_DIR / "webui" / "canvas"
CANVAS_BUNDLE_DIR_ENV = "CODEBRIDGE_CANVAS_BUNDLE_DIR"

# The root script builds `flow-canvas` before `flow-canvas-standalone`, which
# is required: the standalone app imports the library workspace. Targeting the
# standalone alone with `-w flow-canvas-standalone` does not even resolve — the
# workspace is named `@agent-flow/flow-canvas-standalone`.
# `--delete` is not optional: Vite content-hashes filenames, so without it the
# previous chunk survives beside the new one and the page loads whichever the
# stale index.html names. See docs/guide/CANVAS_BUNDLE.md.
_CANVAS_BUILD_COMMAND = (
    "cd ~/VSCodeProject/agent-flow-core/frontend "
    "&& npm install && npm run build "
    "&& rsync -a --delete packages/flow-canvas-standalone/dist/ "
    "$CODE_BRIDGE/server/webui/canvas/"
)
_CANVAS_BUILD_HINT = (
    "The canvas ships as a prebuilt static bundle committed at"
    " server/webui/canvas/ (agent-flow-core T-I1-07); nothing builds it at"
    " install time, because the machines this server installs onto have no"
    " node. Build it from the agent-flow-core checkout — `"
    + _CANVAS_BUILD_COMMAND
    + "` — then copy frontend/packages/flow-canvas-standalone/dist/ into"
    " server/webui/canvas/. The build must set Vite's base to '/canvas/':"
    " the root /assets/* path already belongs to the preview proxy, so a"
    " default-base build 404s against a dev-server proxy instead of loading."
)

# Long-lived only for content-hashed files. Vite writes those into `assets/`
# under the configured base, so the directory is the signal — a hash in the
# name is what makes "immutable" true.
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_HTML_CACHE = "no-cache"
_DEFAULT_CACHE = "public, max-age=3600"

_SECURITY_HEADERS = {
    # The API app serves no other HTML today, so this header is new here. The
    # bundle must reference no external origin for it to hold (T-I1-07's
    # "zero external origins" acceptance criterion is what makes 'self' true).
    # `style-src 'unsafe-inline'` is not laxity, it is what makes the page
    # visible. Emotion injects the canvas's stylesheet at runtime and
    # reactflow positions every node with an inline `style` transform, so
    # under a bare `default-src 'self'` the DOM is correct and the screen is
    # blank — verified both ways in a browser before this line was widened.
    # `img-src data:` covers the inline SVG icons the same way.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
}


def canvas_bundle_dir() -> Path:
    """Resolve the bundle directory: env override, then config, then default."""
    override = os.environ.get(CANVAS_BUNDLE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    configured = getattr(get_config(), "canvas_bundle_dir", None)
    if configured:
        return Path(str(configured)).expanduser()
    return DEFAULT_CANVAS_BUNDLE_DIR


def _bundle_missing(bundle_dir: Path, *, reason: str) -> JSONResponse:
    detail = (
        f"The canvas bundle is not present at {bundle_dir}. "
        f"{_CANVAS_BUILD_HINT}"
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": "canvas_bundle_missing",
            "reason": reason,
            "expected_path": str(bundle_dir),
            "detail": detail,
            "message": detail,
            "remedy": _CANVAS_BUILD_COMMAND,
        },
        headers=dict(_SECURITY_HEADERS),
    )


def _looks_like_asset(asset_path: str) -> bool:
    """True when the request is for a file, not an application route.

    A last segment containing a dot is the practical test — ``main-a1b2.js``,
    ``style.css``, ``favicon.ico`` — and anything under ``assets/`` counts
    regardless, because that is where the hashed build output goes.
    """
    normalized = asset_path.strip("/")
    if not normalized:
        return False
    if normalized.startswith("assets/"):
        return True
    return "." in normalized.rsplit("/", 1)[-1]


def _resolve_within(bundle_dir: Path, asset_path: str) -> Path | None:
    """Join ``asset_path`` onto the bundle, refusing anything that escapes it."""
    candidate = (bundle_dir / asset_path).resolve()
    root = bundle_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _cache_control(relative_path: str) -> str:
    if relative_path.startswith("assets/"):
        return _IMMUTABLE_CACHE
    if relative_path.endswith(".html") or relative_path == "":
        return _HTML_CACHE
    return _DEFAULT_CACHE


def _file_response(path: Path, *, relative_path: str) -> FileResponse:
    media_type, _ = mimetypes.guess_type(path.name)
    headers = dict(_SECURITY_HEADERS)
    headers["Cache-Control"] = _cache_control(relative_path)
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        headers=headers,
    )


def _index_response(bundle_dir: Path) -> Response:
    index = bundle_dir / "index.html"
    if not bundle_dir.is_dir():
        return _bundle_missing(bundle_dir, reason="directory_missing")
    if not index.is_file():
        return _bundle_missing(bundle_dir, reason="index_missing")
    return _file_response(index, relative_path="index.html")


@router.get("", include_in_schema=False, response_model=None)
async def canvas_root_no_slash() -> Response:
    """``/canvas`` — served, not redirected.

    A redirect would not survive: with no route registered here, Starlette's
    slash-redirect never runs, because the preview router's ``/{filename}``
    catch-all matches ``/canvas`` first and answers its own 404.
    """
    return _index_response(canvas_bundle_dir())


@router.get("/", include_in_schema=False, response_model=None)
async def canvas_root() -> Response:
    return _index_response(canvas_bundle_dir())


@router.get("/{asset_path:path}", include_in_schema=False, response_model=None)
async def canvas_asset(asset_path: str) -> Response:
    """One file from the bundle, or the SPA shell for an unknown route."""
    bundle_dir = canvas_bundle_dir()
    if not bundle_dir.is_dir():
        return _bundle_missing(bundle_dir, reason="directory_missing")

    normalized = asset_path.strip("/")
    if not normalized:
        return _index_response(bundle_dir)

    target = _resolve_within(bundle_dir, normalized)
    if target is None:
        # Traversal attempt. Answered as a plain 404: naming the rule would
        # only tell a prober which shape of path the guard is watching for.
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "path": asset_path},
            headers=dict(_SECURITY_HEADERS),
        )
    if target.is_file():
        return _file_response(target, relative_path=normalized)

    if _looks_like_asset(normalized):
        detail = (
            f"No such file in the canvas bundle: {normalized}. Asset paths are"
            " never served the SPA shell — returning index.html here would"
            " hand the browser HTML for a .js request and surface as"
            " \"Unexpected token '<'\" instead of a missing file. Check that"
            " the bundle was built with Vite base '/canvas/' and copied"
            f" whole into {bundle_dir}."
        )
        return JSONResponse(
            status_code=404,
            content={
                "error": "canvas_asset_not_found",
                "path": normalized,
                "bundle_path": str(bundle_dir),
                "detail": detail,
                "message": detail,
            },
            headers=dict(_SECURITY_HEADERS),
        )

    # An application route (``/canvas/agent/abc``): the SPA router owns it.
    return _index_response(bundle_dir)
