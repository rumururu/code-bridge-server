"""``/canvas/*`` static hosting — T-I1-08.

The three behaviours worth a permanent test are the ones a future edit would
plausibly "simplify" away:

1. **No bundle answers with instructions, not a blank page.** A 404 here sends
   the reader hunting for a routing bug that does not exist.
2. **A missing asset never receives ``index.html``.** SPA fallback for a
   mistyped ``.js`` URL turns "file not found" into ``Unexpected token '<'``
   in the browser console, which reads as a bundler failure.
3. **The dashboard/API boundary is a tuple, not a mount.** Removing the router
   from ``_SHARED_ROUTERS`` is all it takes to make the canvas dashboard-only,
   and that is asserted rather than asserted-about-in-prose.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import routes  # noqa: E402
from routes import canvas_static  # noqa: E402

INDEX_HTML = "<!doctype html><title>Canvas</title><div id=root></div>"
ASSET_JS = "console.log('canvas');"


class CanvasStaticRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._bundle = Path(self._tmp.name) / "canvas"
        self._previous_env = os.environ.get(canvas_static.CANVAS_BUNDLE_DIR_ENV)
        os.environ[canvas_static.CANVAS_BUNDLE_DIR_ENV] = str(self._bundle)

        app = FastAPI()
        app.include_router(canvas_static.router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        if self._previous_env is None:
            os.environ.pop(canvas_static.CANVAS_BUNDLE_DIR_ENV, None)
        else:
            os.environ[canvas_static.CANVAS_BUNDLE_DIR_ENV] = self._previous_env
        self._tmp.cleanup()

    def _install_bundle(self) -> None:
        (self._bundle / "assets").mkdir(parents=True, exist_ok=True)
        (self._bundle / "index.html").write_text(INDEX_HTML)
        (self._bundle / "assets" / "app-a1b2c3d4.js").write_text(ASSET_JS)

    # -- 1. missing bundle -------------------------------------------------

    def test_missing_bundle_directory_explains_itself(self) -> None:
        response = self.client.get("/canvas/")
        self.assertEqual(response.status_code, 503, response.text)
        body = response.json()
        self.assertEqual(body["error"], "canvas_bundle_missing")
        self.assertEqual(body["reason"], "directory_missing")
        self.assertEqual(body["expected_path"], str(self._bundle))
        # The remedy is a command, not "see the docs".
        self.assertIn("npm run build", body["remedy"])
        self.assertIn("base", body["message"])

    def test_the_remedy_is_a_command_that_would_actually_work(self) -> None:
        """A remedy that runs and fixes nothing is worse than no remedy.

        Both halves were wrong once. `-w flow-canvas-standalone` exits with
        "No workspaces found" (the package is `@agent-flow/...`), and building
        alone never moves the bundle here, so the reader ran a command, saw it
        succeed, reloaded, and got the same 503.
        """
        remedy = self.client.get("/canvas/").json()["remedy"]
        self.assertNotIn("-w flow-canvas-standalone", remedy)
        self.assertIn("rsync", remedy)
        self.assertIn("--delete", remedy)
        self.assertIn("webui/canvas", remedy)

    def test_missing_index_is_distinguished_from_missing_directory(self) -> None:
        self._bundle.mkdir(parents=True)
        response = self.client.get("/canvas/")
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["reason"], "index_missing")

    def test_asset_request_with_no_bundle_is_503_not_404(self) -> None:
        response = self.client.get("/canvas/assets/app-a1b2c3d4.js")
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["error"], "canvas_bundle_missing")

    # -- 2. serving --------------------------------------------------------

    def test_index_is_served_with_and_without_trailing_slash(self) -> None:
        self._install_bundle()
        for path in ("/canvas/", "/canvas"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("id=root", response.text)

    def test_index_carries_security_headers_and_is_not_cached(self) -> None:
        self._install_bundle()
        response = self.client.get("/canvas/")
        policy = response.headers["content-security-policy"]
        self.assertIn("default-src 'self'", policy)
        self.assertIn("frame-ancestors 'none'", policy)
        # Not laxity — visibility. Emotion injects the canvas stylesheet at
        # runtime and reactflow positions nodes with inline transforms, so a
        # bare default-src renders a correct DOM onto a blank screen. That was
        # verified in a browser both ways; narrowing these two back turns the
        # canvas off without failing anything else.
        self.assertIn("style-src 'self' 'unsafe-inline'", policy)
        self.assertIn("img-src 'self' data:", policy)
        # Everything the page loads is same-origin, so the widening stops here.
        self.assertNotIn("script-src", policy)
        self.assertNotIn("connect-src", policy)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["cache-control"], "no-cache")

    def test_hashed_asset_is_served_immutable(self) -> None:
        self._install_bundle()
        response = self.client.get("/canvas/assets/app-a1b2c3d4.js")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.text, ASSET_JS)
        self.assertIn("immutable", response.headers["cache-control"])
        self.assertIn("javascript", response.headers["content-type"])

    def test_unknown_route_falls_back_to_index(self) -> None:
        self._install_bundle()
        response = self.client.get("/canvas/agent/agent_abc/edit")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("id=root", response.text)

    # -- 3. assets never get the SPA shell ---------------------------------

    def test_missing_asset_404s_instead_of_returning_index_html(self) -> None:
        self._install_bundle()
        response = self.client.get("/canvas/assets/typo-99999999.js")
        self.assertEqual(response.status_code, 404, response.text)
        self.assertNotIn("id=root", response.text)
        body = response.json()
        self.assertEqual(body["error"], "canvas_asset_not_found")
        self.assertIn("Unexpected token", body["message"])

    def test_missing_dotted_file_outside_assets_also_404s(self) -> None:
        self._install_bundle()
        response = self.client.get("/canvas/favicon.ico")
        self.assertEqual(response.status_code, 404, response.text)
        self.assertEqual(response.json()["error"], "canvas_asset_not_found")

    # -- traversal ---------------------------------------------------------

    def test_paths_escaping_the_bundle_are_refused(self) -> None:
        self._install_bundle()
        (Path(self._tmp.name) / "server_info.json").write_text('{"secret": 1}')
        # Called directly: an HTTP client normalises ``..`` out of the URL
        # before the server ever sees it, so the guard has to be exercised at
        # the handler, where a decoded path parameter can still carry one.
        response = asyncio.run(canvas_static.canvas_asset("../server_info.json"))
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(b"secret", bytes(response.body))

    def test_resolve_within_rejects_escapes_and_accepts_children(self) -> None:
        self._install_bundle()
        self.assertIsNone(canvas_static._resolve_within(self._bundle, "../etc/passwd"))
        self.assertIsNotNone(
            canvas_static._resolve_within(self._bundle, "assets/app-a1b2c3d4.js")
        )


class CanvasStaticBoundaryTest(unittest.TestCase):
    """Which listener serves the canvas is decided by one tuple."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._bundle = Path(self._tmp.name) / "canvas"
        self._bundle.mkdir(parents=True)
        (self._bundle / "index.html").write_text(INDEX_HTML)
        self._previous_env = os.environ.get(canvas_static.CANVAS_BUNDLE_DIR_ENV)
        os.environ[canvas_static.CANVAS_BUNDLE_DIR_ENV] = str(self._bundle)

    def tearDown(self) -> None:
        if self._previous_env is None:
            os.environ.pop(canvas_static.CANVAS_BUNDLE_DIR_ENV, None)
        else:
            os.environ[canvas_static.CANVAS_BUNDLE_DIR_ENV] = self._previous_env
        self._tmp.cleanup()

    def test_canvas_router_is_shared_by_both_apps(self) -> None:
        self.assertIn(canvas_static.router, routes._SHARED_ROUTERS)
        self.assertNotIn(canvas_static.router, routes._DASHBOARD_ONLY_ROUTERS)

        for registrar in (routes.register_dashboard_routers, routes.register_api_routers):
            with self.subTest(registrar=registrar.__name__):
                app = FastAPI()
                registrar(app)
                response = TestClient(app).get("/canvas/")
                self.assertEqual(response.status_code, 200, response.text)

    def test_removing_it_from_the_shared_tuple_makes_the_api_app_404(self) -> None:
        """Reverting the boundary decision is a one-line tuple move.

        Built here by registering the shared set *minus* the canvas router,
        which is exactly what moving it into ``_DASHBOARD_ONLY_ROUTERS`` does
        to the API app.
        """
        app = FastAPI()
        for router in routes._SHARED_ROUTERS:
            if router is canvas_static.router:
                continue
            app.include_router(router)
        app.include_router(routes.preview_router)

        response = TestClient(app).get("/canvas/")
        self.assertEqual(response.status_code, 404, response.text)
        self.assertNotIn("id=root", response.text)


class CanvasPreviewCoexistenceTest(unittest.TestCase):
    """The canvas must not have taken anything the preview proxy owns."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._bundle = Path(self._tmp.name) / "canvas"
        (self._bundle / "assets").mkdir(parents=True)
        (self._bundle / "index.html").write_text(INDEX_HTML)
        (self._bundle / "assets" / "app-a1b2c3d4.js").write_text(ASSET_JS)
        self._previous_env = os.environ.get(canvas_static.CANVAS_BUNDLE_DIR_ENV)
        os.environ[canvas_static.CANVAS_BUNDLE_DIR_ENV] = str(self._bundle)

        app = FastAPI()
        routes.register_api_routers(app)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        if self._previous_env is None:
            os.environ.pop(canvas_static.CANVAS_BUNDLE_DIR_ENV, None)
        else:
            os.environ[canvas_static.CANVAS_BUNDLE_DIR_ENV] = self._previous_env
        self._tmp.cleanup()

    def test_root_asset_paths_still_belong_to_the_preview_proxy(self) -> None:
        """``/assets/*`` must never resolve to a canvas file.

        This is why the bundle has to be built with Vite ``base: '/canvas/'``:
        the same filename served at the root is answered by the dev-server
        proxy, not by us.
        """
        response = self.client.get("/assets/app-a1b2c3d4.js")
        self.assertNotEqual(response.status_code, 200)
        self.assertNotIn(ASSET_JS, response.text)

    def test_preview_root_file_allowlist_is_unchanged(self) -> None:
        for filename in ("favicon.ico", "vite.svg", "manifest.json"):
            with self.subTest(filename=filename):
                response = self.client.get(f"/{filename}")
                # Whatever the preview proxy answers (no active session), it
                # must not be canvas HTML.
                self.assertNotIn("id=root", response.text)


if __name__ == "__main__":
    unittest.main()
