import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import app_factory


class AppFactoryTest(unittest.TestCase):
    def test_create_code_bridge_app_registers_routers(self):
        calls = []

        def fake_registrar(app):
            calls.append(app)

        app = app_factory.create_code_bridge_app(
            config=SimpleNamespace(cors_origins=["http://localhost:3000"]),
            router_registrar=fake_registrar,
        )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], app)

    def test_create_code_bridge_app_adds_cors_middleware(self):
        app = app_factory.create_code_bridge_app(
            config=SimpleNamespace(cors_origins=["http://localhost:3000"]),
            router_registrar=lambda _: None,
        )

        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        self.assertIn("CORSMiddleware", middleware_classes)


if __name__ == "__main__":
    unittest.main()
