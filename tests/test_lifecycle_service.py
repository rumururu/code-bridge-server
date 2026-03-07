import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import lifecycle_service


class LifecycleServiceSyncTest(unittest.TestCase):
    def test_display_pairing_qr_for_current_server_noop_without_need(self):
        display_mock = MagicMock()
        builder_mock = MagicMock()
        config = SimpleNamespace(port=8080)

        lifecycle_service.display_pairing_qr_for_current_server(
            config,
            needs_pairing=False,
            qrcode_available=True,
            payload_builder=builder_mock,
            payload_display=display_mock,
        )

        builder_mock.assert_not_called()
        display_mock.assert_not_called()

    def test_start_heartbeat_for_current_server_returns_none_when_not_authenticated(self):
        config = SimpleNamespace(heartbeat_interval_minutes=5)
        firebase_auth = SimpleNamespace(is_authenticated=False)

        task = lifecycle_service.start_heartbeat_for_current_server(config, firebase_auth)

        self.assertIsNone(task)

    def test_start_heartbeat_for_current_server_creates_task(self):
        config = SimpleNamespace(heartbeat_interval_minutes=7)
        firebase_auth = SimpleNamespace(is_authenticated=True, heartbeat=AsyncMock(return_value=True))

        fake_task = object()

        def fake_create_task(coro):
            coro.close()
            return fake_task

        task = lifecycle_service.start_heartbeat_for_current_server(
            config,
            firebase_auth,
            create_task=fake_create_task,
        )

        self.assertIs(task, fake_task)


class LifecycleServiceAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_firebase_for_current_server_registers_local_url(self):
        config = SimpleNamespace(firebase_enabled=True, port=8080)
        firebase_auth = SimpleNamespace(
            is_authenticated=True,
            initialize=AsyncMock(),
            register_device=AsyncMock(),
        )

        pairing = SimpleNamespace(get_local_ip=lambda: "192.168.0.10")

        resolved_auth, needs_pairing = await lifecycle_service.initialize_firebase_for_current_server(
            config,
            firebase_available=True,
            firebase_auth_factory=lambda: firebase_auth,
            pairing_service_factory=lambda: pairing,
        )

        self.assertFalse(needs_pairing)
        self.assertIs(resolved_auth, firebase_auth)
        firebase_auth.initialize.assert_awaited_once_with()
        firebase_auth.register_device.assert_awaited_once_with(None, "http://192.168.0.10:8080")

    async def test_initialize_firebase_for_current_server_handles_init_error(self):
        config = SimpleNamespace(firebase_enabled=True, port=8080)
        firebase_auth = SimpleNamespace(initialize=AsyncMock(side_effect=RuntimeError("boom")))

        resolved_auth, needs_pairing = await lifecycle_service.initialize_firebase_for_current_server(
            config,
            firebase_available=True,
            firebase_auth_factory=lambda: firebase_auth,
        )

        self.assertIsNone(resolved_auth)
        self.assertTrue(needs_pairing)

    async def test_start_remote_tunnel_for_current_server_returns_none_when_disabled(self):
        config = SimpleNamespace(remote_access_enabled=False, port=8080)

        tunnel_service = await lifecycle_service.start_remote_tunnel_for_current_server(
            config,
            firebase_auth=None,
            tunnel_available=True,
        )

        self.assertIsNone(tunnel_service)

    async def test_start_remote_tunnel_for_current_server_updates_device_registration(self):
        config = SimpleNamespace(remote_access_enabled=True, port=8080)
        firebase_auth = SimpleNamespace(
            is_authenticated=True,
            register_device=AsyncMock(),
            update_tunnel_url=AsyncMock(),
        )
        pairing = SimpleNamespace(get_local_ip=lambda: "127.0.0.1")

        fake_tunnel = SimpleNamespace(start=AsyncMock(return_value="https://demo.trycloudflare.com"))

        tunnel_service = await lifecycle_service.start_remote_tunnel_for_current_server(
            config,
            firebase_auth=firebase_auth,
            tunnel_available=True,
            tunnel_service_factory=lambda **_: fake_tunnel,
            pairing_service_factory=lambda: pairing,
            create_task=lambda coro: asyncio.create_task(coro),
        )

        self.assertIs(tunnel_service, fake_tunnel)
        fake_tunnel.start.assert_awaited_once_with()
        firebase_auth.register_device.assert_awaited_once_with(
            "https://demo.trycloudflare.com",
            "http://127.0.0.1:8080",
        )

    async def test_shutdown_runtime_for_current_server_closes_services(self):
        class _FakeTask:
            def __init__(self):
                self.cancel_called = False
                self.awaited = False

            def cancel(self):
                self.cancel_called = True

            def __await__(self):
                async def _done():
                    self.awaited = True
                    return None

                return _done().__await__()

        heartbeat_task = _FakeTask()
        tunnel_service = SimpleNamespace(stop=AsyncMock())
        session_manager = SimpleNamespace(close_all=AsyncMock())
        preview_proxy = SimpleNamespace(close=AsyncMock())

        await lifecycle_service.shutdown_runtime_for_current_server(
            heartbeat_task=heartbeat_task,
            tunnel_service=tunnel_service,
            session_manager=session_manager,
            preview_proxy=preview_proxy,
        )

        self.assertTrue(heartbeat_task.cancel_called)
        self.assertTrue(heartbeat_task.awaited)
        tunnel_service.stop.assert_awaited_once_with()
        session_manager.close_all.assert_awaited_once_with()
        preview_proxy.close.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
