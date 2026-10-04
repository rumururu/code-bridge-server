import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routes import scrcpy_proxy


class ScrcpyProxySecurityTest(unittest.IsolatedAsyncioTestCase):
    async def test_other_device_rejected_before_tango_connection(self):
        websocket = SimpleNamespace(accept=AsyncMock(), close=AsyncMock())
        with patch.dict(os.environ, {"CODEBRIDGE_SCRCPY_ALLOWED_UDID": "leased"}), \
             patch.object(scrcpy_proxy, "is_websocket_from_tunnel", return_value=False), \
             patch.object(scrcpy_proxy, "validate_api_key_for_current_server", return_value=SimpleNamespace(success=True)), \
             patch.object(scrcpy_proxy, "get_scrcpy_manager") as manager, \
             patch.object(scrcpy_proxy.websockets, "connect") as connect:
            await scrcpy_proxy.scrcpy_stream_proxy(websocket, udid="other", api_key="valid")
        websocket.close.assert_awaited_once_with(code=4003, reason="Device not allowed")
        manager.assert_not_called()
        connect.assert_not_called()

    async def test_missing_allowlist_fails_closed(self):
        websocket = SimpleNamespace(accept=AsyncMock(), close=AsyncMock())
        with patch.dict(os.environ, {"CODEBRIDGE_AGENT_ANDROID_DEVICE_ID": "other"}, clear=True), \
             patch.object(scrcpy_proxy, "is_websocket_from_tunnel", return_value=False), \
             patch.object(scrcpy_proxy, "validate_api_key_for_current_server", return_value=SimpleNamespace(success=True)), \
             patch.object(scrcpy_proxy.websockets, "connect") as connect:
            await scrcpy_proxy.scrcpy_stream_proxy(websocket, udid="leased", api_key="valid")
        websocket.close.assert_awaited_once_with(code=4003, reason="Device not allowed")
        connect.assert_not_called()

    async def test_qa_allowlist_only_narrows_configured_device(self):
        websocket = SimpleNamespace(accept=AsyncMock(), close=AsyncMock())
        with patch.dict(os.environ, {"CODEBRIDGE_AGENT_ANDROID_DEVICE_ID": "configured", "CODEBRIDGE_SCRCPY_ALLOWED_UDID": "other"}, clear=True), \
             patch.object(scrcpy_proxy, "is_websocket_from_tunnel", return_value=False), \
             patch.object(scrcpy_proxy, "validate_api_key_for_current_server", return_value=SimpleNamespace(success=True)), \
             patch.object(scrcpy_proxy.websockets, "connect") as connect:
            await scrcpy_proxy.scrcpy_stream_proxy(websocket, udid="other", api_key="valid")
        websocket.close.assert_awaited_once_with(code=4003, reason="Device not allowed")
        connect.assert_not_called()
