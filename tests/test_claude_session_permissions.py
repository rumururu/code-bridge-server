"""The permission bridge between the SDK callback and the app's round-trip.

The SDK asks for a decision by calling ``can_use_tool`` and awaiting its
return value. The app instead receives a ``control_request`` event over its
websocket and answers much later. These tests pin the bridge that joins the
two: the callback parks on a future, the request surfaces on the event queue,
and approve/deny resolves it.

A regression here doesn't raise — the SDK callback simply never returns, so a
tool call hangs forever with the turn stuck "in progress".
"""

import asyncio
import unittest

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from llm.claude_session import ClaudeSession


class PermissionBridgeTest(unittest.IsolatedAsyncioTestCase):
    def _session(self) -> ClaudeSession:
        return ClaudeSession(project_path="/tmp")

    async def test_callback_publishes_a_control_request_and_waits(self):
        session = self._session()
        callback = asyncio.create_task(
            session._on_can_use_tool(
                "Bash",
                {"command": "ls"},
                ToolPermissionContext(tool_use_id="toolu_1"),
            )
        )

        event = await asyncio.wait_for(session._event_queue.get(), timeout=1)
        self.assertEqual(event["type"], "control_request")
        request = event["request"]
        self.assertEqual(request["subtype"], "can_use_tool")
        self.assertEqual(request["tool_name"], "Bash")
        self.assertEqual(request["input"], {"command": "ls"})
        self.assertEqual(request["tool_use_id"], "toolu_1")
        self.assertIsInstance(event["request_id"], str)

        # Still parked — nothing has decided yet.
        self.assertFalse(callback.done())

        session._settle_pending_permission(PermissionResultAllow(updated_input={"command": "ls"}))
        decision = await asyncio.wait_for(callback, timeout=1)
        self.assertIsInstance(decision, PermissionResultAllow)
        self.assertEqual(decision.updated_input, {"command": "ls"})

    async def test_deny_reaches_the_callback(self):
        session = self._session()
        callback = asyncio.create_task(
            session._on_can_use_tool("Bash", {}, ToolPermissionContext())
        )
        await asyncio.wait_for(session._event_queue.get(), timeout=1)

        session._settle_pending_permission(PermissionResultDeny(message="nope"))
        decision = await asyncio.wait_for(callback, timeout=1)
        self.assertIsInstance(decision, PermissionResultDeny)
        self.assertEqual(decision.message, "nope")

    async def test_settling_twice_is_a_no_op(self):
        session = self._session()
        callback = asyncio.create_task(
            session._on_can_use_tool("Bash", {}, ToolPermissionContext())
        )
        await asyncio.wait_for(session._event_queue.get(), timeout=1)

        self.assertTrue(session._settle_pending_permission(PermissionResultAllow()))
        # A second approval (double-tap in the app, or a close racing a reply)
        # must not explode on an already-resolved future.
        self.assertFalse(session._settle_pending_permission(PermissionResultAllow()))
        await asyncio.wait_for(callback, timeout=1)

    async def test_settle_without_a_pending_request(self):
        session = self._session()
        self.assertFalse(session._settle_pending_permission(PermissionResultAllow()))

    async def test_close_releases_a_parked_callback(self):
        session = self._session()
        callback = asyncio.create_task(
            session._on_can_use_tool("Bash", {}, ToolPermissionContext())
        )
        await asyncio.wait_for(session._event_queue.get(), timeout=1)

        # Without this the SDK's task would outlive the connection awaiting a
        # decision that can never arrive.
        await session.close()
        decision = await asyncio.wait_for(callback, timeout=1)
        self.assertIsInstance(decision, PermissionResultDeny)

    async def test_abort_denies_a_permission_parked_turn(self):
        session = self._session()
        session._turn_in_progress = True
        session._client = object()  # abort only needs "a connection exists"
        callback = asyncio.create_task(
            session._on_can_use_tool("Bash", {}, ToolPermissionContext())
        )
        await asyncio.wait_for(session._event_queue.get(), timeout=1)

        aborted = await session.abort_current_turn()
        self.assertTrue(aborted)
        decision = await asyncio.wait_for(callback, timeout=1)
        self.assertIsInstance(decision, PermissionResultDeny)
        self.assertFalse(session._turn_in_progress)
        session._client = None

    async def test_abort_without_a_turn_reports_false(self):
        session = self._session()
        self.assertFalse(await session.abort_current_turn())


class PendingFlagTest(unittest.IsolatedAsyncioTestCase):
    async def test_has_pending_permission_denials_tracks_the_request(self):
        session = ClaudeSession(project_path="/tmp")
        self.assertFalse(session.has_pending_permission_denials)
        session._pending_permission_request = {"type": "control_request"}
        self.assertTrue(session.has_pending_permission_denials)


if __name__ == "__main__":
    unittest.main()
