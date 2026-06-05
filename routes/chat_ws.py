"""WebSocket chat routes for LLM streaming."""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from agent.agent_store import get_agent_store
from chat.chat_stream_service import stream_claude_turn
from chat.chat_ws_service import (
    GLOBAL_CHAT_DISPLAY_NAME,
    GLOBAL_CHAT_PROJECT_NAME,
    create_chat_session_for_current_server,
    process_disconnect_server_message_for_current_server,
    process_firebase_auth_message_for_current_server,
    resolve_chat_provider_selection_for_current_server,
    validate_chat_websocket_access_for_current_server,
)
from pairing.pairing import get_pairing_service
from services.chat.ws_manager import get_ws_manager
from workspaces.workspace_store import get_workspace_store

from .deps import is_websocket_from_tunnel, start_periodic_reauth_task

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Get the global WebSocket connection manager
_ws_manager = get_ws_manager()


GLOBAL_CHAT_USAGE_CONTEXT = """You are the Code Bridge in-app assistant shown inside the main app after the mobile app is already connected to a desktop Code Bridge Server.

Current screen state:
- The server is already registered and connected.
- No project is selected yet.
- The user is on the main split screen, with preview on the left and chat on the right.
- Global help chat is for app guidance only. It cannot inspect files, edit code, run project commands, view git changes, or resume a native CLI session until a project is selected.

What users here are usually trying to understand:
- what to do next after pairing,
- why the center/left preview says no project selected,
- how to select an existing project or add a desktop folder as a project,
- what becomes available after project selection,
- how the model selector affects the right-side chat,
- what the right-side CLI-style chat can do once a project is active,
- where run/preview/log/device controls appear,
- why permissions, abort, history, and native CLI resume matter.

Answer in the user's language from the current app state. Do not start with server installation or pairing steps unless the user explicitly says the server is not connected or asks how to pair another device. If the user asks "what should I do now?", make the next action project selection/addition. Keep the answer short and concrete."""


def _is_global_chat(project_name: str) -> bool:
    return project_name == GLOBAL_CHAT_PROJECT_NAME


def _with_global_chat_context(project_name: str, user_message: str) -> str:
    if not _is_global_chat(project_name):
        return user_message
    return f"{GLOBAL_CHAT_USAGE_CONTEXT}\n\nUser question:\n{user_message}"


# WS-4: backpressure guard tuning.
#
# These thresholds protect the server from a slow consumer pinning unbounded
# memory in the asyncio send buffer. The serialized send wrapper measures the
# JSON-encoded payload length per send, accumulates an "inflight" gauge while
# the underlying ``websocket.send_json`` is awaited, and decrements on
# completion. When the gauge stays above ``WS_BACKPRESSURE_HIGH_WATERMARK_BYTES``
# for longer than ``WS_BACKPRESSURE_SLOW_CONSUMER_TIMEOUT_S``, we close the
# socket with WebSocket code 1011 (server error) and reason ``slow_consumer``.
#
# We do not attempt true OS-level flow control here (that would require socket
# buffer hooks per platform); we instead add a small ``asyncio.sleep`` between
# sends while the gauge is high so the LLM stream loop is throttled.
WS_BACKPRESSURE_HIGH_WATERMARK_BYTES = 4 * 1024 * 1024  # 4 MiB
WS_BACKPRESSURE_SLEEP_S = 0.05
WS_BACKPRESSURE_SLOW_CONSUMER_TIMEOUT_S = 5.0
_WS_SLOW_CONSUMER_CLOSE_CODE = 1011
_WS_SLOW_CONSUMER_CLOSE_REASON = "slow_consumer"


class _SerializedWebSocket:
    """Serialize concurrent sends while a turn task and receive loop coexist.

    Also enforces a soft backpressure guard (WS-4): once the unacked
    inflight-send byte count crosses ``WS_BACKPRESSURE_HIGH_WATERMARK_BYTES``
    we log a warning and introduce a small sleep before the next send to
    throttle the LLM producer loop. If the high-water condition persists for
    longer than ``WS_BACKPRESSURE_SLOW_CONSUMER_TIMEOUT_S`` we close the
    underlying socket with code 1011 and reason ``slow_consumer``.
    """

    def __init__(
        self,
        websocket: WebSocket,
        *,
        high_watermark_bytes: int = WS_BACKPRESSURE_HIGH_WATERMARK_BYTES,
        slow_consumer_timeout_s: float = WS_BACKPRESSURE_SLOW_CONSUMER_TIMEOUT_S,
        backpressure_sleep_s: float = WS_BACKPRESSURE_SLEEP_S,
    ):
        self._websocket = websocket
        self._send_lock = asyncio.Lock()
        self._inflight_bytes = 0
        self._high_watermark_bytes = high_watermark_bytes
        self._slow_consumer_timeout_s = slow_consumer_timeout_s
        self._backpressure_sleep_s = backpressure_sleep_s
        # Monotonic timestamp (asyncio loop time) at which the inflight gauge
        # first crossed the high-water mark; reset to ``None`` once it drops
        # back below the watermark.
        self._high_watermark_since: float | None = None
        self._slow_consumer_closed = False

    def _estimate_payload_bytes(self, data: Any) -> int:
        try:
            return len(json.dumps(data, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            # Fall back to a small constant if data cannot be serialized; the
            # underlying send_json will surface the real error.
            return 0

    async def send_json(self, data: Any) -> None:
        if self._slow_consumer_closed:
            # Once we have decided the consumer is slow we stop trying to push
            # more data; the close handshake tears down the receive loop.
            return

        payload_bytes = self._estimate_payload_bytes(data)
        # Increment _before_ acquiring the lock so payloads queued behind a
        # blocked underlying send_json count toward the inflight gauge.
        self._inflight_bytes += payload_bytes
        loop = asyncio.get_event_loop()
        now = loop.time()
        if self._inflight_bytes >= self._high_watermark_bytes:
            if self._high_watermark_since is None:
                self._high_watermark_since = now
                logger.warning(
                    "WS send buffer high, slow client suspected",
                    extra={
                        "inflight_bytes": self._inflight_bytes,
                        "high_watermark_bytes": self._high_watermark_bytes,
                    },
                )

        try:
            async with self._send_lock:
                if self._slow_consumer_closed:
                    return

                # Re-evaluate after acquiring the lock so we measure how long
                # we have spent above the watermark including any time spent
                # waiting on the serialized send lock.
                now = loop.time()
                if (
                    self._high_watermark_since is not None
                    and now - self._high_watermark_since > self._slow_consumer_timeout_s
                ):
                    self._slow_consumer_closed = True
                    logger.warning(
                        "WS slow consumer threshold exceeded, closing",
                        extra={
                            "inflight_bytes": self._inflight_bytes,
                            "elapsed_high_s": now - self._high_watermark_since,
                        },
                    )
                    try:
                        await self._websocket.close(
                            code=_WS_SLOW_CONSUMER_CLOSE_CODE,
                            reason=_WS_SLOW_CONSUMER_CLOSE_REASON,
                        )
                    except Exception:
                        logger.debug("WS close after slow consumer failed", exc_info=True)
                    return

                if self._inflight_bytes >= self._high_watermark_bytes:
                    # Soft producer throttle while still in the high-water
                    # band: yield 50 ms so the LLM stream loop slows down and
                    # the OS socket buffer has time to drain.
                    await asyncio.sleep(self._backpressure_sleep_s)

                await self._websocket.send_json(data)
        finally:
            self._inflight_bytes = max(0, self._inflight_bytes - payload_bytes)
            if self._inflight_bytes < self._high_watermark_bytes:
                self._high_watermark_since = None

    async def receive_text(self) -> str:
        return await self._websocket.receive_text()

    async def accept(self) -> None:
        await self._websocket.accept()

    async def close(self, *args: Any, **kwargs: Any) -> None:
        await self._websocket.close(*args, **kwargs)

    @property
    def url(self):
        return self._websocket.url


class _AgentRecordingWebSocket:
    """Forward websocket sends while recording them as durable agent events."""

    def __init__(
        self,
        websocket: WebSocket | _SerializedWebSocket,
        *,
        run_id: str,
        agent_store: Any,
    ) -> None:
        self._websocket = websocket
        self._run_id = run_id
        self._agent_store = agent_store

    @property
    def agent_run_id(self) -> str:
        return self._run_id

    async def send_json(self, data: Any) -> None:
        await self._websocket.send_json(data)
        if not isinstance(data, dict):
            return
        try:
            event_type = str(data.get("type") or "websocket")
            provider_id = data.get("provider_id")
            self._agent_store.append_event(
                run_id=self._run_id,
                event_type=event_type,
                provider_id=provider_id if isinstance(provider_id, str) else None,
                provider_event=data if event_type == "provider_event" else None,
                app_event=data if event_type != "provider_event" else data.get("normalized"),
            )
        except Exception:
            logger.exception("failed to record agent event run_id=%s", self._run_id)

    async def receive_text(self) -> str:
        return await self._websocket.receive_text()

    async def accept(self) -> None:
        await self._websocket.accept()

    async def close(self, *args: Any, **kwargs: Any) -> None:
        await self._websocket.close(*args, **kwargs)

    @property
    def url(self):
        return self._websocket.url


async def _handle_user_message(
    websocket: WebSocket | _SerializedWebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
    *,
    agent_store: Any | None = None,
) -> None:
    user_message_raw = message.get("content", "")
    user_message = str(user_message_raw).strip()
    logger.debug("project=%s incoming user_message len=%d", project_name, len(user_message))
    if not user_message:
        await websocket.send_json({"type": "error", "message": "Message content is empty"})
        return

    stream_message = _with_global_chat_context(project_name, user_message)
    store = agent_store or get_agent_store()
    raw_task_id = message.get("task_id")
    task_id = raw_task_id if isinstance(raw_task_id, str) and raw_task_id.strip() else None
    raw_cwd = getattr(session, "project_path", None)
    cwd = raw_cwd if isinstance(raw_cwd, str) else str(raw_cwd) if raw_cwd else None
    workspace_id = None
    if cwd and not _is_global_chat(project_name):
        workspace = get_workspace_store().get_or_create_project_workspace(
            project_name=project_name,
            root_path=cwd,
            display_name=project_name,
        )
        workspace_id = workspace.get("id")
    run = store.create_run(
        project_name=None if _is_global_chat(project_name) else project_name,
        workspace_id=workspace_id if isinstance(workspace_id, str) else None,
        agent_id="agent_adhoc_dev",
        provider_id=getattr(session, "provider_id", None),
        model=getattr(session, "model", None),
        title=user_message[:80],
        goal=user_message,
        cwd=cwd,
        native_session_id=getattr(session, "session_id", None),
        task_id=task_id,
    )
    run_id = run["id"]
    store.add_message(run_id=run_id, role="user", content=user_message)

    recording_ws = _AgentRecordingWebSocket(
        websocket,
        run_id=run_id,
        agent_store=store,
    )
    await recording_ws.send_json({"type": "user_message", "content": user_message})
    try:
        completed = await stream_claude_turn(
            recording_ws,
            session,
            project_name=project_name,
            user_message=stream_message,
        )
    except Exception:
        store.update_run_status(run_id, "failed")
        raise
    store.update_run_status(run_id, "completed" if completed else "failed")


async def _handle_approve_permissions(
    websocket: WebSocket | _SerializedWebSocket,
    session,
    project_name: str,
) -> None:
    if not session.has_pending_permission_denials:
        await websocket.send_json({"type": "error", "message": "No pending permission request to approve"})
        return

    await websocket.send_json(
        {
            "type": "permission_retry_started",
            "message": "Permission approved. Continuing current turn...",
        }
    )
    await stream_claude_turn(
        websocket,
        session,
        project_name=project_name,
        retry_from_permission=True,
    )


async def _handle_deny_permissions(
    websocket: WebSocket | _SerializedWebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
) -> None:
    if not session.has_pending_permission_denials:
        await websocket.send_json({"type": "error", "message": "No pending permission request to deny"})
        return

    deny_message_raw = message.get("message", "Permission denied by user.")
    deny_message = (
        str(deny_message_raw).strip()
        if isinstance(deny_message_raw, str)
        else "Permission denied by user."
    )
    if not deny_message:
        deny_message = "Permission denied by user."

    await websocket.send_json(
        {
            "type": "permission_retry_started",
            "message": "Permission denied. Continuing current turn...",
        }
    )
    await stream_claude_turn(
        websocket,
        session,
        project_name=project_name,
        deny_from_permission_message=deny_message,
    )


async def _handle_abort_turn(
    websocket: WebSocket | _SerializedWebSocket,
    session,
    project_name: str,
) -> None:
    """Handle abort request to stop the current turn."""
    try:
        aborted = await session.abort_current_turn()
        if aborted:
            await websocket.send_json({
                "type": "turn_aborted",
                "message": "Turn aborted by user",
            })
            logger.info("project=%s turn aborted", project_name)
        else:
            await websocket.send_json({
                "type": "error",
                "message": "No turn in progress to abort",
            })
    except OSError as e:
        # Send a generic message to the client; full exception detail
        # belongs in server logs only, not the WebSocket payload.
        logger.exception("Failed to abort turn for project=%s", project_name)
        del e
        await websocket.send_json({
            "type": "error",
            "message": "Failed to abort turn",
        })


async def _handle_firebase_auth_message(
    websocket: WebSocket | _SerializedWebSocket,
    message: dict[str, Any],
    *,
    local_port: int,
) -> None:
    result = await process_firebase_auth_message_for_current_server(
        message,
        local_port=local_port,
    )
    await websocket.send_json(result.payload)
    if result.log_message:
        logger.info("%s", result.log_message)


async def _handle_disconnect_server_message(websocket: WebSocket | _SerializedWebSocket) -> None:
    result = await process_disconnect_server_message_for_current_server()
    await websocket.send_json(result.payload)
    if result.log_message:
        logger.info("%s", result.log_message)


async def _handle_ping(websocket: WebSocket | _SerializedWebSocket, api_key: Optional[str]) -> None:
    """Handle ping message and update client last_used timestamp."""
    if api_key:
        get_pairing_service().touch_api_key(api_key)
    await websocket.send_json({"type": "pong"})


async def _dispatch_chat_message(
    websocket: WebSocket | _SerializedWebSocket,
    session,
    project_name: str,
    message: dict[str, Any],
    *,
    local_port: int,
    api_key: Optional[str] = None,
) -> None:
    """Dispatch websocket chat message to corresponding handler."""
    message_type = message.get("type")
    logger.debug("project=%s message_type=%s", project_name, message_type)
    handlers: dict[str, Callable[[], Awaitable[None]]] = {
        "message": lambda: _handle_user_message(websocket, session, project_name, message),
        "approve_permissions": lambda: _handle_approve_permissions(websocket, session, project_name),
        "deny_permissions": lambda: _handle_deny_permissions(websocket, session, project_name, message),
        "abort": lambda: _handle_abort_turn(websocket, session, project_name),
        "ping": lambda: _handle_ping(websocket, api_key),
        "firebase_auth": lambda: _handle_firebase_auth_message(
            websocket,
            message,
            local_port=local_port,
        ),
        "disconnect_server": lambda: _handle_disconnect_server_message(websocket),
    }

    handler = handlers.get(message_type)
    if handler is None:
        # Keep the JSON error so existing clients still get a payload,
        # but close with policy-violation (1008) so misbehaving senders stop.
        await websocket.send_json(
            {"type": "error", "message": f"Unknown message type: {message_type}"}
        )
        try:
            await websocket.close(code=1008, reason="unknown message type")
        except Exception:
            logger.debug("project=%s close after unknown message failed", project_name)
        return
    await handler()


def _is_turn_message(message_type: Any) -> bool:
    return message_type in {"message", "approve_permissions", "deny_permissions"}


def _consume_turn_task_result(task: asyncio.Task[None], project_name: str) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info("project=%s active turn task cancelled", project_name)
    except Exception:
        logger.exception("project=%s active turn task failed", project_name)


@router.websocket("/ws/chat")
@router.websocket("/ws/claude")
async def global_chat_websocket(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None),
) -> None:
    await _chat_websocket_impl(websocket, GLOBAL_CHAT_PROJECT_NAME, api_key)


@router.websocket("/ws/chat/{project_name}")
@router.websocket("/ws/claude/{project_name}")  # Backwards compatibility alias
async def project_chat_websocket(
    websocket: WebSocket,
    project_name: str,
    api_key: Optional[str] = Query(None),
) -> None:
    await _chat_websocket_impl(websocket, project_name, api_key)


async def _chat_websocket_impl(
    websocket: WebSocket,
    project_name: str,
    api_key: Optional[str],
) -> None:
    """WebSocket endpoint for LLM chat communication.

    Includes connection limits and rate limiting for security.
    """
    # Check connection limits before accepting
    can_connect, reject_reason = _ws_manager.can_connect(websocket)
    if not can_connect:
        await websocket.close(code=1008, reason=reject_reason)
        logger.warning("rejected connection: %s", reject_reason)
        return

    access = validate_chat_websocket_access_for_current_server(
        api_key,
        project_name,
        from_tunnel=is_websocket_from_tunnel(websocket),
    )
    if not access.success:
        if access.close_code is not None:
            await websocket.close(code=access.close_code, reason=access.close_reason or "")
            return

        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": access.error_message or "Project access denied",
            }
        )
        await websocket.close()
        return

    await websocket.accept()
    _ws_manager.register_connection(websocket)
    ws = _SerializedWebSocket(websocket)
    logger.info("accepted project=%s path=%s", project_name, websocket.url.path)

    active_turn_task: asyncio.Task[None] | None = None
    # WS-1: periodic re-auth so revoked api keys terminate the live stream
    # rather than waiting for next handshake.
    reauth_task = start_periodic_reauth_task(ws, api_key)
    try:
        project_path = access.project_path or ""
        local_port = access.local_port or 0

        selection_result = resolve_chat_provider_selection_for_current_server()
        if not selection_result.success or selection_result.selection is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": selection_result.error_message or "Failed to resolve provider",
                }
            )
            await websocket.close()
            return

        provider_name = selection_result.provider_name or "LLM"
        await websocket.send_json({"type": "status", "message": f"Connecting to {provider_name}..."})
        if _is_global_chat(project_name):
            await websocket.send_json(
                {
                    "type": "status",
                    "message": (
                        f"{GLOBAL_CHAT_DISPLAY_NAME}: project-specific file work "
                        "and resume are available after selecting a project."
                    ),
                }
            )

        logger.info("[chat_ws] creating session project=%s path=%r", project_name, project_path)
        session_result = await create_chat_session_for_current_server(
            project_name,
            project_path,
            selection_result.selection,
        )
        logger.info("[chat_ws] session create result success=%s error=%s", session_result.success, session_result.error_message)
        if not session_result.success or session_result.session is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": session_result.error_message or "Failed to create chat session",
                }
            )
            await websocket.close()
            return

        session = session_result.session
        await ws.send_json({"type": "status", "message": f"Connected to {provider_name}"})
        logger.info("project=%s connected provider=%s", project_name, provider_name)
        while True:
            try:
                if active_turn_task is not None and active_turn_task.done():
                    _consume_turn_task_result(active_turn_task, project_name)
                    active_turn_task = None

                data = await ws.receive_text()
                message = json.loads(data)

                # Check message rate limit and re-resolve provider for user messages
                message_type = message.get("type")
                if (
                    _is_turn_message(message_type)
                    and active_turn_task is not None
                    and not active_turn_task.done()
                ):
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "A turn is already in progress.",
                        }
                    )
                    continue

                if message_type == "message":
                    # Rate limit check
                    can_send, rate_reason = _ws_manager.can_send_message(websocket)
                    if not can_send:
                        await ws.send_json({
                            "type": "error",
                            "message": rate_reason,
                        })
                        continue

                    # Re-resolve provider for each new user turn so LLM selection changes
                    # take effect without requiring websocket reconnect.
                    latest_selection = resolve_chat_provider_selection_for_current_server()
                    if not latest_selection.success or latest_selection.selection is None:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": latest_selection.error_message
                                or "Failed to resolve provider",
                            }
                        )
                        continue

                    latest_provider_name = latest_selection.provider_name or "LLM"
                    latest_session_result = await create_chat_session_for_current_server(
                        project_name,
                        project_path,
                        latest_selection.selection,
                    )
                    if not latest_session_result.success or latest_session_result.session is None:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": latest_session_result.error_message
                                or "Failed to create chat session",
                            }
                        )
                        continue

                    session = latest_session_result.session
                    if latest_provider_name != provider_name:
                        provider_name = latest_provider_name
                        await ws.send_json(
                            {
                                "type": "status",
                                "message": f"Switched to {provider_name}",
                            }
                        )
                        logger.info(
                            "project=%s provider_switched=%s", project_name, provider_name
                        )

                if _is_turn_message(message_type):
                    active_turn_task = asyncio.create_task(
                        _dispatch_chat_message(
                            ws,
                            session,
                            project_name,
                            message,
                            local_port=local_port,
                            api_key=api_key,
                        )
                    )
                    active_turn_task.add_done_callback(
                        lambda task: _consume_turn_task_result(task, project_name)
                    )
                    continue

                await _dispatch_chat_message(
                    ws,
                    session,
                    project_name,
                    message,
                    local_port=local_port,
                    api_key=api_key,
                )

            except WebSocketDisconnect:
                logger.info("project=%s disconnected", project_name)
                break
            except json.JSONDecodeError:
                logger.warning("project=%s invalid_json", project_name)
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
    finally:
        if active_turn_task is not None and not active_turn_task.done():
            active_turn_task.cancel()
        if reauth_task is not None and not reauth_task.done():
            reauth_task.cancel()
        # Always unregister connection when done
        _ws_manager.unregister_connection(websocket)
