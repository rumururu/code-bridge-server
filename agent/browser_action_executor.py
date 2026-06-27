"""Workflow-facing browser action execution helper."""

from __future__ import annotations

from typing import Any

from .browser_action_adapter import (
    BrowserActionAdapter,
    BrowserActionAdapterResult,
    get_browser_action_adapter,
)


async def execute_browser_actions(
    actions: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    adapter: BrowserActionAdapter | None = None,
) -> BrowserActionAdapterResult:
    """Execute normalized browser actions through an injectable adapter."""
    if not actions:
        return BrowserActionAdapterResult(
            status="waiting_for_user",
            wait_reason="browser_actions_missing",
            prompt="이 브라우저 단계에 실행할 action이 없습니다. Builder에서 action을 추가한 뒤 다시 실행하세요.",
        )
    runner = adapter or get_browser_action_adapter()
    return await runner.run_actions(actions, context=context)
