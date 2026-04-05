"""Helpers for normalizing and extracting LLM chat payload content."""

from __future__ import annotations

import json
from typing import Any


def format_tool_result_content(content: Any) -> str:
    """Format tool_result content from event payloads."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue

            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_value = item.get("text", "").strip()
                    if text_value:
                        parts.append(text_value)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
                continue

            parts.append(str(item))
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content.get("text", "").strip()
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def extract_assistant_text(message: dict[str, Any]) -> str:
    """Extract plain text blocks from an assistant message payload."""
    blocks = message.get("content")
    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)
