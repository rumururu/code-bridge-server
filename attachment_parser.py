"""Parse attachments from chat messages and convert to multimodal content."""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Supported image MIME types for multimodal
IMAGE_MIME_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})

# Maximum image size (20MB - Claude's limit)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024


@dataclass
class ParsedAttachment:
    """Parsed attachment information."""

    kind: str  # "Photo", "File", "Camera"
    filename: str
    project_path: str
    size_bytes: int


def parse_attachments_from_message(message: str) -> tuple[str, list[ParsedAttachment]]:
    """Extract attachments section from message and return clean message + attachments.

    Returns:
        Tuple of (message_without_attachments, list_of_attachments)
    """
    # Pattern to match the attachments section
    pattern = r"\[Uploaded Attachments\].*?(?=\n\n|\Z)"
    match = re.search(pattern, message, re.DOTALL)

    if not match:
        return message, []

    attachments_section = match.group(0)
    clean_message = message[: match.start()].strip()

    # Parse individual attachments
    attachments: list[ParsedAttachment] = []

    # Pattern for each attachment block
    attachment_pattern = r"- (Photo|File|Camera): (.+?)\n\s+- project_path: (.+?)\n\s+- size_bytes: (\d+)"
    for m in re.finditer(attachment_pattern, attachments_section):
        attachments.append(
            ParsedAttachment(
                kind=m.group(1),
                filename=m.group(2).strip(),
                project_path=m.group(3).strip(),
                size_bytes=int(m.group(4)),
            )
        )

    return clean_message, attachments


def attachment_to_multimodal_block(
    attachment: ParsedAttachment,
    project_path: str,
) -> dict[str, Any] | None:
    """Convert attachment to Claude multimodal content block.

    Returns None if the file cannot be converted (not an image, too large, etc.)
    """
    # Resolve the full file path
    full_path = Path(project_path) / attachment.project_path.lstrip("/")

    if not full_path.exists():
        return None

    # Check file size
    file_size = full_path.stat().st_size
    if file_size > MAX_IMAGE_SIZE_BYTES:
        return None

    # Get MIME type
    mime_type, _ = mimetypes.guess_type(str(full_path))
    if mime_type not in IMAGE_MIME_TYPES:
        # Not an image, keep as text reference for Claude to read via tool
        return None

    # Read and encode as base64
    try:
        with open(full_path, "rb") as f:
            image_data = f.read()
        base64_data = base64.standard_b64encode(image_data).decode("utf-8")
    except Exception:
        return None

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_data,
        },
    }


def build_multimodal_content(
    message: str,
    project_path: str,
) -> str | list[dict[str, Any]]:
    """Parse message for attachments and build multimodal content.

    Returns:
        Either the original message string (no image attachments)
        or a list of content blocks for multimodal
    """
    clean_message, attachments = parse_attachments_from_message(message)

    if not attachments:
        return message

    # Build content blocks
    content_blocks: list[dict[str, Any]] = []

    # Add text block first
    if clean_message:
        content_blocks.append({"type": "text", "text": clean_message})

    # Process attachments
    non_image_attachments: list[ParsedAttachment] = []

    for attachment in attachments:
        image_block = attachment_to_multimodal_block(attachment, project_path)
        if image_block:
            content_blocks.append(image_block)
        else:
            # Not an image or couldn't convert - keep for text reference
            non_image_attachments.append(attachment)

    # Add text reference for non-image attachments
    if non_image_attachments:
        ref_text = "\n\n[Non-image Attachments - use Read tool to inspect]\n"
        for att in non_image_attachments:
            ref_text += f"- {att.kind}: {att.filename}\n"
            ref_text += f"  - project_path: {att.project_path}\n"
        content_blocks.append({"type": "text", "text": ref_text})

    # If only text blocks (no images converted), return plain string
    if all(b.get("type") == "text" for b in content_blocks):
        return message

    return content_blocks
