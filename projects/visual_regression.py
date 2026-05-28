"""Visual comparison helpers for captured preview screenshots."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageChops
except ImportError:  # pragma: no cover - Pillow is installed through qrcode[pil].
    Image = None
    ImageChops = None

WARN_DIFF_RATIO = 0.01
FAIL_DIFF_RATIO = 0.05


def build_screenshot_visual_summary(
    current_path: str,
    *,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    """Summarize one screenshot and optionally compare it to a prior baseline."""
    current = _image_fingerprint(current_path)
    summary: dict[str, Any] = {
        "status": "baseline" if not baseline_path else "compared",
        "method": "pillow" if Image is not None else "hash",
        "current_path": current_path,
        "baseline_path": baseline_path,
        "current": current,
        "changed": False if not baseline_path else None,
        "diff_ratio": 0.0 if not baseline_path else None,
    }

    if not current.get("exists"):
        summary.update(
            {
                "status": "unavailable",
                "changed": None,
                "diff_ratio": None,
                "reason": current.get("error") or "current screenshot unavailable",
            }
        )
        return _with_thresholds(summary)

    if not baseline_path:
        return _with_thresholds(summary)

    baseline = _image_fingerprint(baseline_path)
    summary["baseline"] = baseline
    if not baseline.get("exists"):
        summary.update(
            {
                "status": "baseline_missing",
                "changed": None,
                "diff_ratio": None,
                "reason": baseline.get("error") or "baseline screenshot unavailable",
            }
        )
        return _with_thresholds(summary)

    if baseline.get("sha256") == current.get("sha256"):
        summary.update(
            {
                "changed": False,
                "diff_ratio": 0.0,
                "diff_pixels": 0,
                "max_channel_delta": 0,
                "average_channel_delta": 0.0,
            }
        )
        return _with_thresholds(summary)

    if Image is None or ImageChops is None:
        summary.update(
            {
                "changed": True,
                "diff_ratio": 1.0,
                "reason": "pillow unavailable; hash changed",
            }
        )
        return _with_thresholds(summary)

    return _compare_with_pillow(summary, baseline_path, current_path)


def _image_fingerprint(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path.is_file():
        return {
            "exists": False,
            "error": "file not found",
        }

    try:
        data = path.read_bytes()
    except OSError as exc:
        return {
            "exists": False,
            "error": str(exc),
        }

    info: dict[str, Any] = {
        "exists": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if Image is None:
        return info

    try:
        with Image.open(path) as image:
            info.update(
                {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            )
    except OSError as exc:
        info["decode_error"] = str(exc)
    return info


def _compare_with_pillow(
    summary: dict[str, Any],
    baseline_path: str,
    current_path: str,
) -> dict[str, Any]:
    try:
        with Image.open(baseline_path) as baseline_image, Image.open(current_path) as current_image:
            baseline_rgb = baseline_image.convert("RGB")
            current_rgb = current_image.convert("RGB")
    except OSError as exc:
        summary.update(
            {
                "status": "unavailable",
                "changed": None,
                "diff_ratio": None,
                "reason": str(exc),
            }
        )
        return _with_thresholds(summary)

    if baseline_rgb.size != current_rgb.size:
        summary.update(
            {
                "changed": True,
                "diff_ratio": 1.0,
                "reason": "dimensions changed",
                "baseline_size": {
                    "width": baseline_rgb.width,
                    "height": baseline_rgb.height,
                },
                "current_size": {
                    "width": current_rgb.width,
                    "height": current_rgb.height,
                },
            }
        )
        return _with_thresholds(summary)

    diff = ImageChops.difference(baseline_rgb, current_rgb)
    bbox = diff.getbbox()
    total_pixels = baseline_rgb.width * baseline_rgb.height
    if bbox is None or total_pixels <= 0:
        summary.update(
            {
                "changed": False,
                "diff_ratio": 0.0,
                "diff_pixels": 0,
                "max_channel_delta": 0,
                "average_channel_delta": 0.0,
            }
        )
        return _with_thresholds(summary)

    diff_pixels = 0
    max_channel_delta = 0
    total_channel_delta = 0
    cropped_diff = diff.crop(bbox)
    pixel_data = (
        cropped_diff.get_flattened_data()
        if hasattr(cropped_diff, "get_flattened_data")
        else cropped_diff.getdata()
    )
    for pixel in pixel_data:
        pixel_delta = max(pixel)
        if pixel_delta:
            diff_pixels += 1
            max_channel_delta = max(max_channel_delta, pixel_delta)
            total_channel_delta += sum(pixel)

    summary.update(
        {
            "changed": diff_pixels > 0,
            "diff_ratio": diff_pixels / total_pixels,
            "diff_pixels": diff_pixels,
            "max_channel_delta": max_channel_delta,
            "average_channel_delta": total_channel_delta / (total_pixels * 3),
            "bounding_box": {
                "left": bbox[0],
                "top": bbox[1],
                "right": bbox[2],
                "bottom": bbox[3],
            },
        }
    )
    return _with_thresholds(summary)


def _with_thresholds(summary: dict[str, Any]) -> dict[str, Any]:
    summary["thresholds"] = {
        "warn_diff_ratio": WARN_DIFF_RATIO,
        "fail_diff_ratio": FAIL_DIFF_RATIO,
    }
    ratio = summary.get("diff_ratio")
    if not isinstance(ratio, int | float):
        summary["severity"] = "unknown"
    elif ratio >= FAIL_DIFF_RATIO:
        summary["severity"] = "fail"
    elif ratio >= WARN_DIFF_RATIO:
        summary["severity"] = "warn"
    else:
        summary["severity"] = "ok"
    return summary
