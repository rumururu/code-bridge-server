"""LLM provider/model settings and runtime availability checks."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from database import get_settings_db

SELECTED_COMPANY_KEY = "llm.selected_company"
SELECTED_MODEL_KEY = "llm.selected_model"
CUSTOM_MODELS_PREFIX = "llm.custom_models."

# `chat_supported` means current Code Bridge backend can actually use this
# provider for chat turns today.
LLM_COMPANIES: dict[str, dict[str, Any]] = {
    "anthropic": {
        "name": "Anthropic",
        "command": "claude",
        "chat_supported": True,
        "builtin_models": [
            "claude-sonnet-4-5",
            "claude-opus-4-1",
        ],
    },
    "openai": {
        "name": "OpenAI",
        "command": "codex",
        "chat_supported": False,
        "builtin_models": [
            "codex",
            "gpt-5",
            "gpt-5-codex",
        ],
    },
}


def _custom_models_key(company_id: str) -> str:
    return f"{CUSTOM_MODELS_PREFIX}{company_id}"


def _normalize_model_name(value: str) -> str:
    normalized = value.strip()
    if len(normalized) > 128:
        normalized = normalized[:128]
    return normalized


def _dedupe_models(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw_value in values:
        normalized = _normalize_model_name(raw_value)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


def _load_custom_models(company_id: str) -> list[str]:
    settings_db = get_settings_db()
    raw = settings_db.get_json(_custom_models_key(company_id), [])
    if not isinstance(raw, list):
        return []
    values = [str(item) for item in raw if isinstance(item, str) or isinstance(item, (int, float))]
    return _dedupe_models(values)


def _save_custom_models(company_id: str, values: list[str]) -> None:
    settings_db = get_settings_db()
    settings_db.set_json(_custom_models_key(company_id), _dedupe_models(values))


def _probe_provider_connection(command_name: str) -> tuple[bool, str | None]:
    command_path = shutil.which(command_name)
    if not command_path:
        return False, f"{command_name} CLI not found"

    try:
        result = subprocess.run(
            [command_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=4,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"{command_name} --version timed out"
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        return False, str(exc)

    if result.returncode != 0:
        raw_error = (result.stderr or result.stdout or "").strip()
        if raw_error:
            return False, raw_error.splitlines()[-1][:240]
        return False, f"{command_name} --version failed"

    return True, None


def _build_company_snapshot(company_id: str, config: dict[str, Any]) -> dict[str, Any]:
    builtin_models = _dedupe_models([str(model) for model in config.get("builtin_models", [])])
    custom_models = _load_custom_models(company_id)
    all_models = builtin_models + [value for value in custom_models if value.lower() not in {m.lower() for m in builtin_models}]

    models = [
        {"id": model, "label": model, "source": "builtin"}
        for model in builtin_models
    ] + [
        {"id": model, "label": model, "source": "custom"}
        for model in custom_models
        if model.lower() not in {m.lower() for m in builtin_models}
    ]

    connected, reason = _probe_provider_connection(str(config.get("command", "")))
    chat_supported = bool(config.get("chat_supported", False))

    return {
        "id": company_id,
        "name": str(config.get("name", company_id)),
        "command": str(config.get("command", "")),
        "connected": connected,
        "chat_supported": chat_supported,
        "selectable": connected and chat_supported,
        "status_message": reason,
        "models": models,
        "all_model_ids": all_models,
    }


def _pick_default_selection(companies: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for company in companies:
        if not company.get("selectable"):
            continue
        model_ids = company.get("all_model_ids") or []
        if model_ids:
            return str(company["id"]), str(model_ids[0])
    return None, None


def get_llm_options_snapshot() -> dict[str, Any]:
    """Return provider/model options with runtime availability information."""
    settings_db = get_settings_db()
    companies = [
        _build_company_snapshot(company_id, config)
        for company_id, config in LLM_COMPANIES.items()
    ]

    selected_company = settings_db.get(SELECTED_COMPANY_KEY)
    selected_model = settings_db.get(SELECTED_MODEL_KEY)

    selected_company_snapshot = next(
        (company for company in companies if company["id"] == selected_company),
        None,
    )
    selected_is_valid = (
        selected_company_snapshot is not None
        and bool(selected_company_snapshot.get("selectable"))
        and isinstance(selected_model, str)
        and selected_model in (selected_company_snapshot.get("all_model_ids") or [])
    )

    if not selected_is_valid:
        selected_company, selected_model = _pick_default_selection(companies)
        settings_db.set(SELECTED_COMPANY_KEY, selected_company)
        settings_db.set(SELECTED_MODEL_KEY, selected_model)

    return {
        "selected": {
            "company_id": selected_company,
            "model": selected_model,
        },
        "companies": [
            {
                "id": company["id"],
                "name": company["name"],
                "command": company["command"],
                "connected": company["connected"],
                "chat_supported": company["chat_supported"],
                "selectable": company["selectable"],
                "status_message": company["status_message"],
                "models": company["models"],
            }
            for company in companies
        ],
    }


def set_selected_llm(company_id: str, model: str) -> dict[str, Any]:
    """Persist selected company/model after validating availability."""
    normalized_company = company_id.strip().lower()
    normalized_model = _normalize_model_name(model)

    if normalized_company not in LLM_COMPANIES:
        raise ValueError("Unknown LLM company")
    if not normalized_model:
        raise ValueError("Model is required")

    snapshot = get_llm_options_snapshot()
    companies = snapshot.get("companies", [])
    company_snapshot = next((item for item in companies if item.get("id") == normalized_company), None)
    if company_snapshot is None:
        raise ValueError("Unknown LLM company")

    if not company_snapshot.get("selectable"):
        reason = company_snapshot.get("status_message") or "Provider is not selectable"
        raise ValueError(reason)

    allowed_models = {
        str(item.get("id"))
        for item in (company_snapshot.get("models") or [])
        if isinstance(item, dict) and item.get("id")
    }
    if normalized_model not in allowed_models:
        raise ValueError("Unknown model for selected company")

    settings_db = get_settings_db()
    settings_db.set(SELECTED_COMPANY_KEY, normalized_company)
    settings_db.set(SELECTED_MODEL_KEY, normalized_model)
    return get_llm_options_snapshot()


def add_custom_llm_model(company_id: str, model: str) -> dict[str, Any]:
    """Add a custom model to one company."""
    normalized_company = company_id.strip().lower()
    normalized_model = _normalize_model_name(model)

    if normalized_company not in LLM_COMPANIES:
        raise ValueError("Unknown LLM company")
    if not normalized_model:
        raise ValueError("Model is required")

    builtin_models = {
        value.lower()
        for value in _dedupe_models([str(item) for item in LLM_COMPANIES[normalized_company].get("builtin_models", [])])
    }
    custom_models = _load_custom_models(normalized_company)
    if normalized_model.lower() in builtin_models:
        return get_llm_options_snapshot()

    if normalized_model.lower() not in {value.lower() for value in custom_models}:
        custom_models.append(normalized_model)
        _save_custom_models(normalized_company, custom_models)

    return get_llm_options_snapshot()


def remove_custom_llm_model(company_id: str, model: str) -> dict[str, Any]:
    """Remove one custom model from a company."""
    normalized_company = company_id.strip().lower()
    normalized_model = _normalize_model_name(model)

    if normalized_company not in LLM_COMPANIES:
        raise ValueError("Unknown LLM company")
    if not normalized_model:
        raise ValueError("Model is required")

    custom_models = _load_custom_models(normalized_company)
    next_models = [value for value in custom_models if value.lower() != normalized_model.lower()]
    _save_custom_models(normalized_company, next_models)

    settings_db = get_settings_db()
    selected_company = settings_db.get(SELECTED_COMPANY_KEY)
    selected_model = settings_db.get(SELECTED_MODEL_KEY)
    if selected_company == normalized_company and isinstance(selected_model, str):
        if selected_model.lower() == normalized_model.lower():
            settings_db.set(SELECTED_MODEL_KEY, None)

    return get_llm_options_snapshot()
