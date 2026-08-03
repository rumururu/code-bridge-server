"""LLM provider/model settings and runtime availability checks.

Dynamically detects installed CLI tools (claude, codex) on the user's PC
and provides their supported model aliases. Also reads user config files
to detect configured models.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.database import get_settings_db

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python

# Database keys
SELECTED_COMPANY_KEY = "llm.selected_company"
SELECTED_MODEL_KEY = "llm.selected_model"
CODEX_SANDBOX_MODE_KEY = "llm.codex.sandbox_mode"
# Providers the user has switched off. Stored as the exceptions rather than
# the whole set so a provider added in a later release starts on, which is
# what someone who has never heard of it expects.
DISABLED_COMPANIES_KEY = "llm.disabled_companies"

# Codex sandbox modes
CODEX_SANDBOX_MODES = [
    {"id": "read-only", "label": "Read Only", "description": "Can only read files, no modifications"},
    {"id": "workspace-write", "label": "Workspace Write", "description": "Can modify files in the project (default)"},
    {"id": "danger-full-access", "label": "Full Access", "description": "Full system access (use with caution)"},
]
DEFAULT_CODEX_SANDBOX_MODE = "workspace-write"

LLM_PROVIDER_INSTALL_METHODS: dict[str, list[dict[str, Any]]] = {
    "anthropic": [
        {
            "id": "npm",
            "label": "Install with npm",
            "command": ["npm", "install", "-g", "@anthropic-ai/claude-code"],
        },
    ],
    "openai": [
        {
            "id": "npm",
            "label": "Install with npm",
            "command": ["npm", "install", "-g", "@openai/codex"],
        },
    ],
    "google": [
        {
            "id": "brew",
            "label": "Install with Homebrew",
            "command": ["brew", "install", "gemini-cli"],
        },
        {
            "id": "npm",
            "label": "Install with npm",
            "command": ["npm", "install", "-g", "@google/gemini-cli"],
        },
    ],
    "antigravity": [
        {
            "id": "npm",
            "label": "Install with npm",
            "command": ["npm", "install", "-g", "@google/antigravity"],
        },
    ],
}


@dataclass
class LlmProvider:
    """Configuration for an LLM provider."""

    id: str
    name: str
    command: str  # CLI command to check availability
    chat_supported: bool  # Whether Code Bridge can use this for chat
    models: list[str] = field(default_factory=list)  # Model aliases (always available when CLI is installed)


# Provider configurations with their supported model aliases.
# These are CLI aliases that always work - no need for full model IDs.
BUILTIN_PROVIDERS: list[LlmProvider] = [
    LlmProvider(
        id="anthropic",
        name="Anthropic (Claude)",
        command="claude",
        chat_supported=True,
        models=[
            "sonnet",
            "opus",
            "haiku",
        ],
    ),
    LlmProvider(
        id="openai",
        name="OpenAI (Codex)",
        command="codex",
        chat_supported=True,
        models=[
            "o3",
            "o4-mini",
            "gpt-4.1",
        ],
    ),
    LlmProvider(
        id="google",
        name="Google (Gemini)",
        command="gemini",
        chat_supported=True,
        models=[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
    ),
    LlmProvider(
        id="antigravity",
        name="Google (Antigravity)",
        command="agy",
        chat_supported=True,
        models=[
            "gemini-3.1-pro-high",
            "gemini-3.6-flash-medium",
        ],
    ),
]


PROVIDERS = BUILTIN_PROVIDERS


def _get_provider(provider_id: str) -> LlmProvider | None:
    """Get provider config by ID."""
    normalized = provider_id.strip().lower()
    return next((p for p in PROVIDERS if p.id == normalized), None)


# --- Config File Detection ---


def _detect_codex_models_from_config() -> list[dict[str, str]]:
    """Read ~/.codex/config.toml to detect user's configured model.

    Returns list of dicts with 'id' and 'source' keys.
    """
    config_path = Path.home() / ".codex" / "config.toml"
    detected = []

    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            model = config.get("model")
            if isinstance(model, str) and model.strip():
                detected.append({
                    "id": model.strip(),
                    "source": "config",
                    "config_path": str(config_path),
                })
        except (OSError, tomllib.TOMLDecodeError):
            pass  # Ignore parse errors

    return detected


def _detect_claude_models_from_config() -> list[dict[str, str]]:
    """Read Claude Code settings to detect any model preferences.

    Claude Code uses model aliases (sonnet, opus, haiku) that always work.
    This function checks for any custom model settings in:
    1. ~/.claude/settings.json
    2. Environment variables (ANTHROPIC_MODEL)

    Returns list of dicts with 'id' and 'source' keys.
    """
    detected = []

    # Check environment variable
    env_model = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("CLAUDE_MODEL")
    if env_model and env_model.strip():
        detected.append({
            "id": env_model.strip(),
            "source": "env",
            "config_path": "ANTHROPIC_MODEL or CLAUDE_MODEL",
        })

    # Check ~/.claude/settings.json for model preference
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            # Check for model in settings (might be in different locations)
            model = settings.get("model") or settings.get("defaultModel")
            if isinstance(model, str) and model.strip():
                detected.append({
                    "id": model.strip(),
                    "source": "config",
                    "config_path": str(settings_path),
                })
        except (OSError, json.JSONDecodeError):
            pass  # Ignore parse errors

    return detected


def _detect_gemini_models_from_config() -> list[dict[str, str]]:
    """Read Gemini CLI settings/env to detect configured model preferences."""
    detected = []

    env_model = os.environ.get("GEMINI_MODEL")
    if env_model and env_model.strip():
        detected.append({
            "id": env_model.strip(),
            "source": "env",
            "config_path": "GEMINI_MODEL",
        })

    settings_path = Path.home() / ".gemini" / "settings.json"
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            model = settings.get("model") or settings.get("defaultModel")
            if isinstance(model, str) and model.strip():
                detected.append({
                    "id": model.strip(),
                    "source": "config",
                    "config_path": str(settings_path),
                })
        except (OSError, json.JSONDecodeError):
            pass

    return detected


# How long a model list from the CLI is trusted before asking again. The
# names change when the CLI updates, which is rare; a request-time probe is
# not worth what it costs (see below).
_ANTIGRAVITY_MODELS_TTL_SECONDS = 600.0
_antigravity_models_cache: tuple[float, list[dict[str, Any]]] | None = None
_antigravity_models_lock = threading.Lock()


def _get_antigravity_models() -> list[dict[str, Any]]:
    """Ask the CLI what it can run, at most once every few minutes.

    The first cut of this provider shipped a hand-written list, and every name
    in it was wrong — `agy models` reports the real ones, and they change with
    the CLI rather than with this file.

    The cache is not an optimisation, it is a correctness fix. This runs
    inside `get_llm_options_snapshot`, which sits on request paths (chat
    session creation, LLM commands) that execute on the event loop. A
    synchronous `subprocess.run` there stops the *whole server* answering for
    as long as the CLI takes — every port stays open and every request times
    out, which reads as "the server is up but my phone can't reach it". A
    slow `agy` turned one provider listing into a server-wide outage.
    """
    global _antigravity_models_cache

    fallback = [{"id": m, "label": m, "source": "builtin"} for m in ("gemini-3.1-pro-high", "gemini-3.6-flash-medium")]

    now = time.monotonic()
    cached = _antigravity_models_cache
    if cached is not None and now - cached[0] < _ANTIGRAVITY_MODELS_TTL_SECONDS:
        return cached[1]

    command = shutil.which("agy")
    if not command:
        _antigravity_models_cache = (now, fallback)
        return fallback

    # One prober at a time: without this, N concurrent requests that all miss
    # the cache each spawn their own CLI and each block, which is the same
    # outage with more processes.
    if not _antigravity_models_lock.acquire(blocking=False):
        return cached[1] if cached is not None else fallback
    try:
        result = subprocess.run(
            [command, "models"],
            capture_output=True,
            text=True,
            # Short on purpose: this is time the server spends answering
            # nobody. A CLI slower than this does not get to hold the
            # process hostage — we serve the last known list instead.
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        _antigravity_models_cache = (now, cached[1] if cached is not None else fallback)
        return _antigravity_models_cache[1]
    finally:
        _antigravity_models_lock.release()
    if result.returncode != 0:
        _antigravity_models_cache = (now, cached[1] if cached is not None else fallback)
        return _antigravity_models_cache[1]
    models = [
        {"id": line.strip(), "label": line.strip(), "source": "cli"}
        for line in (result.stdout or "").splitlines()
        if line.strip() and not line.strip().startswith(("Usage", "Available", "-"))
    ]
    resolved = models or fallback
    _antigravity_models_cache = (now, resolved)
    return resolved


def reset_antigravity_models_cache() -> None:
    """Forget the cached model list — for tests, and for a settings reload."""
    global _antigravity_models_cache
    _antigravity_models_cache = None


def _get_codex_models() -> list[dict[str, Any]]:
    """Get Codex models: detected from config + base list.

    Returns list of model dicts with id, label, source.
    """
    base_models = [
        {"id": "o3", "label": "o3 (Reasoning)", "source": "builtin"},
        {"id": "o4-mini", "label": "o4-mini (Fast)", "source": "builtin"},
        {"id": "gpt-4.1", "label": "GPT-4.1", "source": "builtin"},
    ]
    detected = _detect_codex_models_from_config()

    # Combine: detected models first (user's preference), then base models
    all_models = []
    seen_ids = set()

    for m in detected:
        if m["id"] not in seen_ids:
            all_models.append({"id": m["id"], "label": m["id"], "source": m["source"]})
            seen_ids.add(m["id"])

    for m in base_models:
        if m["id"] not in seen_ids:
            all_models.append({
                "id": m["id"],
                "label": m.get("label", m["id"]),
                "source": m["source"],
            })
            seen_ids.add(m["id"])

    return all_models


def _get_claude_models() -> list[dict[str, Any]]:
    """Get Claude models: detected from config + base aliases.

    Returns list of model dicts with id, label, source.
    """
    base_models = [
        {"id": "sonnet", "label": "Sonnet (Recommended)", "source": "builtin"},
        {"id": "opus", "label": "Opus", "source": "builtin"},
        {"id": "haiku", "label": "Haiku (Fast)", "source": "builtin"},
    ]
    detected = _detect_claude_models_from_config()

    # Combine: detected models first, then base models
    all_models = []
    seen_ids = set()

    for m in detected:
        if m["id"] not in seen_ids:
            all_models.append({"id": m["id"], "label": m["id"], "source": m["source"]})
            seen_ids.add(m["id"])

    for m in base_models:
        if m["id"] not in seen_ids:
            all_models.append(m)
            seen_ids.add(m["id"])

    return all_models


def _get_gemini_models() -> list[dict[str, Any]]:
    """Get Gemini models: detected from config/env + base aliases."""
    base_models = [
        {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro", "source": "builtin"},
        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash", "source": "builtin"},
    ]
    detected = _detect_gemini_models_from_config()

    all_models = []
    seen_ids = set()

    for m in detected:
        if m["id"] not in seen_ids:
            all_models.append({"id": m["id"], "label": m["id"], "source": m["source"]})
            seen_ids.add(m["id"])

    for m in base_models:
        if m["id"] not in seen_ids:
            all_models.append(m)
            seen_ids.add(m["id"])

    return all_models


# --- Provider Connection Check ---


# Cache for CLI availability checks (command -> (available, error_msg, timestamp))
_cli_cache: dict[str, tuple[bool, str | None, float]] = {}
_CLI_CACHE_TTL = 60.0  # Cache for 60 seconds
_cli_warmup_done = False


def _check_cli_available(command: str, *, use_cache: bool = True) -> tuple[bool, str | None]:
    """Check if CLI tool is installed and working.

    Results are cached for 60 seconds to improve dashboard responsiveness.
    """
    import time

    now = time.time()

    # Check cache first
    if use_cache and command in _cli_cache:
        available, error_msg, cached_at = _cli_cache[command]
        if now - cached_at < _CLI_CACHE_TTL:
            return available, error_msg

    path = shutil.which(command)
    if not path:
        result = (False, f"{command} CLI not installed")
        _cli_cache[command] = (result[0], result[1], now)
        return result

    try:
        proc_result = subprocess.run(
            [path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=4,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result = (False, f"{command} --version timed out")
        _cli_cache[command] = (result[0], result[1], now)
        return result
    except OSError as e:
        result = (False, str(e))
        _cli_cache[command] = (result[0], result[1], now)
        return result

    if proc_result.returncode != 0:
        error = (proc_result.stderr or proc_result.stdout or "").strip()
        if error:
            result = (False, error.splitlines()[-1][:240])
        else:
            result = (False, f"{command} --version failed")
        _cli_cache[command] = (result[0], result[1], now)
        return result

    result = (True, None)
    _cli_cache[command] = (result[0], result[1], now)
    return result


def warmup_cli_cache() -> None:
    """Pre-warm CLI cache by checking all provider commands.

    Called at server startup in background to ensure dashboard loads fast.
    """
    global _cli_warmup_done
    if _cli_warmup_done:
        return

    for provider in PROVIDERS:
        _check_cli_available(provider.command, use_cache=False)

    _cli_warmup_done = True


# --- Snapshot Building ---


def get_disabled_company_ids() -> set[str]:
    raw = get_settings_db().get(DISABLED_COMPANIES_KEY)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return set()
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if isinstance(item, str)}


def set_company_enabled(company_id: str, enabled: bool) -> dict[str, Any]:
    """Switch a provider on or off for everything that picks a model.

    Disabling clears `selectable`, which is the single gate every model list
    already passes through — the dev chat, the Configurator and the agent
    runner all read it — so a provider that is out of quota or that you simply
    do not want used stops being offered everywhere at once instead of in the
    one place someone remembered to filter.
    """
    provider = _get_provider(company_id)
    if not provider:
        raise ValueError("Unknown LLM provider")

    disabled = get_disabled_company_ids()
    if enabled:
        disabled.discard(provider.id)
    else:
        disabled.add(provider.id)

    remaining = [
        p for p in PROVIDERS
        if p.id not in disabled and _check_cli_available(p.command)[0] and p.chat_supported
    ]
    if not remaining:
        # Disabling the last usable provider leaves nothing to answer with,
        # and the failure would surface later as an unrelated chat error.
        raise ValueError("At least one LLM provider must stay enabled")

    get_settings_db().set(DISABLED_COMPANIES_KEY, json.dumps(sorted(disabled)))
    return get_llm_options_snapshot()


def _build_provider_snapshot(provider: LlmProvider) -> dict[str, Any]:
    """Build status snapshot for a provider.

    Models are dynamically detected from config files on each call.
    """
    installed, error_msg = _check_cli_available(provider.command)
    enabled = provider.id not in get_disabled_company_ids()
    # `selectable` is what every model picker consults, so switching a provider
    # off here removes it from all of them rather than from whichever list the
    # caller remembered to filter.
    selectable = installed and provider.chat_supported and enabled

    # Get models - dynamic detection for known providers
    if provider.id == "openai":
        models = _get_codex_models()
    elif provider.id == "anthropic":
        models = _get_claude_models()
    elif provider.id == "google":
        models = _get_gemini_models()
    elif provider.id == "antigravity":
        models = _get_antigravity_models()
    else:
        # Fallback - use defined models
        models = [{"id": m, "label": m, "source": "builtin"} for m in provider.models]

    # Extract just IDs for validation
    model_ids = [m["id"] for m in models]
    capabilities = {
        "chat": provider.chat_supported,
        "chat_supported": provider.chat_supported,
        "sessions": provider.id in ("anthropic", "openai", "antigravity"),
        "session_history": provider.id in ("anthropic", "openai"),
        "resumable_sessions": provider.id in ("anthropic", "openai"),
        "resume": provider.id in ("anthropic", "openai"),
        "project_scoped_history": provider.id in ("anthropic", "openai"),
        "raw_activity_details": True,
        "raw_events": True,
        "tool_events": True,
        "permissions": provider.id == "anthropic",
        "live_permission_prompts": provider.id == "anthropic",
        "async_install": True,
        "install_jobs": True,
        "sandbox_mode": provider.id == "openai",
        "abort": True,
    }

    snapshot = {
        "id": provider.id,
        "name": provider.name,
        "command": provider.command,
        "connected": installed,  # "connected" means CLI is installed
        "chat_supported": provider.chat_supported,
        "enabled": enabled,
        "selectable": selectable,
        "status_message": error_msg,
        "models": models,
        "all_model_ids": model_ids,
        "install_methods": LLM_PROVIDER_INSTALL_METHODS.get(provider.id, []),
        "capabilities": capabilities,
    }

    # Add provider-specific settings
    if provider.id == "openai":
        db = get_settings_db()
        sandbox_mode = db.get(CODEX_SANDBOX_MODE_KEY) or DEFAULT_CODEX_SANDBOX_MODE
        snapshot["settings"] = {
            "sandbox_mode": sandbox_mode,
            "sandbox_modes": CODEX_SANDBOX_MODES,
        }

    return snapshot


def _pick_default_selection(snapshots: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Pick first selectable provider and its first model."""
    for snap in snapshots:
        if snap.get("selectable") and snap.get("all_model_ids"):
            return snap["id"], snap["all_model_ids"][0]
    return None, None


# --- Public API ---


def get_llm_options_snapshot() -> dict[str, Any]:
    """Get current LLM options with availability status."""
    db = get_settings_db()

    snapshots = [_build_provider_snapshot(p) for p in PROVIDERS]

    # Load saved selection
    selected_company = db.get(SELECTED_COMPANY_KEY)
    selected_model = db.get(SELECTED_MODEL_KEY)

    # Validate selection
    company_snap = next((s for s in snapshots if s["id"] == selected_company), None)
    is_valid = (
        company_snap is not None
        and company_snap.get("selectable")
        and isinstance(selected_model, str)
        and selected_model in (company_snap.get("all_model_ids") or [])
    )

    if not is_valid:
        selected_company, selected_model = _pick_default_selection(snapshots)
        db.set(SELECTED_COMPANY_KEY, selected_company)
        db.set(SELECTED_MODEL_KEY, selected_model)

    companies = []
    for s in snapshots:
        company = {
            "id": s["id"],
            "name": s["name"],
            "command": s["command"],
            "connected": s["connected"],
            "chat_supported": s["chat_supported"],
            "enabled": s["enabled"],
            "selectable": s["selectable"],
            "status_message": s["status_message"],
            "models": s["models"],
            "install_methods": s.get("install_methods", []),
            "capabilities": s.get("capabilities", {}),
        }
        # Include provider-specific settings if present
        if "settings" in s:
            company["settings"] = s["settings"]
        companies.append(company)

    return {
        "selected": {
            "company_id": selected_company,
            "model": selected_model,
        },
        "companies": companies,
    }


def set_selected_llm(company_id: str, model: str) -> dict[str, Any]:
    """Set the active LLM company and model."""
    provider = _get_provider(company_id)
    if not provider:
        raise ValueError("Unknown LLM provider")

    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("Model name is required")

    snapshot = get_llm_options_snapshot()
    company_snap = next(
        (c for c in snapshot.get("companies", []) if c["id"] == provider.id),
        None,
    )

    if not company_snap:
        raise ValueError("Unknown LLM provider")

    if not company_snap.get("selectable"):
        reason = company_snap.get("status_message") or "Provider not available"
        raise ValueError(reason)

    allowed = {m["id"] for m in company_snap.get("models", []) if isinstance(m, dict)}
    if normalized_model not in allowed:
        raise ValueError(f"Model '{normalized_model}' not available for {provider.name}")

    db = get_settings_db()
    db.set(SELECTED_COMPANY_KEY, provider.id)
    db.set(SELECTED_MODEL_KEY, normalized_model)

    return get_llm_options_snapshot()


def resolve_llm_provider_install(provider_id: str, method: str = "brew") -> dict[str, Any]:
    """Resolve a supported provider CLI install command without running it."""
    provider = _get_provider(provider_id)
    if not provider:
        raise ValueError("unknown_provider: Unknown LLM provider")

    normalized_method = method.strip().lower() if isinstance(method, str) else ""
    methods = LLM_PROVIDER_INSTALL_METHODS.get(provider.id, [])
    install_method = next((m for m in methods if m.get("id") == normalized_method), None)
    if install_method is None:
        available = ", ".join(str(m.get("id")) for m in methods) or "none"
        raise ValueError(
            f"invalid_install_method: Install method '{method}' not available for "
            f"{provider.name}. Available: {available}"
        )

    command = install_method.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError("invalid_install_method: Install command is not configured")

    return {
        "provider_id": provider.id,
        "provider_name": provider.name,
        "provider_command": provider.command,
        "method": normalized_method,
        "command": [str(part) for part in command],
    }


def clear_llm_provider_cli_cache(provider_id: str) -> None:
    """Clear cached CLI availability for a provider after install attempts."""
    provider = _get_provider(provider_id)
    if provider:
        _cli_cache.pop(provider.command, None)


def check_llm_provider_cli_available(provider_id: str) -> tuple[bool, str | None]:
    """Check whether a provider's runtime CLI is currently available."""
    provider = _get_provider(provider_id)
    if not provider:
        raise ValueError("unknown_provider: Unknown LLM provider")
    return _check_cli_available(provider.command, use_cache=False)


def get_codex_sandbox_mode() -> str:
    """Get current Codex sandbox mode."""
    db = get_settings_db()
    mode = db.get(CODEX_SANDBOX_MODE_KEY)
    if mode and any(m["id"] == mode for m in CODEX_SANDBOX_MODES):
        return mode
    return DEFAULT_CODEX_SANDBOX_MODE


def set_codex_sandbox_mode(mode: str) -> dict[str, Any]:
    """Set Codex sandbox mode."""
    normalized = mode.strip().lower()
    valid_modes = {m["id"] for m in CODEX_SANDBOX_MODES}

    if normalized not in valid_modes:
        raise ValueError(f"Invalid sandbox mode: {mode}. Valid: {', '.join(valid_modes)}")

    db = get_settings_db()
    db.set(CODEX_SANDBOX_MODE_KEY, normalized)

    return {
        "sandbox_mode": normalized,
        "sandbox_modes": CODEX_SANDBOX_MODES,
    }
