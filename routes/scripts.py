"""Registered-script APIs.

A ``shell`` workflow step names a script from this registry; it can never
carry a command line of its own. Registering is therefore the security
boundary, and it is deliberately a separate, audited act from running.

Registration is dashboard-only (localhost). The phone can list what is
registered — so a run's steps are readable — but cannot add one: pairing a
phone should not hand out the ability to register arbitrary executables.
"""

import asyncio
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

import re
import stat
from pathlib import Path

from agent.script_store import ScriptRegistrationError, get_script_store
from agent.script_models import (
    ScriptDraftRequest,
    ScriptDraftSave,
    ScriptRegister,
    ScriptUpdate,
)
from core.runtime_paths import runtime_dir

from .deps import verify_api_key

# Drafted scripts land here rather than anywhere the intent mentions, so a
# generated script can never overwrite something else on the machine.
#
# Deliberately NOT "scripts": on a script install the server tree *is*
# ~/.code-bridge, whose scripts/ holds the server's own tooling. Writing
# generated files in there mixes them with shipped code and invites a name
# collision with something that matters.
_MANAGED_SCRIPTS_DIR = runtime_dir(
    "generated_scripts", Path.home() / ".code-bridge" / "generated_scripts"
)

_SCRIPT_PROMPT = """Write a single {interpreter} script for this job:

{intent}

Rules:
- Output ONLY the script. No prose, no explanation, no markdown fences.
- Start with the shebang.
- `set -euo pipefail` near the top unless the job genuinely needs to continue past errors.
- Exit non-zero with a clear message on stderr when the job fails — the exit code and stderr are what a later step reads to diagnose it.
- Do not invent paths, device ids or credentials. Take them as arguments ($1, $2, ...) or environment variables, and fail loudly with usage if they are missing.
- No destructive commands (rm -rf, disk formatting, killing unrelated processes).
"""


def _slugify(name: str) -> str:
    """Filename for a drafted script.

    Keeps letters from any script, not just ASCII — dropping them turned every
    Korean name into the same handful of slugs, so two different scripts could
    collide on one filename.
    """
    slug = re.sub(r"[^\w.-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return slug[:60] or "script"


def _suggested_name(intent: str) -> str:
    """A short, editable name — the filename is derived from it.

    The whole intent sentence made for a name like "Take the android device
    serial as $1, check it is connected", and therefore a filename to match.
    """
    words = re.findall(r"[\w'-]+", intent.strip(), flags=re.UNICODE)
    if not words:
        return "script"
    return " ".join(words[:5])[:60]


def _strip_code_fences(text: str) -> str:
    """Models add ```bash fences even when told not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

router = APIRouter(prefix="/api/agent/scripts", tags=["scripts"])


@router.get("", dependencies=[Depends(verify_api_key)], response_model=None)
async def list_scripts(limit: int = Query(default=100, ge=1, le=200)) -> dict[str, Any]:
    return {"scripts": get_script_store().list_scripts(limit=limit)}


@router.get("/{script_id}", dependencies=[Depends(verify_api_key)], response_model=None)
async def get_script(script_id: str) -> dict[str, Any]:
    script = get_script_store().get(script_id)
    if script is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return {"script": script}


async def register_script(body: ScriptRegister) -> dict[str, Any]:
    """Register a script. Dashboard-only — see module docstring."""
    try:
        script = get_script_store().register(
            name=body.name,
            path=body.path,
            description=body.description,
            interpreter=body.interpreter,
            default_args=body.default_args,
            timeout_seconds=body.timeout_seconds,
            created_by=body.created_by,
        )
    except ScriptRegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"script": script}


async def update_script(script_id: str, body: ScriptUpdate) -> dict[str, Any]:
    try:
        script = get_script_store().update(
            script_id, body.model_dump(exclude_unset=True)
        )
    except ScriptRegistrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if script is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return {"script": script}


async def delete_script(script_id: str) -> dict[str, Any]:
    if not get_script_store().delete(script_id):
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return {"deleted": True, "script_id": script_id}


__all__ = [
    "router",
    "register_script",
    "update_script",
    "delete_script",
    "draft_script",
    "save_drafted_script",
]


async def draft_script(body: ScriptDraftRequest) -> dict[str, Any]:
    """Have the LLM write a script for a described job.

    Nothing is written to disk and nothing is registered — the caller reviews
    (and edits) the text first. A generated script that registered itself would
    make "write me a script" equivalent to "run whatever you just wrote".
    """
    # Imported here to keep the module importable without the LLM stack.
    from chat.chat_session_service import create_chat_session, get_chat_provider_selection
    from routes.agents import _collect_llm_response_text

    interpreter = "bash"
    prompt = _SCRIPT_PROMPT.format(interpreter=interpreter, intent=body.intent.strip())
    selection = get_chat_provider_selection()
    llm_session = await create_chat_session(
        "script-writer", str(_MANAGED_SCRIPTS_DIR), selection
    )
    try:
        raw = await asyncio.wait_for(
            _collect_llm_response_text(llm_session, prompt), timeout=180.0
        )
    except asyncio.TimeoutError as exc:
        with suppress(Exception):
            await llm_session.abort_current_turn()
        raise HTTPException(
            status_code=504, detail="Script drafting timed out"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Script drafting failed: {exc}") from exc

    script_body = _strip_code_fences(raw)
    if not script_body:
        # No rule-based stand-in: an empty draft is a failure, and inventing a
        # plausible script would be worse than saying so.
        raise HTTPException(status_code=502, detail="The model returned an empty script")
    return {
        "name": _suggested_name(body.intent),
        "body": script_body,
        "interpreter": interpreter,
    }


async def save_drafted_script(body: ScriptDraftSave) -> dict[str, Any]:
    """Write a reviewed draft into the managed directory and register it."""
    _MANAGED_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(body.name)}.sh"
    path = _MANAGED_SCRIPTS_DIR / filename
    if path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"A script called {filename} already exists — rename it or remove the old one",
        )
    path.write_text(body.body if body.body.endswith("\n") else body.body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IRWXU)
    try:
        script = get_script_store().register(
            name=body.name,
            path=str(path),
            description=body.description,
            interpreter=body.interpreter,
            timeout_seconds=body.timeout_seconds,
            created_by="llm_draft",
        )
    except ScriptRegistrationError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"script": script}
