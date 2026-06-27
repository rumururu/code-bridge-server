"""REST endpoints for managing ``~/.code-bridge/.env`` secrets.

Contract (Flutter UI in TASK_003 will consume this):

- ``GET    /api/secrets``        -> ``{"items": [{"name": str, "has_value": bool}, ...]}``
- ``POST   /api/secrets``        -> body ``{"name": str, "value": str}`` -> ``{"name", "has_value": true}``
- ``DELETE /api/secrets/{key}``  -> ``{"name", "deleted": true}`` or 404

Security:
  * No endpoint ever returns secret values.
  * Audit log records the key name only — never the value.
  * Authentication uses the standard ``verify_api_key`` dependency.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from audit.route_audit import record_api_action
from secret_store_pkg import secret_store
from secret_store_pkg.secret_models import SecretUpsert

from .deps import verify_api_key

router = APIRouter(prefix="/api/secrets", tags=["secrets"])


@router.get("", dependencies=[Depends(verify_api_key)])
async def list_secrets() -> dict[str, list[dict[str, Any]]]:
    """List secret names + ``has_value`` flag.

    Values are intentionally **never** included in the response. The
    Flutter UI only needs to know which secrets are configured so it can
    render a masked-state checklist (NAVER_ID set, ANTHROPIC_API_KEY
    missing, etc.).
    """
    return {"items": secret_store.list_keys()}


@router.post("", dependencies=[Depends(verify_api_key)])
async def upsert_secret(body: SecretUpsert) -> dict[str, Any]:
    """Create or replace a secret in ``~/.code-bridge/.env``.

    The store layer also propagates the new value to ``os.environ`` so
    consumers (LLM client, integrations) see it without a server
    restart. We log only the key name to the audit trail.
    """
    try:
        secret_store.upsert(body.name, body.value)
    except secret_store.SecretStoreError as exc:
        # Defence in depth: the pydantic model already rejects bad
        # patterns at the boundary, but the store layer has the final
        # say (newline-in-value rule is enforced there, not in pydantic).
        _audit("secrets.upsert", body.name, success=False, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc))
    _audit("secrets.upsert", body.name, success=True, status_code=200)
    return {"name": body.name, "has_value": True}


@router.delete("/{key}", dependencies=[Depends(verify_api_key)])
async def delete_secret(key: str) -> dict[str, Any]:
    """Delete a secret from ``~/.code-bridge/.env`` and ``os.environ``."""
    try:
        existed = secret_store.delete(key)
    except secret_store.SecretStoreError as exc:
        _audit("secrets.delete", key, success=False, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc))
    if not existed:
        _audit("secrets.delete", key, success=False, status_code=404)
        raise HTTPException(status_code=404, detail=f"Secret '{key}' not found")
    _audit("secrets.delete", key, success=True, status_code=200)
    return {"name": key, "deleted": True}


# Internal --------------------------------------------------------------

def _audit(operation: str, name: str, *, success: bool, status_code: int) -> None:
    """Record an audit event with the key name only — never the value.

    Wrapping ``record_api_action`` keeps the redaction discipline in one
    place so future maintainers can't accidentally pass ``{"value": ...}``.
    """
    record_api_action(
        operation=operation,
        details={"name": name},
        success=success,
        status_code=status_code,
    )
