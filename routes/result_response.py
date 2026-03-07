"""Helpers for translating typed service results into route responses."""

from typing import Any, Protocol

from fastapi.responses import JSONResponse


class RouteResult(Protocol):
    """Protocol for route-mappable typed results."""

    success: bool
    status_code: int

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize result payload for HTTP responses."""


def as_route_response(result: RouteResult) -> dict[str, Any] | JSONResponse:
    """Return JSON payload for success, JSONResponse for error."""
    payload = result.as_response_fields()
    if result.success:
        return payload
    return JSONResponse(status_code=result.status_code, content=payload)


def as_flagged_response(
    payload: dict[str, Any],
    *,
    success_key: str = "success",
    error_status_code: int = 400,
) -> dict[str, Any] | JSONResponse:
    """Return payload if success flag is truthy, otherwise JSONResponse."""
    if payload.get(success_key):
        return payload
    return JSONResponse(status_code=error_status_code, content=payload)
