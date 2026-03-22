"""Helpers for translating typed service results into route responses."""

from typing import Any, Protocol, TypeVar

from fastapi.responses import JSONResponse

# Import BaseRouteResult from base module to avoid circular imports
from base_result import BaseRouteResult


class RouteResult(Protocol):
    """Protocol for route-mappable typed results."""

    success: bool
    status_code: int

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize result payload for HTTP responses."""


# Type variable for generic result factory
T = TypeVar("T", bound=BaseRouteResult)


def create_error_result(
    result_class: type[T],
    status_code: int,
    message: str,
    **extra: Any,
) -> T:
    """Generic factory for creating error results.

    Args:
        result_class: The result dataclass type to instantiate.
        status_code: HTTP status code for the error.
        message: Error message.
        **extra: Additional fields to include in the payload.

    Returns:
        Instance of result_class with error payload.
    """
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return result_class(success=False, status_code=status_code, payload=payload)


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
