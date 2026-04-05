"""Base result class for service layer results."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseRouteResult:
    """Base dataclass for typed route results.

    All service result classes should inherit from this to ensure consistent
    structure and eliminate code duplication.

    Example:
        @dataclass(frozen=True)
        class MyServiceResult(BaseRouteResult):
            pass

        # Create success result
        result = MyServiceResult.ok({"data": "value"})

        # Create error result
        result = MyServiceResult.error(404, "Not found")
    """

    success: bool
    status_code: int
    payload: dict[str, Any]

    def as_response_fields(self) -> dict[str, Any]:
        """Serialize for route response helpers."""
        return self.payload

    @classmethod
    def ok(cls, payload: dict[str, Any], status_code: int = 200) -> "BaseRouteResult":
        """Create a success result."""
        return cls(success=True, status_code=status_code, payload=payload)

    @classmethod
    def error(cls, status_code: int, message: str, **extra: Any) -> "BaseRouteResult":
        """Create an error result."""
        payload: dict[str, Any] = {"error": message}
        payload.update(extra)
        return cls(success=False, status_code=status_code, payload=payload)
