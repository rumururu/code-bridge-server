"""Server-side entitlement enforcement for Code Bridge.

This package provides a simple ``EntitlementService`` that validates whether
a given RevenueCat app user has an active Code Bridge subscription. The
result is consumed by the ``verify_subscription`` FastAPI dependency in
``routes.deps`` so that gated routers reject unentitled callers with HTTP
402 Payment Required.
"""

from .entitlement_service import (
    EntitlementResult,
    EntitlementService,
    get_entitlement_service,
)

__all__ = [
    "EntitlementResult",
    "EntitlementService",
    "get_entitlement_service",
]
