"""Prompt-to-local-app APIs."""

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from app_builder.app_builder_models import LocalAppCreate
from app_builder.app_builder_service import create_local_app_for_current_server

from .deps import verify_api_key
from .result_response import as_route_response

router = APIRouter(prefix="/api/agent/apps", tags=["agent-apps"])


@router.post("", dependencies=[Depends(verify_api_key)], response_model=None)
async def create_local_app(body: LocalAppCreate) -> dict[str, Any] | Response:
    """Create a local app workspace and seed an agent task/run."""
    result = create_local_app_for_current_server(
        root_path=body.root_path,
        app_name=body.app_name,
        prompt=body.prompt,
        template=body.template,
        provider_id=body.provider_id,
        model=body.model,
    )
    response = as_route_response(result)
    if result.success and result.status_code != 200:
        return JSONResponse(status_code=result.status_code, content=response)
    return response
