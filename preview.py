"""Preview proxy for dev servers."""

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse


class PreviewProxy:
    """Reverse proxy for dev server previews."""

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def proxy_request(
        self,
        request: Request,
        target_port: int,
        path: str = "",
    ) -> Response:
        """Proxy request to target dev server."""
        # Build URL with query string if present
        target_url = f"http://localhost:{target_port}/{path}"
        if request.query_params:
            target_url = f"{target_url}?{request.query_params}"

        # Build headers (filter out host)
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("Host", None)

        try:
            # Make request to target server
            if request.method == "GET":
                resp = await self._client.get(target_url, headers=headers)
            elif request.method == "POST":
                body = await request.body()
                resp = await self._client.post(target_url, headers=headers, content=body)
            elif request.method == "PUT":
                body = await request.body()
                resp = await self._client.put(target_url, headers=headers, content=body)
            elif request.method == "DELETE":
                resp = await self._client.delete(target_url, headers=headers)
            else:
                resp = await self._client.request(
                    request.method,
                    target_url,
                    headers=headers,
                    content=await request.body(),
                )

            # Build response headers
            response_headers = dict(resp.headers)
            # Remove headers that shouldn't be forwarded
            for header in ["transfer-encoding", "content-encoding", "content-length"]:
                response_headers.pop(header, None)
                response_headers.pop(header.title(), None)

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )

        except httpx.ConnectError:
            return Response(
                content="Dev server not running or not accessible",
                status_code=503,
                media_type="text/plain",
            )
        except Exception as e:
            return Response(
                content=f"Proxy error: {str(e)}",
                status_code=500,
                media_type="text/plain",
            )

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


# Global proxy instance
_preview_proxy: PreviewProxy | None = None


def get_preview_proxy() -> PreviewProxy:
    """Get global preview proxy instance."""
    global _preview_proxy
    if _preview_proxy is None:
        _preview_proxy = PreviewProxy()
    return _preview_proxy
