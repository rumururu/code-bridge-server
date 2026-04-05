"""mDNS/Bonjour service for local network discovery."""

import asyncio
import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import zeroconf, but make it optional
try:
    from zeroconf import ServiceInfo
    from zeroconf.asyncio import AsyncZeroconf

    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    ServiceInfo = None
    AsyncZeroconf = None


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        # Create a socket and connect to an external address
        # This doesn't actually send any data, just determines the route
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError as exc:
        logger.debug("Could not determine local IP: %s, using fallback", exc)
        return "127.0.0.1"


class MDNSService:
    """mDNS service for advertising Code Bridge server on local network.

    Service Type: _codebridge._tcp.local.
    TXT Records:
        - server_id: Unique server identifier
        - version: Server version
        - api_port: API server port
    """

    SERVICE_TYPE = "_codebridge._tcp.local."

    def __init__(
        self,
        server_id: str,
        server_name: str,
        api_port: int,
        dashboard_port: int,
    ):
        """Initialize mDNS service.

        Args:
            server_id: Unique server identifier
            server_name: Human-readable server name
            api_port: Port for API server (exposed to network)
            dashboard_port: Port for dashboard server (localhost only)
        """
        self.server_id = server_id
        self.server_name = server_name
        self.api_port = api_port
        self.dashboard_port = dashboard_port
        self.async_zeroconf: Optional["AsyncZeroconf"] = None
        self.service_info: Optional["ServiceInfo"] = None
        self._started = False

    async def start_async(self) -> bool:
        """Start advertising the service via mDNS (async version).

        Returns:
            True if started successfully, False otherwise.
        """
        if not ZEROCONF_AVAILABLE:
            logger.info("zeroconf not installed, skipping mDNS advertisement")
            return False

        if self._started:
            return True

        try:
            self.async_zeroconf = AsyncZeroconf()
            local_ip = get_local_ip()

            # Sanitize server name for mDNS (only alphanumeric, hyphen, underscore)
            safe_name = "".join(
                c if c.isalnum() or c in "-_" else "-" for c in self.server_name
            )
            if not safe_name:
                safe_name = "CodeBridge"

            # Include server_id suffix to make name unique
            short_id = self.server_id[:8] if len(self.server_id) >= 8 else self.server_id
            unique_name = f"{safe_name}-{short_id}"

            # Get hostname for proper mDNS resolution
            hostname = socket.gethostname()
            # Ensure hostname ends with .local. but avoid double .local.
            if hostname.endswith(".local"):
                hostname = f"{hostname}."
            elif not hostname.endswith(".local."):
                hostname = f"{hostname}.local."

            # Create service info with explicit server hostname
            self.service_info = ServiceInfo(
                self.SERVICE_TYPE,
                f"{unique_name}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.api_port,
                server=hostname,  # Required for iOS Bonjour resolution
                properties={
                    "server_id": self.server_id,
                    "version": "1.0",
                    "api_port": str(self.api_port),
                    "dashboard_port": str(self.dashboard_port),
                    "local_url": f"http://{local_ip}:{self.api_port}",
                },
            )

            # allow_name_change=True will append number if name still conflicts
            await self.async_zeroconf.async_register_service(
                self.service_info, allow_name_change=True
            )
            self._started = True
            logger.info("Advertising as '%s' on %s:%d", unique_name, local_ip, self.api_port)
            return True

        except OSError as e:
            logger.warning("mDNS failed to start: %s", e)
            await self._cleanup_async()
            return False
        except (RuntimeError, AttributeError, ImportError, TypeError) as e:
            logger.exception("Unexpected error starting mDNS: %s", e)
            await self._cleanup_async()
            return False

    async def stop_async(self) -> None:
        """Stop advertising the service (async version)."""
        if not self._started:
            return

        await self._cleanup_async()
        logger.info("mDNS service stopped")

    async def _cleanup_async(self) -> None:
        """Clean up zeroconf resources (async version)."""
        try:
            if self.async_zeroconf and self.service_info:
                await self.async_zeroconf.async_unregister_service(self.service_info)
        except OSError as exc:
            logger.debug("Error unregistering mDNS service: %s", exc)

        try:
            if self.async_zeroconf:
                await self.async_zeroconf.async_close()
        except OSError as exc:
            logger.debug("Error closing zeroconf: %s", exc)

        self.async_zeroconf = None
        self.service_info = None
        self._started = False


# Global mDNS service instance
_mdns_service: Optional[MDNSService] = None


async def start_mdns_service(
    server_id: str,
    server_name: str,
    api_port: int,
    dashboard_port: int,
) -> bool:
    """Start the global mDNS service (async).

    Args:
        server_id: Unique server identifier
        server_name: Human-readable server name
        api_port: Port for API server
        dashboard_port: Port for dashboard server

    Returns:
        True if started successfully, False otherwise.
    """
    global _mdns_service

    if _mdns_service is not None:
        await _mdns_service.stop_async()

    _mdns_service = MDNSService(
        server_id=server_id,
        server_name=server_name,
        api_port=api_port,
        dashboard_port=dashboard_port,
    )
    return await _mdns_service.start_async()


async def stop_mdns_service() -> None:
    """Stop the global mDNS service (async)."""
    global _mdns_service

    if _mdns_service is not None:
        await _mdns_service.stop_async()
        _mdns_service = None
