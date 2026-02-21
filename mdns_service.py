"""mDNS Service Registration for Code Bridge Server.

Registers a _codebridge._tcp service on the local network using Zeroconf,
allowing apps on the same WiFi to discover the server automatically.
"""

import asyncio
import logging
import socket
from typing import Optional

from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

logger = logging.getLogger(__name__)

# Service type for Code Bridge
SERVICE_TYPE = "_codebridge._tcp.local."
SERVICE_NAME = "Code Bridge._codebridge._tcp.local."


class MDNSService:
    """Manages mDNS service registration for local network discovery."""

    def __init__(
        self,
        port: int,
        server_name: str = "Code Bridge",
        api_key: Optional[str] = None,
    ):
        """Initialize mDNS service.

        Args:
            port: Server port number
            server_name: Human-readable server name (e.g., "MacBook Pro")
            api_key: Optional API key (only first 8 chars are advertised for verification)
        """
        self._port = port
        self._server_name = server_name
        self._api_key = api_key
        self._zeroconf: Optional[AsyncZeroconf] = None
        self._service_info: Optional[ServiceInfo] = None
        self._registered = False

    def _get_local_ip(self) -> str:
        """Get the local IP address for service registration."""
        try:
            # Create a socket to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Connect to an external address (doesn't actually send data)
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            # Fallback to localhost
            return "127.0.0.1"

    def _get_hostname(self) -> str:
        """Get machine hostname."""
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    async def start(self) -> bool:
        """Start mDNS service registration.

        Returns:
            True if registration successful, False otherwise
        """
        if self._registered:
            logger.warning("mDNS service already registered")
            return True

        try:
            local_ip = self._get_local_ip()
            hostname = self._get_hostname()

            # Build properties for service discovery
            properties = {
                "version": "1.0",
                "server_name": self._server_name,
                "hostname": hostname,
            }

            # Include partial API key hash for verification (if set)
            if self._api_key:
                # Only expose first 8 chars for client verification
                properties["api_key_hint"] = self._api_key[:8] if len(self._api_key) >= 8 else self._api_key

            # Create unique service name based on hostname
            service_name = f"{self._server_name} ({hostname})._codebridge._tcp.local."

            self._service_info = ServiceInfo(
                type_=SERVICE_TYPE,
                name=service_name,
                port=self._port,
                properties=properties,
                server=f"{hostname}.local.",
                addresses=[socket.inet_aton(local_ip)],
            )

            # Create and start AsyncZeroconf
            self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)
            await self._zeroconf.async_register_service(self._service_info)

            self._registered = True
            logger.info(
                f"mDNS service registered: {service_name} at {local_ip}:{self._port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register mDNS service: {e}")
            return False

    async def stop(self) -> None:
        """Stop mDNS service and unregister from network."""
        if not self._registered or self._zeroconf is None:
            return

        try:
            if self._service_info:
                await self._zeroconf.async_unregister_service(self._service_info)
            await self._zeroconf.async_close()
            self._registered = False
            logger.info("mDNS service unregistered")
        except Exception as e:
            logger.error(f"Error stopping mDNS service: {e}")

    @property
    def is_registered(self) -> bool:
        """Check if service is currently registered."""
        return self._registered

    def get_status(self) -> dict:
        """Get current mDNS service status."""
        return {
            "registered": self._registered,
            "service_type": SERVICE_TYPE,
            "port": self._port,
            "server_name": self._server_name,
            "local_ip": self._get_local_ip() if self._registered else None,
        }


# Singleton instance
_mdns_service: Optional[MDNSService] = None


def get_mdns_service() -> Optional[MDNSService]:
    """Get the current mDNS service instance."""
    return _mdns_service


def create_mdns_service(
    port: int,
    server_name: str = "Code Bridge",
    api_key: Optional[str] = None,
) -> MDNSService:
    """Create and store mDNS service singleton.

    Args:
        port: Server port
        server_name: Human-readable server name
        api_key: Optional API key for authentication

    Returns:
        MDNSService instance
    """
    global _mdns_service
    _mdns_service = MDNSService(port, server_name, api_key)
    return _mdns_service
