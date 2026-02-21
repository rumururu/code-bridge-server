"""Cloudflare Tunnel Service for Code Bridge Server.

Manages Cloudflare Quick Tunnels (trycloudflare.com) for secure remote access.
No account required - tunnels are automatically created with temporary URLs.
"""

import asyncio
import logging
import re
import shutil
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TunnelService:
    """Manages Cloudflare Quick Tunnel for remote access."""

    def __init__(
        self,
        local_port: int,
        on_url_change: Optional[Callable[[str], None]] = None,
    ):
        """Initialize tunnel service.

        Args:
            local_port: Local server port to tunnel
            on_url_change: Callback when tunnel URL changes
        """
        self._local_port = local_port
        self._on_url_change = on_url_change
        self._process: Optional[asyncio.subprocess.Process] = None
        self._tunnel_url: Optional[str] = None
        self._running = False
        self._restart_task: Optional[asyncio.Task] = None

    @staticmethod
    def is_cloudflared_installed() -> bool:
        """Check if cloudflared CLI is installed."""
        return shutil.which("cloudflared") is not None

    async def start(self) -> Optional[str]:
        """Start Cloudflare Quick Tunnel.

        Returns:
            Tunnel URL if started successfully, None otherwise
        """
        if self._running:
            return self._tunnel_url

        if not self.is_cloudflared_installed():
            logger.error(
                "cloudflared not installed. Install with: brew install cloudflared"
            )
            return None

        try:
            # Start cloudflared quick tunnel
            self._process = await asyncio.create_subprocess_exec(
                "cloudflared",
                "tunnel",
                "--url",
                f"http://localhost:{self._local_port}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self._running = True

            # Start background task to read tunnel output
            asyncio.create_task(self._monitor_tunnel())

            # Wait for URL to be extracted (with timeout)
            for _ in range(30):  # 30 second timeout
                await asyncio.sleep(1)
                if self._tunnel_url:
                    logger.info(f"Tunnel started: {self._tunnel_url}")
                    return self._tunnel_url

            logger.warning("Tunnel started but URL not detected within timeout")
            return self._tunnel_url

        except Exception as e:
            logger.error(f"Failed to start tunnel: {e}")
            self._running = False
            return None

    async def _monitor_tunnel(self) -> None:
        """Monitor tunnel process output for URL and errors."""
        if not self._process or not self._process.stderr:
            return

        url_pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

        try:
            while self._running and self._process.returncode is None:
                line = await self._process.stderr.readline()
                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.debug(f"[tunnel] {text}")

                    # Extract tunnel URL from output
                    match = url_pattern.search(text)
                    if match:
                        new_url = match.group(0)
                        if new_url != self._tunnel_url:
                            self._tunnel_url = new_url
                            logger.info(f"Tunnel URL: {new_url}")

                            # Notify callback
                            if self._on_url_change:
                                try:
                                    self._on_url_change(new_url)
                                except Exception as e:
                                    logger.error(f"URL change callback error: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Tunnel monitor error: {e}")
        finally:
            if self._running and self._process.returncode is not None:
                logger.warning(
                    f"Tunnel process exited with code {self._process.returncode}"
                )
                self._running = False

                # Auto-restart if unexpected exit
                if self._restart_task is None or self._restart_task.done():
                    self._restart_task = asyncio.create_task(self._auto_restart())

    async def _auto_restart(self) -> None:
        """Automatically restart tunnel after unexpected exit."""
        await asyncio.sleep(5)  # Wait before restart
        if not self._running:
            logger.info("Auto-restarting tunnel...")
            await self.start()

    async def stop(self) -> None:
        """Stop the tunnel."""
        self._running = False

        if self._restart_task:
            self._restart_task.cancel()
            self._restart_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.error(f"Error stopping tunnel: {e}")
            finally:
                self._process = None

        self._tunnel_url = None
        logger.info("Tunnel stopped")

    @property
    def tunnel_url(self) -> Optional[str]:
        """Get current tunnel URL."""
        return self._tunnel_url

    @property
    def is_running(self) -> bool:
        """Check if tunnel is running."""
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get current tunnel status."""
        return {
            "installed": self.is_cloudflared_installed(),
            "running": self._running,
            "url": self._tunnel_url,
            "local_port": self._local_port,
        }


# Singleton instance
_tunnel_service: Optional[TunnelService] = None


def get_tunnel_service() -> Optional[TunnelService]:
    """Get the current tunnel service instance."""
    return _tunnel_service


def create_tunnel_service(
    local_port: int,
    on_url_change: Optional[Callable[[str], None]] = None,
) -> TunnelService:
    """Create and store tunnel service singleton.

    Args:
        local_port: Local server port
        on_url_change: Callback when tunnel URL changes

    Returns:
        TunnelService instance
    """
    global _tunnel_service
    _tunnel_service = TunnelService(local_port, on_url_change)
    return _tunnel_service
