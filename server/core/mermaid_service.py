"""Mermaid diagram server-side rendering service.

This service renders Mermaid diagrams to SVG using mermaid-cli (mmdc),
avoiding the need for client-side JavaScript loading from external CDNs.
"""

import asyncio
import base64
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class MermaidService:
    """Service for rendering Mermaid diagrams to SVG."""

    def __init__(self):
        self._mmdc_path: str | None = None
        self._checked = False

    def _find_mmdc(self) -> str | None:
        """Find mermaid-cli (mmdc) executable."""
        if self._checked:
            return self._mmdc_path

        self._checked = True

        # Try common locations
        mmdc = shutil.which("mmdc")
        if mmdc:
            self._mmdc_path = mmdc
            logger.info("Found mermaid-cli at: %s", mmdc)
            return mmdc

        # Try npm global locations
        npm_paths = [
            Path.home() / ".npm-global" / "bin" / "mmdc",
            Path.home() / "node_modules" / ".bin" / "mmdc",
            Path("/usr/local/bin/mmdc"),
            Path("/opt/homebrew/bin/mmdc"),
        ]

        for path in npm_paths:
            if path.exists():
                self._mmdc_path = str(path)
                logger.info("Found mermaid-cli at: %s", path)
                return self._mmdc_path

        logger.warning("mermaid-cli (mmdc) not found. Install with: npm install -g @mermaid-js/mermaid-cli")
        return None

    @property
    def is_available(self) -> bool:
        """Check if mermaid-cli is available."""
        return self._find_mmdc() is not None

    async def render_svg(self, code: str, theme: str = "default") -> tuple[str | None, str | None]:
        """Render Mermaid diagram to SVG.

        Args:
            code: Mermaid diagram code
            theme: Theme name ('default', 'dark', 'forest', 'neutral')

        Returns:
            Tuple of (svg_content, error_message)
            - On success: (svg_content, None)
            - On failure: (None, error_message)
        """
        mmdc = self._find_mmdc()
        if not mmdc:
            return None, "mermaid-cli not installed"

        # Validate theme
        valid_themes = {"default", "dark", "forest", "neutral"}
        if theme not in valid_themes:
            theme = "default"

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.mmd"
            output_file = Path(tmpdir) / "output.svg"
            config_file = Path(tmpdir) / "config.json"

            # Write input file
            input_file.write_text(code, encoding="utf-8")

            # Write config file with theme
            config_content = f'{{"theme": "{theme}"}}'
            config_file.write_text(config_content, encoding="utf-8")

            # Run mermaid-cli
            cmd = [
                mmdc,
                "-i", str(input_file),
                "-o", str(output_file),
                "-c", str(config_file),
                "-b", "transparent",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)

                if proc.returncode != 0:
                    error_msg = stderr.decode("utf-8", errors="replace").strip()
                    if not error_msg:
                        error_msg = f"mermaid-cli exited with code {proc.returncode}"
                    logger.error("Mermaid render error: %s", error_msg)
                    return None, error_msg

                if not output_file.exists():
                    return None, "SVG output file not created"

                svg_content = output_file.read_text(encoding="utf-8")
                return svg_content, None

            except asyncio.TimeoutError:
                logger.error("Mermaid render timeout")
                return None, "Render timeout (30s)"
            except Exception as e:
                logger.error("Mermaid render exception: %s", e)
                return None, str(e)

    async def render_svg_base64(self, code: str, theme: str = "default") -> tuple[str | None, str | None]:
        """Render Mermaid diagram to base64-encoded SVG.

        Args:
            code: Mermaid diagram code
            theme: Theme name ('default', 'dark', 'forest', 'neutral')

        Returns:
            Tuple of (base64_svg, error_message)
            - On success: (base64_encoded_svg, None)
            - On failure: (None, error_message)
        """
        svg_content, error = await self.render_svg(code, theme)
        if error:
            return None, error

        if svg_content:
            encoded = base64.b64encode(svg_content.encode("utf-8")).decode("ascii")
            return encoded, None

        return None, "Empty SVG content"


# Global service instance
_service: MermaidService | None = None


def get_mermaid_service() -> MermaidService:
    """Get the global MermaidService instance."""
    global _service
    if _service is None:
        _service = MermaidService()
    return _service
