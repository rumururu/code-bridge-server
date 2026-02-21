"""Terminal QR Code Display for Code Bridge.

Displays QR codes in the terminal using Unicode block characters.
Supports both modern terminals (with Unicode) and ASCII fallback.
"""

import sys
from typing import Optional

try:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_L

    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False


def generate_qr_terminal(
    data: str,
    border: int = 2,
    use_unicode: bool = True,
) -> str:
    """Generate QR code as terminal-printable string.

    Args:
        data: The data to encode in the QR code
        border: Number of quiet zone modules around QR code
        use_unicode: Use Unicode block characters (smaller, cleaner)

    Returns:
        String representation of QR code for terminal display
    """
    if not QRCODE_AVAILABLE:
        return "[QR Code library not installed. Run: pip install qrcode[pil]]"

    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=ERROR_CORRECT_L,
        box_size=1,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    matrix = qr.get_matrix()

    if use_unicode:
        return _render_unicode(matrix)
    else:
        return _render_ascii(matrix)


def _render_unicode(matrix: list[list[bool]]) -> str:
    """Render QR code using Unicode half-block characters.

    Uses the upper-half block character to fit 2 rows per line,
    making the QR code more compact and square-looking.
    """
    lines = []
    height = len(matrix)
    width = len(matrix[0]) if height > 0 else 0

    # Process two rows at a time
    for row in range(0, height, 2):
        line = ""
        for col in range(width):
            # Get values for top and bottom cells
            top = matrix[row][col]
            bottom = matrix[row + 1][col] if row + 1 < height else False

            if top and bottom:
                line += "\u2588"  # Full block (both black)
            elif top and not bottom:
                line += "\u2580"  # Upper half block
            elif not top and bottom:
                line += "\u2584"  # Lower half block
            else:
                line += " "  # Both white

        lines.append(line)

    return "\n".join(lines)


def _render_ascii(matrix: list[list[bool]]) -> str:
    """Render QR code using ASCII characters.

    Uses double characters for better aspect ratio.
    """
    lines = []
    for row in matrix:
        line = ""
        for cell in row:
            if cell:
                line += "##"  # Black module
            else:
                line += "  "  # White module
        lines.append(line)
    return "\n".join(lines)


def print_qr_to_terminal(
    data: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    use_color: bool = True,
) -> None:
    """Print QR code to terminal with optional title and subtitle.

    Args:
        data: The data to encode in the QR code
        title: Optional title above QR code
        subtitle: Optional subtitle below QR code
        use_color: Use ANSI colors for better visibility
    """
    if not QRCODE_AVAILABLE:
        print("\n[Error] QR Code library not installed.")
        print("Run: pip install qrcode[pil]\n")
        return

    # ANSI color codes
    if use_color and sys.stdout.isatty():
        BOLD = "\033[1m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        RESET = "\033[0m"
        DIM = "\033[2m"
    else:
        BOLD = GREEN = CYAN = RESET = DIM = ""

    # Generate QR code
    qr_str = generate_qr_terminal(data, border=2, use_unicode=True)

    # Print with formatting
    print()
    if title:
        print(f"{BOLD}{GREEN}{title}{RESET}")
        print()

    # Print QR code with background for better visibility
    for line in qr_str.split("\n"):
        print(f"  {line}")

    print()
    if subtitle:
        print(f"{DIM}{subtitle}{RESET}")
    print()


def display_pairing_qr(
    qr_url: str,
    local_url: str,
    tunnel_url: Optional[str] = None,
    server_name: Optional[str] = None,
) -> None:
    """Display pairing QR code with connection info.

    Args:
        qr_url: The codebridge://pair/... URL to encode
        local_url: Local network URL for display
        tunnel_url: Remote tunnel URL if available
        server_name: Server display name
    """
    if not QRCODE_AVAILABLE:
        print("\n" + "=" * 60)
        print(" QR Code library not installed")
        print(" Run: pip install qrcode[pil]")
        print("=" * 60 + "\n")
        return

    # ANSI codes
    if sys.stdout.isatty():
        BOLD = "\033[1m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        DIM = "\033[2m"
    else:
        BOLD = GREEN = CYAN = YELLOW = RESET = DIM = ""

    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  Code Bridge - Scan QR Code to Pair{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()

    # Print QR code
    qr_str = generate_qr_terminal(qr_url, border=2, use_unicode=True)
    for line in qr_str.split("\n"):
        print(f"  {line}")

    print()
    print(f"  {CYAN}Server:{RESET} {server_name or 'PC Server'}")
    print(f"  {CYAN}Local:{RESET}  {local_url}")
    if tunnel_url:
        print(f"  {CYAN}Remote:{RESET} {tunnel_url}")
    print()
    print(f"  {DIM}Open Code Bridge app and scan this QR code{RESET}")
    print(f"  {DIM}QR code expires in 5 minutes{RESET}")
    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()


def display_install_complete(
    local_url: str,
    tunnel_url: Optional[str] = None,
) -> None:
    """Display installation complete message.

    Args:
        local_url: Local network URL
        tunnel_url: Remote tunnel URL if available
    """
    if sys.stdout.isatty():
        BOLD = "\033[1m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        RESET = "\033[0m"
    else:
        BOLD = GREEN = CYAN = RESET = ""

    print()
    print(f"{BOLD}{GREEN}Installation Complete!{RESET}")
    print()
    print(f"  {CYAN}Local URL:{RESET}  {local_url}")
    if tunnel_url:
        print(f"  {CYAN}Remote URL:{RESET} {tunnel_url}")
    print()
    print("  To start the server later, run:")
    print(f"  {CYAN}~/.code-bridge/start.sh{RESET}")
    print()
