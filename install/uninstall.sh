#!/bin/bash
# Code Bridge Server - Uninstall Script for macOS/Linux
# Usage: curl -fsSL https://raw.githubusercontent.com/rumururu/code-bridge-server/<sha>/install/uninstall.sh | bash
#
# Removes the installation directory created by install.sh (default
# ~/.code-bridge) along with its venv and any auto-generated start
# scripts. cloudflared and mermaid-cli are NOT removed because they may
# be shared with other tooling on this machine.
#
# Environment overrides:
#   CODE_BRIDGE_INSTALL_DIR    target directory (default: ~/.code-bridge)
#   CODE_BRIDGE_ASSUME_YES     1 to skip the confirmation prompt

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

INSTALL_DIR="${CODE_BRIDGE_INSTALL_DIR:-$HOME/.code-bridge}"

echo ""
echo -e "${CYAN}Code Bridge Server — Uninstall${NC}"
echo ""

if [ ! -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Nothing to remove at $INSTALL_DIR${NC}"
    exit 0
fi

# Refuse to delete obvious mistakes such as / or $HOME directly.
real_dir="$(cd "$INSTALL_DIR" 2>/dev/null && pwd -P || echo "")"
case "$real_dir" in
    ""|"/"|"$HOME"|"$HOME/")
        echo -e "${RED}Refusing to delete $real_dir — set CODE_BRIDGE_INSTALL_DIR to a dedicated directory.${NC}"
        exit 1
        ;;
esac

echo "About to remove:"
echo "  $real_dir"
echo

if [[ "${CODE_BRIDGE_ASSUME_YES:-0}" != "1" ]]; then
    if [[ -r /dev/tty ]]; then
        read -r -n 1 -p "Continue? [y/N] " reply < /dev/tty || reply=""
        echo
    else
        reply=""
    fi
    if [[ ! $reply =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Drop the login item before the tree goes away, otherwise launchd (or
# the XDG autostart entry) keeps pointing at a deleted launcher.
if [ -x "$real_dir/venv/bin/python" ] && [ -f "$real_dir/desktop_server_app/launcher.py" ]; then
    "$real_dir/venv/bin/python" "$real_dir/desktop_server_app/launcher.py" --disable-autostart >/dev/null 2>&1 || true
fi
rm -f "$HOME/Library/LaunchAgents/com.mkideabox.codebridge.server.plist"
rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/autostart/code-bridge-server.desktop"

rm -rf "$real_dir"
echo -e "${GREEN}✓ Removed $real_dir${NC}"

# Surface any backup directories install.sh produced so the user can
# clean those up too if desired.
backups="$(find "$(dirname "$real_dir")" -maxdepth 1 -type d -name "$(basename "$real_dir").bak.*" 2>/dev/null || true)"
if [ -n "$backups" ]; then
    echo ""
    echo -e "${YELLOW}Leftover backups from previous installs:${NC}"
    echo "$backups"
    echo "Delete manually if you no longer need them."
fi

echo ""
echo "Note: cloudflared / mermaid-cli were left in place."
