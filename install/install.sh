#!/bin/bash
# Code Bridge Server - Installation Script for macOS/Linux
# Usage: curl -fsSL https://raw.githubusercontent.com/rumururu/code-bridge-server/<sha>/install/install.sh | bash
#
# Environment overrides:
#   CODE_BRIDGE_INSTALL_DIR    target directory (default: ~/.code-bridge)
#   CODE_BRIDGE_REF            git ref to install (default: pinned SHA below)
#   CODE_BRIDGE_AUTO_START     0 to skip auto-start after install
#   CODE_BRIDGE_AUTOSTART      1/0 to register the login item without prompting
#   CODE_BRIDGE_FORCE_RESET    1 to overwrite local changes during upgrade

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

INSTALL_DIR="${CODE_BRIDGE_INSTALL_DIR:-$HOME/.code-bridge}"
REPO_URL="https://github.com/rumururu/code-bridge-server.git"
# Pinned upstream commit. Override with CODE_BRIDGE_REF=main for HEAD.
CODE_BRIDGE_REF_DEFAULT="e08d9c3e8e1e99088f3df11f00bf5213d7520ab3"
CODE_BRIDGE_REF="${CODE_BRIDGE_REF:-$CODE_BRIDGE_REF_DEFAULT}"
MIN_PYTHON_VERSION="3.10"

# cloudflared release pin (override with CODE_BRIDGE_CLOUDFLARED_VERSION).
CLOUDFLARED_VERSION="${CODE_BRIDGE_CLOUDFLARED_VERSION:-2026.5.2}"
CLOUDFLARED_DEB_SHA256="f7378c11f55a061b4f1f7d1bccdd07bdfd947ed95634c5f6f4ba71a20d5b1d1d"

echo ""
echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Code Bridge Server - Installation    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
echo ""

prompt_yes_no() {
    local prompt="$1"
    local reply=""

    if [[ -r /dev/tty ]]; then
        read -r -n 1 -p "$prompt" reply < /dev/tty || reply=""
        echo
    else
        echo "No interactive terminal available; skipping optional install."
        return 1
    fi

    [[ $reply =~ ^[Yy]$ ]]
}

# Function to compare version numbers
version_ge() {
    local IFS=.
    local i
    local left=($1)
    local right=($2)

    for ((i=${#left[@]}; i<3; i++)); do left[i]=0; done
    for ((i=${#right[@]}; i<3; i++)); do right[i]=0; done

    for ((i=0; i<3; i++)); do
        if ((10#${left[i]} > 10#${right[i]})); then
            return 0
        fi
        if ((10#${left[i]} < 10#${right[i]})); then
            return 1
        fi
    done

    return 0
}

# Check for Python 3.10+
check_python() {
    echo -e "${CYAN}Checking Python installation...${NC}"

    # Try different Python commands
    for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if version_ge "$version" "$MIN_PYTHON_VERSION"; then
                PYTHON_CMD="$cmd"
                echo -e "${GREEN}✓ Found $cmd version $version${NC}"
                return 0
            fi
        fi
    done

    echo -e "${RED}✗ Python $MIN_PYTHON_VERSION or higher is required${NC}"
    echo ""
    echo "Please install Python 3.10+:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu/Debian: sudo apt install python3.11"
    echo "  Fedora: sudo dnf install python3.11"
    exit 1
}

# Check for Git
check_git() {
    echo -e "${CYAN}Checking Git installation...${NC}"

    if command -v git &> /dev/null; then
        echo -e "${GREEN}✓ Git is installed${NC}"
        return 0
    fi

    echo -e "${RED}✗ Git is required${NC}"
    echo ""
    echo "Please install Git:"
    echo "  macOS: xcode-select --install"
    echo "  Ubuntu/Debian: sudo apt install git"
    echo "  Fedora: sudo dnf install git"
    exit 1
}

# Optional: Install cloudflared for remote access
check_cloudflared() {
    echo -e "${CYAN}Checking cloudflared (optional, for remote access)...${NC}"

    if command -v cloudflared &> /dev/null; then
        echo -e "${GREEN}✓ cloudflared is installed${NC}"
        return 0
    fi

    echo -e "${YELLOW}! cloudflared not found (optional for remote access)${NC}"

    if ! prompt_yes_no "Install cloudflared $CLOUDFLARED_VERSION for remote access? [y/N] "; then
        return 0
    fi

    echo -e "${CYAN}Installing cloudflared $CLOUDFLARED_VERSION...${NC}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install cloudflare/cloudflare/cloudflared
        else
            echo -e "${YELLOW}Homebrew not found. Please install cloudflared manually:${NC}"
            echo "  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        fi
        return 0
    fi

    if [[ ! -f /etc/debian_version ]]; then
        echo -e "${YELLOW}Automatic install only available on Debian/Ubuntu via .deb.${NC}"
        echo "Manual download: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        return 0
    fi

    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    local deb_url="https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-amd64.deb"
    local deb_path="$tmp/cloudflared.deb"

    echo "  Downloading $deb_url"
    if ! curl -fL --proto '=https' --tlsv1.2 -o "$deb_path" "$deb_url"; then
        echo -e "${RED}  Failed to download cloudflared. Skipping.${NC}"
        return 0
    fi

    # Verify SHA256 before touching dpkg. This catches a tampered mirror
    # OR a maintainer who bumped CLOUDFLARED_VERSION without updating the
    # pin in this script.
    local actual_sha
    actual_sha="$(shasum -a 256 "$deb_path" | awk '{print $1}')"
    if [[ "$actual_sha" != "$CLOUDFLARED_DEB_SHA256" ]]; then
        echo -e "${RED}  cloudflared SHA256 mismatch — refusing to install.${NC}"
        echo "    expected: $CLOUDFLARED_DEB_SHA256"
        echo "    got:      $actual_sha"
        return 0
    fi
    echo -e "${GREEN}  SHA256 verified${NC}"

    # Try non-interactive sudo first so curl|bash users don't sit on a
    # silent password prompt. Fall back to a manual instruction.
    if sudo -n true 2>/dev/null; then
        sudo dpkg -i "$deb_path" || echo -e "${YELLOW}  dpkg returned non-zero; cloudflared may be partially installed.${NC}"
        return 0
    fi

    # No passwordless sudo: keep the verified deb where the user can find
    # it after the script exits and surface a manual install command.
    mkdir -p "$INSTALL_DIR"
    local persisted="$INSTALL_DIR/cloudflared-${CLOUDFLARED_VERSION}.deb"
    cp "$deb_path" "$persisted"
    echo -e "${YELLOW}  sudo password required; skipping automatic install.${NC}"
    echo "  Verified deb saved at: $persisted"
    echo "  Run manually: sudo dpkg -i $persisted"
}

# Check for Node.js and install mermaid-cli for diagram rendering
check_mermaid_cli() {
    echo -e "${CYAN}Checking mermaid-cli (for diagram rendering)...${NC}"

    if command -v mmdc &> /dev/null; then
        echo -e "${GREEN}✓ mermaid-cli is installed${NC}"
        return 0
    fi

    echo -e "${YELLOW}! mermaid-cli not found${NC}"

    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        echo -e "${YELLOW}npm not found. Skipping mermaid-cli installation.${NC}"
        echo "To enable diagram rendering, install Node.js and run: npm install -g @mermaid-js/mermaid-cli"
        return 0
    fi

    if prompt_yes_no "Install mermaid-cli for diagram rendering? [y/N] "; then
        echo -e "${CYAN}Installing mermaid-cli...${NC}"
        npm install -g @mermaid-js/mermaid-cli
        if command -v mmdc &> /dev/null; then
            echo -e "${GREEN}✓ mermaid-cli installed successfully${NC}"
        else
            echo -e "${YELLOW}mermaid-cli installation may have failed. You can install it manually later.${NC}"
        fi
    fi
}

# Clone or update repository
setup_repository() {
    echo ""
    echo -e "${CYAN}Setting up Code Bridge Server (ref: $CODE_BRIDGE_REF)...${NC}"

    if [ -d "$INSTALL_DIR/.git" ]; then
        echo "Updating existing installation..."
        cd "$INSTALL_DIR"

        # Guard local edits before any destructive git operation. Users
        # piping curl|bash should not silently lose customizations.
        if ! git diff --quiet HEAD 2>/dev/null || ! git diff --quiet --cached 2>/dev/null; then
            if [[ "${CODE_BRIDGE_FORCE_RESET:-0}" != "1" ]]; then
                local backup="$INSTALL_DIR.bak.$(date +%Y%m%d-%H%M%S)"
                echo -e "${YELLOW}! Local changes detected in $INSTALL_DIR${NC}"
                echo "  Backing up working tree to: $backup"
                cp -R "$INSTALL_DIR" "$backup"
                echo "  Set CODE_BRIDGE_FORCE_RESET=1 to skip this backup next time."
            else
                echo -e "${YELLOW}! CODE_BRIDGE_FORCE_RESET=1 — discarding local changes${NC}"
            fi
        fi

        git fetch --tags origin
        git checkout --force "$CODE_BRIDGE_REF"
    elif [ -d "$INSTALL_DIR" ] && [ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
        # The directory exists but carries no repo — either a pre-git
        # install or a plain data dir holding .env / api_keys.json /
        # paired_accounts.json. `git clone` refuses a non-empty
        # destination, so attach the repo in place and keep the data.
        echo -e "${YELLOW}! $INSTALL_DIR exists but is not a git repository${NC}"

        if [[ "${CODE_BRIDGE_FORCE_RESET:-0}" != "1" ]]; then
            local backup="$INSTALL_DIR.bak.$(date +%Y%m%d-%H%M%S)"
            echo "  Backing up existing contents to: $backup"
            cp -R "$INSTALL_DIR" "$backup"
            echo "  Set CODE_BRIDGE_FORCE_RESET=1 to skip this backup next time."
        fi

        echo "  Attaching repository in place (existing files are kept)"
        cd "$INSTALL_DIR"
        git init -q
        if git remote get-url origin >/dev/null 2>&1; then
            git remote set-url origin "$REPO_URL"
        else
            git remote add origin "$REPO_URL"
        fi
        git fetch --tags origin
        git checkout --force "$CODE_BRIDGE_REF"
    else
        echo "Installing to $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
        git checkout --force "$CODE_BRIDGE_REF"
    fi

    echo -e "${GREEN}✓ Repository ready (HEAD: $(git rev-parse --short HEAD))${NC}"
}

# Create virtual environment and install dependencies
setup_venv() {
    echo ""
    echo -e "${CYAN}Setting up Python environment...${NC}"

    cd "$INSTALL_DIR"

    # Detect requirements.txt changes between runs and rebuild the venv
    # when they differ so an upgrade doesn't leave stale packages around.
    local req_hash
    req_hash="$(shasum -a 256 requirements.txt | awk '{print $1}')"
    local marker="venv/.requirements.sha256"

    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    elif [ ! -f "$marker" ] || [ "$(cat "$marker" 2>/dev/null)" != "$req_hash" ]; then
        echo "requirements.txt changed since last install — recreating venv..."
        rm -rf venv
        $PYTHON_CMD -m venv venv
    fi

    echo "Installing dependencies..."
    # shellcheck disable=SC1091
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install --upgrade -r requirements.txt -q
    deactivate

    echo "$req_hash" > "$marker"

    echo -e "${GREEN}✓ Python environment ready${NC}"
}

# Create start script
create_start_script() {
    echo ""
    echo -e "${CYAN}Creating start script...${NC}"

    cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main.py --show-qr "$@"
EOF

    chmod +x "$INSTALL_DIR/start.sh"

    # Menu-bar/tray mode. pystray and the platform bindings already come
    # from requirements.txt, so this only needs the launcher the repo
    # ships in desktop_server_app/.
    if [ -f "$INSTALL_DIR/desktop_server_app/launcher.py" ]; then
        cat > "$INSTALL_DIR/start-menubar.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
exec python desktop_server_app/launcher.py "$@"
EOF
        chmod +x "$INSTALL_DIR/start-menubar.sh"
        echo -e "${GREEN}✓ Start scripts created (terminal + menu bar)${NC}"
    else
        echo -e "${GREEN}✓ Start script created${NC}"
    fi
}

# Register the menu-bar launcher as a login item
configure_autostart() {
    [ -f "$INSTALL_DIR/desktop_server_app/launcher.py" ] || return 0

    local choice="${CODE_BRIDGE_AUTOSTART:-}"
    if [ -z "$choice" ]; then
        if prompt_yes_no "Start Code Bridge Server at login (menu bar mode)? [y/N] "; then
            choice=1
        else
            choice=0
        fi
    fi

    [ "$choice" = "1" ] || return 0

    echo ""
    echo -e "${CYAN}Registering login item...${NC}"
    if "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/desktop_server_app/launcher.py" --enable-autostart; then
        echo -e "${GREEN}✓ Code Bridge Server will start at login${NC}"
        echo "  Turn it off from the menu-bar item, or re-run with CODE_BRIDGE_AUTOSTART=0"
    else
        echo -e "${YELLOW}! Could not register the login item — start it manually with start-menubar.sh${NC}"
    fi
}

# Run the server
run_server() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   Installation Complete!               ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo ""
    echo "Starting Code Bridge Server..."
    echo ""

    cd "$INSTALL_DIR"
    source venv/bin/activate
    python main.py --show-qr
}

# Main installation flow
main() {
    check_python
    check_git
    check_cloudflared
    check_mermaid_cli
    setup_repository
    setup_venv
    create_start_script
    configure_autostart

    if [[ "${CODE_BRIDGE_AUTO_START:-1}" == "1" ]]; then
        run_server
    else
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║   Installation Complete!               ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
        echo ""
        echo "Start Code Bridge Server with:"
        echo "  $INSTALL_DIR/start.sh            (terminal, prints the pairing QR)"
        if [ -f "$INSTALL_DIR/start-menubar.sh" ]; then
            echo "  $INSTALL_DIR/start-menubar.sh    (menu bar / tray icon)"
        fi
    fi
}

main "$@"
