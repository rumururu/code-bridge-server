#!/bin/bash
# Code Bridge Server - Installation Script for macOS/Linux
# Usage: curl -fsSL https://raw.githubusercontent.com/rumururu/code-bridge-server/8817ce8/install/install.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

INSTALL_DIR="${CODE_BRIDGE_INSTALL_DIR:-$HOME/.code-bridge}"
REPO_URL="https://github.com/rumururu/code-bridge-server.git"
MIN_PYTHON_VERSION="3.10"

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

    if prompt_yes_no "Install cloudflared for remote access? [y/N] "; then
        echo -e "${CYAN}Installing cloudflared...${NC}"

        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install cloudflare/cloudflare/cloudflared
            else
                echo -e "${YELLOW}Homebrew not found. Please install cloudflared manually.${NC}"
                echo "See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            fi
        elif [[ -f /etc/debian_version ]]; then
            # Debian/Ubuntu
            curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
            sudo dpkg -i cloudflared.deb
            rm cloudflared.deb
        else
            echo -e "${YELLOW}Please install cloudflared manually for your OS.${NC}"
            echo "See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        fi
    fi
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
    echo -e "${CYAN}Setting up Code Bridge Server...${NC}"

    if [ -d "$INSTALL_DIR" ]; then
        echo "Updating existing installation..."
        cd "$INSTALL_DIR"
        git fetch origin
        git reset --hard origin/main
    else
        echo "Installing to $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    echo -e "${GREEN}✓ Repository ready${NC}"
}

# Create virtual environment and install dependencies
setup_venv() {
    echo ""
    echo -e "${CYAN}Setting up Python environment...${NC}"

    cd "$INSTALL_DIR"

    # Create venv if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    fi

    # Activate venv and install dependencies
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q

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
    echo -e "${GREEN}✓ Start script created${NC}"
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

    if [[ "${CODE_BRIDGE_AUTO_START:-1}" == "1" ]]; then
        run_server
    else
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║   Installation Complete!               ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
        echo ""
        echo "Start Code Bridge Server with:"
        echo "  $INSTALL_DIR/start.sh"
    fi
}

main "$@"
