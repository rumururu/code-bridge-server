# Install Code Bridge Server

This guide is for people who want to use Code Bridge without becoming terminal users. Code Bridge needs a small server running on your computer so the mobile app can talk to your local projects and AI coding CLI tools.

## Choose an install path

| Path | Best for | What you do |
| --- | --- | --- |
| Desktop Server App | Most users | Download the app, sign in to an AI CLI if needed, scan the QR code. |
| Ask an AI CLI to install it | Users who already use Claude, Codex, or Gemini | Paste one prompt into the AI CLI and let it run the setup. |
| Advanced terminal install | Developers and support staff | Install Python dependencies and start the server manually. |

## Path 1: Desktop Server App

Use this path when an official Code Bridge Server App installer is available for your operating system.

1. Download the installer from the official Code Bridge release page.
2. Open the installer:
   - macOS: open the `.dmg` or `.pkg`, then drag or install Code Bridge Server.
   - Windows: run the signed `.msi` installer.
   - Linux: install the `.AppImage`, `.deb`, or `.rpm` package for your distribution.
3. Start **Code Bridge Server**.
4. Open the server dashboard from the app, or visit `http://127.0.0.1:8766/dashboard`.
5. In **AI Providers**, choose the CLI you want Code Bridge to use:
   - Claude Code: `claude`
   - OpenAI Codex: `codex`
   - Google Gemini CLI: `gemini`
6. If the dashboard says a provider is missing, use the dashboard install button or follow the provider instructions below.
7. Open **Pair Device** in the dashboard.
8. In the Code Bridge mobile app, choose **Add Server** and scan the QR code.
9. Confirm the new device appears in the dashboard under paired devices.

The dashboard runs locally on your computer. The default dashboard address is `http://127.0.0.1:8766/dashboard`. The app API normally uses the next port, `8767`.

## Install and sign in to AI CLI providers

Code Bridge does not replace the AI CLI account setup. The server can only use providers that are installed and authenticated on the same computer.

### Claude Code

Install one of these official options:

```bash
# Native installer, macOS/Linux/WSL
curl -fsSL https://claude.ai/install.sh | bash

# Homebrew, macOS/Linux
brew install --cask claude-code

# npm, Node.js 18+
npm install -g @anthropic-ai/claude-code
```

Then sign in:

```bash
claude
```

If prompted, use `/login` and complete the browser sign-in. Confirm installation with:

```bash
claude --version
```

### OpenAI Codex

Install one official option:

```bash
# npm
npm install -g @openai/codex

# Homebrew, macOS
brew install --cask codex
```

Then sign in:

```bash
codex
```

Choose **Sign in with ChatGPT** when prompted, or configure API key auth if your organization requires it. Confirm installation with:

```bash
codex --version
```

### Google Gemini CLI

Install one official option:

```bash
# npm
npm install -g @google/gemini-cli

# Homebrew, macOS/Linux
brew install gemini-cli
```

Then sign in:

```bash
gemini
```

Choose **Login with Google** unless your organization requires an API key or Vertex AI. Confirm installation with:

```bash
gemini --version
```

Security note: install Gemini CLI only from `@google/gemini-cli` or the official `google-gemini/gemini-cli` repository. Avoid similarly named packages.

## Path 2: Ask an AI CLI to install Code Bridge

Use this path if you already have Claude Code, Codex, or Gemini CLI installed and logged in. Open your terminal in the folder where you want Code Bridge installed, start your AI CLI, and paste this prompt:

```text
Install and start Code Bridge Server on this computer.

Use the official repository or release bundle provided by the user. Do not install unofficial packages. Prefer the Desktop Server App when available. If installing from source, use Python 3.11 or newer, create a virtual environment, install server/requirements.txt, and start server/main.py. When the server starts, open or print http://127.0.0.1:8766/dashboard and http://127.0.0.1:8766/pair. Verify that the dashboard loads, check whether claude, codex, or gemini is installed and authenticated, and explain exactly how to pair the mobile app by QR code.
```

Let the AI CLI run commands only after you read the command it proposes. Do not approve commands that download from unknown websites, disable security software, expose all ports, or ask for passwords outside the normal OS/browser sign-in flow.

## Path 3: Advanced terminal install

Use this path when you are installing from a source checkout.

### macOS or Linux

```bash
cd server
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### Windows PowerShell

```powershell
cd server
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

When the server is running, open:

- Dashboard: `http://127.0.0.1:8766/dashboard`
- Pairing page: `http://127.0.0.1:8766/pair`
- Health check: `http://127.0.0.1:8767/api/health`

Useful server options:

```bash
python main.py --port 8766
python main.py --show-qr
python main.py --qr-only
```

## QR pairing

1. Keep Code Bridge Server running on your computer.
2. Open `http://127.0.0.1:8766/pair`.
3. Open the Code Bridge mobile app.
4. Choose **Add Server**.
5. Scan the QR code or enter the 6-digit pairing code.
6. Wait for the mobile app to show the server name.
7. Return to the dashboard and confirm the device appears as paired.

The QR code contains a short-lived pairing token. If it expires, refresh the pairing page and scan again.

## Dashboard checklist

After setup, the dashboard should show:

- Server status is running.
- Dashboard port is `8766`.
- API port is `8767`.
- At least one AI provider is installed and authenticated.
- Your mobile device appears under paired devices.
- Remote access or tunnel status matches your intended use.
- Projects are visible or can be imported.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| Dashboard does not open | Make sure the server app is running. Try `http://127.0.0.1:8766/dashboard`. If you changed the port, use that port. |
| QR scan fails | Refresh `/pair`, keep the computer and mobile device online, and scan the newest QR code. |
| Mobile app cannot connect on local Wi-Fi | Confirm both devices are on the same network. Check OS firewall prompts and allow Code Bridge Server. |
| Remote access does not work | Check the dashboard tunnel status. Pair locally first, then enable remote access. |
| Claude/Codex/Gemini is missing | Install the provider CLI, restart Code Bridge Server, then check the dashboard again. |
| Provider is installed but not authenticated | Run `claude`, `codex`, or `gemini` once in a terminal and complete browser login. |
| `command not found` | Restart the terminal or desktop server app. Make sure the CLI install directory is in `PATH`. |
| Port already in use | Stop the other server or start Code Bridge with another dashboard port, for example `python main.py --port 8770`. |
| Windows blocks the app | Use the signed installer. If Windows Defender SmartScreen appears, verify the publisher before continuing. |
| macOS says the app is damaged or unidentified | Use a signed and notarized build. Avoid bypassing Gatekeeper for unknown downloads. |

## Official provider references

- Claude Code quickstart: https://docs.claude.com/en/docs/claude-code/quickstart
- Claude Code CLI reference: https://code.claude.com/docs/en/cli-reference
- OpenAI Codex CLI repository: https://github.com/openai/codex
- OpenAI Codex CLI help: https://help.openai.com/en/articles/11096431-openai-codex-cli-getting-started
- Gemini CLI documentation: https://google-gemini.github.io/gemini-cli/docs/get-started/
- Gemini CLI authentication: https://google-gemini.github.io/gemini-cli/docs/get-started/authentication.html
