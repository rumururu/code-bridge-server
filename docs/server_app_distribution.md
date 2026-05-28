# Code Bridge Server App Distribution

This document defines the target distribution strategy for a Code Bridge Desktop Server App. The app exists to make the server installable by people who do not use terminals.

## Product goal

The Desktop Server App should provide one obvious flow:

1. Install Code Bridge Server.
2. Start the local server in the background.
3. Show the dashboard.
4. Help install or detect Claude Code, OpenAI Codex, and Google Gemini CLI.
5. Guide provider login.
6. Show QR pairing.
7. Keep the server updated and recoverable.

The app should not hide security-sensitive actions. When it needs to install dependencies, open browser login, add firewall permissions, start a tunnel, or register autostart, it should explain the action before requesting OS approval.

## What the app should bundle

Bundle:

- Code Bridge server code.
- A known-compatible Python runtime or standalone server binary.
- Python dependencies from `server/requirements.txt`.
- A launcher/tray process.
- Dashboard deep links for `/dashboard` and `/pair`.
- Logs viewer and diagnostics export.
- Optional helper binaries only when licenses and update policy are clear.

Do not bundle user-authenticated AI CLIs by default:

- Claude Code, Codex, and Gemini each have their own accounts, update channels, auth stores, and terms.
- The Server App should detect them, offer official install instructions, and verify login status.
- If a future partner distribution bundles a provider, it must keep provider updates and auth flows intact.

## Platform matrix

| Platform | Primary package | Background start | Notes |
| --- | --- | --- | --- |
| macOS | Signed, notarized `.dmg` or `.pkg` | LaunchAgent | Ship universal or separate arm64/x64 builds. |
| Windows | Signed `.msi` installer | Startup task or Windows service | Prefer per-user install first; request firewall permission only for API access. |
| Linux | `.AppImage` plus `.deb`/`.rpm` | systemd user service | Support Ubuntu/Debian first, then Fedora/RHEL. |

## macOS strategy

Recommended distribution:

- Universal signed `.dmg` for direct download.
- Optional `.pkg` for managed environments.
- Homebrew cask after stable signing and update URLs exist.

Requirements:

- Apple Developer ID signing.
- Notarization and stapling.
- Hardened runtime.
- Clear first-run prompt for local network access.
- LaunchAgent for "Start at Login".
- Dashboard opens at `http://127.0.0.1:8766/dashboard`.

macOS app behavior:

- Menu bar item shows Running, Stopped, Needs Login, or Needs Pairing.
- **Open Dashboard** opens `/dashboard`.
- **Pair Device** opens `/pair`.
- **Check Providers** scans `PATH` and common install paths for `claude`, `codex`, and `gemini`.
- If a CLI is missing, show official install choices and warn against unofficial packages.

## Windows strategy

Recommended distribution:

- Signed `.msi` installer for the default Windows download.
- Avoid `.exe` as the primary artifact because browser and security tooling can interrupt executable downloads more often than MSI installer packages.
- Optional `winget` package after stable releases.

Requirements:

- Authenticode signing.
- Per-user install path by default.
- Start Menu shortcut.
- Optional startup registration.
- Firewall prompt only when local network API access is enabled.
- WebView2 or default-browser dashboard launch.

Windows app behavior:

- Tray icon shows server state.
- **Open Dashboard** opens `http://127.0.0.1:8766/dashboard`.
- **Pair Device** opens `http://127.0.0.1:8766/pair`.
- Provider detection checks user and system `PATH`.
- PowerShell execution policy issues should be handled with signed installers rather than telling users to weaken policy globally.

## Linux strategy

Recommended distribution:

- `.AppImage` for broad desktop compatibility.
- `.deb` for Ubuntu/Debian.
- `.rpm` for Fedora/RHEL.
- Package repository later, after update and signing processes are stable.

Requirements:

- x86_64 first, arm64 when CI coverage exists.
- systemd user service for start at login.
- Desktop entry and tray/status indicator where supported.
- Clear fallback when desktop portals or tray icons are unavailable.

Linux app behavior:

- Open dashboard in the default browser.
- Detect `claude`, `codex`, and `gemini` from the user's login shell environment where possible.
- If the GUI process has a reduced `PATH`, show the exact detected path and a repair action.

## First-run UX

First run should be a short checklist:

1. **Server**: start local dashboard and API.
2. **AI Provider**: detect Claude, Codex, and Gemini.
3. **Login**: ask the user to complete provider login if needed.
4. **Pair**: show QR code and 6-digit code.
5. **Verify**: confirm mobile app pairing and provider readiness.

The app should always offer:

- Open Dashboard.
- Pair Device.
- Provider Settings.
- View Logs.
- Restart Server.
- Stop Server.
- Start at Login toggle.
- Export Diagnostics.

## Provider install handoff

The app should prefer official install methods and show source links.

Claude Code:

- Native installer: `curl -fsSL https://claude.ai/install.sh | bash`
- Homebrew: `brew install --cask claude-code`
- npm: `npm install -g @anthropic-ai/claude-code`
- Login check: run `claude` and guide `/login` if prompted.

OpenAI Codex:

- npm: `npm install -g @openai/codex`
- Homebrew: `brew install --cask codex`
- Login check: run `codex` and choose ChatGPT sign-in or configured API key flow.

Google Gemini CLI:

- npm: `npm install -g @google/gemini-cli`
- Homebrew: `brew install gemini-cli`
- Login check: run `gemini` and choose Google login, API key, or Vertex AI.

For non-technical users, the Desktop App should show these as buttons, not as a wall of commands. If a button runs a command, show the command and ask for confirmation.

## AI CLI delegated install

The distribution page should include an "Ask your AI CLI to install Code Bridge" option for users who already trust one provider. This flow should:

- Provide a copyable prompt.
- Tell the AI CLI to use official Code Bridge releases or the user-provided source checkout.
- Tell the AI CLI to avoid unofficial downloads.
- Require verification of dashboard, provider login, and QR pairing.
- Leave a human-readable install log.

This is a support path, not the primary consumer path. The Desktop App remains the preferred experience.

## Updates

Recommended channels:

- Stable: default for all users.
- Preview: opt-in for users helping test packaging or provider changes.
- Development: local builds only.

Update rules:

- Never update while an active chat turn is running.
- Preserve `server_info.json`, paired clients, provider auth, project list, and user settings.
- Show release notes before major upgrades.
- Keep one rollback version where platform conventions allow it.
- Verify server health after update.

## Security and privacy

The Server App should:

- Bind the dashboard to localhost by default.
- Keep dashboard and API ports separate.
- Require QR pairing for mobile access.
- Store per-device API keys in the existing server data store.
- Show paired devices and allow revocation.
- Avoid exposing dashboard routes through tunnels.
- Make remote access opt-in or clearly visible when enabled.
- Avoid asking users to paste secrets into support chats.

Installer security:

- Sign every release artifact.
- Publish checksums.
- Use HTTPS release downloads.
- Do not recommend disabling Gatekeeper, SmartScreen, antivirus, or Linux package verification.

## Diagnostics

The app should collect a support bundle with:

- Server version.
- OS version and CPU architecture.
- Dashboard/API port values.
- Server health response.
- Provider detection results.
- Recent server logs with secrets redacted.
- Paired device count, not API keys.
- Tunnel status, not private tokens.

The bundle should not include project source files, provider credentials, API keys, or chat transcripts unless the user explicitly chooses to attach them.

## Release checklist

- macOS artifact is signed, notarized, and opens without Gatekeeper bypass.
- Windows artifact is signed and does not trigger unknown-publisher warnings.
- Linux artifacts install on supported distributions.
- Fresh install opens dashboard.
- Upgrade preserves pairing and settings.
- Uninstall leaves user data only when the user chooses to keep it.
- Claude, Codex, and Gemini detection works when installed through npm and Homebrew.
- QR pairing works locally.
- Firewall and local network prompts are documented.
- Logs and diagnostics are accessible from the app.
