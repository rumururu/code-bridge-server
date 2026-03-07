# Code Bridge Server

A bridge server that connects Claude Code (or other LLM CLI tools) with the Code Bridge mobile app, enabling remote development workflows.

## Mobile App

Download the Code Bridge app to connect to this server:

[![Google Play](https://img.shields.io/badge/Google%20Play-Download-green?logo=google-play)](https://play.google.com/store/apps/details?id=com.mkideabox.codeBridge)
[![App Store](https://img.shields.io/badge/App%20Store-Download-blue?logo=apple)](https://apps.apple.com/app/code-bridge/id6740008029)

## Requirements

- Python 3.11+
- ADB (Android Debug Bridge) - for device mirroring features

## Installation

```bash
cd server
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

This starts two servers:
- **Dashboard**: `http://127.0.0.1:8766` (localhost only)
- **API**: `http://0.0.0.0:8767` (tunnel-exposed for external access)

### CLI Options

```bash
python main.py --port 8766        # Custom dashboard port (API = port + 1)
python main.py --single           # Legacy single-server mode
python main.py --show-qr          # Show QR code before starting
python main.py --qr-only          # Show QR code only (don't start server)
```

## Architecture

### Dual-Server Design

```
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│      Dashboard Server           │  │         API Server               │
│      127.0.0.1:8766             │  │      0.0.0.0:8767                │
│  ┌───────────────────────────┐  │  │  ┌───────────────────────────┐   │
│  │  /dashboard, /pair, /     │  │  │  │  /api/*, /ws/*, /health   │   │
│  │  /api/dashboard/auth/*    │  │  │  │  /preview/*               │   │
│  │  /api/filesystem/*        │  │  │  └───────────────────────────┘   │
│  └───────────────────────────┘  │  │                │                  │
│                                  │  │     Cloudflare Tunnel            │
│  localhost only (not exposed)    │  │     (this port is exposed)       │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

**Security Benefits:**
- Dashboard is physically isolated from tunnel exposure
- No URL path filtering required (Quick Tunnel doesn't support it)
- Local-only endpoints cannot be accessed externally

## Configuration

Create `config.yaml` in the project root (parent of `server/`):

```yaml
server:
  host: "0.0.0.0"
  port: 8766
  server_name: "My Dev Machine"
  debug: true
  log_level: "info"
  cors_origins: ["*"]

  # Remote access via Cloudflare Tunnel
  remote_access_enabled: true

  # Firebase authentication (optional)
  firebase_enabled: false

  # LLM usage tracking
  weekly_budget_usd: 100.0
  usage_window_days: 7

  # Heartbeat interval for presence updates (minutes)
  heartbeat_interval_minutes: 15
```

## Features

### Core Features

- **Project Management**: Create, import, list, and manage development projects
- **LLM Chat Sessions**: WebSocket-based chat with Claude Code, Codex, or other LLM CLI tools
- **Dev Server Proxy**: Preview Next.js, Vite, Flutter web, and other dev servers remotely
- **File Operations**: Browse, read, write, search files within projects
- **Android Device Mirroring**: Stream device screen via scrcpy integration

### Security Features

- **QR Pairing**: Secure device pairing via QR code or 6-digit code
- **API Key Authentication**: Per-device API keys for authorized access
- **IP Login Mode**: Optional anonymous access for development/testing
- **Preview Tokens**: Time-limited tokens for browser-based preview access
- **Cloudflare Tunnel**: Secure remote access without port forwarding

### Dashboard

Web-based dashboard at `/dashboard` showing:
- Server status and version
- Connected/paired devices with status
- Project list and dev server status
- LLM configuration
- Network/tunnel status
- Server logs

## Security

### Dual-Server Security Model

The server uses physical port separation for security:

| Server | Binding | Tunnel Exposed | Purpose |
|--------|---------|----------------|---------|
| Dashboard | `127.0.0.1:8766` | ❌ No | Local admin UI, pairing page, filesystem |
| API | `0.0.0.0:8767` | ✅ Yes | Mobile app API, preview, WebSocket |

**Benefits:**
- Dashboard is physically isolated from tunnel exposure
- No URL path filtering required (Quick Tunnel doesn't support it)
- Local-only endpoints cannot be accessed externally

### Authentication Methods

| Method | Use Case | Mechanism |
|--------|----------|-----------|
| API Key | REST API, WebSocket | `X-API-Key` header or `api_key` query param |
| Preview Token | Browser/WebView preview | URL query param + IP-based session |
| Localhost | Dashboard settings | No auth for `127.0.0.1` requests |
| IP Login | Development/testing | No authentication required (local only) |

### Access Control Dependencies

| Dependency | Behavior |
|------------|----------|
| `require_local_access` | Blocks tunnel requests (403), allows local only |
| `verify_api_key` | Requires valid API key (blocks tunnel without key) |
| `verify_api_key_or_localhost` | Allows localhost without key, requires key for others |

### Access Control Flow

1. **Dashboard Server** (localhost only):
   - All requests must come from `127.0.0.1`
   - Tunnel requests are physically impossible (different port)

2. **API Server** (tunnel exposed):
   - **Tunnel Access**: Always requires API key (QR pairing)
   - **Local Access**: IP Login mode allows anonymous access
   - **Preview Access**: Token-based session for WebView

### Dashboard Server Endpoints (Localhost Only)

- `GET /` - Redirect to dashboard
- `GET /dashboard` - Admin dashboard page
- `GET /pair` - QR pairing page (HTML)
- `GET /api/dashboard/auth/*` - Dashboard authentication
- `GET /api/filesystem/*` - File browser
- `GET /api/system/ip-login` - IP login settings
- `PUT /api/system/ip-login` - Update IP login
- `GET /api/system/network-status` - Network/tunnel status
- `POST /api/system/tunnel/start` - Start tunnel
- `POST /api/system/tunnel/stop` - Stop tunnel

### API Server Endpoints (Tunnel Exposed)

**Public (No Auth Required):**
- `GET /api/health` - Health check
- `GET /api/pair/qr` - QR code data for pairing
- `GET /api/pair/qr-image` - QR code image
- `POST /api/pair/verify` - Verify pairing token

**Protected (API Key Required):**
- `GET /api/projects` - List projects
- `POST /api/projects/{name}/start` - Start dev server
- `GET /ws/chat/{project}` - WebSocket chat
- All other `/api/*` endpoints

### Preview Token System

Browsers cannot add custom headers to static resource requests (JS, CSS, images).
The preview token system solves this:

1. App requests preview token with API key (`POST /api/preview/token`)
2. Token included in WebView URL (`/preview/project/?preview_token=xxx`)
3. Server validates token and creates IP-based session
4. Subsequent requests from same IP use session auth

**Token Properties:**
- Validity: 15 minutes
- Single-use: Token role ends after session creation
- Auto-refresh: App requests new token on reload/project switch

## API Endpoints

### Health & System

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/health` | GET | No | Server health check |
| `/api/system/overview` | GET | No | System overview (server, projects, devices) |
| `/api/system/server-log` | GET | No | Server log tail |
| `/api/system/check-update` | POST | No | Check for updates via git pull |

### Pairing

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/pair/qr` | GET | No | Get QR code data |
| `/api/pair/qr-image` | GET | No | Get QR code as PNG image |
| `/api/pair/code` | POST | No | Request 6-digit pairing code |
| `/api/pair/verify` | POST | No | Verify pairing and get API key |
| `/api/pair/status` | GET | Yes | Get pairing/client status |
| `/api/pair/clients/{id}` | DELETE | Yes | Remove paired client |

### Projects

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/projects` | GET | Yes | List all projects |
| `/api/projects` | POST | Yes | Create project |
| `/api/projects/import` | POST | Yes | Bulk import projects |
| `/api/projects/{name}` | GET | Yes | Get project details |
| `/api/projects/{name}` | PUT | Yes | Update project |
| `/api/projects/{name}` | DELETE | Yes | Delete project |
| `/api/projects/{name}/start` | POST | Yes | Start dev server |
| `/api/projects/{name}/stop` | POST | Yes | Stop dev server |
| `/api/projects/{name}/build` | POST | Yes | Build project (Flutter web) |
| `/api/projects/{name}/build-status` | GET | Yes | Get build status |
| `/api/projects/{name}/run-device` | POST | Yes | Run on Android device |
| `/api/projects/{name}/stop-device-run` | POST | Yes | Stop device run |
| `/api/projects/{name}/device-run-log` | GET | Yes | Get device run logs |

### Files

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/projects/{name}/files` | GET | Yes | List files in project |
| `/api/projects/{name}/files/content` | GET | Yes | Read file content |
| `/api/projects/{name}/files/content` | PUT | Yes | Write file content |
| `/api/projects/{name}/files` | POST | Yes | Create file/directory |
| `/api/projects/{name}/files` | DELETE | Yes | Delete file/directory |
| `/api/projects/{name}/files/rename` | POST | Yes | Rename file |
| `/api/projects/{name}/files/copy` | POST | Yes | Copy file |
| `/api/projects/{name}/files/move` | POST | Yes | Move file |
| `/api/projects/{name}/files/search` | GET | Yes | Search files by name |
| `/api/projects/{name}/files/search-content` | GET | Yes | Search file contents |
| `/api/projects/{name}/files/upload` | POST | Yes | Upload file |

### Devices

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/devices` | GET | Yes | List connected Android devices |
| `/api/scrcpy/status` | GET | Yes | Get scrcpy streaming status |
| `/api/scrcpy/start` | POST | Yes | Start scrcpy streaming |
| `/api/scrcpy/stop` | POST | Yes | Stop scrcpy streaming |

### Preview

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/preview/token` | POST | Yes | Get preview token |
| `/preview/{project}/{path}` | GET | Token | Proxy to dev server |
| `/build-preview/{name}/{path}` | GET | Token | Serve built files |

### LLM Settings

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/system/llm/options` | GET | Yes | Get LLM provider options |
| `/api/system/llm/selection` | PUT | Yes | Set LLM provider/model |
| `/api/system/llm/codex/settings` | GET | Yes | Get Codex settings |
| `/api/system/llm/codex/settings` | PUT | Yes | Update Codex settings |
| `/api/system/usage` | GET | Yes | Get LLM usage statistics |

### System Settings

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/system/ip-login` | GET | Yes | Get IP login setting |
| `/api/system/ip-login` | PUT | Yes | Set IP login (true/false) |
| `/api/system/heartbeat` | GET | Yes | Get heartbeat settings |
| `/api/system/heartbeat` | PUT | Yes | Set heartbeat interval |
| `/api/system/network-status` | GET | Yes | Get network/tunnel status |
| `/api/system/tunnel/start` | POST | Yes | Start Cloudflare tunnel |
| `/api/system/tunnel/stop` | POST | Yes | Stop Cloudflare tunnel |
| `/api/system/directories` | GET | Yes | Browse server directories |
| `/api/system/project-candidates` | GET | Yes | Scan for project candidates |

### Filesystem

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/filesystem/accessible-folders` | GET | No | List accessible folders |
| `/api/filesystem/accessible-folders` | POST | No | Add accessible folder |
| `/api/filesystem/accessible-folders` | DELETE | No | Remove accessible folder |
| `/api/filesystem/browse` | GET | No | Browse directory |
| `/api/filesystem/quick-access` | GET | No | Get quick access paths |

### WebSocket

| Endpoint | Auth | Description |
|----------|------|-------------|
| `/ws/chat/{project}` | Query param | LLM chat session |
| `/ws/claude/{project}` | Query param | Alias for chat (backwards compat) |

## Architecture

```
server/
├── main.py                 # Entry point
├── app_factory.py          # FastAPI app creation
├── config.py               # Configuration loader
├── database.py             # SQLite database access
├── pairing.py              # QR pairing service
├── auth_service.py         # API key validation
├── routes/                 # API route modules
│   ├── health.py
│   ├── pairing.py
│   ├── projects.py
│   ├── files.py
│   ├── devices.py
│   ├── preview.py
│   ├── chat_ws.py
│   ├── dashboard.py
│   ├── system_settings.py
│   └── ...
├── *_service.py            # Business logic services
├── *_session.py            # LLM session handlers
└── tests/                  # Test files
```

## Data Storage

- **SQLite Database**: `server/data/code_bridge.db`
  - Projects
  - System settings
  - Accessible folders

- **JSON Files**: `server/data/`
  - `api_keys.json` - Paired client API keys
  - `device_info.json` - Firebase device info
  - `preview_tokens.json` - Active preview tokens

## License

MIT
