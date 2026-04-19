"""
Code Bridge Server Domain Organization
=======================================

Server modules are organized by domain for maintainability.

Domain Structure
----------------

**core/** - Core utilities and configuration
    - config.py           # Server configuration
    - database.py         # SQLite database management
    - server_logging.py   # Logging configuration
    - base_result.py      # Base result types
    - rate_limiter.py     # Rate limiting

**chat/** - Chat services and WebSocket handling
    - chat_session_service.py   # Session management
    - chat_stream_service.py    # SSE streaming
    - chat_ws_service.py        # WebSocket service
    - chat_event_utils.py       # Event utilities

**llm/** - AI model integrations
    - llm_session.py      # Base LLM session
    - llm_settings.py     # LLM settings management
    - claude_session.py   # Claude (Anthropic) integration
    - claude_usage.py     # Usage tracking
    - codex_session.py    # Codex (OpenAI) integration

**pairing/** - Device pairing and QR code services
    - pairing.py              # Route definitions
    - pairing_service.py      # Core pairing logic
    - pairing_models.py       # Pairing data models
    - pairing_qr_service.py   # QR code generation
    - pairing_page_service.py # Web page service
    - pairing_page.py         # Page rendering
    - qr_display.py           # Terminal QR display
    - mdns_service.py         # mDNS discovery

**projects/** - Project management services
    - project_manager.py          # Core project management
    - project_models.py           # Project data models
    - project_query_service.py    # Query operations
    - project_action_service.py   # Action operations
    - project_device_run_service.py  # Device run service
    - project_device_run_plan.py     # Run planning
    - project_device_logs.py         # Log management
    - project_dev_server.py          # Dev server management
    - project_dev_server_start.py    # Server startup
    - project_server_detection.py    # Server type detection
    - project_server_process.py      # Process management
    - project_utils.py               # Utilities
    - project_process_utils.py       # Process utilities
    - project_runtime_helpers.py     # Runtime helpers
    - project_view.py                # View generation
    - projects.py                    # Route definitions

**files/** - File system operations and git
    - file_manager.py         # Core file operations
    - file_manager_helpers.py # File utilities
    - file_action_service.py  # Action service
    - filesystem_service.py   # Filesystem operations
    - files.py                # Route definitions
    - git_manager.py          # Git operations
    - attachment_parser.py    # Attachment parsing

**dashboard/** - Web dashboard
    - dashboard_service.py      # Dashboard logic
    - dashboard_auth_service.py # Dashboard authentication
    - dashboard_page.py         # Page rendering

**preview/** - Web preview services
    - preview_service.py       # Preview management
    - preview_access.py        # Access control
    - preview_route_service.py # Route handling
    - preview.py               # Route definitions

**devices/** - Device management
    - device_action_service.py # Device actions
    - scrcpy_manager.py        # Scrcpy streaming

**auth/** - Authentication
    - auth_service.py   # Core auth service
    - firebase_auth.py  # Firebase integration

**remote/** - Remote access and tunnel
    - remote_access_service.py  # Remote access management
    - remote_access_models.py   # Data models
    - tunnel_service.py         # Tunnel service
    - heartbeat_settings.py     # Heartbeat configuration

**system/** - System services
    - system_inspect_service.py   # System inspection
    - system_settings_service.py  # Settings management
    - system_status_service.py    # Status reporting
    - lifecycle_service.py        # Lifecycle management
    - optional_services.py        # Optional service loading
    - autostart_service.py        # Autostart management

**models.py** - Shared API request/response models (server root)
    - Project models (ProjectCreate, ProjectUpdate, etc.)
    - File models (FileWrite, FileCreate)
    - Git models (GitCommit, GitBranch, etc.)
    - LLM models (LlmSelectionUpdate, CodexSettingsUpdate)
    - Pairing models (PairVerifyRequest, SSOPairRequest, etc.)

Import Style
------------
Import directly from domain modules:
    from core.config import get_config
    from projects.project_manager import get_project_manager
    from files.file_manager import get_file_manager

Backward Compatibility
----------------------
The services/ package provides re-exports for legacy imports:
    from services import PairingService, get_pairing_service
"""

__all__ = []
