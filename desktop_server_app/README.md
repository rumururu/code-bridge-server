# Desktop server app packaging

This directory contains the cross-platform tray/menu-bar launcher for the Code
Bridge server app. The launcher starts/stops the existing Python server, opens
the existing dashboard, opens the QR pairing page, and can register itself to
start at login without requiring the user to type terminal commands.

## Build script

Script:

```sh
python scripts/build_desktop_server_app.py --dry-run
```

Default entry point:

```text
desktop_server_app/launcher.py
```

## Prerequisites

Install server dependencies in the Python environment used for packaging:

```sh
python -m pip install -r server/requirements.txt
python -m pip install pyinstaller
```

The script fails early with a clear error if `pyinstaller` is missing. DMG
creation additionally requires macOS `hdiutil`.

PyInstaller does not cross-compile. Build each platform artifact on that
platform. The macOS artifact can be built on macOS; Windows `.msi`
artifacts must be built on Windows.

```sh
# macOS .app with Android streaming runtime
python scripts/build_desktop_server_app.py --format app --include-scrcpy

# macOS .dmg with Android streaming runtime
python scripts/build_desktop_server_app.py --format dmg --include-scrcpy

# Windows MSI (requires WiX Toolset v4 on Windows)
python scripts/build_desktop_server_app.py --format msi

# Linux one-folder build
python scripts/build_desktop_server_app.py --format onedir
```

When `--include-scrcpy` is enabled, the build also packages an official Node.js
runtime and official Android platform-tools by default. The packaged server
uses `server/vendor/node/bin/node` and `server/vendor/platform-tools/adb` before
checking the user's PATH, so Android streaming does not require separate Node or
ADB installations on the user's machine. Override the bundled Node runtime with
`--node-version`, override platform-tools with `--platform-tools-version`, and
disable either only for developer diagnostics with `--skip-node-runtime` or
`--skip-platform-tools`.

Bundled downloads are verified before extraction. Node.js archives are checked
against the official Node `SHASUMS256.txt`; Android platform-tools archives are
checked against Google SDK repository metadata. Reusable staged Node and
platform-tools directories also carry marker metadata for the source archive, so
a changed or mismatched archive is re-extracted instead of silently reusing an
older staging directory.

Default output directory:

```text
dist/desktop_server_app
```

## Tray and menu-bar behavior

Packaged desktop builds run as a tray/menu-bar app, not a foreground control
window:

- macOS `.app` and `.dmg` builds set `LSUIElement=true`, so the app appears in
  the top menu bar as `CB` and does not stay in the Dock.
- Windows `.msi` builds use PyInstaller windowed mode and show a
  system tray icon instead of a console window.
- The tray/menu-bar menu contains status, API URL, Start Server, Stop Server,
  Open Dashboard, Show QR Pairing, Start at Login, and Quit.

## Packaged server paths

The script adds `server` to PyInstaller's import path and stages a clean copy of
the server source tree into the app bundle. The launcher starts the server by
re-invoking the same packaged executable in `--server-child` mode.

Packaged data paths:

```text
build/desktop_server_app/packaged_server   -> server
```

Device mirroring assets are optional. The app packages a filtered staging copy
of `server/scrcpy/dist`, not the full `server/scrcpy` development tree. This
keeps WebDriverAgent/XCTest test bundles and other non-runtime development
assets out of the notarized desktop app while preserving the Android ws-scrcpy
runtime:

```sh
python scripts/build_desktop_server_app.py --include-scrcpy
```

PyInstaller can reclassify native addons such as `pty.node` into
`Contents/Frameworks`. The build script restores Node-loadable native addons to
the original `Contents/Resources/server/scrcpy/dist` module path after the
PyInstaller build so the runtime can resolve them normally. The bundled Node
executable and Android platform-tools are similarly restored to
`Contents/Resources/server/vendor` on macOS before signing/notarization.

The script does not package local runtime/configuration state such as
`server/firebase_config.json`, `server/server_info.json`, `server/config.yaml`,
API key files, local SQLite databases, backup files, `.env` files, PID files,
logs, virtualenvs, tests, the `server/scrcpy` development tree, or matching
state files found under the optional scrcpy dist staging copy. Runtime
dependencies required by scrcpy dist, including production `node_modules`
content, remain packageable after this filter. Packaged desktop launches set
`CODEBRIDGE_APP_SUPPORT_DIR`, and current server runtime path helpers route
mutable state there instead of into the signed app bundle.

## macOS signing

The bundled Node.js runtime uses V8 JIT memory, so the macOS app must be signed
with the entitlements file in this directory before creating and notarizing the
DMG. The helper script signs the app, recreates/signs the DMG, submits it for
notarization, staples the ticket, validates the result, and prints the SHA-256:

```sh
CODEBRIDGE_CODESIGN_IDENTITY="Developer ID Application: mkideabox Co. Ltd. (3SAMRT9KZD)" \
CODEBRIDGE_NOTARY_KEY="ios/fastlane/AuthKey.p8" \
CODEBRIDGE_NOTARY_KEY_ID="..." \
CODEBRIDGE_NOTARY_ISSUER="..." \
python scripts/sign_notarize_desktop_app.py
```

Manual signing uses the same entitlements:

```sh
codesign --force --deep --options runtime --timestamp \
  --entitlements desktop_server_app/macos_entitlements.plist \
  --sign "Developer ID Application: mkideabox Co. Ltd. (3SAMRT9KZD)" \
  "dist/desktop_server_app/Code Bridge Server.app"
```

## Release upload

Built packages can be uploaded to GitHub Releases:

```sh
python scripts/publish_desktop_release.py desktop-server-v1.0.0 \
  --title "Code Bridge Server v1.0.0" \
  --draft
```

The script requires the GitHub CLI:

```sh
gh auth login
```

The uploader accepts generated `.dmg`, `.pkg`, `.msi`, `.zip`,
`.AppImage`, `.deb`, and `.rpm` artifacts. It does not build missing platforms;
run the build script on each target OS first, then upload the outputs to one
release tag.

## Current limitations

- macOS `.app` builds use PyInstaller windowed mode and `LSUIElement=true`, so
  the app opens without a terminal or Dock icon.
- If `pystray` cannot load on a platform, the launcher falls back to a headless
  mode that starts the server and opens the dashboard in the browser.
- Runtime state is expected to go through `CODEBRIDGE_APP_SUPPORT_DIR`; new
  mutable state files should use the server runtime path helpers before public
  distribution.
- Signing, notarization, Windows code signing, and Linux desktop entries are
  release pipeline work before a public non-warning download.
