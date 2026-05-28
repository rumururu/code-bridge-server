# Desktop Packaging — DMG + MSI Build Reference

Concrete, runnable build instructions for the Code Bridge Server desktop
installers. The design rationale, what to bundle, and update strategy live
in `docs/server_app_distribution.md`; this document is just the recipe.

There are three scripts. They are deliberately separated so each step can
be re-run on its own when a previous step succeeded.

| Script | Role | Host requirement |
|---|---|---|
| `scripts/build_desktop_server_app.py` | Freeze the server with PyInstaller, package native helpers, optionally produce DMG (macOS) or MSI (Windows). | macOS for `.app`/`.dmg`. Windows for `.exe`/`.msi`. PyInstaller does **not** cross-compile. |
| `scripts/sign_notarize_desktop_app.py` | Sign + notarize + staple the macOS `.app`/`.dmg`. | macOS with Apple Developer ID Application certificate in the login keychain and App Store Connect API key on disk. |
| `scripts/create_developer_id_application_cert.py` | Create / register a new Apple Developer ID Application certificate non-interactively via App Store Connect API. | macOS. Run once when bootstrapping a new build machine. |

Windows MSI signing is **not** wired into a script yet. The plan is to
delegate it to SignPath Foundation through a GitHub Actions workflow; see
the "MSI signing" section below and
`docs/release_signing/signpath_application.md`.

---

## Output layout

All build outputs land under `dist/desktop_server_app/`.

```text
dist/desktop_server_app/
├── Code Bridge Server.app/              # macOS app bundle (PyInstaller BUNDLE)
├── Code Bridge Server.dmg               # macOS disk image (hdiutil UDZO)
├── Code Bridge Server/                  # Windows / onedir frozen tree
├── Code Bridge Server.exe               # Windows one-file build (if --onefile)
├── Code Bridge Server.msi               # Windows installer (WiX v4)
└── Code Bridge Server.wxs               # Generated WiX source (kept for audit)
```

Intermediate state lives under `build/desktop_server_app/` (PyInstaller
work files, packaged Node.js runtime, Android platform-tools cache, the
generated `.spec`, generated icons). Safe to delete; will be regenerated.

---

## 1. Prerequisites

### macOS host

```bash
# 1. Python 3.13.x (matches server/.venv/lib/python3.13)
brew install python@3.13

# 2. PyInstaller — usually picked up from the server venv
./server/.venv/bin/pip install pyinstaller

# 3. create-dmg replacement is NOT used. We use the built-in `hdiutil`.
xcode-select --install   # installs hdiutil if missing

# 4. Apple Developer ID Application certificate
#    Either:
#      a) Import an existing .p12 into the login keychain, or
#      b) Run scripts/create_developer_id_application_cert.py to mint one
#         via App Store Connect API.

# 5. App Store Connect API key (.p8) on disk and the matching key id + issuer
#    id available, for notarization.
```

### Windows host

```powershell
# 1. Python 3.13.x (matching CPython tag)
winget install Python.Python.3.13

# 2. PyInstaller into the build venv
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pyinstaller -r server\requirements.txt

# 3. WiX Toolset v4 (provides the `wix` CLI)
#    The build script honours the CODEBRIDGE_WIX_PATH env var if `wix`
#    is not on PATH.
dotnet tool install --global wix

# 4. (Optional later) signtool.exe from Windows SDK — only needed if you
#    are signing locally rather than delegating to SignPath. Not required
#    for the SignPath GitHub Actions flow.
```

### Both hosts

The build pulls Node.js and Android platform-tools from official upstream
mirrors the first time. Network access required on the build machine.

---

## 2. Build the unsigned artifact

### macOS `.dmg`

```bash
cd /path/to/code_bridge
./server/.venv/bin/python scripts/build_desktop_server_app.py \
    --format dmg \
    --name "Code Bridge Server" \
    --include-scrcpy
```

Output: `dist/desktop_server_app/Code Bridge Server.app` and
`dist/desktop_server_app/Code Bridge Server.dmg`.

Useful flags:

| Flag | Effect |
|---|---|
| `--include-scrcpy` | Bundle `server/scrcpy/dist` so device mirroring works without a separate Node.js install. Implies `--include-node-runtime` and `--include-platform-tools` by default. |
| `--skip-node-runtime` | Do not bundle Node.js (smaller artifact, requires the user to install Node.js separately). |
| `--skip-platform-tools` | Do not bundle Android `platform-tools` (requires user-installed `adb`). |
| `--node-version 20.18.0` | Pin a different Node.js LTS. Default lives in `DEFAULT_NODE_VERSION` near the top of the script. |
| `--platform-tools-version latest` | Pull the latest Google `platform-tools` instead of the pinned version. |
| `--icon path/to/file.icns` | Override the generated icon (default `build/desktop_server_app/generated_icons/app_icon.icns`). |
| `--bundle-identifier` | Override the macOS bundle id. Default is `com.mkideabox.codebridge.server`. |
| `--dry-run` | Print the PyInstaller command without executing it. Useful for CI debugging. |

### Windows `.msi`

```powershell
py -3.13 scripts\build_desktop_server_app.py `
    --format msi `
    --name "Code Bridge Server" `
    --include-scrcpy
```

Output:

- `dist/desktop_server_app/Code Bridge Server/` — PyInstaller onedir tree.
- `dist/desktop_server_app/Code Bridge Server.wxs` — generated WiX v4
  source. Inspect it to audit which files end up in the MSI.
- `dist/desktop_server_app/Code Bridge Server.msi` — installer.

The MSI is **not signed** at this point. Run SignPath signing in the CI
workflow (see "MSI signing" below) before publishing.

### Build host detection

If you forget the platform check and run `--format dmg` on Windows (or
vice versa) the script aborts with a clear message:

```text
macOS .app/.dmg builds must be run on macOS.
Windows MSI builds must be run on Windows.
```

This is intentional: PyInstaller does not support cross-compilation, so
producing a working cross-built bundle is not possible from a single host.
CI runs both jobs in parallel on platform-matched runners.

---

## 3. macOS — sign + notarize + staple

After `build_desktop_server_app.py --format dmg` produces an **unsigned**
`.app` and `.dmg`, run:

```bash
export NOTARY_KEY_PATH="$HOME/.appstoreconnect/private_keys/AuthKey_XXXX.p8"
export NOTARY_KEY_ID="XXXXXXXXXX"
export NOTARY_ISSUER_ID="00000000-0000-0000-0000-000000000000"
export CODESIGN_IDENTITY="Developer ID Application: Mankil Seo (TEAMID)"

./server/.venv/bin/python scripts/sign_notarize_desktop_app.py \
    --identity "$CODESIGN_IDENTITY" \
    --notary-key "$NOTARY_KEY_PATH" \
    --notary-key-id "$NOTARY_KEY_ID" \
    --notary-issuer "$NOTARY_ISSUER_ID"
```

What this does (in order):

1. `codesign --force --deep --options runtime --timestamp --sign ...` on
   the `.app` bundle.
2. Recreate the DMG from the signed `.app` (so the DMG payload matches
   what was signed).
3. `codesign` the DMG itself.
4. `xcrun notarytool submit --wait` with the App Store Connect API key.
5. `xcrun stapler staple` on both `.app` and `.dmg`.
6. `spctl --assess --type install` as a final Gatekeeper sanity check.

Useful flags:

| Flag | Effect |
|---|---|
| `--skip-dmg-create` | Reuse the existing DMG instead of recreating it. Use when you have already produced a signed DMG and only want to repeat notarization. |
| `--skip-notarize` | Sign and verify only. Useful when you are debugging signing offline. The resulting artifact will still show Gatekeeper warnings to end users; do not ship it. |
| `--app`, `--dmg` | Point at a non-default path (e.g. when building in a CI work directory). |
| `--entitlements` | Use a custom `*.plist`. Default is `desktop_server_app/macos_entitlements.plist`. |

All secret-bearing flags (`--notary-key`, `--notary-key-id`,
`--notary-issuer`, `--identity`) are redacted in the script's command echo,
so CI logs are safe to retain.

### Apple Developer ID certificate bootstrap

If the macOS keychain on a fresh build machine has no `Developer ID
Application` cert, mint one without going through the App Store Connect
web UI:

```bash
./server/.venv/bin/python scripts/create_developer_id_application_cert.py \
    --key "$NOTARY_KEY_PATH" \
    --key-id "$NOTARY_KEY_ID" \
    --issuer "$NOTARY_ISSUER_ID" \
    --team-id "$TEAM_ID" \
    --keychain login.keychain-db
```

The script generates a private key, builds a CSR, calls
`POST /v1/certificates` against App Store Connect, downloads the issued
cert, packages it as `.p12`, and imports it into the requested keychain.
After it succeeds, `sign_notarize_desktop_app.py` will find the identity.

---

## 4. Windows — MSI signing (SignPath)

There is no `sign_msi.py` script today. The plan, documented in
`docs/release_signing/signpath_application.md`, is to do signing from a
GitHub Actions workflow that invokes the official
`signpath-io/github-action-submit-signing-request@v1` action.

Skeleton workflow (`.github/workflows/release-desktop.yml`) — **not yet
checked in**. Add when the SignPath project is approved and the
`SIGNPATH_API_TOKEN` secret has been set:

```yaml
name: Release desktop installers

on:
  push:
    tags: ['v*']

jobs:
  windows-msi:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install WiX v4
        run: dotnet tool install --global wix
      - name: Install server deps + PyInstaller
        run: |
          python -m pip install --upgrade pip
          pip install -r server/requirements.txt pyinstaller
      - name: Build unsigned MSI
        run: |
          python scripts/build_desktop_server_app.py --format msi --include-scrcpy
      - name: Upload unsigned MSI
        uses: actions/upload-artifact@v4
        with:
          name: codebridge-msi-unsigned
          path: dist/desktop_server_app/Code Bridge Server.msi
      - name: Submit to SignPath
        uses: signpath-io/github-action-submit-signing-request@v1
        with:
          api-token: ${{ secrets.SIGNPATH_API_TOKEN }}
          organization-id: ${{ vars.SIGNPATH_ORG_ID }}
          project-slug: code-bridge-server
          signing-policy-slug: release-signing
          artifact-configuration-slug: msi
          github-artifact-id: ${{ steps.upload.outputs.artifact-id }}
          wait-for-completion: true
          output-artifact-directory: dist/desktop_server_app
      - name: Attach signed MSI to release
        uses: softprops/action-gh-release@v2
        with:
          files: dist/desktop_server_app/Code Bridge Server.msi

  macos-dmg:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - name: Install build deps
        run: |
          python -m pip install --upgrade pip
          pip install -r server/requirements.txt pyinstaller
      - name: Import Developer ID cert
        env:
          DEVELOPER_ID_P12_BASE64: ${{ secrets.DEVELOPER_ID_P12_BASE64 }}
          DEVELOPER_ID_P12_PASSWORD: ${{ secrets.DEVELOPER_ID_P12_PASSWORD }}
        run: |
          security create-keychain -p actions build.keychain
          security default-keychain -s build.keychain
          security unlock-keychain -p actions build.keychain
          echo "$DEVELOPER_ID_P12_BASE64" | base64 --decode > cert.p12
          security import cert.p12 \
              -k build.keychain \
              -P "$DEVELOPER_ID_P12_PASSWORD" \
              -T /usr/bin/codesign
          security set-key-partition-list -S apple-tool:,apple:,codesign: \
              -s -k actions build.keychain
      - name: Build DMG
        run: |
          python scripts/build_desktop_server_app.py --format dmg --include-scrcpy
      - name: Sign + notarize + staple
        env:
          NOTARY_KEY_PATH: ${{ secrets.NOTARY_KEY_PATH }}
          NOTARY_KEY_ID: ${{ secrets.NOTARY_KEY_ID }}
          NOTARY_ISSUER_ID: ${{ secrets.NOTARY_ISSUER_ID }}
          CODESIGN_IDENTITY: ${{ secrets.CODESIGN_IDENTITY }}
        run: |
          python scripts/sign_notarize_desktop_app.py \
              --identity "$CODESIGN_IDENTITY" \
              --notary-key "$NOTARY_KEY_PATH" \
              --notary-key-id "$NOTARY_KEY_ID" \
              --notary-issuer "$NOTARY_ISSUER_ID"
      - name: Attach signed DMG to release
        uses: softprops/action-gh-release@v2
        with:
          files: dist/desktop_server_app/Code Bridge Server.dmg
```

When SignPath approves the project, set these in the
`rumururu/code-bridge-server` repository settings:

- Secret `SIGNPATH_API_TOKEN`
- Repository variable `SIGNPATH_ORG_ID`
- (Existing macOS secrets) `DEVELOPER_ID_P12_BASE64`,
  `DEVELOPER_ID_P12_PASSWORD`, `NOTARY_KEY_PATH`, `NOTARY_KEY_ID`,
  `NOTARY_ISSUER_ID`, `CODESIGN_IDENTITY`.

Until SignPath approves, an unsigned MSI can still be produced (omit the
SignPath job). Users will see SmartScreen warnings; flag this in the
GitHub Release description so it does not look like a regression.

---

## 5. Local sanity checks before publishing

```bash
# macOS
codesign --verify --deep --strict --verbose=2 "dist/desktop_server_app/Code Bridge Server.app"
spctl --assess --type install --verbose=2 "dist/desktop_server_app/Code Bridge Server.dmg"
stapler validate "dist/desktop_server_app/Code Bridge Server.dmg"
```

```powershell
# Windows
Get-AuthenticodeSignature "dist\desktop_server_app\Code Bridge Server.msi" | Format-List *
# StatusMessage should be "Signature verified."
```

```bash
# Both
shasum -a 256 dist/desktop_server_app/Code\ Bridge\ Server.dmg
shasum -a 256 "dist/desktop_server_app/Code Bridge Server.msi"
```

Publish the SHA-256 hashes alongside the GitHub Release artifacts so users
can verify what they downloaded matches what was signed.

---

## 6. Smoke test the installed product

1. Move the signed DMG / MSI to a clean machine (or fresh VM snapshot).
2. Install through the OS double-click flow. There must be **no**
   Gatekeeper / SmartScreen blocker. On Windows you should see "Verified
   publisher: SignPath Foundation" (or the certificate's CN if SignPath
   policy uses a different label).
3. Launch the menu-bar / tray icon.
4. Confirm the dashboard opens on `http://127.0.0.1:8766/dashboard`.
5. Confirm provider detection finds `claude`, `codex`, `gemini` if they
   are installed.
6. Confirm QR pairing with the mobile app works on localhost.
7. Stop the server from the tray; confirm the local port closes.

Items that should be reviewed but typically need real users are documented
in `docs/server_app_distribution.md` under "Release checklist".

---

## 7. Common failure modes

| Symptom | Most likely cause | Fix |
|---|---|---|
| `pyinstaller` complains about missing `hiddenimports` for `uvicorn.loops.auto` etc. | Spec file was regenerated and lost hiddenimports list. | Restore from `build/desktop_server_app/spec/Code Bridge Server.spec`; keep the `hiddenimports` block intact. |
| `hdiutil: create failed - Resource busy` | A Finder window still has the previous DMG mounted. | `hdiutil detach /Volumes/Code\ Bridge\ Server` and rerun. |
| `WiX v4 'wix' command was not found` | `wix` not on `PATH`. | Install with `dotnet tool install --global wix` and reopen the shell, or set `CODEBRIDGE_WIX_PATH` to the absolute path. |
| `xcrun notarytool` returns `Invalid` | Hardened runtime missing, or a binary inside the bundle is signed with a wrong identity. | Re-sign with `--options runtime --timestamp`, ensure every nested binary is signed. The script already does this for the top-level binary — extend it if you start bundling new helpers. |
| Build is reproducible on host A but fails on host B | Different Python patch version, different PyInstaller version, or a stale `build/desktop_server_app/` cache. | Pin Python + PyInstaller in the build venv and delete `build/desktop_server_app/work/` before retrying. |
| Windows MSI installs but the service does not start | Firewall blocked port 8767, or the install path includes a non-ASCII character. | Try installing to `C:\Program Files\Code Bridge Server\`. If it works, file a bug; the WiX template should be sanitising the install path already. |

---

## 8. What is NOT in scope of this document

- The design rationale and feature set of the desktop app
  (`docs/server_app_distribution.md`).
- The product-level "agent cockpit" architecture
  (`docs/AGENT_COCKPIT_IMPLEMENTATION_PLAN.md`).
- Mobile app build / release. See `README.md` and `FASTLANE.md` for the
  Flutter side.
- The SignPath Foundation application itself
  (`docs/release_signing/signpath_application.md`) and the project's
  public code signing policy
  (`docs/release_signing/code_signing_policy.md`).
