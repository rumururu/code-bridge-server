# SignPath Foundation Application — Code Bridge Server

Draft answers for the SignPath Foundation open-source application form
(<https://signpath.org/apply>). Copy each section into the matching field on
the form. Where the form asks something not listed here, fall back to the
information in `docs/server_app_distribution.md` and this repository.

Submit from an email address tied to the GitHub account that owns
`rumururu/code-bridge-server`. SignPath verifies maintainer identity against
the source repository.

---

## 1. Project identity

| Field | Value |
|---|---|
| **Project name** | Code Bridge Server |
| **Short tagline** | Local-first AI Agent Cockpit server. Runs your own Claude / Codex / Gemini CLI against your local repositories under a policy + approval + audit boundary. Mobile clients (iPad / Android) drive it remotely. |
| **Project homepage** | <https://github.com/rumururu/code-bridge-server> |
| **Source repository** | <https://github.com/rumururu/code-bridge-server> (public, MIT) |
| **License** | MIT (OSI-approved, no commercial dual-licensing) |
| **Programming languages** | Python (server), TypeScript (scrcpy bundle), PowerShell + Bash (install scripts) |
| **Primary platforms requested** | Windows (MSI), macOS (DMG — already signed via Apple Developer ID; SignPath is requested for Windows only) |

## 2. Maintainer / signing team

| Field | Value |
|---|---|
| **Primary maintainer name** | Mankil Seo |
| **Primary maintainer GitHub** | rumururu |
| **Primary maintainer email** | mkideabox@gmail.com |
| **Organization / sponsor** | Individual maintainer (mkideabox brand). No commercial entity owns the project. |
| **Country of residence** | South Korea |
| **MFA on GitHub** | Enabled |
| **MFA on SignPath account** | Will be enabled before first signing request |

SignPath requires distinct **Author / Reviewer / Approver** roles for signing
governance. Current single-maintainer state:

- **Author** — Mankil Seo (writes code, opens PRs)
- **Reviewer** — Mankil Seo (will recruit at least one external reviewer
  before going to general availability; pre-GA releases may be self-reviewed
  with rationale logged on the release PR)
- **Approver** — Mankil Seo (manual approval per release on the SignPath
  dashboard)

If SignPath requires a second person before activation, contact will be made
with contributors from the Code Bridge mobile-client repository to take the
Reviewer role.

## 3. Project description and audience

Code Bridge Server is the desktop component of a multi-piece system:

1. **Code Bridge Server** (this project) — a FastAPI server that runs on a
   user's Mac or Windows PC. Hosts a local dashboard, an authenticated API
   for the paired mobile client, a durable Agent Run / Task / Approval /
   Audit platform, and adapter integrations for the user's local Claude
   Code, OpenAI Codex, and Google Gemini CLIs.

2. **Code Bridge mobile app** (separate private repo) — Flutter app for
   iPad and Android. Acts as the operator UI for runs, tasks, approvals,
   artifacts, and workspaces. Communicates with the server over LAN
   (mDNS-discovered) or remote Cloudflare Tunnel using a per-device API key
   issued during QR pairing.

3. **End-user installer** (the artifact requested for signing) — a packaged
   bundle of the server, a frozen Python runtime, a packaged Node.js
   runtime for the scrcpy device-mirroring service, and platform-specific
   helper binaries. Distributed as `.dmg` on macOS and `.msi` on Windows.

**Audience**: software engineers and small dev teams who want to run AI
coding agents against their existing local repositories without sending
source code to a cloud sandbox. Distribution is direct download from GitHub
Releases.

## 4. Why code signing is needed

Without Authenticode signing, the MSI installer triggers Windows
SmartScreen "Unknown publisher" warnings, blocking non-technical users from
installing the server. Mobile-only operators (the primary persona) frequently
hand the installer to colleagues or family members to run on shared
machines; SmartScreen warnings cause those installs to fail.

The macOS DMG is already signed and notarized with an Apple Developer ID
certificate (see `scripts/sign_notarize_desktop_app.py`). Windows is the
remaining gap.

## 5. Build reproducibility

The full installer build runs entirely from `scripts/build_desktop_server_app.py`
(1,356 lines, MIT-licensed in the same repo). For Windows MSI:

1. PyInstaller bundles `server/` and `desktop_server_app/launcher.py` into a
   `dist/Code Bridge Server/` directory using
   `build/desktop_server_app/spec/Code Bridge Server.spec`.
2. The script generates a WiX v4 `.wxs` source (`generate_wix_source()` at
   `scripts/build_desktop_server_app.py:1092`) deterministically — every
   file under the PyInstaller output gets a stable component id derived
   from its path.
3. `wix build` is invoked (WiX v4) to produce `Code Bridge Server.msi`.

No prebuilt or third-party binaries are pulled into the MSI beyond:

- Python runtime — frozen via PyInstaller from CPython on the build host.
- Node.js runtime — downloaded from official `nodejs.org` releases during
  build (recorded in `build/desktop_server_app/node_extract/`).
- scrcpy server / ADB platform tools — downloaded from Google's official
  Android platform tools archive; checksum-verified.

Every dependency source URL and checksum is recorded in the build log so
the binary can be reproduced from a checkout at the tagged commit.

## 6. CI integration plan

Signing will be wired into a GitHub Actions workflow under
`.github/workflows/release-desktop.yml` (to be added; skeleton tracked in
`docs/release_signing/ci_signing_workflow.md`). Outline:

1. Tag push (`v*`) triggers the workflow.
2. Windows job:
   - `windows-latest` runner.
   - `pip install -r server/requirements.txt`, `pyinstaller`, WiX v4.
   - `python scripts/build_desktop_server_app.py --format msi`.
   - Upload unsigned MSI artifact.
3. SignPath signing step:
   - `signpath-io/github-action-submit-signing-request@v1`
   - Inputs: `api-token` (from repo secret `SIGNPATH_API_TOKEN`),
     `organization-id`, `project-slug`, `signing-policy-slug`, the unsigned
     MSI artifact name.
   - Waits for the SignPath dashboard approver (Mankil Seo) to confirm.
4. Signed MSI is downloaded back as a workflow artifact and attached to the
   GitHub Release.

The GitHub Action `signpath-io/github-action-submit-signing-request` is the
official integration and is the only place where the SignPath API token is
referenced; the token never appears in source or in the install bundle.

## 7. Release cadence and metadata

| Item | Value |
|---|---|
| **Current release stage** | Pre-1.0, monthly releases during build-out |
| **Estimated annual signing volume** | <100 MSI signatures / year (one per tagged release plus occasional hotfix) |
| **Versioning** | Semantic versioning. `version` and `productName` set in the WiX `.wxs` from the git tag |
| **Distribution channel** | GitHub Releases, direct download from project homepage |
| **Embedded metadata** | `ProductName=Code Bridge Server`, `Manufacturer=mkideabox`, `Version=<git tag>` — all enforced in `generate_wix_source()` |

## 8. Code signing policy (public page)

SignPath requires a `Code Signing Policy` page reachable from the project
homepage. A draft is checked in at
`docs/release_signing/code_signing_policy.md` and will be promoted to the
project README + a stable URL on the GitHub Pages site before activation.

## 9. Active maintenance evidence

Recent commit / release evidence will be linked in the application:

- Project commit history: <https://github.com/rumururu/code-bridge-server/commits/main>
- Mobile-client repository (private, but referenced for context): code-bridge

If the form requires download statistics or community size, current numbers
are below SignPath's typical thresholds — the project is at the
soft-launch stage. Explain this directly: the request is for early-stage
trust, not for an existing user base. Acknowledge that approval may be
declined and Azure Artifact Signing ($9.99/mo) will be used as fallback.

## 10. Things to attach

- Link to LICENSE file: <https://github.com/rumururu/code-bridge-server/blob/main/LICENSE>
- Link to README: <https://github.com/rumururu/code-bridge-server/blob/main/README.md>
- Link to docs/server_app_distribution.md
- Link to docs/release_signing/code_signing_policy.md (this repo)
- Link to scripts/build_desktop_server_app.py (showing the deterministic
  WiX generation)

## 11. Acknowledgements to include in submission

- The project is MIT-licensed with no commercial dual-licensing.
- No paid features, no proprietary components from maintainers or affiliates.
- All team members will enable MFA on GitHub and SignPath.
- Signed binaries will carry SignPath Foundation attribution where the
  platform requires it.
- Release notes for every signed build will be published on GitHub Releases.
- Project will cooperate fully with any abuse / impersonation investigation
  by SignPath Foundation.
