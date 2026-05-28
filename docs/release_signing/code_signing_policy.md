# Code Bridge Server — Code Signing Policy

This policy describes how Code Bridge Server binaries are signed and who is
authorized to sign them. It exists to satisfy the SignPath Foundation
transparency requirement and to give end users a way to verify that an
installer they downloaded actually came from this project.

This page must remain reachable from the project homepage as long as the
project uses SignPath Foundation certificates.

## Scope

This policy covers the following binaries built from the `code-bridge-server`
public repository:

- `Code Bridge Server-<version>.msi` (Windows installer)
- `Code Bridge Server.exe` and packaged content inside the MSI
- Any future `.exe` or `.dll` artifacts produced by the same build pipeline

The macOS `.dmg` is signed separately with an Apple Developer ID
certificate and is **not** covered by this SignPath policy.

## Identity

| Field | Value |
|---|---|
| **Product name on signed binaries** | Code Bridge Server |
| **Publisher attribution** | SignPath Foundation (sponsoring) on behalf of the Code Bridge Server maintainers |
| **Primary maintainer** | Mankil Seo (<mkideabox@gmail.com>, GitHub: `rumururu`) |
| **Project home** | <https://github.com/rumururu/code-bridge-server> |

## Roles

SignPath governance separates three roles. They are listed here for the
public record so users can see who can authorize a release.

- **Author** — anyone whose pull request is accepted into `main`. Currently:
  Mankil Seo.
- **Reviewer** — at least one maintainer who reviews and approves the PR
  before merge. Currently: Mankil Seo (a second reviewer will be onboarded
  before general availability; pre-GA releases are merged with the reviewer
  role logged in the PR description).
- **Approver** — the only role allowed to push the *Approve* button on the
  SignPath dashboard for a given signing request. Currently: Mankil Seo.

All role holders are required to use multi-factor authentication on both
GitHub and SignPath.

## What we sign

We only sign artifacts that satisfy **all** of the following:

1. The artifact is produced by the GitHub Actions workflow
   `.github/workflows/release-desktop.yml` running on a tagged commit
   (`v*`) in the `rumururu/code-bridge-server` repository.
2. The artifact is built from source code already on the public default
   branch at that tag.
3. The build uses no third-party binaries beyond:
   - The CPython interpreter (frozen by PyInstaller from the build host).
   - Official Node.js runtime downloads from <https://nodejs.org/>.
   - Official Android `platform-tools` and `scrcpy` releases.
   Each third-party dependency is recorded in the build log with its
   source URL and checksum.
4. The workflow attaches the unsigned artifact to a SignPath signing
   request; the Approver reviews the workflow run, the source diff since
   the last release, and the build log before clicking **Approve** in the
   SignPath dashboard.

We will not sign:

- Local developer builds.
- Builds from forks of the repository.
- Builds from branches other than the tagged release commit.
- Builds that include any binary that is not produced by the workflow or
  downloaded from the official sources above.

## What signed binaries do

Every signed binary embeds:

- `ProductName=Code Bridge Server`
- `Manufacturer=mkideabox`
- `Version=<release semver>`
- An RFC 3161 timestamp from the SignPath time-stamping authority so that
  the signature remains valid after certificate rotation.

The signed installer asks for:

- Local administrator rights to install the server, its bundled Python
  runtime, the bundled Node.js runtime, and the scrcpy support binaries.
- Windows Firewall exception for the local API port (`8767` by default)
  only if the user explicitly opts in to remote LAN access.
- Optionally, registration of a Startup task so the server launches at
  user login.

The installer does **not** download additional code at install time. All
runtime dependencies needed to start the server are inside the MSI.

## What signed binaries do not do

The signed installer will never:

- Bundle the user's AI provider account credentials (Claude / Codex /
  Gemini login is performed separately, after install, by the official
  provider CLIs).
- Send local source code to any remote service operated by this project.
- Auto-install or auto-update without the user pressing an explicit
  "Update" button on the dashboard.
- Disable or bypass Windows SmartScreen, Defender, or any other security
  control.

## Verification

Users who want to verify a downloaded MSI is genuinely signed by this
project should check both:

1. **Authenticode signature** is present and chains to a SignPath
   Foundation issuing certificate. On Windows:

   ```powershell
   Get-AuthenticodeSignature 'Code Bridge Server.msi' | Format-List *
   ```

   The signer field should reference the SignPath Foundation certificate
   issued for the Code Bridge Server project, and the timestamp should be
   present.

2. **Release artifact provenance** matches a published GitHub Release on
   <https://github.com/rumururu/code-bridge-server/releases>. The release
   page lists the SHA-256 hash of every signed artifact. If your local
   hash does not match the release page, do not run the installer.

## Reporting abuse or compromise

If a binary appears to be signed by this project but was not produced by
our workflow — for example, a third party has somehow obtained access to
the signing pipeline and signed an unrelated artifact — report it
immediately:

- Email <mkideabox@gmail.com>.
- Open an issue on <https://github.com/rumururu/code-bridge-server/issues>
  with the binary's SHA-256 and where it was found.
- Notify SignPath Foundation at <abuse@signpath.org> so the certificate
  can be revoked.

## Privacy

This project does not collect personal information through the signing
pipeline. SignPath Foundation may keep logs of signing requests as
described in their own privacy statement
(<https://about.signpath.io/legal/privacy>). No content of the user's local
repository is ever included in a signing request.

## Attribution

Code signing certificates for this project are sponsored by the **SignPath
Foundation** (<https://signpath.org>). The Foundation issues certificates
to qualifying open-source projects at no cost. We acknowledge their
sponsorship on the project README and on every download page that links to
a signed Windows installer.

## Changes to this policy

Material changes (new role holders, change of project ownership, signing
pipeline change) are recorded in the git history of
`docs/release_signing/code_signing_policy.md` in this repository. The
canonical version is whichever commit is currently checked out on the
public default branch.
