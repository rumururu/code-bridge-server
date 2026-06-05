# Secret rotation runbook

Audit item BUILD-1 / BUILD-2 from
`docs/audit_findings/2026-06-05_full_stack_audit.md` identified signing
material and service-account credentials still reachable in this
repository's git history. The keys themselves listed in `.gitignore`
going forward, but anything anyone with read access to a historical
revision can still extract them.

This document is the **operational** runbook for actually rotating
each secret. Code changes alone don't fix the leak — you have to
revoke the leaked key in the upstream service AND scrub the history.
Do these in the order below; later steps depend on earlier ones.

> ⚠️ Every step in section 1 is **destructive in upstream services**.
> Once you revoke an App Store Connect key, every Fastlane lane that
> still references it will fail until you swap in the new one. Plan
> a downtime window of at least 30 minutes.

---

## 1. Rotate upstream credentials (do this first)

### 1.1 App Store Connect API key

Affected files: `ios/fastlane/AuthKey_DGS7ANBS8T.p8`,
`ios/fastlane/api_key.json`.

1. Sign in to **App Store Connect → Users and Access → Integrations →
   App Store Connect API**.
2. Find the key with ID `DGS7ANBS8T` and click **Revoke**.
3. Click **+ Generate API Key**. Give it a meaningful name
   (`code-bridge-fastlane-YYYYMMDD`). Download the new `.p8` file
   (you only get one chance — save it immediately).
4. Note the new **Key ID** and **Issuer ID**. They will replace the
   hardcoded values currently in `ios/fastlane/Fastfile`.
5. Store the new `.p8` outside the repo (suggested:
   `~/.code-bridge-secrets/AuthKey_<new-id>.p8` with mode `0600`).
6. Update `ios/fastlane/Fastfile`:
   - Replace every occurrence of `key_id: "DGS7ANBS8T"` with the new
     Key ID, or better, read both Key ID and Issuer ID from env vars
     (`ENV.fetch("ASC_KEY_ID")`, `ENV.fetch("ASC_ISSUER_ID")`).
   - Replace `key_filepath: "AuthKey_DGS7ANBS8T.p8"` with a path
     resolved from `ENV["ASC_KEY_PATH"]`.

### 1.2 App Store Connect Subscription key

Affected file: `ios/fastlane/SubscriptionKey_66J9885ZPA.p8`.

Same procedure as 1.1, but in App Store Connect navigate to **In-App
Purchases → API Keys**. Revoke the key with ID `66J9885ZPA`, generate
a new one, and reference it via `ENV["ASC_SUBSCRIPTION_KEY_PATH"]`.

### 1.3 Google Play service account

Affected file: `android/fastlane/play-store-key.json`.

1. Sign in to **Google Cloud Console → IAM & Admin → Service Accounts**.
2. Find the service account currently used for Play uploads
   (check the `client_email` field in the leaked JSON).
3. Either **Disable** the existing key (under the Keys tab, delete the
   leaked key) or rotate the entire service account.
4. Generate a new JSON key. Save outside the repo
   (`~/.code-bridge-secrets/play-store-<new-id>.json`, mode `0600`).
5. Update `android/fastlane/Fastfile` to read the JSON path from
   `ENV["PLAY_STORE_JSON_KEY_PATH"]` instead of the in-repo path.

### 1.4 Android signing keystore

Affected files: `android/app/code_bridge-release.jks.bak`,
`android/key.properties` (the password is in plaintext locally even
though the file itself isn't checked in).

> ⚠️ Rotating the **release** keystore means Google Play Store will
> reject the next upload unless you go through the **Play App Signing
> key reset** flow (a 7-day delay). Only do this if you believe the
> keystore was actually leaked. The `.bak` file in the repo history is
> suspicious but does not prove the active keystore was compromised —
> check the contents to be sure.

If you decide to rotate:

1. Generate a new keystore: `keytool -genkey -v -keystore
   ~/.code-bridge-secrets/code_bridge-release.jks -alias code_bridge
   -keyalg RSA -keysize 2048 -validity 25000`.
2. Generate strong passwords (e.g. `openssl rand -base64 24`); do NOT
   reuse the previous one (`Tlsrlfn1@3`).
3. Update `android/key.properties` (still gitignored) to point at the
   new keystore path and use the new passwords.
4. Initiate **Play App Signing key reset** in the Play Console under
   **Setup → App integrity**.
5. Wait 7 days. During that window, builds signed with the old key
   are still accepted.
6. Delete the `code_bridge-release.jks.bak` from the working tree, then
   purge it from git history (section 2).

If you choose NOT to rotate (acceptable if the `.bak` file's contents
match the active keystore and you simply never want to ship that
backup): just `git rm` it and history-purge.

### 1.5 RevenueCat keys (informational)

The audit flagged `.env` containing RevenueCat keys
(`goog_iaLuSod...`, `appl_KFPWcdb...`). These are **public** keys by
RevenueCat's design — they are read by the client and meant to be
extractable. Rotation is low-value unless the underlying product /
offering configuration also changed. No action required unless you
want a fresh set as part of a broader product reset.

---

## 2. Scrub git history

Once new keys are in place and Fastlane has been updated to consume
them from env vars or out-of-repo paths, purge the leaked files from
all history.

```bash
# Make sure you have an up-to-date mirror of both repos.
git clone --mirror git@github.com:rumururu/code-bridge.git
git clone --mirror git@github.com:rumururu/code-bridge-server.git

# Install BFG: brew install bfg

cd code-bridge.git
bfg --delete-files 'AuthKey_*.p8'
bfg --delete-files 'SubscriptionKey_*.p8'
bfg --delete-files 'api_key.json'
bfg --delete-files 'play-store-key.json'
bfg --delete-files 'code_bridge-release.jks.bak'

git reflog expire --expire=now --all && git gc --prune=now --aggressive

git push --force
```

Repeat for `code-bridge-server.git` (the public mirror); the same
files exist there if Fastlane ever ran against that clone.

After force-push, **every collaborator must reclone** — their local
copies will diverge irreparably from the rewritten history. Notify
them before the force-push.

---

## 3. Add a defense-in-depth check

Drop the following script in `scripts/check-secrets.sh` and wire it
into `.git/hooks/pre-commit` (or your CI) so anyone who tries to add
a `.p8`, `play-store-key.json`, or `*.jks*` gets blocked at commit
time:

```bash
#!/usr/bin/env bash
set -euo pipefail
suspicious=$(git diff --cached --name-only | grep -E \
    '(\.p8$|^.*api_key\.json$|^.*play-store-key\.json$|\.jks(\.bak)?$|^\.env$)' \
    || true)
if [ -n "$suspicious" ]; then
    echo "Refusing commit — files look like signing material:"
    echo "$suspicious" | sed 's/^/  /'
    exit 1
fi
```

This script is committed in this repository alongside this runbook;
see `scripts/check-secrets.sh`.

---

## 4. Verification checklist

- [ ] Old App Store Connect key (`DGS7ANBS8T`) shows status **Revoked**
      in App Store Connect.
- [ ] Old subscription key (`66J9885ZPA`) is **Revoked**.
- [ ] Old Play Console service-account key is **Disabled** (or the
      whole account replaced).
- [ ] `ios/fastlane/Fastfile` no longer hardcodes Key IDs or Issuer
      IDs; everything comes from env vars.
- [ ] `android/fastlane/Fastfile` reads the JSON path from
      `ENV["PLAY_STORE_JSON_KEY_PATH"]`.
- [ ] `git log -- ios/fastlane/AuthKey_DGS7ANBS8T.p8` returns nothing
      (history purged in both repos).
- [ ] All collaborators have re-cloned.
- [ ] A test Fastlane lane (`ios beta`, `android internal`) completes
      successfully end-to-end with the new credentials.
- [ ] `scripts/check-secrets.sh` is wired into a pre-commit hook.
