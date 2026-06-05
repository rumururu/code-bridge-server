#!/usr/bin/env bash
#
# Defense-in-depth pre-commit guard. Refuses any commit that would add
# files matching the patterns we historically leaked into this repo
# (see docs/runbooks/secret_rotation.md).
#
# Wire-up:
#   ln -s ../../scripts/check-secrets.sh .git/hooks/pre-commit
#
# CI:
#   - name: Check for committed secrets
#     run: ./scripts/check-secrets.sh --staged-against-head
#
# Exit codes:
#   0 — staged set is clean
#   1 — at least one suspicious file detected; commit refused

set -euo pipefail

# Pattern fragments are joined with `|` for one regex pass. Update this
# list whenever a new shape of credential gets identified.
patterns=(
    '\.p8$'                                  # App Store Connect / push keys
    '(^|/)api_key\.json$'                    # App Store Connect JSON wrapper
    '(^|/)play-store-key\.json$'             # Google Play service-account key
    '(^|/)firebase_config\.json$'            # Firebase admin SDK config
    '(^|/)server_info\.json$'                # server-side pairing/JWT secrets
    '\.jks(\.bak)?$'                         # Android signing keystores
    '(^|/)\.env$'                            # client-side env (RevenueCat etc.)
    '(^|/)\.env\.local$'
    '(^|/)key\.properties$'                  # Android signing passwords
    '\.mobileprovision$'                     # iOS provisioning profiles
)

joined="$(IFS='|'; echo "${patterns[*]}")"

if [[ "${1:-}" == "--staged-against-head" ]]; then
    candidates="$(git diff --cached --name-only --diff-filter=AM)"
else
    candidates="$(git diff --cached --name-only --diff-filter=AM)"
fi

if [[ -z "$candidates" ]]; then
    exit 0
fi

suspicious="$(printf '%s\n' "$candidates" | grep -E "$joined" || true)"

if [[ -z "$suspicious" ]]; then
    exit 0
fi

cat >&2 <<EOF

Refusing commit — these staged files look like signing material or
service credentials. They must not be tracked:

EOF

printf '%s\n' "$suspicious" | sed 's/^/  /' >&2

cat >&2 <<EOF

If a match is a false positive (e.g. an example fixture under
\`test/fixtures/\`), add the path to .gitignore or rename it. To
override this check for an emergency commit, run:

    git commit --no-verify   # (do not push without review)

See docs/runbooks/secret_rotation.md for the rotation procedure.
EOF

exit 1
