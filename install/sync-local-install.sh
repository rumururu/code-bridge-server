#!/bin/bash
#
# sync-local-install.sh — deploy this checkout to the local Code Bridge install.
#
# The install directory (~/.code-bridge by default) is NOT a plain copy of this
# repository. install.sh git-clones a separate, flat-layout deployment repo at a
# pinned SHA, and developers have since hand-copied files from this repo on top
# of it. That hand-copying is what produces partial-deployment drift, and this
# script exists to replace it.
#
# Two transfers:
#     server/             ->  $INSTALL_DIR/
#     desktop_server_app/ ->  $INSTALL_DIR/desktop_server_app/
#
# ---------------------------------------------------------------------------
#  SAFETY: THIS SCRIPT MUST NEVER DELETE ANYTHING IN THE INSTALL DIRECTORY
# ---------------------------------------------------------------------------
# --delete is deliberately NOT used, and --delete-excluded is FORBIDDEN. The
# install directory is not disposable: it holds the deployment .git checkout,
# api_keys.json, paired_accounts.json (paired user accounts), the Firebase
# service account key (push credentials, NOT reissuable), an Apple AuthKey .p8,
# server logs and user-generated content. Combining --delete with the exclude
# list below would mark every one of those as "not present in the source" and
# remove it -- an unrecoverable loss of production credentials.
#
# --delete-excluded is worse still: it deletes precisely the paths the exclude
# list was written to keep. Any --delete* flag passed to this script is rejected
# at startup (see reject_delete_flags below), and the protected paths are also
# emitted as rsync "P" (protect) rules so they would survive even a --delete
# added by hand in a future edit.
#
# Leftover modules from the old flat layout (database.py, chat_session_service.py,
# task_orchestrator.py, ...) are REPORTED as orphan candidates, never removed.
# Removing them is a human decision.
# ---------------------------------------------------------------------------
#
# Browser runtime:
#   install.sh downloads the Chromium build that browser_action steps run on;
#   this script never did, and this is the deploy path actually in daily use.
#   The result was a machine with the `playwright` Python package importable
#   and no browser behind it, so every browser step failed on the spot. The
#   dependency block below now ends with the same --ensure the installers run:
#   it probes first and downloads nothing when Chromium is already there.
#
# Usage:
#   ./install/sync-local-install.sh                 # dry run (default)
#   ./install/sync-local-install.sh --apply         # actually copy
#   ./install/sync-local-install.sh --apply --with-scrcpy
#   ./install/sync-local-install.sh --install-dir /tmp/rehearsal --apply
#   ./install/sync-local-install.sh --browser-only  # report browser runtime only
#   ./install/sync-local-install.sh --browser-only --apply   # ...and install it
#   ./install/sync-local-install.sh --apply --no-browser     # skip the check
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${CODE_BRIDGE_INSTALL_DIR:-$HOME/.code-bridge}"
VERIFY_PY="$REPO_ROOT/install/verify_install.py"

APPLY=0
WITH_SCRCPY=0
SKIP_PIP=0
SKIP_BROWSER=0
BROWSER_ONLY=0

if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    CYAN=$'\033[0;36m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; NC=''
fi

usage() {
    sed -n '3,53p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# Refuse every deletion flag, however it is spelled. See the SAFETY block above.
reject_delete_flags() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            --delete|--delete-*|--del|--remove-source-files|--force)
                echo "${RED}refusing '$arg': this script must never delete anything in" >&2
                echo "$INSTALL_DIR (it holds .git, api keys, paired accounts and the" >&2
                echo "Firebase service account). See the SAFETY block in this file.${NC}" >&2
                exit 2
                ;;
        esac
    done
}
reject_delete_flags "$@"

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)        APPLY=1 ;;
        --dry-run)      APPLY=0 ;;
        --with-scrcpy)  WITH_SCRCPY=1 ;;
        --no-pip)       SKIP_PIP=1 ;;
        --no-browser)   SKIP_BROWSER=1 ;;
        --browser-only) BROWSER_ONLY=1 ;;
        --install-dir)  INSTALL_DIR="${2:?--install-dir needs a path}"; shift ;;
        -h|--help)      usage 0 ;;
        *)              echo "${RED}unknown option: $1${NC}" >&2; usage 2 ;;
    esac
    shift
done

# RSYNC_EXTRA is an escape hatch for humans; it must not smuggle a delete in.
reject_delete_flags ${RSYNC_EXTRA:-}

[ -d "$INSTALL_DIR" ] || { echo "${RED}install directory not found: $INSTALL_DIR${NC}" >&2; exit 2; }
[ -f "$VERIFY_PY" ]   || { echo "${RED}missing $VERIFY_PY${NC}" >&2; exit 2; }

PYTHON_CMD="$(command -v python3 || command -v python || true)"
[ -n "$PYTHON_CMD" ] || { echo "${RED}python3 not found (needed for the transfer rules)${NC}" >&2; exit 2; }

# The exclude/protect rules live in verify_install.py so the two tools cannot
# drift. Pull them in rather than restating them here.
# Kept as a plain string (not an array): macOS ships bash 3.2, where expanding
# an empty array under `set -u` is an "unbound variable" error.
SCRCPY_FLAG=""
[ "$WITH_SCRCPY" -eq 1 ] && SCRCPY_FLAG="--with-scrcpy"

FILTER_ARGS=()
while IFS= read -r rule; do
    [ -n "$rule" ] && FILTER_ARGS+=("--filter=$rule")
done < <("$PYTHON_CMD" "$VERIFY_PY" --print-rsync-filters $SCRCPY_FLAG)

[ "${#FILTER_ARGS[@]}" -gt 0 ] || { echo "${RED}no transfer rules produced — refusing to run${NC}" >&2; exit 2; }

# Plain string again for bash 3.2 (empty arrays + `set -u` do not mix).
RSYNC_MODE="--dry-run"
MODE_LABEL="${YELLOW}DRY RUN${NC} (nothing is written; pass --apply to deploy)"
if [ "$APPLY" -eq 1 ]; then
    RSYNC_MODE=""
    MODE_LABEL="${GREEN}APPLY${NC}"
fi

# --- browser runtime --------------------------------------------------------
# pip installs the `playwright` package; it never installs the Chromium build
# that package drives. install.sh has downloaded Chromium since C4, but this
# script is the deploy path in daily use and it did not, so a machine could run
# with the import present and no browser behind it — every `browser_action`
# step then failed within seconds of starting.
#
# The decision is NOT made here. It is made once, in the install directory's
# own copy of system/browser_runtime_setup.py (rsynced by the server pass), so
# this script, install.sh, install.ps1 and the dashboard's one-click install
# cannot disagree about what "present" means. --ensure probes first and
# downloads nothing when Chromium is already on disk, so re-running this script
# never re-fetches 450MB.
sync_browser_runtime() {
    local python="$INSTALL_DIR/venv/bin/python"
    local helper="$INSTALL_DIR/system/browser_runtime_setup.py"

    if [ ! -x "$python" ]; then
        echo "${YELLOW}venv python not found at $python — skipping browser runtime check${NC}"
        return 0
    fi
    if [ ! -f "$helper" ]; then
        echo "${YELLOW}$helper is not deployed yet — run this script with --apply first${NC}"
        return 0
    fi

    # A dry run reports; only --apply is allowed to download anything.
    if [ "$APPLY" -eq 1 ]; then
        # Never fail the deploy over a browser: a server without one starts
        # fine and says what it is missing.
        "$python" "$helper" --ensure || true
    else
        "$python" "$helper" --ensure --dry-run || true
    fi
}

echo "${CYAN}Code Bridge local install sync${NC}"
echo "  repo    : $REPO_ROOT"
echo "  install : $INSTALL_DIR"
echo "  scrcpy  : $([ "$WITH_SCRCPY" -eq 1 ] && echo included || echo 'excluded (--with-scrcpy to include)')"
echo "  mode    : $MODE_LABEL"
echo ""

# --browser-only exists for the machine that is already deployed and merely
# missing Chromium — the common case after this gap was found. It runs the
# browser check and nothing else: no rsync, no pip, no integrity report.
if [ "$BROWSER_ONLY" -eq 1 ]; then
    echo "${CYAN}--- browser runtime${NC}"
    sync_browser_runtime
    echo ""
    exit 0
fi

# Flags, and why each one is (or is not) here:
#   -r  recurse            -l  copy symlinks as symlinks
#   -i  itemize exactly what would change
#   -c  compare by checksum, not size+mtime. The install directory came from a
#       git clone, so nearly every file has a different mtime despite identical
#       content; without -c the dry run lists the entire tree and is worthless
#       as a drift report. With it, only genuine content drift appears.
#   -E  preserve the execute bit only.
#   NO -p: -p would push the source's directory permissions onto the target,
#       and server/ is 755 while $INSTALL_DIR is 700. Applying it would widen
#       the directory that holds api_keys.json, paired_accounts.json, the
#       Firebase service account and the Apple .p8 to world-readable. -E gives
#       us the one permission bit that actually matters.
#   NO -t: with -c, timestamps carry no information, and preserving them would
#       rewrite the mtime of every already-identical file — noise in the report
#       and pointless writes against a live install.
#   NO -D (devices/specials). NO --delete, ever.
run_pass() {
    local src="$1" dest="$2" label="$3"
    echo "${CYAN}--- $label${NC}"
    [ -d "$src" ] || { echo "  ${YELLOW}source missing, skipped: $src${NC}"; return 0; }
    # Only create the destination when actually applying: a dry run must not
    # touch the filesystem at all.
    if [ "$APPLY" -eq 1 ]; then
        mkdir -p "$dest"
    elif [ ! -d "$dest" ]; then
        echo "  ${YELLOW}(destination $dest does not exist yet; --apply would create it)${NC}"
        return 0
    fi
    rsync -rlEc --itemize-changes $RSYNC_MODE \
        "${FILTER_ARGS[@]}" \
        ${RSYNC_EXTRA:-} \
        "$src/" "$dest/"
    echo ""
}

run_pass "$REPO_ROOT/server"             "$INSTALL_DIR"                     "server/ -> $INSTALL_DIR/"
run_pass "$REPO_ROOT/desktop_server_app" "$INSTALL_DIR/desktop_server_app"  "desktop_server_app/ -> $INSTALL_DIR/desktop_server_app/"

# --- dependency refresh -----------------------------------------------------
# Reuse install.sh's marker convention: venv/.requirements.sha256 holds the
# sha256 of the requirements.txt the venv was last built from. Only reinstall
# when it actually changed. install.sh recreates the venv on a change; we do
# not -- recreating a live server's venv from a sync script is too destructive.
sync_requirements() {
    local req="$INSTALL_DIR/requirements.txt"
    local marker="$INSTALL_DIR/venv/.requirements.sha256"
    local pip="$INSTALL_DIR/venv/bin/pip"

    [ -f "$req" ] || { echo "${YELLOW}no requirements.txt in install dir; skipping${NC}"; return 0; }

    local req_hash
    req_hash="$(shasum -a 256 "$req" | awk '{print $1}')"
    local old_hash=""
    [ -f "$marker" ] && old_hash="$(cat "$marker" 2>/dev/null || true)"

    if [ "$req_hash" = "$old_hash" ]; then
        echo "${GREEN}requirements.txt unchanged since last install — no pip run needed${NC}"
        return 0
    fi

    echo "${YELLOW}requirements.txt changed${NC} (marker ${old_hash:0:12}... -> ${req_hash:0:12}...)"
    if [ "$APPLY" -ne 1 ]; then
        echo "  would run: $pip install -r requirements.txt, then update $marker"
        return 0
    fi
    [ -x "$pip" ] || { echo "${RED}venv pip not found at $pip — install dependencies manually${NC}" >&2; return 1; }
    "$pip" install -r "$req"
    echo "$req_hash" > "$marker"
    echo "${GREEN}dependencies updated and marker refreshed${NC}"
}

echo "${CYAN}--- dependencies${NC}"
if [ "$SKIP_PIP" -eq 1 ]; then
    echo "skipped (--no-pip)"
else
    sync_requirements
fi
echo ""

# The step this script was missing. See sync_browser_runtime above for why a
# server can have the `playwright` import and no browser behind it.
echo "${CYAN}--- browser runtime${NC}"
if [ "$SKIP_BROWSER" -eq 1 ]; then
    echo "skipped (--no-browser)"
else
    sync_browser_runtime
fi
echo ""

# --- integrity report -------------------------------------------------------
# Section 3 of this report is the orphan list. Nothing here deletes.
echo "${CYAN}--- integrity report${NC}"
verify_status=0
"$PYTHON_CMD" "$VERIFY_PY" \
    --repo-root "$REPO_ROOT" --install-dir "$INSTALL_DIR" $SCRCPY_FLAG || verify_status=$?

echo ""
if [ "$APPLY" -eq 1 ]; then
    if [ "$verify_status" -eq 0 ]; then
        echo "${GREEN}Deployed. Restart the server for the changes to take effect.${NC}"
    else
        echo "${YELLOW}Deployed, but drift remains (see sections 1 and 2 above).${NC}"
    fi
else
    echo "${YELLOW}Dry run only — nothing was written. Re-run with --apply to deploy.${NC}"
fi

exit "$verify_status"
