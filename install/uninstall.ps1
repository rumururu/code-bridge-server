# Code Bridge Server - Uninstall Script for Windows
# Usage: iwr -useb https://raw.githubusercontent.com/rumururu/code-bridge-server/<sha>/install/uninstall.ps1 | iex
#
# Removes the installation directory created by install.ps1 (default
# %USERPROFILE%\.code-bridge) along with its venv and any auto-generated
# start scripts. cloudflared and mermaid-cli are NOT removed because
# they may be shared with other tooling on this machine.
#
# Environment overrides:
#   $env:CODE_BRIDGE_INSTALL_DIR    target directory (default: ~/.code-bridge)
#   $env:CODE_BRIDGE_ASSUME_YES     "1" to skip the confirmation prompt

$ErrorActionPreference = "Stop"

$INSTALL_DIR = if ($env:CODE_BRIDGE_INSTALL_DIR) {
    $env:CODE_BRIDGE_INSTALL_DIR
} else {
    "$env:USERPROFILE\.code-bridge"
}

Write-Host ""
Write-Host "Code Bridge Server - Uninstall" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $INSTALL_DIR)) {
    Write-Host "Nothing to remove at $INSTALL_DIR" -ForegroundColor Yellow
    return
}

$resolved = (Resolve-Path -LiteralPath $INSTALL_DIR).Path
$home = (Resolve-Path -LiteralPath $env:USERPROFILE).Path
if ($resolved -eq $home -or $resolved -match "^[A-Za-z]:\\?$") {
    Write-Host "Refusing to delete $resolved - set `$env:CODE_BRIDGE_INSTALL_DIR to a dedicated directory." -ForegroundColor Red
    exit 1
}

Write-Host "About to remove:"
Write-Host "  $resolved"
Write-Host ""

if ($env:CODE_BRIDGE_ASSUME_YES -ne "1") {
    $response = Read-Host "Continue? [y/N]"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Aborted."
        exit 1
    }
}

# Drop the login item before the tree goes away, otherwise the Run key
# keeps pointing at a deleted launcher.
if ((Test-Path "$resolved\venv\Scripts\python.exe") -and (Test-Path "$resolved\desktop_server_app\launcher.py")) {
    & "$resolved\venv\Scripts\python.exe" "$resolved\desktop_server_app\launcher.py" --disable-autostart *> $null
}
Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" `
    -Name "Code Bridge Server" -ErrorAction SilentlyContinue

Remove-Item -Recurse -Force -LiteralPath $resolved
Write-Host "[OK] Removed $resolved" -ForegroundColor Green

# Surface any backup directories install.ps1 produced so the user can
# clean those up too if desired.
$parent = Split-Path -Parent $resolved
$leaf = Split-Path -Leaf $resolved
$backups = Get-ChildItem -LiteralPath $parent -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "$leaf.bak.*" }
if ($backups) {
    Write-Host ""
    Write-Host "Leftover backups from previous installs:" -ForegroundColor Yellow
    $backups | ForEach-Object { Write-Host "  $($_.FullName)" }
    Write-Host "Delete manually if you no longer need them."
}

Write-Host ""
Write-Host "Note: cloudflared / mermaid-cli were left in place."
