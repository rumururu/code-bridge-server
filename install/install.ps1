# Code Bridge Server - Installation Script for Windows
# Usage: iwr -useb https://raw.githubusercontent.com/rumururu/code-bridge-server/<sha>/install/install.ps1 | iex
#
# Environment overrides:
#   $env:CODE_BRIDGE_INSTALL_DIR    target directory (default: ~/.code-bridge)
#   $env:CODE_BRIDGE_REF            git ref to install (default: pinned SHA below)
#   $env:CODE_BRIDGE_AUTO_START     "0" to skip auto-start after install
#   $env:CODE_BRIDGE_FORCE_RESET    "1" to overwrite local changes during upgrade

$ErrorActionPreference = "Stop"

$INSTALL_DIR = if ($env:CODE_BRIDGE_INSTALL_DIR) { $env:CODE_BRIDGE_INSTALL_DIR } else { "$env:USERPROFILE\.code-bridge" }
$REPO_URL = "https://github.com/rumururu/code-bridge-server.git"
# Pinned upstream commit. Override with $env:CODE_BRIDGE_REF = "main" for HEAD.
$CODE_BRIDGE_REF_DEFAULT = "93e2b8ef9cbeacfee9525563e883a72e14ad23b1"
$CODE_BRIDGE_REF = if ($env:CODE_BRIDGE_REF) { $env:CODE_BRIDGE_REF } else { $CODE_BRIDGE_REF_DEFAULT }
$MIN_PYTHON_VERSION = [version]"3.10"

# cloudflared release pin (override with $env:CODE_BRIDGE_CLOUDFLARED_VERSION).
$CLOUDFLARED_VERSION = if ($env:CODE_BRIDGE_CLOUDFLARED_VERSION) {
    $env:CODE_BRIDGE_CLOUDFLARED_VERSION
} else { "2026.5.2" }
$CLOUDFLARED_EXE_SHA256 = "20b9638f685333d623798e733effbad2487093f15ba592f6c7752360ff3b7ab7"

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   Code Bridge Server - Installation  " -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

function Test-PythonVersion {
    Write-Host "Checking Python installation..." -ForegroundColor Cyan

    # Try different Python commands
    $pythonCommands = @(
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3",
        "python",
        "py -3.13",
        "py -3.12",
        "py -3.11",
        "py -3.10",
        "py -3"
    )

    foreach ($cmd in $pythonCommands) {
        try {
            $cmdParts = $cmd -split ' '
            $exe = $cmdParts[0]
            $args = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] + "--version" } else { @("--version") }

            $output = & $exe $args 2>&1
            if ($output -match "Python (\d+\.\d+)") {
                $version = [version]$Matches[1]
                if ($version -ge $MIN_PYTHON_VERSION) {
                    Write-Host "[OK] Found Python $version" -ForegroundColor Green
                    return $cmd
                }
            }
        } catch {
            # Command not found, try next
        }
    }

    Write-Host "[ERROR] Python $MIN_PYTHON_VERSION or higher is required" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.10+ from:"
    Write-Host "  https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

function Test-Git {
    Write-Host "Checking Git installation..." -ForegroundColor Cyan

    try {
        $null = & git --version 2>&1
        Write-Host "[OK] Git is installed" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[ERROR] Git is required" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install Git from:"
        Write-Host "  https://git-scm.com/download/win"
        exit 1
    }
}

function Test-Cloudflared {
    Write-Host "Checking cloudflared (optional, for remote access)..." -ForegroundColor Cyan

    try {
        $null = & cloudflared --version 2>&1
        Write-Host "[OK] cloudflared is installed" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[!] cloudflared not found (optional for remote access)" -ForegroundColor Yellow

        $response = Read-Host "Install cloudflared $CLOUDFLARED_VERSION for remote access? [y/N]"
        if ($response -ne "y" -and $response -ne "Y") { return $false }

        Write-Host "Installing cloudflared $CLOUDFLARED_VERSION..." -ForegroundColor Cyan

        if (-not (Test-Path $INSTALL_DIR)) {
            New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
        }

        $cloudflaredUrl = "https://github.com/cloudflare/cloudflared/releases/download/$CLOUDFLARED_VERSION/cloudflared-windows-amd64.exe"
        $cloudflaredPath = Join-Path $INSTALL_DIR "cloudflared.exe"

        try {
            Invoke-WebRequest -Uri $cloudflaredUrl -OutFile $cloudflaredPath -UseBasicParsing
        } catch {
            Write-Host "[!] Failed to download cloudflared. Please install manually." -ForegroundColor Yellow
            Write-Host "See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            return $false
        }

        # Verify SHA256 before recommending the binary to the user. A
        # mismatch means either the mirror was tampered with or the
        # maintainer bumped the version without updating the pin.
        $actualSha = (Get-FileHash -Algorithm SHA256 -Path $cloudflaredPath).Hash.ToLower()
        if ($actualSha -ne $CLOUDFLARED_EXE_SHA256.ToLower()) {
            Write-Host "[ERROR] cloudflared SHA256 mismatch — removing untrusted download." -ForegroundColor Red
            Write-Host "  expected: $CLOUDFLARED_EXE_SHA256"
            Write-Host "  got:      $actualSha"
            Remove-Item -LiteralPath $cloudflaredPath -Force -ErrorAction SilentlyContinue
            return $false
        }

        Write-Host "[OK] cloudflared verified and saved to $cloudflaredPath" -ForegroundColor Green
        Write-Host "You may want to add $INSTALL_DIR to your PATH." -ForegroundColor Yellow
        return $true
    }
}

function Test-MermaidCli {
    Write-Host "Checking mermaid-cli (for diagram rendering)..." -ForegroundColor Cyan

    try {
        $null = & mmdc --version 2>&1
        Write-Host "[OK] mermaid-cli is installed" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[!] mermaid-cli not found" -ForegroundColor Yellow

        # Check if npm is available
        try {
            $null = & npm --version 2>&1
        } catch {
            Write-Host "npm not found. Skipping mermaid-cli installation." -ForegroundColor Yellow
            Write-Host "To enable diagram rendering, install Node.js and run: npm install -g @mermaid-js/mermaid-cli" -ForegroundColor Yellow
            return $false
        }

        $response = Read-Host "Install mermaid-cli for diagram rendering? [y/N]"
        if ($response -eq "y" -or $response -eq "Y") {
            Write-Host "Installing mermaid-cli..." -ForegroundColor Cyan

            try {
                & npm install -g @mermaid-js/mermaid-cli
                Write-Host "[OK] mermaid-cli installed successfully" -ForegroundColor Green
            } catch {
                Write-Host "[!] mermaid-cli installation may have failed. You can install it manually later." -ForegroundColor Yellow
            }
        }
        return $false
    }
}

function Setup-Repository {
    Write-Host ""
    Write-Host "Setting up Code Bridge Server (ref: $CODE_BRIDGE_REF)..." -ForegroundColor Cyan

    $gitDir = Join-Path $INSTALL_DIR ".git"
    if (Test-Path $gitDir) {
        Write-Host "Updating existing installation..."
        Set-Location $INSTALL_DIR

        # Guard local edits before any destructive git operation. Users
        # piping iwr|iex should not silently lose customizations.
        $dirty = (& git status --porcelain) -ne $null -and (& git status --porcelain).Trim() -ne ""
        if ($dirty) {
            if ($env:CODE_BRIDGE_FORCE_RESET -ne "1") {
                $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
                $backup = "$INSTALL_DIR.bak.$stamp"
                Write-Host "[!] Local changes detected in $INSTALL_DIR" -ForegroundColor Yellow
                Write-Host "  Backing up working tree to: $backup"
                Copy-Item -Recurse -Force -LiteralPath $INSTALL_DIR -Destination $backup
                Write-Host "  Set `$env:CODE_BRIDGE_FORCE_RESET = '1' to skip this backup next time."
            } else {
                Write-Host "[!] CODE_BRIDGE_FORCE_RESET=1 — discarding local changes" -ForegroundColor Yellow
            }
        }

        & git fetch --tags origin
        & git checkout --force $CODE_BRIDGE_REF
    } elseif ((Test-Path $INSTALL_DIR) -and ((Get-ChildItem -Force -LiteralPath $INSTALL_DIR | Measure-Object).Count -gt 0)) {
        # The directory exists but carries no repo — either a pre-git
        # install or a plain data dir holding .env / api_keys.json /
        # paired_accounts.json. `git clone` refuses a non-empty
        # destination, so attach the repo in place and keep the data.
        Write-Host "[!] $INSTALL_DIR exists but is not a git repository" -ForegroundColor Yellow

        if ($env:CODE_BRIDGE_FORCE_RESET -ne "1") {
            $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
            $backup = "$INSTALL_DIR.bak.$stamp"
            Write-Host "  Backing up existing contents to: $backup"
            Copy-Item -Recurse -Force -LiteralPath $INSTALL_DIR -Destination $backup
            Write-Host "  Set `$env:CODE_BRIDGE_FORCE_RESET = '1' to skip this backup next time."
        }

        Write-Host "  Attaching repository in place (existing files are kept)"
        Set-Location $INSTALL_DIR
        & git init -q
        & git remote get-url origin *> $null
        if ($LASTEXITCODE -eq 0) {
            & git remote set-url origin $REPO_URL
        } else {
            & git remote add origin $REPO_URL
        }
        & git fetch --tags origin
        & git checkout --force $CODE_BRIDGE_REF
    } else {
        Write-Host "Installing to $INSTALL_DIR..."
        & git clone $REPO_URL $INSTALL_DIR
        Set-Location $INSTALL_DIR
        & git checkout --force $CODE_BRIDGE_REF
    }

    $head = (& git rev-parse --short HEAD).Trim()
    Write-Host "[OK] Repository ready (HEAD: $head)" -ForegroundColor Green
}

function Setup-Venv {
    param([string]$PythonCmd)

    Write-Host ""
    Write-Host "Setting up Python environment..." -ForegroundColor Cyan

    Set-Location $INSTALL_DIR

    $reqHash = (Get-FileHash -Algorithm SHA256 -Path "requirements.txt").Hash.ToLower()
    $marker = Join-Path "venv" ".requirements.sha256"

    $needsCreate = -not (Test-Path "venv")
    if (-not $needsCreate -and (Test-Path $marker)) {
        $oldHash = (Get-Content -LiteralPath $marker -Raw).Trim().ToLower()
        if ($oldHash -ne $reqHash) {
            Write-Host "requirements.txt changed since last install — recreating venv..."
            Remove-Item -Recurse -Force -LiteralPath "venv"
            $needsCreate = $true
        }
    } elseif (-not $needsCreate) {
        # venv exists but has no marker — treat as legacy install and rebuild.
        Write-Host "Legacy venv detected — recreating to record dependency state..."
        Remove-Item -Recurse -Force -LiteralPath "venv"
        $needsCreate = $true
    }

    if ($needsCreate) {
        Write-Host "Creating virtual environment..."
        $cmdParts = $PythonCmd -split ' '
        $exe = $cmdParts[0]
        $venvArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] + "-m", "venv", "venv" } else { @("-m", "venv", "venv") }
        & $exe $venvArgs
    }

    Write-Host "Installing dependencies..."
    & "$INSTALL_DIR\venv\Scripts\pip.exe" install --upgrade pip -q
    & "$INSTALL_DIR\venv\Scripts\pip.exe" install --upgrade -r requirements.txt -q

    Set-Content -LiteralPath $marker -Value $reqHash -NoNewline

    Write-Host "[OK] Python environment ready" -ForegroundColor Green
}

function Create-StartScript {
    Write-Host ""
    Write-Host "Creating start scripts..." -ForegroundColor Cyan

    # Create batch file
    @"
@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python main.py --show-qr %*
"@ | Out-File -FilePath "$INSTALL_DIR\start.bat" -Encoding ASCII

    # Create PowerShell script
    @"
Set-Location `$PSScriptRoot
& `$PSScriptRoot\venv\Scripts\Activate.ps1
python main.py --show-qr @args
"@ | Out-File -FilePath "$INSTALL_DIR\start.ps1" -Encoding UTF8

    # Tray mode. pystray already comes from requirements.txt (its win32
    # backend is ctypes-only), so this only needs the launcher the repo
    # ships in desktop_server_app/.
    if (Test-Path "$INSTALL_DIR\desktop_server_app\launcher.py") {
        @"
@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
pythonw desktop_server_app\launcher.py %*
"@ | Out-File -FilePath "$INSTALL_DIR\start-tray.bat" -Encoding ASCII

        Write-Host "[OK] Start scripts created (console + tray)" -ForegroundColor Green
    } else {
        Write-Host "[OK] Start scripts created" -ForegroundColor Green
    }
}

function Show-InstallComplete {
    Write-Host ""
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host "   Installation Complete!             " -ForegroundColor Green
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Start Code Bridge Server with:"
    Write-Host "  $INSTALL_DIR\start.bat          (console, prints the pairing QR)"
    if (Test-Path "$INSTALL_DIR\start-tray.bat") {
        Write-Host "  $INSTALL_DIR\start-tray.bat     (system tray icon)"
    }
}

function Start-Server {
    Write-Host ""
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host "   Installation Complete!             " -ForegroundColor Green
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Starting Code Bridge Server..."
    Write-Host ""

    Set-Location $INSTALL_DIR
    & "$INSTALL_DIR\venv\Scripts\python.exe" main.py --show-qr
}

# Main installation flow
$PythonCmd = Test-PythonVersion
Test-Git
Test-Cloudflared
Test-MermaidCli
Setup-Repository
Setup-Venv -PythonCmd $PythonCmd
Create-StartScript

if ($env:CODE_BRIDGE_AUTO_START -eq "0") {
    Show-InstallComplete
} else {
    Start-Server
}
