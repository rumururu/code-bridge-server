# Code Bridge Server - Installation Script for Windows
# Usage: iwr -useb https://raw.githubusercontent.com/rumururu/code-bridge-server/main/install/install.ps1 | iex

$ErrorActionPreference = "Stop"

$INSTALL_DIR = "$env:USERPROFILE\.code-bridge"
$REPO_URL = "https://github.com/rumururu/code-bridge-server.git"
$MIN_PYTHON_VERSION = [version]"3.10"

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   Code Bridge Server - Installation  " -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

function Test-PythonVersion {
    Write-Host "Checking Python installation..." -ForegroundColor Cyan

    # Try different Python commands
    $pythonCommands = @("python", "python3", "py -3")

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

        $response = Read-Host "Install cloudflared for remote access? [y/N]"
        if ($response -eq "y" -or $response -eq "Y") {
            Write-Host "Installing cloudflared..." -ForegroundColor Cyan

            # Download cloudflared
            $cloudflaredUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            $cloudflaredPath = "$env:USERPROFILE\.code-bridge\cloudflared.exe"

            # Create directory if it doesn't exist
            if (-not (Test-Path "$env:USERPROFILE\.code-bridge")) {
                New-Item -ItemType Directory -Path "$env:USERPROFILE\.code-bridge" -Force | Out-Null
            }

            try {
                Invoke-WebRequest -Uri $cloudflaredUrl -OutFile $cloudflaredPath
                Write-Host "[OK] cloudflared downloaded to $cloudflaredPath" -ForegroundColor Green
                Write-Host "You may want to add this to your PATH." -ForegroundColor Yellow
            } catch {
                Write-Host "[!] Failed to download cloudflared. Please install manually." -ForegroundColor Yellow
                Write-Host "See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            }
        }
        return $false
    }
}

function Setup-Repository {
    Write-Host ""
    Write-Host "Setting up Code Bridge Server..." -ForegroundColor Cyan

    if (Test-Path $INSTALL_DIR) {
        Write-Host "Updating existing installation..."
        Set-Location $INSTALL_DIR
        & git fetch origin
        & git reset --hard origin/main
    } else {
        Write-Host "Installing to $INSTALL_DIR..."
        & git clone $REPO_URL $INSTALL_DIR
        Set-Location $INSTALL_DIR
    }

    Write-Host "[OK] Repository ready" -ForegroundColor Green
}

function Setup-Venv {
    param([string]$PythonCmd)

    Write-Host ""
    Write-Host "Setting up Python environment..." -ForegroundColor Cyan

    Set-Location $INSTALL_DIR

    # Create venv if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..."
        $cmdParts = $PythonCmd -split ' '
        $exe = $cmdParts[0]
        $args = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] + "-m", "venv", "venv" } else { @("-m", "venv", "venv") }
        & $exe $args
    }

    # Activate venv and install dependencies
    Write-Host "Installing dependencies..."
    & "$INSTALL_DIR\venv\Scripts\pip.exe" install --upgrade pip -q
    & "$INSTALL_DIR\venv\Scripts\pip.exe" install -r requirements.txt -q

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

    Write-Host "[OK] Start scripts created" -ForegroundColor Green
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
Setup-Repository
Setup-Venv -PythonCmd $PythonCmd
Create-StartScript
Start-Server
