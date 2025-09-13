<#
  windows.ps1
  - Checks/installs Chocolatey, CMake, OpenCL SDK (tries Khronos release), and MinGW (g++).
  - Self-elevates.
#>

# --- Self-elevate ---
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
      [Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    if ($PSCommandPath) {
        Write-Host "Script needs elevation. Relaunching as Administrator..."
        Start-Process -FilePath powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
        exit
    } else {
        Write-Warning "Please run this script from an elevated PowerShell (Run as Administrator)."
        # Pause
        Write-Host "Press Enter to exit..."
        [void][System.Console]::ReadLine()
        exit 1
    }
}

# --- Ensure TLS1.2 ---
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Chocolatey
if (Get-Command choco -ErrorAction SilentlyContinue) {
    Write-Host "Chocolatey: installed"
} else {
    Write-Host "Chocolatey: missing"
    Write-Host "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "Chocolatey installed successfully."
    } else {
        Write-Warning "Chocolatey installation failed. Please install it manually from https://chocolatey.org/install"
        exit 1
    }
}

# CMake
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    Write-Host "CMake: installed"
} else {
    Write-Host "CMake: missing"
    Write-Host "Installing CMake via Chocolatey..."
    choco install cmake -y --installargs 'ADD_CMAKE_TO_PATH=System'
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        Write-Host "CMake installed successfully."
    } else {
        Write-Warning "CMake installation failed. Please install it manually from https://cmake.org/download/"
        exit 1
    }
}


# Check for OpenCL dlls
# System32/opencl.dll
# Program Files\OpenCL-*
$openclFound = $false
if (Test-Path "$env:SystemRoot\System32\OpenCL.dll") {
    $openclFound = $true
}
if (Test-Path "$env:ProgramFiles\OpenCL-SDK") {
    $openclFound = $true
}
# OpenCL
if ($openclFound) {
    Write-Host "OpenCL runtime: found/installed"
} else {
    Write-Host "OpenCL runtime: missing"
    Write-Warning "OpenCL runtime installation failed or OpenCL.dll not found. Please install it manually from your GPU vendor."
    Write-Warning "NVIDIA: https://developer.nvidia.com/cuda-downloads"
    Write-Warning "AMD: https://www.amd.com/en/support/kb/faq/gpu-579"
    Write-Warning "Intel: https://www.intel.com/content/www/us/en/developer/articles/opencl-drivers.html"
    # Give user a pause to read
    Write-Host "Press Enter to exit..."
    [void][System.Console]::ReadLine()
    exit 1
}

# Compiler (Unix Makefiles ; g++)
if (Get-Command make -ErrorAction SilentlyContinue) {
    Write-Host "Compiler (make): installed"
} else {
    Write-Host "Compiler (make): missing"
    Write-Host "Installing make via Chocolatey..."
    choco install make -y
}

# --- Summary ---
Write-Host "If new tools were installed, restart PowerShell to refresh PATH."

# -- Pause and let me read errors
Write-Host "Press Enter to exit..."
[void][System.Console]::ReadLine()