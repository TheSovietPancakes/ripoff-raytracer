@echo off
setlocal enabledelayedexpansion

:: ==============================
:: Check if Chocolatey is installed
:: ==============================
choco -v >nul 2>&1
if %errorlevel% neq 0 (
    echo Chocolatey not found. Installing Chocolatey...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
     "Set-ExecutionPolicy Bypass -Scope Process -Force; ^
      [System.Net.ServicePointManager]::SecurityProtocol = 'Tls12'; ^
      iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    if %errorlevel% neq 0 (
        echo Failed to install Chocolatey. Exiting.
        exit /b 1
    )
    refreshenv
)

:: ==============================
:: Check if CMake is installed
:: ==============================
cmake --version >nul 2>&1
if %errorlevel% neq 0 (
    echo CMake not found. Installing CMake...
    choco install cmake -y
) else (
    echo CMake already installed.
)

:: ==============================
:: Check if OpenCL headers / libs are present
:: (Assume Khronos OpenCL SDK from GitHub if missing)
:: ==============================
if exist "%ProgramFiles%\OpenCL SDK" (
    echo OpenCL SDK already installed.
) else (
    echo OpenCL SDK not found. Downloading...
    choco install opencl-headers -y
    choco install opencl-icd-loader -y
    :: optional C++ bindings
    choco install opencl-icd-headers -y
)

:: ==============================
:: Check if MinGW or another compiler is installed
:: ==============================
g++ --version >nul 2>&1
if %errorlevel% neq 0 (
    echo No compiler found. Installing MinGW...
    choco install mingw -y
) else (
    echo Compiler already installed.
)

echo =====================================
echo All dependencies are installed.
echo =====================================

pause