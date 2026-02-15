@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   FrameWright Installer for Windows
echo ============================================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found.
    echo     Please install Python 3.9+ from https://www.python.org/downloads/
    echo     Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

:: Create virtual environment
if not exist ".venv" (
    echo.
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [X] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: Activate virtual environment
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated.

:: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install FrameWright
echo.
echo Installing FrameWright...
pip install -e ".[full,progress]"
if %errorlevel% neq 0 (
    echo [X] Installation failed. Check the error messages above.
    pause
    exit /b 1
)
echo [OK] FrameWright installed.

:: Check FFmpeg
echo.
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [!!] FFmpeg not found in PATH.
    echo.
    echo     Option 1: Install via winget (recommended^):
    echo       winget install Gyan.FFmpeg
    echo.
    echo     Option 2: Install via Chocolatey (requires admin^):
    echo       choco install ffmpeg
    echo.
    echo     Option 3: Download manually from https://ffmpeg.org/download.html
    echo       and add the bin folder to your PATH.
    echo.
) else (
    echo [OK] FFmpeg found.
)

:: Run hardware check
echo.
echo Running hardware check...
python -m framewright --help >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] FrameWright is ready.
) else (
    echo [!!] FrameWright installed but help command returned an error.
    echo     Try: python -m framewright --help
)

echo.
echo ============================================================
echo   Installation complete!
echo.
echo   To activate the environment in a new terminal:
echo     .venv\Scripts\activate
echo.
echo   Quick start:
echo     framewright --help
echo     framewright-check
echo     framewright video.mp4
echo ============================================================
pause
