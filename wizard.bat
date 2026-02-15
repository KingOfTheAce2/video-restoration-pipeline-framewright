@echo off
REM FrameWright Interactive Wizard Launcher for Windows

cd /d "%~dp0"

echo.
echo ============================================================
echo   FrameWright Interactive Setup Wizard
echo ============================================================
echo.

if "%1"=="" (
    python run_wizard.py
) else (
    python run_wizard.py "%~1"
)

pause
