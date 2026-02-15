# FrameWright Interactive Wizard Launcher for PowerShell

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FrameWright Interactive Setup Wizard" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if ($args.Count -eq 0) {
    python run_wizard.py
} else {
    python run_wizard.py $args[0]
}

Write-Host ""
Read-Host -Prompt "Press Enter to exit"
