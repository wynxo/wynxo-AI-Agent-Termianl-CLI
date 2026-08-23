@echo off
REM One-command setup for wynxo on Windows.
REM
REM   install.bat              interactive
REM   install.bat --yes        accept every recommendation
REM   install.bat --no-ollama  just install wynxo
REM
REM A .bat file rather than PowerShell on purpose: PowerShell refuses to run
REM .ps1 scripts under its default Restricted execution policy, which is the
REM single most common reason a Windows setup appears to do nothing at all.
REM Batch files are not subject to that policy, so this always runs.

setlocal
set "DIR=%~dp0"

REM The py launcher is the reliable way to reach a modern Python on Windows.
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py -3 "%DIR%install.py" %*
    exit /b %ERRORLEVEL%
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python "%DIR%install.py" %*
    exit /b %ERRORLEVEL%
)

python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python3 "%DIR%install.py" %*
    exit /b %ERRORLEVEL%
)

echo.
echo   wynxo needs Python 3.10 or newer, and none was found.
echo.
echo   Install it from https://python.org/downloads
echo   Tick "Add python.exe to PATH" in the installer, then reopen this window.
echo.
exit /b 1
