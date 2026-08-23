# One-command setup for wynxo (Windows).
#
#   .\install.ps1              interactive
#   .\install.ps1 --yes        accept every recommendation
#   .\install.ps1 --no-ollama  just install wynxo
#
# All it does is find a Python 3.10+ and hand over to install.py.

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-Python($exe, $prefix) {
    try {
        $check = 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'
        if ($prefix) { & $exe $prefix -c $check 2>$null } else { & $exe -c $check 2>$null }
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

# The py launcher is the reliable way to reach a modern Python on Windows.
if (Get-Command py -ErrorAction SilentlyContinue) {
    if (Test-Python "py" "-3") {
        & py -3 (Join-Path $dir "install.py") @args
        exit $LASTEXITCODE
    }
}
foreach ($exe in @("python", "python3")) {
    if ((Get-Command $exe -ErrorAction SilentlyContinue) -and (Test-Python $exe $null)) {
        & $exe (Join-Path $dir "install.py") @args
        exit $LASTEXITCODE
    }
}

Write-Host "wynxo needs Python 3.10 or newer, and none was found."
Write-Host ""
Write-Host "  Install it from https://python.org/downloads"
Write-Host "  Tick 'Add python.exe to PATH' in the installer."
exit 1
