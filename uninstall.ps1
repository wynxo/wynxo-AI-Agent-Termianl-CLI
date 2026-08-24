# Uninstall wynxo (Windows).
#
#   .\uninstall.ps1              interactive
#   .\uninstall.ps1 --yes        accept every prompt
#   .\uninstall.ps1 --dry-run    list what would go, change nothing
#
# Run it with:  powershell -ExecutionPolicy Bypass -File .\uninstall.ps1
# PowerShell refuses to run unsigned .ps1 files under its default policy,
# which is why that flag is there rather than a plain .\uninstall.ps1.

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-Python {
    foreach ($try in @(@("py", "-3"), @("python", $null), @("python3", $null))) {
        $exe, $flag = $try
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            $check = 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'
            if ($flag) { & $exe $flag -c $check 2>$null } else { & $exe -c $check 2>$null }
            if ($LASTEXITCODE -eq 0) { return ,@($exe, $flag) }
        }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host "  Python 3.10 or newer is required and was not found."
    Write-Host "  wynxo lives in these places; remove them by hand:"
    Write-Host "    $HOME\.wynxo-src"
    Write-Host "    $env:LOCALAPPDATA\Microsoft\WindowsApps\wynxo.cmd"
    Write-Host "    $env:APPDATA\wynxo  and  $env:LOCALAPPDATA\wynxo"
    exit 1
}
$exe, $flag = $python

$uninstaller = Join-Path $dir "uninstall.py"
if ($flag) { & $exe $flag $uninstaller @args } else { & $exe $uninstaller @args }
exit $LASTEXITCODE
