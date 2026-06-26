$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ProjectRoot 'outputs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Start-Transcript -Path (Join-Path $LogDir 'setup_new_pc_admin.log') -Append
Write-Output "=== Admin setup started $(Get-Date) ==="
Write-Output "User: $env:USERNAME"
Write-Output "Is admin: $(([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"
Write-Output "--- Installing VC++ Redistributable ---"
winget install --id Microsoft.VCRedist.2015+.x64 -e --accept-package-agreements --accept-source-agreements --silent
Write-Output "VC++ winget exit code: $LASTEXITCODE"
Write-Output "--- Enabling WSL optional features ---"
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
Write-Output "WSL feature exit code: $LASTEXITCODE"
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
Write-Output "VMP feature exit code: $LASTEXITCODE"
Write-Output "--- Installing/updating WSL Ubuntu ---"
wsl.exe --install -d Ubuntu --no-launch
Write-Output "WSL install exit code: $LASTEXITCODE"
wsl.exe --set-default-version 2
Write-Output "WSL default version exit code: $LASTEXITCODE"
wsl.exe -l -v
Write-Output "=== Admin setup ended $(Get-Date) ==="
Stop-Transcript
Read-Host 'Admin setup done. Press Enter to close'
