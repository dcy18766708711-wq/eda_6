$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

Set-Location $Root

python -m venv $Venv
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Root "requirements-cpu.txt")

Write-Host "安装完成，请运行: .\start_cpu.ps1"
