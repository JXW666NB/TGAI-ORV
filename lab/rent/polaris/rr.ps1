# rr.ps1 -- push one python script to the V100 box and run it.
# NOTE: the remote default shell is PowerShell, so bootstrap commands use PS syntax.
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\rr.ps1 -Script p01_xxx.py [-PyArgs "a b"]
param(
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$PyArgs = "",
    [string]$Py = "D:\orv_pip_0962\Scripts\python.exe",
    [string[]]$Extra = @(),
    [switch]$NoPush
)
$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$k = "$env:USERPROFILE\.ssh\v100_tgai"
$h = "Administrator@192.168.0.103"
$src = Split-Path -Parent $MyInvocation.MyCommand.Path
$remote = "D:\orv_bench\polaris"

ssh -i $k -o StrictHostKeyChecking=no $h "New-Item -ItemType Directory -Force -Path $remote | Out-Null; New-Item -ItemType Directory -Force -Path $remote\out | Out-Null" | Out-Null
if (-not $NoPush) {
    scp -i $k -o StrictHostKeyChecking=no -q "$src\$Script" "${h}:D:/orv_bench/polaris/$Script"
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $Script" }
    foreach ($e in $Extra) {
        scp -i $k -o StrictHostKeyChecking=no -q "$src\$e" "${h}:D:/orv_bench/polaris/$e"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $e" }
    }
}
$cmd = "`$env:PYTHONUTF8='1'; `$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUNBUFFERED='1'; $Py $remote\$Script $PyArgs"
Write-Host ("[rr] $Script") -ForegroundColor DarkGray
ssh -i $k -o StrictHostKeyChecking=no $h $cmd
$code = $LASTEXITCODE
Write-Host ("[rr] exit=" + $code) -ForegroundColor DarkGray
exit $code
