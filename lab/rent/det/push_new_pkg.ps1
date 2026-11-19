# 把工作区当前 python/tgai_go 打包成"新包"并推到远端 D:\orv_det_work\new_pkg_x
# 用法（在仓库根目录）：powershell -ExecutionPolicy Bypass -File lab\rent\det\push_new_pkg.ps1
# 注意：Windows PowerShell 5.1 按 ANSI 读无 BOM 的 .ps1，所以本文件里**不能出现中文路径**；
#       $src 用相对路径，依赖当前目录 = 仓库根。
$ErrorActionPreference = 'Stop'
$src  = 'python\tgai_go'
if (-not (Test-Path $src)) { throw "cwd 不是仓库根（找不到 $src）：$(Get-Location)" }
$work = 'D:\orv_det_work'
$stage = Join-Path $work 'new_pkg\tgai_go'
$zip  = Join-Path $work 'new_pkg.zip'
$key  = "$env:USERPROFILE\.ssh\v100_tgai"
$dest = 'Administrator@192.168.0.103'

if (-not (Test-Path $work)) { New-Item -ItemType Directory -Force -Path $work | Out-Null }
if (Test-Path $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
robocopy $src $stage /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed rc=$LASTEXITCODE" }
if (Test-Path $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal

scp -i $key $zip "${dest}:D:/orv_det_work/" 2>&1 | Out-Null
ssh -i $key $dest "rmdir /s /q D:\orv_det_work\new_pkg_x" 2>&1 | Out-Null
ssh -i $key $dest "mkdir D:\orv_det_work\new_pkg_x" 2>&1 | Out-Null
ssh -i $key $dest "tar -xf D:\orv_det_work\new_pkg.zip -C D:\orv_det_work\new_pkg_x" 2>&1 | Out-Null
$size = (ssh -i $key $dest "dir D:\orv_det_work\new_pkg_x\tgai_go\train\cuda.py" 2>&1) -join "`n"
Write-Host "[push_new_pkg] zip=$((Get-Item $zip).Length) bytes"
Write-Host $size
