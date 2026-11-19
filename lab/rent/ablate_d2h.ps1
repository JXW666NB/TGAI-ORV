# Run the D2H ablation matrix on the remote V100 (seq2048, 20 steps).
# Usage: powershell -NoProfile -File lab\rent\ablate_d2h.ps1
$ErrorActionPreference = "Continue"
$k = Join-Path $env:USERPROFILE ".ssh\v100_tgai"
$py = "D:\orv_pip_0962\Scripts\python.exe"
$remote = "Administrator@192.168.0.103"
$bench = "D:\orv_bench"
$common = "--data D:\orv_bench\data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --amp gemm --batch 1 --steps 20 --log-every 99 --no-final-save"
$modes = @("none", "route", "gatebwd", "both", "scalar")
$log = Join-Path $PSScriptRoot "ablate_d2h.log"
Remove-Item $log -ErrorAction SilentlyContinue

foreach ($m in $modes) {
    foreach ($r in 1, 2) {
        $cmd = "$py -X utf8 $bench\probe_d2h_ablation.py $common --ablate $m"
        $out = ssh -i $k $remote $cmd 2>&1
        $joined = ($out | Out-String)
        Add-Content $log "===== MODE=$m RUN=$r ====="
        Add-Content $log $joined
        $mSteady = [regex]::Match($joined, '"step_s_steady": ([0-9.]+)')
        $mByp = [regex]::Match($joined, "bypassed=\{([^}]*)\}")
        if ($mSteady.Success) {
            Write-Output ("{0,-8} run{1}  steady={2} s   bypassed={3}" -f `
                $m, $r, $mSteady.Groups[1].Value, $mByp.Groups[1].Value)
        } else {
            Write-Output ("{0,-8} run{1}  FAILED (see {2})" -f $m, $r, $log)
        }
    }
}
Write-Output "log -> $log"
