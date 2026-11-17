# 在单张 V100 上用 2 个进程验证数据并行路径（world=2）。
# 用 Start-Process：子进程**继承当前环境变量**，避免 Start-Job 的数组反序列化问题。
$py = 'D:\orv_pip_0962\Scripts\python.exe'
$arg = '-m tgai_go.train.v3_pretrain --data D:\TGAI-V3-1.5B\01_corpus\packed512\s512 ' +
       '--vocab 96000 --seq 512 --d 1024 --layers 2 --E 4 --n_q 16 --d_ff 2816 ' +
       '--top_k 2 --csa 4 --steps 3 --log-every 1 --no-final-save'
$dir = 'D:\orv_src\tools\prof'

$env:TGAI_DP_WORLD = '2'
$env:TGAI_DP_JOB = 'v2probe'
$env:TGAI_DP_RANK = '0'
$p0 = Start-Process -FilePath $py -ArgumentList $arg -NoNewWindow -PassThru `
      -RedirectStandardOutput "$dir\r0.txt" -RedirectStandardError "$dir\e0.txt"
$env:TGAI_DP_RANK = '1'
$p1 = Start-Process -FilePath $py -ArgumentList $arg -NoNewWindow -PassThru `
      -RedirectStandardOutput "$dir\r1.txt" -RedirectStandardError "$dir\e1.txt"

$null = $p0.WaitForExit(1200000)
$null = $p1.WaitForExit(1200000)

foreach ($r in @('0', '1')) {
    "--- rank$r (exit=$($null)) ---"
    if (Test-Path "$dir\r$r.txt") {
        Get-Content "$dir\r$r.txt" | Select-String 'dp\]|loss=|Traceback|Error' | Select-Object -Last 6 | ForEach-Object { $_.Line.Trim() }
    }
    if (Test-Path "$dir\e$r.txt") {
        $err = Get-Content "$dir\e$r.txt" | Select-Object -Last 3
        if ($err) { "  stderr: $($err -join ' | ')" }
    }
}
