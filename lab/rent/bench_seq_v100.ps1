# V100 上的 seq 扫描 + 分块阈值 A/B
# 目的：验证「长 seq 优势变小」到底是不是因为未走分块（cq 默认 2048，seq=2048 时正好不触发）
$ErrorActionPreference = 'Continue'
$ROOT = 'D:\orv_bench'
$orv = 'D:\orv_pip_0962\Scripts\python.exe'
$pyt = 'C:\Python311\python.exe'
New-Item -ItemType Directory -Force $ROOT | Out-Null
Set-Location $ROOT

$STEPS = 4

function Run-Orv($name, $L, $DFF, $E, $TK, $SEQ, $DATA, $CQ) {
    if ($CQ -ne $null) { $env:TG_CSA_ATTN_CQ = "$CQ" } else { Remove-Item Env:TG_CSA_ATTN_CQ -ErrorAction SilentlyContinue }
    $out = & $orv -m tgai_go.train.v3_pretrain --data $DATA --vocab 96000 --seq $SEQ `
        --d 1024 --layers $L --E $E --n_q 16 --d_ff $DFF --top_k $TK --csa 4 `
        --opt int8gpu --steps $STEPS --log-every 1 --no-final-save 2>&1 | Out-String
    if ($out -match '"status": "ok"') {
        $s = [regex]::Match($out, '"step_s_steady": ([0-9.]+)').Groups[1].Value
        $t = [regex]::Match($out, '"tok_s_steady": ([0-9.]+)').Groups[1].Value
        $m = [regex]::Match($out, '"pool_peak_live_gib": ([0-9.]+)').Groups[1].Value
        $cq = if ($CQ -ne $null) { $CQ } else { 'default' }
        'ORV   {0,-24} cq={1,-8} {2,7} s  {3,8} tok/s  {4,6} GiB' -f $name, $cq, $s, $t, $m
    } else {
        $why = ($out -split "`n" | Where-Object { $_ -match '显存不足|CudaError|Error' } | Select-Object -First 1)
        'ORV   {0,-24} FAIL: {1}' -f $name, $why
    }
}

function Run-Pyt($name, $L, $DFF, $E, $TK, $SEQ) {
    $out = & $pyt "$ROOT\v3_pyt_train.py" --d 1024 --layers $L --E $E --vocab 96000 `
        --d_ff $DFF --n_heads 16 --top_k $TK --seq $SEQ --steps $STEPS 2>&1 | Out-String
    if ($out -match '"impl": "pytorch"') {
        $s = [regex]::Match($out, '"step_s": ([0-9.]+)').Groups[1].Value
        $m = [regex]::Match($out, '"peak_mib": ([0-9]+)').Groups[1].Value
        $tok = [double]$SEQ / [double]$s
        'PyT   {0,-24}          {1,7} s  {2,8:N1} tok/s  {3,6:N2} GiB' -f $name, $s, $tok, ([double]$m / 1024)
    } else {
        'PyT   {0,-24} FAIL: OOM' -f $name
    }
}

# 数据：按最大行宽 2048 生成，seq 小的切片仍可用（宽度 // seq 得窗口数）
if (-not (Test-Path "$ROOT\data2048")) {
    & $orv "$ROOT\make_packed.py" --out "$ROOT\data2048" --seq 2048 --vocab 96000 --shards 8 --rows 16
}

'===== MoE 12 层 / seq 扫描 ====='
foreach ($S in 512, 1024, 2048) {
    Run-Orv "moe12-seq$S" 12 3328 4 2 $S "$ROOT\data2048" $null
    Run-Pyt "moe12-seq$S" 12 3328 4 2 $S
}

''
'===== 分块阈值 A/B（MoE 12 层，seq 2048 / 1024）====='
foreach ($S in 2048, 1024) {
    foreach ($CQ in 512, 256) {
        Run-Orv "moe12-seq$S-cq$CQ" 12 3328 4 2 $S "$ROOT\data2048" $CQ
    }
}

''
'===== 模型大小的影响（seq 2048）====='
foreach ($L in 8, 12, 24) {
    Run-Orv "moe${L}-seq2048" $L 3328 4 2 2048 "$ROOT\data2048" 256
    Run-Pyt "moe${L}-seq2048" $L 3328 4 2 2048
}
