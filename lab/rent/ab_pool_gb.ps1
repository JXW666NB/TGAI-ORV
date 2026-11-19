# TG_POOL_GB 的 A/B（seq2048，20 步稳态）
$env:PYTHONIOENCODING = 'utf-8'
$py = 'D:\orv_pip_0962\Scripts\python.exe'
$common = @(
  '-m','tgai_go.train.v3_pretrain',
  '--data','D:\orv_bench\data2048_v96000','--vocab','96000','--seq','2048',
  '--d','1024','--layers','12','--E','4','--n_q','16','--d_ff','3328',
  '--top_k','2','--csa','4','--opt','int8gpu','--amp','gemm','--batch','1',
  '--steps','20','--log-every','99','--no-final-save'
)
foreach ($g in @('12','16','20')) {
    $env:TG_POOL_GB = $g
    $out = & $py @common 2>&1 | Out-String
    $m = [regex]::Match($out, '"step_s_steady":\s*([0-9.]+)')
    if ($m.Success) {
        Write-Output ("TG_POOL_GB={0,-3} step_s_steady = {1} s" -f $g, $m.Groups[1].Value)
    } else {
        $tail = ($out -split "`n" | Select-Object -Last 2) -join ' | '
        Write-Output ("TG_POOL_GB={0,-3} FAIL: {1}" -f $g, $tail)
    }
}
