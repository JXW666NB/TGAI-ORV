# Probe candidate pretraining corpora on the V100 (ASCII only: PS 5.1 reads scripts as ANSI)
$dirs = @('D:\tgai_corpus', 'D:\TGAI-V3-1.5B\01_corpus', 'D:\TGAI-V3-1.5B\02_tokenizer', 'D:\tgai_nlp')
foreach ($d in $dirs) {
    Write-Output ("=== " + $d + " ===")
    if (-not (Test-Path $d)) { Write-Output "  (missing)"; continue }
    $all = Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue
    if (-not $all) { Write-Output "  (empty)"; continue }
    $total = ($all | Measure-Object Length -Sum).Sum
    Write-Output ("  total " + [math]::Round($total / 1GB, 2) + " GB, files " + $all.Count)
    $top = $all | Sort-Object Length -Descending | Select-Object -First 10
    foreach ($f in $top) {
        $rel = $f.FullName.Substring($d.Length)
        Write-Output ("  " + [math]::Round($f.Length / 1MB, 1) + " MB  " + $rel)
    }
}
