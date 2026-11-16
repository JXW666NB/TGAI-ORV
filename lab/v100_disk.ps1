# Disk and run-directory probe on the V100 (ASCII only)
$d = Get-PSDrive D
Write-Output ("D: free = {0:N2} GB, used = {1:N2} GB" -f ($d.Free / 1GB), ($d.Used / 1GB))
if (Test-Path 'D:\orv_runs') {
    Get-ChildItem 'D:\orv_runs' -Directory | ForEach-Object {
        $s = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
              Measure-Object Length -Sum).Sum
        Write-Output ("  {0} = {1:N2} GB" -f $_.Name, ($s / 1GB))
    }
} else {
    Write-Output "  D:\orv_runs missing"
}
