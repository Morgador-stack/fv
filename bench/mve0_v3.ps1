# mve0 corpus runner v3 (PS native, no bash dependency)
# 30 images x fvseg --tune (default v3), records baseline/best/headroom
$ErrorActionPreference = "Continue"
$FV = "D:\WorkSpace\ZCWorkspace\forza-vinyl"
$EXE = "$FV\fv\target-b1\release\fvseg.exe"
$DIR = "$FV\bench\mve0"
$CSV = "$DIR\headroom_v3.csv"
"img,baseline,best,headroom" | Set-Content $CSV

$manifest = Get-Content "$DIR\manifest.txt" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$n = 0
foreach ($img in $manifest) {
    $name = $img -replace "\.png$", ""
    $out = "$DIR\v3_$name"
    $log = "$DIR\v3_$name.log"
    & $EXE "$DIR\$img" --model u2net --tune --gpu --tune-algo v3 `
        --total 240 --soft 120 --samples 5000 --seed 7 --out $out > $log 2>&1
    $txt = Get-Content $log -Raw -ErrorAction SilentlyContinue
    $base = $null; $best = $null
    if ($txt -match "baseline\s+psnr=([\d.]+|inf)") { $base = $Matches[1] }
    if ($txt -match "best psnr=([\d.]+|inf) shapes") { $best = $Matches[1] }
    if ($base -and $best) {
        $b = if ($base -eq "inf") { 99.0 } else { [double]$base }
        $t = if ($best -eq "inf") { 99.0 } else { [double]$best }
        $hr = [Math]::Round($t - $b, 3)
        "$name,$base,$best,$hr" | Add-Content $CSV
        Write-Output "$name base=$base best=$best headroom=$hr"
    } else {
        Write-Output "$name PARSE_FAIL"
    }
    $n++
}
Write-Output "---- stats ($n images) ----"
$rows = Get-Content $CSV | Select-Object -Skip 1 | ForEach-Object {
    $f = $_ -split ","
    [pscustomobject]@{ img = $f[0]; hr = [double]$f[3] }
}
$vals = $rows.hr | Sort-Object
$mean = ($rows.hr | Measure-Object -Average).Average
$p50 = $vals[[int][Math]::Floor($rows.Count * 0.5)]
$p95ix = [Math]::Min($rows.Count - 1, [int][Math]::Floor($rows.Count * 0.95))
$p95 = $vals[$p95ix]
$mx = ($rows.hr | Measure-Object -Maximum).Maximum
Write-Output ("n={0} mean={1:F3} p50={2:F3} p95={3:F3} max={4:F3}" -f $rows.Count, $mean, $p50, $p95, $mx)
