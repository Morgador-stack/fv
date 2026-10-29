# 经 api.github.com Git Data API 单提交全量上传（PS 原生，无 git/python 依赖）。
# 用法: powershell -File gh_upload.ps1 [repo名=fv]
# token 读 <repo根>/.token。排除: .gitignore 主干规则 + .token/.mimosa/fv.bundle/workflows。
param([string]$RepoName = "fv")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$token = (Get-Content (Join-Path $root ".token") -Raw).Trim()
$H = @{ Authorization = "Bearer $token"; Accept = "application/vnd.github+json";
        "X-GitHub-Api-Version" = "2022-11-28"; "User-Agent" = "fv-upload" }

function Api($method, $path, $body) {
    $uri = "https://api.github.com/" + $path.TrimStart("/")
    $args = @{ Uri = $uri; Method = $method; Headers = $H; ContentType = "application/json" }
    if ($body -ne $null) { $args.Body = ($body | ConvertTo-Json -Depth 8 -Compress) }
    try { return Invoke-RestMethod @args } catch {
        $resp = $_.Exception.Response
        if ($resp) {
            $sr = New-Object IO.StreamReader($resp.GetResponseStream())
            $det = $sr.ReadToEnd()
            throw "API $method $path -> HTTP $([int]$resp.StatusCode): $det"
        }
        throw
    }
}

# 1) 鉴权 + owner
$me = Api GET "/user"
$owner = $me.login
Write-Host "[auth] $owner OK"

# 2) 建仓/复用
Write-Host "[repo] Morgador-stack/fv（已确认存在，跳过建仓）"

# 3) 枚举文件（.gitignore 主干语义）
$exDirs  = @("\\refs\\", "\\deps\\", "\\build\\", "\\bundle-5090d\\", "\\target\\", "\\target-", "\.mimosa\\", "\\mve0_dut\\", "\\results\\", "\\pqdump\\target\\")
$exPats  = @("^\.token$", "\.log$", "^bench/mve0/t_", "^bench/vis_", "\.preview\.png$", "^tools/", "^aqtinstall", "^vinylizer", " Cargo.lock$|^fv/Cargo.lock$", "^PUSH.md")
$files = Get-ChildItem $root -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($root.Length + 1)
    if ($rel -like ".git*") { return $false }
    if ($rel -eq ".token" -or $rel -like "*.log" -or $rel -like "fv.bundle") { return $false }
    if ($rel -like "refs*" -or $rel -like "deps*" -or $rel -like "build*" -or $rel -like "bundle-5090d*") { return $false }
    if ($rel -match "\\\\.mimosa\\\\") { return $false }
    if ($rel -like "fv\target*") { return $false }
    if ($rel -like "bench\results*") { return $false }
    if ($rel -like "bench\mve0\t_*") { return $false }
    if ($rel -like "bench\vis_*") { return $false }
    if ($rel -like "bench\mve0_dut*") { return $false }
    if ($rel -like ".github\workflows*") { return $false }
    if ($rel -like "tools*") { return $false }
    if ($rel -eq "PUSH.md" -or $rel -eq "fv/Cargo.lock" -or $rel -eq "Cargo.lock") { return $false }
    return $true
}
Write-Host ("[tree] " + $files.Count + " 个文件待传")

# 4) 空仓 prime：第一个根级文件走 Contents API
$first = $files | Where-Object { $_.FullName.Substring($root.Length + 1) -notmatch "[\\/]" } | Select-Object -First 1
if (-not $first) { throw "无根级文件可 prime" }
$rel1 = $first.FullName.Substring($root.Length + 1) -replace "/", "_"
# （prime 用原文件名，路径替换只防意外斜杠）
$rel1 = [IO.Path]::GetFileName($first.FullName)
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($first.FullName))
try {
    $null = Api PUT "/repos/$owner/$RepoName/contents/$rel1" @{
        message = "bootstrap (empty-repo prime)"; branch = "main"; content = $b64 }
    Write-Host "[prime] $rel1 OK"
} catch { Write-Host "[prime] 跳过（可能已有提交）: $($_.Exception.Message.Substring(0,[Math]::Min(80,$_.Exception.Message.Length)))" }

# 5) blob → tree → commit → ref
$tree = @(); $i = 0
foreach ($f in $files) {
    $rel = ($f.FullName.Substring($root.Length + 1)) -replace "\\","/"
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($f.FullName))
    $blob = Api POST "/repos/$owner/$RepoName/git/blobs" @{ content = $b64; encoding = "base64" }
    $tree += @{ path = $rel; mode = "100644"; type = "blob"; sha = $blob.sha }
    $i++
    if ($i % 25 -eq 0) { Write-Host "  blobs $i/$($files.Count)" }
}
Write-Host ("[blobs] " + $tree.Count + " OK")
$t = Api POST "/repos/$owner/$RepoName/git/trees" @{ tree = $tree }
$c = Api POST "/repos/$owner/$RepoName/git/commits" @{
    message = "fv: full import via Git Data API (PS)`n`nLocal history (12 commits) preserved in fv.bundle; see STUDY.md."
    tree = $t.sha; parents = @() }
try {
    $null = Api PATCH "/repos/$owner/$RepoName/git/refs/heads/main" @{ sha = $c.sha; force = $true }
} catch {
    $null = Api POST "/repos/$owner/$RepoName/git/refs" @{ ref = "refs/heads/main"; sha = $c.sha }
}
Write-Host ("[done] https://github.com/$owner/$RepoName  commit " + $c.sha.Substring(0,8) + "（" + $tree.Count + " 文件）")
