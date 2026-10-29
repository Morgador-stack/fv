# Incremental sync: diff local HEAD vs remote HEAD via GitHub API, push changed blobs.
# ASCII-only (PS 5.1 no-BOM reads as ANSI).
$ErrorActionPreference = "Stop"
$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
$token = (Get-Content ".token" -Raw).Trim()
$H = @{ Authorization = "Bearer " + $token; "User-Agent" = "fv"; Accept = "application/vnd.github+json" }

function Api($method, $path, $body) {
    $uri = "https://api.github.com/" + $path.TrimStart("/")
    $a = @{ Uri = $uri; Method = $method; Headers = $H; ContentType = "application/json" }
    if ($null -ne $body) { $a.Body = ($body | ConvertTo-Json -Depth 8 -Compress) }
    try { return Invoke-RestMethod @a } catch {
        $code = 0
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        return @{ __status = $code; __error = $_.Exception.Message }
    }
}

$me = Api GET "/user"
if ($me.__status -eq 401) { throw "token invalid (401)" }
$owner = $me.login
$repo = "fv"
Write-Output ("[auth] " + $owner)

$localHead = (& git rev-parse HEAD).Trim()
$remoteCommits = Api GET ("/repos/" + $owner + "/" + $repo + "/commits?per_page=1")
$remoteHeadSha = $remoteCommits[0].sha
Write-Output ("local HEAD " + $localHead.Substring(0,8) + "  remote HEAD " + $remoteHeadSha.Substring(0,8))

if ($remoteHeadSha -eq $localHead) { Write-Output "already in sync"; exit 0 }

$rt = Api GET ("/repos/" + $owner + "/" + $repo + "/git/trees/" + $remoteHeadSha + "?recursive=1")
$remoteFiles = @{}
foreach ($e in $rt.tree) { if ($e.type -eq "blob") { $remoteFiles[$e.path] = $e.sha } }

$lsOut = & git ls-tree -r HEAD
$localFiles = @{}
foreach ($line in $lsOut) {
    $meta, $path = $line -split "`t", 2
    $parts = $meta.Split(" ")
    $localFiles[$path.Trim()] = $parts[2]
}
Write-Output ("local files: " + $localFiles.Count + "  remote files: " + $remoteFiles.Count)

$changed = @()
foreach ($k in $localFiles.Keys) {
    if (-not $remoteFiles.ContainsKey($k) -or $remoteFiles[$k] -ne $localFiles[$k]) { $changed += $k }
}
$deleted = @($remoteFiles.Keys | Where-Object { -not $localFiles.ContainsKey($_) })
Write-Output ("changed/added: " + $changed.Count + "  deleted: " + $deleted.Count)

$treeItems = New-Object System.Collections.Generic.List[object]
foreach ($k in $localFiles.Keys) {
    $treeItems.Add(@{ path = $k; mode = "100644"; type = "blob"; sha = $localFiles[$k] })
}
foreach ($k in $deleted) { $treeItems.Add(@{ path = $k; mode = "100644"; type = "blob"; sha = $null }) }

$i = 0
foreach ($k in $changed) {
    $tmp = New-TemporaryFile
    cmd /c "git cat-file blob HEAD:$k > ""$tmp""" 2>$null
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($tmp))
    Remove-Item $tmp
    $blob = Api POST ("/repos/" + $owner + "/" + $repo + "/git/blobs") @{ content = $b64; encoding = "base64" }
    if ($blob.__status) { throw "blob upload failed for $k" }
    for ($j = 0; $j -lt $treeItems.Count; $j++) {
        if ($treeItems[$j].path -eq $k) { $treeItems[$j].sha = $blob.sha }
    }
    $i++
    if ($i % 10 -eq 0) { Write-Output ("  blobs " + $i + "/" + $changed.Count) }
}
$finalTree = $treeItems | Where-Object { $null -ne $_.sha }
Write-Output ("[blobs] " + $i + " uploaded")

$t = Api POST ("/repos/" + $owner + "/" + $repo + "/git/trees") @{ tree = $finalTree }
$c = Api POST ("/repos/" + $owner + "/" + $repo + "/git/commits") @{
    message = "sync: wave-1 agent deliverables (audit fixes, fv-bench step-1, docs)"
    tree = $t.sha
    parents = @($remoteHeadSha) }
$null = Api PATCH ("/repos/" + $owner + "/" + $repo + "/git/refs/heads/main") @{ sha = $c.sha }
Write-Output ("[done] remote main -> " + $c.sha.Substring(0,8))
