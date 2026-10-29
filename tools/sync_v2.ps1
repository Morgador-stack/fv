# Delta sync v2: upload changed blobs then full tree POST, commit, move ref.
$ErrorActionPreference = "Stop"
$git = "D:\Tools\MinGit\cmd\git.exe"
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
$token = (Get-Content ".token" -Raw).Trim()
$H = @{ Authorization = "Bearer " + $token; "User-Agent" = "fv"; Accept = "application/vnd.github+json" }
$owner = "Morgador-stack"
$repo = "fv"

function Api($method, $path, $body) {
    $uri = "https://api.github.com/" + $path.TrimStart("/")
    $a = @{ Uri = $uri; Method = $method; Headers = $H; ContentType = "application/json" }
    if ($null -ne $body) { $a.Body = ($body | ConvertTo-Json -Depth 8 -Compress) }
    try { return Invoke-RestMethod @a } catch {
        $code = 0; $det = ""
        if ($_.Exception.Response) {
            $code = [int]$_.Exception.Response.StatusCode
            try { $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream()); $det = $sr.ReadToEnd() } catch {}
        }
        return @{ __status = $code; __det = $det }
    }
}

$localHead = (& $git rev-parse HEAD).Trim()
$localTreeSha = ((& $git cat-file -p HEAD) | Where-Object { $_ -match "^tree " }) -split " " | Select-Object -Index 1
$remoteCommits = Api GET "/repos/$owner/$repo/commits?per_page=1"
if ($remoteCommits.__status) { throw "remote fetch failed" }
$remoteHead = $remoteCommits[0].sha
Write-Output ("local " + $localHead.Substring(0,8) + " tree " + $localTreeSha.Substring(0,8))
Write-Output ("remote " + $remoteHead.Substring(0,8))

$treeCheck = Api GET ("/repos/$owner/$repo/git/trees/" + $localTreeSha)
if (-not $treeCheck.__status) {
    Write-Output ("local tree EXISTS on remote: " + $treeCheck.sha.Substring(0,8))
    $treeToUse = $treeCheck.sha
} else {
    Write-Output "local tree missing on remote; uploading changed blobs + full tree"
    $rt = Api GET ("/repos/$owner/$repo/git/trees/" + $remoteHead + "?recursive=1")
    $remoteBlobs = @{}
    foreach ($e in $rt.tree) { if ($e.type -eq "blob") { $remoteBlobs[$e.path] = $e.sha } }

    $lsOut = & $git ls-tree -r HEAD
    $changed = @()
    $treeItems = @()
    foreach ($line in $lsOut) {
        $meta, $path = $line -split "`t", 2
        $parts = $meta.Split(" ")
        $sha = $parts[2]
        $p = $path.Trim()
        $treeItems += @{ path = $p; mode = "100644"; type = "blob"; sha = $sha }
        if (-not $remoteBlobs.ContainsKey($p) -or $remoteBlobs[$p] -ne $sha) {
            $changed += @{ path = $p; sha = $sha }
        }
    }
    Write-Output ("blobs to upload: " + $changed.Count)
    $i = 0
    foreach ($c in $changed) {
        $tmp = New-TemporaryFile
        & cmd /c "`"$git`" cat-file blob HEAD:$($c.path) > ""$tmp""" 2>$null
        $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($tmp))
        Remove-Item $tmp
        $blob = Api POST ("/repos/$owner/$repo/git/blobs") @{ content = $b64; encoding = "base64" }
        if ($blob.__status) { throw "blob fail $($c.path): $($blob.__status)" }
        foreach ($t in $treeItems) { if ($t.path -eq $c.path) { $t.sha = $blob.sha } }
        $i++
        if ($i % 10 -eq 0) { Write-Output ("  " + $i + "/" + $changed.Count) }
    }
    $t = Api POST ("/repos/$owner/$repo/git/trees") @{ tree = $treeItems }
    if ($t.__status) { throw "tree POST failed: $($t.__status) $($t.__det)" }
    $treeToUse = $t.sha
    Write-Output ("[tree] " + $treeToUse.Substring(0,8))
}

$c = Api POST ("/repos/$owner/$repo/git/commits") @{
    message = "sync: wave-1 (audit P1 fixes, fv-bench step-1, tune v3, docs)"
    tree = $treeToUse
    parents = @($remoteHead) }
if ($c.__status) { throw "commit failed: $($c.__status) $($c.__det)" }
Write-Output ("[commit] " + $c.sha.Substring(0,8))
$null = Api PATCH ("/repos/$owner/$repo/git/refs/heads/main") @{ sha = $c.sha }
Write-Output ("[done] main -> " + $c.sha.Substring(0,8))
