# Finish sync v2: base_tree + delta entries (avoids full-tree reconstruction).
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
        $code = 0; $det = ""
        if ($_.Exception.Response) {
            $code = [int]$_.Exception.Response.StatusCode
            try { $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream()); $det = $sr.ReadToEnd() } catch {}
        }
        throw "API $method $path -> HTTP $code : $det"
    }
}

$localHead = (& git rev-parse HEAD).Trim()
$remoteCommits = Api GET "/repos/Morgador-stack/fv/commits?per_page=1"
$remoteHead = $remoteCommits[0].sha
$remoteTree = $remoteCommits[0].commit.tree.sha
Write-Output ("local " + $localHead.Substring(0,8) + "  remote " + $remoteHead.Substring(0,8) + " tree " + $remoteTree.Substring(0,8))

# local tree entries of HEAD (recursive, from git)
$lsOut = & git ls-tree -r HEAD
$localFiles = @{}
foreach ($line in $lsOut) {
    $meta, $path = $line -split "`t", 2
    $parts = $meta.Split(" ")
    $localFiles[$path.Trim()] = @{ sha = $parts[2]; mode = $parts[0] }
}

# remote tree entries
$rt = Api GET ("/repos/Morgador-stack/fv/git/trees/" + $remoteTree + "?recursive=1")
$remoteFiles = @{}
foreach ($e in $rt.tree) { if ($e.type -eq "blob") { $remoteFiles[$e.path] = $e.sha } }

# delta: local entries whose sha differs or missing remotely
$delta = New-Object System.Collections.Generic.List[object]
foreach ($k in $localFiles.Keys) {
    $l = $localFiles[$k]
    if (-not $remoteFiles.ContainsKey($k) -or $remoteFiles[$k] -ne $l.sha) {
        $delta.Add(@{ path = ($k -replace "\\","/"); mode = "100" + "644"; type = "blob"; sha = $l.sha })
    }
}
Write-Output ("delta entries: " + $delta.Count)

if ($delta.Count -eq 0) {
    # trees match; just move ref to a commit pointing at remote tree with local message? No-op.
    Write-Output "trees identical - nothing to sync"
    exit 0
}

$t = Api POST "/repos/Morgador-stack/fv/git/trees" @{ base_tree = $remoteTree; tree = $delta }
Write-Output ("[tree] " + $t.sha.Substring(0,8))
$c = Api POST "/repos/Morgador-stack/fv/git/commits" @{
    message = "sync: wave-1 (audit P1 fixes, fv-bench step-1, docs sync, research docs)"
    tree = $t.sha
    parents = @($remoteHead) }
Write-Output ("[commit] " + $c.sha.Substring(0,8))
$null = Api PATCH "/repos/Morgador-stack/fv/git/refs/heads/main" @{ sha = $c.sha }
Write-Output ("[done] main -> " + $c.sha.Substring(0,8))
