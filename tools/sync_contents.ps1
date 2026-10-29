# Incremental sync via Contents API (PUT per file = 1 commit each, then fast-forward not needed).
# Strategy: PUT changed files (each creates a commit on main); deletions via DELETE.
# Token needs only Contents: Read/Write (which fv-upload has).
$ErrorActionPreference = "Stop"
$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
$token = (Get-Content ".token" -Raw).Trim()
$H = @{ Authorization = "Bearer " + $token; "User-Agent" = "fv"; Accept = "application/vnd.github+json" }
$owner = "Morgador-stack"; $repo = "fv"

function Api($method, $path, $body) {
    $uri = "https://api.github.com/" + $path.TrimStart("/")
    $a = @{ Uri = $uri; Method = $method; Headers = $H; ContentType = "application/json" }
    if ($null -ne $body) { $a.Body = ($body | ConvertTo-Json -Depth 8 -Compress) }
    try { return Invoke-RestMethod @a } catch {
        $code = 0
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        return @{ __status = $code }
    }
}

# local files from git ls-tree HEAD
$lsOut = & git ls-tree -r HEAD
$localFiles = @{}
foreach ($line in $lsOut) {
    $meta, $path = $line -split "`t", 2
    $parts = $meta.Split(" ")
    $localFiles[$path.Trim()] = $parts[2]
}

# remote files from HEAD tree
$head = Api GET "/repos/$owner/$repo/commits/main"
if ($head.__status) { throw "cannot read remote HEAD" }
$remoteHeadSha = $head.sha
$rt = Api GET ("/repos/$owner/$repo/git/trees/" + $head.commit.tree.sha + "?recursive=1")
if ($rt.__status) { throw "cannot read remote tree" }
$remoteFiles = @{}
foreach ($e in $rt.tree) { if ($e.type -eq "blob") { $remoteFiles[$e.path] = $e.sha } }

# changed/added
$changed = @()
foreach ($k in $localFiles.Keys) {
    if ($k -like ".github*") { continue }   # workflow files need workflow-scope token; skip
    if (-not $remoteFiles.ContainsKey($k) -or $remoteFiles[$k] -ne $localFiles[$k]) { $changed += $k }
}
# deleted
$deleted = @($remoteFiles.Keys | Where-Object { -not $localFiles.ContainsKey($_) })

Write-Output ("changed/added: " + $changed.Count + "  deleted: " + $deleted.Count)

$i = 0
foreach ($k in $changed) {
    $tmp = New-TemporaryFile
    & "D:\Tools\MinGit\cmd\git.exe" cat-file blob ("HEAD:" + $k) > $tmp
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($tmp))
    Remove-Item $tmp
    $body = @{ message = "sync: $k"; content = $b64; branch = "main" }
    $r = Api PUT ("/repos/$owner/$repo/contents/" + ($k -replace "\\", "/")) $body
    if ($r.__status -and $r.__status -ne 0) { throw "PUT failed for $k : HTTP " + $r.__status }
    $i++
    Write-Output ("  [$i/$($changed.Count)] $k")
    Start-Sleep -Milliseconds 300
}

foreach ($k in $deleted) {
    $cur = Api GET ("/repos/$owner/$repo/contents/" + ($k -replace "\\", "/") + "?ref=main")
    if ($cur.__status) { continue }
    $body = @{ message = "sync: delete $k"; sha = $cur.sha; branch = "main" }
    $null = Api DELETE ("/repos/$owner/$repo/contents/" + ($k -replace "\\", "/")) $body
    $i++
    Write-Output ("  [del] $k")
    Start-Sleep -Milliseconds 300
}

Write-Output ("[done] " + $i + " operations. Remote main now matches local files (as 1-commit-per-file history).")
