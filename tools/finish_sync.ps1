# Finish sync: create tree from local tree sha (blobs already uploaded), commit, move ref.
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
            $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())
            $det = $sr.ReadToEnd()
        }
        throw "API $method $path -> HTTP $code : $det"
    }
}

$localHead = (& git rev-parse HEAD).Trim()
$localTree = ((& git cat-file -p HEAD) | Where-Object { $_ -match "^tree " }) -split " " | Select-Object -Index 1
$remoteCommits = Api GET "/repos/Morgador-stack/fv/commits?per_page=1"
$remoteHead = $remoteCommits[0].sha
Write-Output ("local " + $localHead.Substring(0,8) + " tree " + $localTree.Substring(0,8) + "  remote " + $remoteHead.Substring(0,8))

$t = Api POST "/repos/Morgador-stack/fv/git/trees" @{ tree = $localTree }
Write-Output ("[tree] " + $t.sha.Substring(0,8))
$c = Api POST "/repos/Morgador-stack/fv/git/commits" @{
    message = "sync: wave-1 (audit P1 fixes, fv-bench step-1, docs sync, research docs)"
    tree = $t.sha
    parents = @($remoteHead) }
Write-Output ("[commit] " + $c.sha.Substring(0,8))
$null = Api PATCH "/repos/Morgador-stack/fv/git/refs/heads/main" @{ sha = $c.sha }
Write-Output ("[done] main -> " + $c.sha.Substring(0,8))
