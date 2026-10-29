$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
$token = (Get-Content ".token" -Raw).Trim()
$remote = "https://Morgador-stack:" + $token + "@github.com/Morgador-stack/fv"
git push $remote main
Write-Output ("exit=" + $LASTEXITCODE)
