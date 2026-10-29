$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
git add -A
git commit -m "fix(security): scrub local paths per audit P1 findings

- replay.rs: temp_dir() instead of hardcoded user profile
- bench flagship scripts: TMPDIR fallback + self-locating FV
- fv-seg probes: required CLI args, no local defaults
Verified: replay test green (skip path), sh -n syntax all pass,
grep 17479/Lenovo zero hits in tracked sources"
git log --oneline | Select-Object -First 2
