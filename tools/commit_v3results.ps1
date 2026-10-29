$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
git add bench/mve0/headroom_v3.csv bench/mve0_v3.ps1 tools/sync_github.ps1 tools/sync_v2.ps1 tools/finish_sync.ps1 tools/finish_sync2.ps1 tools/commit_wave1.ps1 tools/commit_i3.ps1 tools/commit_p1.ps1 tools/commit_i5i6.ps1 tools/tune_stats.py 2>$null
git add bench/mve0/knob_stats.json
git commit -m "bench: tune v3 corpus validation - mean headroom 5.13dB (v2: 4.88)

- 30-img corpus, PS-native runner (no bash dependency)
- p95 24.96 (v2: 20.4), max 28.07 (lower_letters_5_7: 50.89->78.96)
- vertex adoption contributed gains beyond v2 grid bounds
- validates TUNE_V3_DESIGN acceptance criteria
- tools: GitHub incremental sync scripts (Git Data API), PS-native
  corpus runners, commit helpers"
git log --oneline | Select-Object -First 2
