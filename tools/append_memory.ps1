# ASCII-only (PS 5.1 reads no-BOM files as ANSI; CJK corrupts here-strings)
$add = @"
**2026-10-28 fv session (env rebuild + MSVC):**
- Git: MinGit 2.55 x64 at D:\Tools\MinGit\cmd (add to PATH manually). Python 3.14/3.15 ARM64 at %LOCALAPPDATA%\Programs\Python\. Rust 1.99.1 + rsproxy crates mirror config restored from backup.
- MSVC Build Tools INSTALLED (exit 0): VCTools.ARM64 + Win11SDK.26100; cargo build/test fully restored (fv-common+fv-fit 53 tests green). Installer gotchas: (1) --include <id> syntax exits 87; each component needs its own --add (correct script: tools/install_msvc.ps1); (2) silent UAC decline shows as "User may have declined" in box log.
- tune knob importance (59 logs, tools/tune_stats.py + TUNE_V2_ANALYSIS.md): band_radius 27% win-rate = top lever; subject_* only 11.8% (v3 fast tier can drop them); second coordinate-descent round contributes 35% of wins.
- fv GitHub upload still waiting for valid PAT (old .token 401): powershell -File tools\gh_upload.ps1 fv completes it in one shot.
"@
[IO.File]::AppendAllText(
    "C:\Users\17479\.zcode\cli\memories\projects\zcworkspace-d6e910a5f54e006e\memory\workstation-rtx-spark-n1x.md",
    $add + [Environment]::NewLine
)
Write-Output "appended"
