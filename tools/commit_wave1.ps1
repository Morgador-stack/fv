$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
git add -A
git commit -m "docs+fix: agent wave-1 deliverables

- Research: TUNE_V3_DESIGN (continuous knobs + early-stop, CD beats CEM 2.41x),
  REFINE_FORENSICS_PLAN (4-experiment discriminator package),
  FV_BENCH_DESIGN (judge library, zero-geometry principle),
  FM2023_PRESTUDY (anti-cheat = MS proprietary, G0=CE survey),
  SVG_PIPELINE_PRESTUDY (3-20x layer efficiency, CPU-only),
  AUDIT_REPORT (no P0, P1x3 local-path leaks)
- fix(fv-fit): intake gate rejects non-finite data with position
- fix(fv-inject): pre-write gate as defense-in-depth layer
- docs: README cross-caliber fix, PUSH counts, tune headroom +4.88dB added"
git log --oneline | Select-Object -First 2
