$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
git add fv/crates/fv-seg/src/bin/fvseg.rs tools/tune_stats.py
git commit -m "feat(fvseg): tune v3 - continuous knobs + early-stop + eval cache

- Kind{Grid,Neighbors,GridVertex,Cont,SnapCont} DimSpec engine per
  TUNE_V3_DESIGN (dual probes + vertex3 lagrange + verify-before-adopt)
- E1 gain/floor skip gate (zero-loss verified on 30-img corpus)
- E3 inf-baseline early exit (1 eval vs 39 wasted)
- E5 param-key HashSet dedup (stripes_21: 15/18 cache hits)
- flags: --tune-algo v2|v3 (default v3), --tune-rounds, --tune-fast
- tears_38: headroom 6.49 -> 8.65 dB (+33%), 35 evals vs 39, 75.7s
- v2 regression: bit-identical rerun; 21 tests green, clippy 0"
git log --oneline | Select-Object -First 2
