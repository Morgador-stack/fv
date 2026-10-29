$env:Path = "D:\Tools\MinGit\cmd;" + $env:Path
Set-Location "D:\WorkSpace\ZCWorkspace\forza-vinyl"
git add fv/crates/fv-fit/src/engine.rs fv/crates/fv-fit/tests/gpu_parity.rs fv/crates/fv-gpu/tests/replay.rs fv/crates/fv-seg/examples/ bench/flagship_chain.sh bench/flagship_final.sh bench/flagship_v3.sh
git commit -m "refactor(engine): split fit_impl 542-line monolith into staged functions

- prepare_workspace / solid_greedy_loop / polish_phase / soft_phase /
  finalize_outputs, intermediate state in Workspace struct
- 61 tests green (52 behavioral anchors unchanged + 9 new unit tests
  covering RNG contract, GPU/CPU candidate stream parity, verify top-k,
  to_full lineage, soft accept/rollback)
- NEW tests/gpu_parity.rs: CpuMirrorEvaluator proves fit_gpu == fit
  bit-identical across solid/polish+soft/auto_soft channels
- 12/12 CLI artifacts byte-identical vs pre-refactor binary (fc /b)
- audit P1 fixes: replay.rs temp_dir, bench scripts TMPDIR+self-locate,
  fv-seg probes required args (local paths scrubbed)"
git log --oneline | Select-Object -First 3
