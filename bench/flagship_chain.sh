#!/usr/bin/env bash
# 旗舰冲高链 v2：refine↔resume 交替（纯实心阶段反复精修，最后才 auto-soft）。
# s300(已精修 26.97) → resume 1000 → refine → resume 2000 → refine → resume 3000 +auto-soft → 判定
# 全程 bg auto / 1M 采样 / opt-res 328。日志 tee 到 flagship_chain.log。
set -eu
FV="$(cd "$(dirname "$0")/.." && pwd)"
EXE=$FV/fv/target-l/release
IMG=$FV/refs/forza-painter-fh6/assets/imgs/ayylmao.png
T=${TMPDIR:-/tmp}
REFINE="--diff-tau0 32 --diff-lr-geo 0.2 --diff-lr-col 0.8"

run() { echo "=== $(date +%H:%M:%S) $*" | tee -a $T/flagship_chain.log; "$@" 2>&1 | tail -2 | tee -a $T/flagship_chain.log; }

cd $FV/fv
# 基座自产（Temp 会被清理，不依赖旧产物）：solid300 → refine
run $EXE/fvfit-gpu.exe $IMG --layers 300 --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fc_s300 --seed 20261026
run $EXE/fvfit-gpu.exe --refine-json $T/fc_s300.json $IMG --opt-res 328 --bg auto --diff-iters 300 $REFINE
cp $T/fc_s300.refined.json $T/fc_s300r.json

run $EXE/fvfit-gpu.exe $IMG --resume $T/fc_s300r.json --layers 1000 --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fc_r1000
run $EXE/fvfit-gpu.exe --refine-json $T/fc_r1000.json $IMG --opt-res 328 --bg auto --diff-iters 300 $REFINE
cp $T/fc_r1000.refined.json $T/fc_f1000.json

run $EXE/fvfit-gpu.exe $IMG --resume $T/fc_f1000.json --layers 2000 --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fc_r2000
run $EXE/fvfit-gpu.exe --refine-json $T/fc_r2000.json $IMG --opt-res 328 --bg auto --diff-iters 300 $REFINE
cp $T/fc_r2000.refined.json $T/fc_f2000.json

run $EXE/fvfit-gpu.exe $IMG --resume $T/fc_f2000.json --layers 3000 --auto-soft --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fc_full

echo "=== $(date +%H:%M:%S) JUDGE" | tee -a $T/flagship_chain.log
$EXE/examples/eval_design.exe $IMG $T/fc_full.json --bg auto 2>/dev/null | tail -1 | tee -a $T/flagship_chain.log
echo ALLDONE | tee -a $T/flagship_chain.log
