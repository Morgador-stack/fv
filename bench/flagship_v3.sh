#!/usr/bin/env bash
# 旗舰链 v3：B1 结构导引采样版交替链（s300→refine→r1000→refine→r2000→refine→3000 auto-soft）。
# 独占 GPU（串行纪律）。
set -eu
FV="$(cd "$(dirname "$0")/.." && pwd)"
EXE=$FV/fv/target-b1/release
IMG=$FV/refs/forza-painter-fh6/assets/imgs/ayylmao.png
T=${TMPDIR:-/tmp}
REFINE="--diff-tau0 32 --diff-lr-geo 0.2 --diff-lr-col 0.8"
LOG=$T/flagship_v3.log

run() { echo "=== $(date +%H:%M:%S) $*" | tee -a $LOG; "$@" 2>&1 | tail -1 | tee -a $LOG; }

cd $FV/fv
run $EXE/fvfit-gpu.exe $IMG --layers 300 --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fv3_s300
run $EXE/fvfit-gpu.exe --refine-json $T/fv3_s300.json $IMG --opt-res 328 --bg auto --diff-iters 300 $REFINE
cp $T/fv3_s300.refined.json $T/fv3_s300r.json
run $EXE/fvfit-gpu.exe $IMG --resume $T/fv3_s300r.json --layers 1000 --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fv3_r1000
run $EXE/fvfit-gpu.exe --refine-json $T/fv3_r1000.json $IMG --opt-res 328 --bg auto --diff-iters 300 $REFINE
cp $T/fv3_r1000.refined.json $T/fv3_f1000.json
run $EXE/fvfit-gpu.exe $IMG --resume $T/fv3_f1000.json --layers 2000 --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fv3_r2000
run $EXE/fvfit-gpu.exe --refine-json $T/fv3_r2000.json $IMG --opt-res 328 --bg auto --diff-iters 300 $REFINE
cp $T/fv3_r2000.refined.json $T/fv3_f2000.json
run $EXE/fvfit-gpu.exe $IMG --resume $T/fv3_f2000.json --layers 3000 --auto-soft --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fv3_full

echo "=== $(date +%H:%M:%S) JUDGE" | tee -a $LOG
$EXE/examples/eval_design.exe $IMG $T/fv3_full.json --bg auto 2>/dev/null | tail -1 | tee -a $LOG
echo ALLDONE | tee -a $LOG
