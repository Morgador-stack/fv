#!/usr/bin/env bash
# 旗舰链末段（重启后恢复）：resume fc_f2000(已精修 2000 实心) → 3000 + auto-soft → 判定。
# 独占 GPU（nvlddmkm 崩溃教训：串行纪律）。
set -eu
FV=/d/WorkSpace/ZCWorkspace/forza-vinyl
EXE=$FV/fv/target-l/release
IMG=$FV/refs/forza-painter-fh6/assets/imgs/ayylmao.png
T=/c/Users/Lenovo/AppData/Local/Temp

echo "=== $(date +%H:%M:%S) final stage" | tee -a $T/flagship_chain.log
$EXE/fvfit-gpu.exe $IMG --resume $T/fc_f2000.json --layers 3000 --auto-soft \
  --samples 1000000 --opt-res 328 --bg auto --out-prefix $T/fc_full \
  2>&1 | tail -1 | tee -a $T/flagship_chain.log
echo "=== $(date +%H:%M:%S) JUDGE" | tee -a $T/flagship_chain.log
$EXE/examples/eval_design.exe $IMG $T/fc_full.json --bg auto 2>/dev/null | tail -1 | tee -a $T/flagship_chain.log
echo ALLDONE | tee -a $T/flagship_chain.log
