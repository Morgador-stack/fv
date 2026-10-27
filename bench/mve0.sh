#!/usr/bin/env bash
# MVE-0 headroom 探针（L13_RESEARCH §D）：30 图 × fvseg --tune（每图 13 评估），
# 记录 baseline/best PSNR 与 headroom。判据：p95 headroom ≥0.5dB → L1.3 继续；
# 全体 <0.3dB → 止损。轻预算（total 240 / samples 5000 / soft 120）。
# 产出：bench/mve0/headroom.csv
set -u
FV=/d/WorkSpace/ZCWorkspace/forza-vinyl
EXE=${EXE:-$FV/fv/target-b1/release}
DIR=$FV/bench/mve0
CSV=$DIR/headroom.csv
echo "img,baseline,best,headroom" > "$CSV"

for img in $(cat $DIR/manifest.txt | tr -d '\r'); do
  name=${img%.png}
  out=$DIR/t_${name}
  log=$DIR/t_${name}.log
  # U2Net matte + 全管线坐标搜索（GPU）
  "$EXE/fvseg.exe" "$DIR/$img" --model u2net --tune --gpu \
      --total 240 --soft 120 --samples 5000 --seed 7 --out "$out" > "$log" 2>&1
  base=$(command grep "baseline" "$log" | tail -1 | sed -E 's/.*psnr=([0-9.]+|inf).*/\1/')
  best=$(command grep "done in" "$log" | tail -1 | sed -E 's/.*best psnr=([0-9.]+|inf).*/\1/')
  # 完美重构（inf）按 99 记——headroom 语义不受影响（已是上界）
  base=${base/inf/99}; best=${best/inf/99}
  if [ -n "$base" ] && [ -n "$best" ]; then
    hr=$(echo "$best $base" | awk '{printf "%.3f", $1-$2}')
    echo "$name,$base,$best,$hr" >> "$CSV"
    echo "$name base=$base best=$best headroom=$hr"
  else
    echo "$name PARSE_FAIL (see $log)"
  fi
done
echo "---- headroom stats ----"
awk -F, 'NR>1 {n++; v[n]=$4; s+=$4; if($4>mx)mx=$4} END {
  asort(v); p50=v[int(n/2)+1]; p95=v[int(n*0.95)+1>n?n:int(n*0.95)+1];
  printf "n=%d mean=%.3f p50=%.3f p95=%.3f max=%.3f\n", n, s/n, p50, p95, mx}' "$CSV"
