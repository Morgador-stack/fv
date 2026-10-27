#!/usr/bin/env bash
# MVE-0 DUT 域（真实照片）：10 对 (jpg, mask) × fvseg --tune --matte。
# 判据与 vinyl 域一致（p95≥0.5dB go / 全体<0.3dB 止损）。串行纪律。
set -u
FV=/d/WorkSpace/ZCWorkspace/forza-vinyl
EXE=$FV/fv/target-b1/release
DIR=$FV/bench/mve0_dut
CSV=$FV/bench/mve0/headroom_dut.csv
echo "img,baseline,best,headroom" > "$CSV"

for i in 0 1 2 3 4 5 6 7 8 9; do
  img=$DIR/$(printf '%03d' $i)_image.jpg
  msk=$DIR/$(printf '%03d' $i)_mask.png
  [ -f "$img" ] || continue
  log=$DIR/t_$i.log
  "$EXE/fvseg.exe" "$img" --matte "$msk" --tune --gpu \
      --total 240 --soft 120 --samples 5000 --seed 7 --out "$DIR/t_$i" > "$log" 2>&1
  base=$(command grep "baseline" "$log" | tail -1 | sed -E 's/.*psnr=([0-9.]+).*/\1/')
  best=$(command grep "done in" "$log" | tail -1 | sed -E 's/.*best psnr=([0-9.]+).*/\1/')
  if [ -n "$base" ] && [ -n "$best" ]; then
    hr=$(echo "$best $base" | awk '{printf "%.3f", $1-$2}')
    echo "dut_$i,$base,$best,$hr" >> "$CSV"
    echo "dut_$i base=$base best=$best headroom=$hr"
  else
    echo "dut_$i PARSE_FAIL"
  fi
done
echo "---- DUT headroom stats ----"
awk -F, 'NR>1 {n++; v[n]=$4; s+=$4; if($4+0>mx)mx=$4+0} END {
  if(n>0){asort(v); p50=v[int(n/2)+1]; ix=int(n*0.95)+1; if(ix>n)ix=n; printf "n=%d mean=%.3f p50=%.3f p95=%.3f max=%.3f\n", n, s/n, p50, v[ix], mx}}' "$CSV"
