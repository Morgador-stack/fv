#!/usr/bin/env bash
# fv 多图基准：FH 社区涂装真实语料（forza-painter 内嵌 1400 张，精选 12 类各一）。
# 用法：bench.sh [layers] [samples] [mode]   默认 300 100000 solid
#   mode: solid（纯实心）| soft（同预算 --auto-soft：实心三振后余量给软层）
# 产出：bench/results/<mode>/<img>.{json,preview.png} + summary.csv（统一裁判 --bg auto）
set -u
FV=/d/WorkSpace/ZCWorkspace/forza-vinyl
EXE=${EXE:-$FV/fv/target-l/release}
IMGS=$FV/refs/forza-painter-fh6/src/data/fh6_vinyl_resources/Vinyls
LAYERS=${1:-300}
SAMPLES=${2:-100000}
MODE=${3:-solid}
OUT=$FV/bench/results/$MODE
EXTRA=""
[ "$MODE" = "soft" ] && EXTRA="--auto-soft"
mkdir -p "$OUT"

# 12 类各一：社区涂装×2 + 火焰/渐变形/小写字母/自然/泼溅/基元/赛车图标/条纹/部落
LIST="
Community_Vinyls_1/16.png
Community_Vinyls_3/22.png
Flames/14.png
Gradient_Shapes/11.png
Lower_Letters_1/10.png
Nature/15.png
Paint_Splats/31.png
Primitives/15.png
Racing_Icons/3.png
Stripes/24.png
Tribal/16.png
Upper_Letters_1/14.png
"

CSV="$OUT/summary.csv"
echo "img,layers,shapes,psnr_db,secs" > "$CSV"
for rel in $LIST; do
  name=$(basename "$rel" .png)_$(dirname "$rel" | tr 'A-Z' 'a-z')
  img="$IMGS/$rel"
  [ -f "$img" ] || { echo "skip missing $rel"; continue; }
  # 剪影模板 → 白底压平（否则 bg=目标，mse 恒 0 无意义）
  flat="$OUT/${name}_flat.png"
  "$EXE/examples/flatten.exe" "$img" "$flat" >/dev/null 2>&1 || { echo "FAIL flatten $rel"; continue; }
  t0=$(date +%s.%N)
  "$EXE/fvfit-gpu.exe" "$flat" --layers "$LAYERS" --samples "$SAMPLES" $EXTRA \
      --opt-res 328 --bg auto --out-prefix "$OUT/$name" >/dev/null 2>&1 || { echo "FAIL fit $rel"; continue; }
  t1=$(date +%s.%N)
  line=$("$EXE/examples/eval_design.exe" "$flat" "$OUT/$name.json" --bg auto 2>/dev/null | tail -1)
  shapes=$(echo "$line" | sed -E 's/shapes=([0-9]+).*/\1/')
  psnr=$(echo "$line" | sed -E 's/.*PSNR=([0-9.]+) dB.*/\1/')
  secs=$(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')
  echo "$name,$LAYERS,$shapes,$psnr,$secs" >> "$CSV"
  echo "$name shapes=$shapes psnr=$psnr ${secs}s"
done
echo "---- summary ----"
awk -F, 'NR>1 {n++; s+=$4; if($4>mx){mx=$4}; if(mn==0||$4<mn){mn=$4}} END {printf "n=%d mean=%.2f min=%.2f max=%.2f\n", n, s/n, mn, mx}' "$CSV"
