# bench · 基准与调优协议

## 语料获取（不入库，~1400 张 FH 社区涂装模板）

语料来自 [forza-painter](https://github.com/the-adawg/forza-painter)（MIT）内嵌资源：
`src/data/fh6_vinyl_resources/Vinyls/`（14 类：Community_Vinyls ×4 / Flames / Gradient_Shapes /
Letters ×22 / Nature / Paint_Splats / Primitives / Racing_Icons / Stripes / Tears / Tribal）。

模板是单色剪影 on 透明（游戏内再上色）——直接拟合会退化（bg=字形均值=目标，MSE 恒 0）。
统一流程：`flatten` 工具智能压平（字形亮度 ≥128 → 黑底，否则白底）：

```bash
cargo build --release -p fv-fit --example flatten
fv/target/release/examples/flatten.exe in.png out.png   # 底色自动
```

## 脚本

| 脚本 | 用途 |
|---|---|
| `bench.sh [layers] [samples] [solid\|soft]` | 12 类各一（固定清单）统一基准；`EXE=` 可换二进制目录；输出 results/<mode>/summary.csv |
| `mve0.sh` | MVE-0 规划参数调优协议（30 图 × `fvseg --tune` 坐标搜索 → headroom 统计） |
| `flagship_v3.sh` 等 | 冲榜链（refine↔resume 交替）的精确复现脚本 |

## 数据（已入库）

- `mve0/features.csv`：30 图特征（墨迹 chamfer 宽度分位 / 边界密度 / 走向各向异性；
  背景取**灰度中位数**——均值会被 >50% 墨量图污染）。
- `mve0/headroom.csv` / `headroom_dut.csv`：两域调优余量实测（vinyl 域均值 +4.88dB / 照片域 ≤0.33dB 止损）。
- `mve0/manifest.txt`：30 图清单（seed 20261026 每类定额抽取，可复现）。

## 参考（post-B1，2026-10-27）

| 指标 | 值 |
|---|---|
| 12 图 solid@300 均值 | 30.05 dB（B1 前 27.78） |
| 完美重构（MSE=0） | 5/12 图 |
| 逐图 tune（59s/图）headroom | 均值 +4.88 dB、p95 20.4 dB |
