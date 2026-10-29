# fv · 图片 → Forza 涂装图层

把任意图片转成 Forza Horizon 6 可导入的 **vinyl 图层栈**（旋转椭圆 / 228 软椭圆），并可经内存注入直接写入游戏内涂装编辑器。

纯 Rust 自研（算法 / GPU / 注入全链路），GPU 走 Vulkan compute（wgpu）——同一代码在 NVIDIA GB10 (ARM Windows)、RTX 50 系与未来 Arc/Radeon 上运行。

```
图片 ──fvfit──▶ 几何/颜色拟合──┬─ 3000 层旋转椭圆设计 JSON（geometrize 格式）
                              ├─ fvseg 语义分层（人像/主体/背景路由，MODNet/U2Net matting）
                              └─ fv-inject ──▶ FH6 涂装编辑器（内存写入，游戏内即见即存）
```

## 核心数字

**旗舰**（ayylmao 328²，~3000 层，统一裁判）：**40.24 dB PSNR / MAE 0.74**（亚像素级残差） —— 接近视觉无损（误差 95%+ 像素 ≤8 灰阶，见 `visdiff` 热图）。

质量演进（同一图、同一裁判、每一行都有可复现命令，详见 [STUDY.md](STUDY.md)）：

| 阶段 | 关键改动 | PSNR |
|---|---|---|
| v0 贪心 | 积分图 O(高) 评分 + 误差加权采样 | 21.0 |
| +抛光/软层 | 遮挡感知精修 + 228 软椭圆混合语义 | 30.28* |
| +可微精修 (M1.8) | GPU 前向/反向 WGSL + Adam + τ 退火，300 层 +1.1dB | 30.83 |
| +B1 结构采样 | 对数半轴+Pareto 长宽比+Sobel 走向（细结构命中 ↑10⁶ 倍） | 36.11 |
| +polish 循环 | 遮挡感知离散记账 ×4 轮（E1 判别实验，CPU 秒级） | **40.24** |

\* 裁判 bbox 勘误后回溯校正（旧裁判口径原记 27.6），见 STUDY.md §"裁判勘误"。

**12 图真实基准**（FH 社区涂装语料，`bench/bench.sh`）：B1 结构采样使三难图 +4.4~+7.8dB，**5/12 图达到完美重构（MSE=0）**——社区涂装本来就是椭圆艺术，采样盲区一除即解。

**逐图 `--tune` 调优余量**（fvseg 规划参数坐标搜索，30 图社区语料）：平均再 **+4.88dB**（p95 +20.4dB；逐图数据 `bench/mve0/headroom.csv`）。

## 快速开始

```bash
cargo build --release            # CPU 全功能
# GPU 批量评估（需 Vulkan + SHADER_INT64，N/A/Intel 近年驱动均支持）
cargo build --release -p fv-gpu --bin fvfit-gpu

# 拟合（贪心 + B1 结构采样 + 可微精修，全默认开）
fv/target/release/fvfit-gpu myimg.png --layers 3000 --auto-soft --samples 1000000 --bg auto

# 语义分层（人像走 MODNet，通用主体走 U2Net，规划参数自动调优 --tune）
fvseg myimg.png --model u2net --gpu --tune --total 1000 --soft 300

# 质量评估（统一裁判）
cargo run --release -p fv-fit --example eval_design -- myimg.png out.json --bg auto
```

游戏内注入（Windows x64 + FH6）：见 [INJECT.md](INJECT.md)——三步安全流程（dry-run 抽样核对 → 确认写入 → 逐字段回滚），M2 起全链路 Mock 测试覆盖。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `fv/crates/fv-fit` | 拟合引擎：行前缀和评分、误差网格采样、爬山精修、遮挡抛光、软椭圆、结构导引采样（B1）、Marching-Squares 描边（M1.7） |
| `fv/crates/fv-gpu` | wgpu/Vulkan compute：i64 精确批评估内核、软椭圆两遍内核、**可微渲染器**（前向双扫描快照 + 反向梯度 + 窗口化大栈支持） |
| `fv/crates/fv-seg` | ONNX matting（ort load-dynamic；MODNet 人像 / U2Net 通用显著性）、分层规划（PlanParams 可调）、`fvseg` CLI（`--model/--gpu/--tune/--strokes`） |
| `fv/crates/fv-inject` | FH6 涂装组内存注入器（数据驱动布局 + 严格校验 + 回滚） |
| `fv/crates/fv-common` | geometrize JSON、形状类型 |
| `bench/` | 12 图基准脚本、MVE-0 规划调优协议、标签/特征数据 |
| [STUDY.md](STUDY.md) | 完整研发日志：RE 方法论、算法推导、每一步的坑与勘误 |
| [M18_DESIGN.md](M18_DESIGN.md) | 可微渲染器数学（唯一事实来源，含 FD 验证协议） |
| [L13_RESEARCH.md](L13_RESEARCH.md) | 语义微调研究（go/no-go 判据与数据） |

## 特色技术（详文档）

- **B1 结构导引采样**：从图像量出墨迹宽度分布与 Sobel 走向场，候选生成为对数均匀半轴 + Pareto 长宽比 + 走向 ±5°——对 2-10px 细结构的命中率提升约 6 个数量级（STUDY §13）。
- **GPU 可微渲染器 (M1.8)**：软椭圆覆盖 β=σ((1−q)τ)，前向双扫描写 Cbelow/Tabove 快照、反向一 workgroup 一层；有限差分验证 rel ≤ 5.5e-4；accept-guard 全程保护（劣化即回滚，零风险叠加到任何管线）。
- **量化快照设计教训**（STUDY §v2.2）：u8 量化在近收敛区被噪声劫持梯度——快照位宽必须按信号动态范围选，教科书级反例存档。
- **228 软椭圆混合语义**：α = α₂₂₈(r)·opacity 的解析 S 曲线（牛顿解），与游戏渲染逐位一致；与 B1 正交互补（条纹类再 +11dB）。

## 基准与复现

- `bench/bench.sh [layers] [samples] [solid|soft]`：12 类真实社区涂装语料上的统一基准（语料获取见 bench/README）。
- 每个质量主张附带的命令与裁判口径（`--bg` 必须与拟合一致）都在 STUDY.md 对应章节。

## 许可

代码 MIT（见 [LICENSE](LICENSE)）。模型（MODNet/U2Net ONNX）与 onnxruntime 遵循各自上游许可，不入库——获取方式见 `fv/crates/fv-seg/README`。

## 风险须知

内存注入本质是修改在线游戏进程，理论上违反 ToS；社区同类工具自 2021 起大规模使用未见系统性封号，**风险自担**。注入器三重防线：严格校验 → dry-run 抽样 → 逐字段回滚；实验性写入后不要保存设计。
