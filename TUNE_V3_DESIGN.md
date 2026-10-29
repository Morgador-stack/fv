# TUNE_V3 设计：候选连续化 + 不敏感旋钮早停（2026-10-28）

> 目标：在 **≤40 次评估/图** 的预算下，把 `fvseg --tune` 的搜索从"固定离散网格坐标
> 下降"升级为"局部连续化（三点抛物线找峰）+ 数据驱动的早停"，并论证为什么在
> 该预算下继续用坐标下降（CD）而不是交叉熵法（CEM）。
> 本文只做设计，不改源码。现行实现：
> - `fv/crates/fv-seg/src/pipeline.rs` 的 `tune_search`（L253-339，v2 六旋钮两轮坐标下降）
> - `fv/crates/fv-seg/src/plan.rs` 的 `PlanParams`（L137-167）、`allocate_budget_with`（L286）
> - `fv/crates/fv-seg/src/bin/fvseg.rs` 的 `--tune` 分支（L202-211，只调 `tune_search`）

## 0. 数据口径

| 数据源 | 口径 | 用途 |
|---|---|---|
| `bench/mve0/t_*.log`（30 份 = 30 图语料；其中 4 份 baseline=inf，另有 2 份在搜索中命中 inf） | 逐图逐轮解析 | 本文全部逐图统计（§B、§C、§E） |
| `bench/mve0/knob_stats.json` | 59 份日志宽口径（v1/v2 混合，跨二进制不可比，仅排序有效） | 旋钮胜率表（TUNE_V2_ANALYSIS §1） |
| `bench/mve0/headroom.csv` | 30 图 baseline→best | 收益绝对值（含零增益 inf 图，均值 4.87 dB） |

关键实测（30 图语料，v2 网格，每图 39 评估 = baseline 1 + 每轮 18 × 2）：

- round-0 胜出 120 次（4.0 次/图），round-1 胜出 35 次——二轮贡献 22.6%（本语料口径；
  宽口径 35%，结论一致：一轮方案丢 1/5~1/3 收益）。
- **路径总增益 4.578 dB/图（26 张有限基线图均值；其中 2 张终评 inf 使其路径增益略被
  低估），而最佳单跳（任一候选相对其评估时点 incumbent 的最大跳变）仅 1.899 dB
  ——组合比 2.41×**。收益的主体是"逐维采纳的累加"，不是任何一个孤立的候选点。
  这是 §C 的核心证据。
- 评估是**确定性的**：同 `PlanParams` + 同 seed 复评 PSNR 逐位一致（stripes_21 两轮
  重复候选的 PSNR 完全相等可证）→ 缓存去重（§B-E5）与抛物线拟合（§A）都成立。
- 每评估 ~1.5 s（59.4 s / 39）；耗时 6.9~117.5 s/图随尺寸复杂度浮动。

---

## §A 连续化设计（候选连续化）

### A.1 旋钮分类与依据

| 旋钮 | 语义 | v3 处理 | 依据 |
|---|---|---|---|
| band_radius | Chebyshev 膨胀半径，usize px | **SnapCont**：f64 潜变量搜索、整数取整评估（§E-R1） | 胜率 27% 第一杠杆；终值 38% 落在域边界 7 → 内部峰假设常不成立，须配"凸延伸" |
| edge_frac | f64 预算占比 | **Grid+顶点**：保留 v2 网格做探针 + 抛物线顶点 | 胜出值分散（0.10/0.35/0.45 都有），固定网格保覆盖 |
| subject_gain | f64 预算增益 | **Cont**：±δ 对称探针 + 顶点 | 胜率 11.8% 且增益小（+0.85 dB），不值得 3 个固定点 |
| subject_floor | f64 预算下限 | **Cont**：±δ 探针 + 顶点 | 同上；且存在惰性平台（stripes_21 两候选 PSNR 完全相等） |
| thresh_bias | 整数像素偏置 | **Grid 保留**（天然离散：只在跨越直方图占用 bin 时掩码才变，亚像素无意义） | 任务约定 + 语义 |
| smooth_rounds | u32 轮数 0-3 | **Grid 保留**，round-1 只测 incumbent 的邻值 | 4 个取值全枚举即完备 |

### A.2 三点抛物线协议（连续化的最小安全形态）

对连续维（Cont / SnapCont / Grid+顶点）， incumbent `x0` 的 PSNR `y0` 已知（它就是
当前 best），协议为：

1. **双侧探针**：评估 `x− = clamp(x0 − δ)` 与 `x+ = clamp(x0 + δ)`（2 次评估；
   被 clamp 成 `x0` 的一侧跳过）。
2. **顶点求解**：三点 `(x−,y−), (x0,y0), (x+,y+)` 拟合二次曲线。
   - 等距（`h = x+ − x0 = x0 − x−`）时：
     `denom = y− − 2·y0 + y+`，**仅当 `denom < 0`（内凹）**才有内部峰：
     `x* = x0 + h·(y− − y+) / (2·denom)`
   - 探针被域 clamp 导致不等距时用拉格朗日一般式：
     `x* = [y−(x0²−x+²) + y0(x+²−x−²) + y+(x−²−x0²)] / [2·(y−(x0−x+) + y0(x+−x−) + y+(x−−x0))]`，
     凹性用二阶差分符号复核（分子/分母比值的 `a < 0`）。
   - `denom ≥ 0`（单调/凸/平台）：无内部峰。采纳较优侧；**仅极致档补差轮**沿更优方向
     再延伸一步 `x0 ± 2δ`（band_radius 常态如此，见 §E-R1）。
   - 任一点 PSNR 为 inf/NaN：跳过顶点（不拟合）。
3. **顶点验证**（防模型误设，永不倒退的兜底）：
   - SnapCont：`x̂ = round(x*)`，clamp 进 `[0, 12]`；若 `x̂ ∈ {x−, x0, x+}` 已评估过 →
     跳过验证（顶点只贡献方向信息，供补差轮延伸用）；否则评估 `x̂`（1 次）。
   - Cont：若 `|x* − x0| < max(ε_abs, δ/10)`（ε_abs：edge/gain/floor 取 0.005）→ 顶点
     与中心无区分度，跳过；否则评估 `x*`（1 次）。
   - **采纳判据 = 实测 `f(x̂) > best.psnr` 才采纳**。顶点预测值只用来决定"要不要花
     这 1 次评估"，绝不直接采纳未实测的点。
4. **平台/惰性判定**：若 `y− = y0 = y+`（确定性评估下严格相等）→ 该维惰性
   （如 subject_floor 在 `gain·(0.5+α)` 恒大于 floor 时根本不起作用），后续轮次跳过。

成本：每维每轮 = 2（探针）+ 0..1（顶点验证）。相比 v2 的 3 个固定候选，评估数相同或
更少，但第三个点的位置由前三点决定而非网格预表——这是"连续化"的全部增量：
**同样的钱，买自适应峰值定位，且夹带了免费的方向信息（denom ≥ 0 时知道往哪边延伸）**。

局部二次模型的合理性：评估确定、无随机噪声（§0），PSNR 对单维在固定其余维时是
分段光滑的（预算经 `round(··T)` 量化，T=1500 时量化步 0.00067，远细于 δ≥0.05），
三点抛物线是"每维 2 次新评估"预算下唯一可估的二阶模型。五点/样条在 40 预算下
是浪费（每维 +2 评估 × 4 维 = +8，占预算 20%）。

### A.3 各维参数表（round-0 / round-1）

| 旋钮 | kind | round-0 探针（incumbent=默认值） | δ0 | round-1（δ 减半） | 域 clamp |
|---|---|---|---|---|---|
| band_radius | SnapCont | `3±2 → {1, 5}` + 顶点 | 2 px | `r±1` + 顶点 | [0, 12] |
| thresh_bias | Grid | `{-32, -16, +16, +32}` | — | 同网格（极致档改 `w±8` 细分） | 掩码 [1,255] 内 |
| smooth_rounds | Grid | `{0, 2, 3}`（1=incumbent 天然覆盖） | — | incumbent 的相邻值 ≤2 个 | [0, 3] |
| edge_frac | Grid+顶点 | `{0.10, 0.35, 0.45}`（v2 网格保覆盖）+ 拉格朗日顶点 | — | incumbent ±0.10 + 顶点 | [0.02, 0.60] |
| subject_gain | Cont | `0.45±0.15 → {0.30, 0.60}` + 顶点 | 0.15 | ±0.075 + 顶点 | [0.20, 0.90] |
| subject_floor | Cont | `0.30±0.10 → {0.20, 0.40}` + 顶点 | 0.10 | ±0.05 + 顶点 | [0.10, 0.50] |

设计说明：
- band_radius/thresh_bias/smooth_rounds 的 round-0 探针与 v2 网格逐点重合（{1,5} 对
  默认 3 对称 = v2 的 {1,5,7} 的前两个；thresh/smooth 原样）——**v2 已验证有效的候选
  一点不丢**，v3 只是把"第三点"从固定网格值换成自适应顶点。
- edge_frac 保留 v2 的不对称三点（0.10/0.35/0.45）而非 ±δ：胜出值跨 0.10-0.45 分散，
  保覆盖优先，顶点用一般式。δ_e1 = 0.10 的 round-1 探针从新 incumbent 出发可到达
  0.35/0.45 区域的邻域。
- subject_gain δ=0.15 的探针 {0.30, 0.60} 覆盖了 v2 网格 {0.35, 0.55, 0.65} 的主区间；
  0.65 类端点胜出（lower_letters_5_7、upper_letters_5_21）依赖顶点验证或补差轮到达，
  这是 §E-R5 明示的覆盖风险，用验收标准兜底（§D.4）。

### A.4 维顺序（坐标下降的顺序即先验）

`band_radius → thresh_bias → smooth_rounds → edge_frac → subject_gain → subject_floor`

- band_radius 提前：胜率/增益积最高（0.27×3.20 = 0.86，次名 smooth 0.158×5.39 的
  增益是极端图拉高的，可靠性低），且它**只影响边界带、与阈值/掩码完全独立**
  （plan.rs L219-222 从原始 matte 计算），放第一位零依赖风险。
- thresh/smooth 保持 v2 相对顺序（掩码侧），edge/gain/floor 殿后（预算侧依赖 alpha，
  alpha 由前两者决定）。
- 后维在更优 incumbent 邻域探查 = 隐式利用已采纳增益，这是 §C 中 CD"组合收益"的
  机制本体。

---

## §B 早停规则（全部可用 30 图语料逐图验证）

| 规则 | 内容 | 触发率（实测） | 损失（实测） | 省 |
|---|---|---|---|---|
| **E3 inf 即出** | baseline PSNR = inf（完美重构）→ 立即返回默认参数，不做任何搜索 | 4/30 图（baseline=inf） | 0（inf 不可被有限值超过，v2 白跑 38 次——`r.psnr > inf` 恒假但仍逐个评估；另 2 张在搜索中命中 inf 属正常搜索收益，E3 不干预） | 38 评估/图 × 4 图（这 4 图 v2 仅耗 7-10 s，省的主要是无谓计算而非墙钟大头） |
| **E1 低敏维门控** | subject_gain 与 subject_floor **第一轮都无胜出** → 第二轮两者全跳 | 10/30 图（33%） | **0**：本语料 round-1 的 9 次 gain/floor 胜出（8 gain + 1 floor）全部落在第一轮**有**胜出的图上；10 张触发图中 round-1 gain/floor 胜出 = 0 | 触发图省 5~6 评估/图 |
| **E5 评估缓存去重** | `HashSet<[i64;6]>`（六参位模式键）记录已评估 PlanParams，命中即跳过 | 收敛图全轮命中（stripes_21 round-1 的 18 评估中 15 次与 round-0 逐位同参） | 0（确定性评估，同参同 PSNR，必不优于已记录的 best） | 7 张零胜出图各省 ~15 评估 |
| **E2 预算硬顶** | 总评估 ≤ 40；补差轮（极致档）只花剩余额度 | — | — | 保证上界 |
| E4 全零轮跳过（**否决**） | round-0 零胜出 → 跳过 round-1 | 本语料 0 张触发（最少 1 胜出） | — | 无数据支持，不实现 |

E1 的机制解释（为什么零损失成立）：gain/floor 只有在"当前 budget 分配恰好卡在
floor/gain 约束上"时才起作用；round-0 无胜出 ≈ 该图预算不在约束边界附近，而 round-1
的其他维改动（band/edge/thresh）改变 alpha 的幅度不足以把它推回边界。反之第一轮
有胜出的图（16/30）说明该图预算敏感，第二轮保留全探查——条件正确分层。

早停省出的评估不退回，而是交给 E2 补差轮（极致档）按 `p_i·Ē_i` 排序再投资
（band_radius 延伸 → thresh ±8 细分 → edge 二次顶点），标准档直接落袋为时间节省。

---

## §C CEM vs 坐标下降：≤40 评估预算下的数学论证

**命题**：population 12 / elite 4 / iter 3（36 评估 + 1 次 μ_T 解码评估 = 37）的 CEM，
在本问题上的期望终增益显著低于同预算的坐标下降 v3。

### C.1 实测锚点（30 图语料）

- **F1 组合比**：路径总增益 / 最佳单跳 = 4.578 / 1.899 = **2.41×**。
- **F2**：精选网格候选胜率 155/910 = 17%；每图采纳 5.17 次，单次采纳均值 0.94 dB。
- **F3**：round-1（条件化重探）胜率仍有 35/540 = 6.5%——"在已改进点上重探邻域"
  持续产出。
- **F4**：band_radius 终值 38% 落在域边界（7），目标函数对单维呈边界极值/平台结构。

### C.2 CD 的期望模型：收益 = 路径和（Σ）

坐标下降每采纳一次，incumbent 前移，后续探针在更好的基线上进行：

```
E[G_CD] = Σ_k E[g_k] ≈ m · μ_g = 5.17 × 0.94 ≈ 4.87 dB   ← 与实测 4.58（26 图口径）吻合
```

v3 的连续化只在此之上做非负修正：顶点验证"实测超三点最优才采纳"保证每维收益
≥ v2 同维（v2 的第三固定候选被自适应顶点替换，探索域单调不减）；E1/E5/E3 省出的
评估再投资于高杠杆维。故 `E[G_v3] ≥ E[G_v2] ≈ 4.6 dB`（38 vs 39 评估，差 1 次可忽略）。

### C.3 CEM 的期望模型：收益 = 样本最大值（max）+ 漂移受限

CEM 返回的是 37 次评估中的最优样本（μ_T 解码点是第 37 个候选，未经验证）。
**它没有任何路径组合机制**——每次样本在 6 维上同时扰动，单样本增益被 5 个负期望维
拖累。两条独立推导给出同一个量级：

**(a) 极值上界**。`E[G_CEM] ≤ E[max of 37 个候选]`。实测 18 个精选候选的
`E[max] = 1.90 dB`；极值放缩 `E[max_n] ∝ √(2 ln n)` 给
`E[max of 37] ≈ 1.90 × √(ln37/ln18) ≈ 1.90 × 1.11 ≈ 2.1 dB`。
CEM 前 1-2 轮的样本是"μ 周围高斯采样"而非精选网格，命中质量只会更低——这是上界，
且是宽松的上界。

**(b) 维级漂移-收缩分析**。标准 CEM（β=1）取 elite 经验矩：
- 选择强度：top-4-of-12 ≈ 上三分之一截断，`E[elite均值 − μ] = λ·σ = 1.09σ`，
  `Var[elite] = (1 + z·λ − λ²)σ² = 0.28σ²` → **每轮 σ 收缩至 0.53σ**，三轮后 0.28σ₀。
- μ 每轮最多走 1.1σ_t（α=1），三轮累计走 `1.1·σ₀(1+0.53+0.28) ≈ 2.0σ₀`——且这要求
  elite 在**同一维**上连续三轮一致偏向。K=4 的 elite 要同时对 6 个维做归因，
  得分只揭示 6 维的联合和：归因信噪比 ∝ 1/√6，只有信号最强的 1-2 个维
  （edge_frac / band_radius，F4）能维持一致方向，其余维的 μ 漂移是选择噪声下的
  有偏随机游走。
- 结果：三轮结束时 CEM ≈ "默认参数 + 1~2 个强维的部分位移，σ 已塌缩"。
  按强维胜出增益标定（edge_frac 类 +1.3 dB、band 类 +3.2 dB 的 1~2 个），
  `E[G_CEM] ≈ 1.5~2.5 dB`。与 (a) 的 2.1 dB 上界互相印证。
- 离散维额外罚：thresh_bias/smooth_rounds/band_radius 三维量化，elite 均值落在格点
  之间，解码取整又是一次未验证采样；阶跃型目标（掩码只在跨 bin 时变化）使高斯
  采样大量落在平台上，样本有效率进一步低于 17% 的精选网格口径。

### C.4 结论

| 方案 | 评估数 | 期望增益（本库标定） | 机制 |
|---|---|---|---|
| CD v2（现行） | 39 | 4.6 dB（实测） | 路径 Σ |
| **CD v3（本设计）** | ≤39 最坏 / ~33 典型 | **≥ 4.6 dB** | 路径 Σ + 自适应顶点 + 早停再投资 |
| CD v3 快速档 | ≤12 | ~2.8 dB（TUNE_V2_ANALYSIS §3 的 60% 口径） | 同上、砍维度 |
| CEM 12/4/3 | 37 | ~2.1 dB（上界推算） | 样本 max，无组合 |

比值 ≈ 2.2×，与 F1 组合比 2.41× 同源：**CD 的本质优势是把 m≈5 次采纳的增益相加，
而 CEM 的本质是对样本取 max；只要路径增益准可加（本库实测强可加），Σ 对 max 的
优势就是 2 倍量级，且这个差距不是调参能弥合的**——CEM 要追平，只能靠更长的迭代
让 μ 逐维到位（预算 ≥100 评估量级）。

CEM 反超的三个前提在此全部不成立：(1) 预算 ≥ 40（population 法需要样本密度覆盖
6 维）；(2) 评估可大规模并行摊薄墙钟（本管线每次评估已占满 GPU，墙钟 ∝ 评估数）；
(3) 全连续维 + 强联合交互（3/6 维离散，交互已由 round-1 条件化重探收割，F3）。
**决定：tune v3 采用坐标下降；CEM 不实现默认路径**（若未来做，挂 `--tune-cem`
实验开关、只在 ≥120 评估的离线研究档启用）。

---

## §D 实现规格

### D.1 改动面

| 文件 | 改动 |
|---|---|
| `pipeline.rs` | 重写 `tune_search` 函数体（签名不变）；新增 `DimSpec`/`TuneCtx` 结构与 `vertex3`/`probe_dim` 辅助。`run_pipeline`/`run_with_params` 不动 |
| `fvseg.rs` | `--tune` 分支不动；新增可选 `--tune-algo <v2\|v3>`（默认 v3）用于 A/B 回退与验收 |
| `plan.rs` | 不动 |
| `tools/tune_stats.py` | 不动（日志格式兼容，见 D.2 约束） |

### D.2 伪码（tune v3 主干，Rust 形状）

```rust
// ---- 类型 ----
enum Kind { Grid, Cont, SnapCont }          // SnapCont = band_radius（f64 搜、usize 评）
struct DimSpec {
    name: &'static str, kind: Kind,
    get: fn(&PlanParams) -> f64,
    set: fn(&mut PlanParams, f64),
    d0: f64,                                 // round-0 步长（round-1 自动减半）
    probes0: &'static [f64],                 // Grid / Grid+顶点 维的固定探针
    lo: f64, hi: f64,
}

const DIMS: [DimSpec; 6] = [                 // 顺序 = §A.4
    band_radius  SnapCont d0=2.0 probes0=&[] (0.0, 12.0),
    thresh_bias  Grid    probes0=&[-32.,-16.,16.,32.],
    smooth_rounds Grid   probes0=&[0., 2., 3.],
    edge_frac    Grid    probes0=&[0.10, 0.35, 0.45] (0.02, 0.60),   // Grid+顶点：Grid 种类但 probe 后加顶点步
    subject_gain Cont    d0=0.15 (0.20, 0.90),
    subject_floor Cont   d0=0.10 (0.10, 0.50),
];

struct TuneCtx {
    cache: HashSet<[i64; 6]>,                // E5：六参位模式键
    won_r0: [bool; 6],                       // E1 记账（gain=4, floor=5 号位）
    inert: [bool; 6],                        // §A.2-4 平台判定
    evals: usize,
}

fn key(p: &PlanParams) -> [i64; 6] {
    [p.thresh_bias as i64, p.smooth_rounds as i64, p.band_radius as i64,
     p.edge_frac.to_bits() as i64, p.subject_gain.to_bits() as i64,
     p.subject_floor.to_bits() as i64]
}

fn eval_cached(ctx, p) -> Option<PipelineResult> {   // None = 缓存命中跳过
    if !ctx.cache.insert(key(p)) { return None; }
    ctx.evals += 1;
    Some(eval(p, gpu)?)
}

// ---- 三点抛物线顶点（等距特例；不等距走拉格朗日一般式）----
fn vertex3(xm, ym, x0, y0, xp, yp) -> Option<f64> {
    let denom = ym - 2.0 * y0 + yp;
    if !(denom < -1e-9) { return None; }                    // 单调/凸/平台 → 无内部峰
    if !(ym.is_finite() && y0.is_finite() && yp.is_finite()) { return None; }
    Some(x0 + (xp - x0) * (ym - yp) / (2.0 * denom))        // h = xp - x0
}

// ---- 单维连续化探查（Cont / SnapCont / Grid+顶点共用）----
fn probe_dim(ctx, dim, round, best_p: &mut PlanParams, best: &mut PipelineResult) {
    if ctx.inert[dim.i] { return; }
    let x0 = (dim.get)(best_p);
    let y0 = best.psnr;
    let mut tried = vec![x0];

    // 1. 探针集：Grid → probes0；Cont/SnapCont → x0 ± δ（clamp，跳过塌缩侧）
    let probes = match dim.kind {
        Grid => dim.probes0.to_vec(),
        _ => [x0 - delta, x0 + delta].iter().map(|&x| x.clamp(dim.lo, dim.hi))
                                       .filter(|&x| x != x0).collect(),
    };
    let mut pts = vec![(x0, y0)];
    for x in probes {
        let mut p = *best_p; (dim.set)(&mut p, x);
        let Some(r) = eval_cached(ctx, &p) else { continue };
        log_probe(dim.name, x, r);                            // v2 日志格式
        if r.psnr > best.psnr { *best = r; *best_p = p; ctx.won_r0[dim.i] |= round == 0; }
        pts.push((x, r.psnr));
    }

    // 2. 顶点（Grid 维 edge_frac 也做；纯 Grid 的 thresh/smooth 不做）
    if dim.kind != Grid && pts.len() == 3 {
        let Some(xs_raw) = vertex3(...) else {
            // denom ≥ 0：采纳较优侧（上面已做）；极致档补差轮负责延伸，此处只记方向
            ctx.dir[dim.i] = if y_左侧 > y_右侧 { -1 } else { 1 };
            return;
        };
        let xs = match dim.kind {
            SnapCont => (xs_raw.round() as i64 as f64).clamp(dim.lo, dim.hi),  // §E-R1 取整
            _ => xs_raw,
        };
        let eps = if dim.kind == SnapCont { 0.5 } else { (delta * 0.1).max(0.005) };
        if !tried.iter().any(|&t| (t - xs).abs() < 1e-9) && (xs - x0).abs() >= eps {
            let mut p = *best_p; (dim.set)(&mut p, xs);
            if let Some(r) = eval_cached(ctx, &p) {
                log_probe_vertex(dim.name, xs, r);
                if r.psnr > best.psnr { *best = r; *best_p = p; ctx.won_r0[dim.i] |= round == 0; }
            }
        }
    }

    // 3. 平台判定（确定性评估下严格相等才有意义）
    if 三点 PSNR 严格全等 { ctx.inert[dim.i] = true; }
}

// ---- 主流程 ----
fn tune_search(...) -> Result<(PipelineResult, PlanParams, f64), String> {
    let t0 = Instant::now();
    let mut ctx = TuneCtx::default();
    let mut best_p = PlanParams::default();
    let mut best = eval_cached(&mut ctx, &best_p)?;            // 评估 #1
    eprintln!("[tune] baseline ...");
    if best.psnr.is_infinite() {                               // E3：inf 即出
        eprintln!("[tune] baseline inf → 完美重构，跳过搜索");
        return Ok((best, best_p, t0.elapsed().as_secs_f64()));
    }

    // round 0：全六维，δ = d0
    for dim in DIMS { probe_dim(&mut ctx, dim, 0, &mut best_p, &mut best)?; }

    // round 1：δ 减半；E1 门控 gain/floor；smooth 只测邻值
    let e1_skip = !ctx.won_r0[4] && !ctx.won_r0[5];            // gain & floor 第一轮均无胜出
    for dim in DIMS {
        if e1_skip && (dim 是 gain || dim 是 floor) { continue; }
        if dim 是 smooth_rounds { 改用 incumbent 邻值探针（≤2 个）； }
        probe_dim(&mut ctx, dim, 1, &mut best_p, &mut best)?;   // 内部 delta = d0 / 2
    }

    // 极致档补差轮（--tune-tier extreme 时）：花满 40 额度
    while ctx.evals < 40 {
        优先级：band_radius 凸侧延伸（x0 ± 2d0）→ thresh_bias 细分 {w±8}
              → edge_frac 二次顶点（δ/4）→ gain/floor 第二顶点（E1 触发则不参与）
        若最高优先级维无可评新点 → break;
    }

    eprintln!("[tune] done in {:.1}s ... params={best_p:?}");
    Ok((best, best_p, t0.elapsed().as_secs_f64()))
}
```

日志兼容约束：所有评估行保持 v2 正则可解析格式
`[tune] {name:<14}={value:<5.2} psnr={:.3} shapes={} [← best]`；顶点行在该格式尾部追加
` (vertex)` 标记。`tools/tune_stats.py` 的正则对尾随文本不敏感（L12-17），knob_stats.json
可继续重建。

### D.3 评估预算表（每图；~1.5 s/评估为 30 图语料均值）

| 档位 | v2 评估数 | v2 时间 | v3 评估数（最坏 / 典型） | v3 时间（典型） | 说明 |
|---|---|---|---|---|---|
| **快速** | 11（1+4+3+3） | ~17 s | **≤12 / ~11** | ~17 s | 只跑 band_radius+edge_frac+thresh_bias；评估数与 v2 持平，但 band/edge 的第三点换成自适应顶点 |
| **标准（默认）** | 39（1+18+18） | ~59 s | **≤39 / ~33** | ~50 s | 最坏 = 1 + R0≤20 + R1≤18；E1（33% 图省 5-6）+ E5（收敛图省 ~15）把典型拉到 ~33 |
| **极致** | ~45-60 | ~90 s | **≤40（E2 硬顶）** | ≤60 s | 标准 + 补差轮花满剩余额度（band 延伸 → thresh ±8 → edge 二次顶点） |
| **inf 基线图**（4 张） | 39（全浪费） | 7-10 s | **1** | <2 s | E3 即出；v2 缺陷修复（另有 2 张搜索中命中 inf，v3 照常早停于该候选） |

R0/R1 明细（标准档）：

| 轮 | 维 | 评估 |
|---|---|---|
| R0 | band 2(+1) · thresh 4 · smooth 3 · edge 3(+1) · gain 2(+1) · floor 2(+1) | 16 探针 + ≤4 顶点 = **≤20** |
| R1（全跑） | thresh 4 · band 2(+1) · smooth ≤2 · edge 2(+1) · gain 2(+1) · floor 2(+1) | 14 探针 + ≤4 顶点 = **≤18** |
| R1（E1 触发） | 前 4 维 | 10 探针 + ≤2 顶点 = **≤12** |
| 合计 | | 最坏 39 ≤ 40（E2 兜底截断），典型 ~33 |

### D.4 验证计划与验收标准

```
# A/B：同一 30 图语料，v2 存档日志已在 bench/mve0/t_*.log
fvseg <img>.png --tune --gpu                    # v3（默认）
fvseg <img>.png --tune --tune-algo v2 --gpu     # 回退对照
python tools/tune_stats.py                      # 重建 knob_stats.json，校验日志兼容
```

验收（对比 26 非.inf 图的存档 v2）：
1. **收益不回退**：路径总增益均值 ≥ 4.58 × 0.95 = 4.35 dB（最坏评估数与 v2 持平、
   分配更优，预期持平或正；0.95 容忍个别图网格覆盖差动，见 §E-R5）。
2. **预算达线**：非 inf 图评估数均值 ≤ 34、最坏 ≤ 39；inf 图 = 1。
3. **连续化有效性**：顶点验证采纳率 > 5%（否则砍掉 SnapCont/Cont 的顶点评估，
   每维每轮再省 1 次——用 tune_stats 按 `(vertex)` 标记统计）。
4. **E1 零损失复验**：新语料上"round-1 gain/floor 胜出 ⊆ round-0 gain/floor 有胜出"
   仍成立；若不成立（>1 图违反），E1 改为保守版（仅快速档剔除，标准档保留）。

---

## §E 风险

**R1 连续化在离散语义旋钮上的取整（band_radius，任务点名的头号风险）**
- 策略：f64 潜变量 `ρ` 做三点搜索，**评估永远在 `round(ρ)` 上**；顶点 `x̂ = round(x*)`
  后 clamp [0,12]，与已评估点重合则不再花验证评估。
- 论据：膨胀半径只以整数进入 `dilate_radius`（plan.rs L364），亚像素位移最多改变
  1 px 的带缘行，0.4 px 级的顶点精度无语义；rounding 误差 |frac| ≤ 0.5 px 造成的
  峰值损失 ≤ |a|·0.25（等距网格 δ=2 的最佳格点损失期望 |a|·δ²/12 = 0.33|a|），
  净收益为正但小——**band_radius 上抛物线的主要价值是方向判定（denom ≥ 0 时知道
  往 1 还是往 7 延伸），不是亚像素定位**。
- 实测预警：终值 38% 落在域边界 7（F4）→ 内凹假设经常不成立，故配凸侧延伸
  （补差轮沿 `denom ≥ 0` 判定的方向探 `x0 ± 2δ`），并设 D.4-3 的采纳率检查点：
  若顶点对该维无益，可单独关掉 band_radius 的顶点验证，回到纯 ±δ 邻域搜索。

**R2 惰性维与平台**：subject_floor 在预算不贴约束时完全失效（stripes_21 两候选
PSNR 严格相等）。确定性评估使"严格相等"是可信信号而非噪声 → 平台判定安全；
误判后果上限 = 少探一个本就无效的维，无收益损失。

**R3 模型误设（PSNR 非二次）**：顶点只在实测通过后才采纳，任何情况下不劣于
"三点取最优"（= v2 同维减一个候选）。最坏退化 = 浪费 1 次顶点评估。

**R4 inf/NaN**：inf 图由 E3 在 baseline 处截断；顶点公式对含 inf/NaN 的三点返回
None（D.2 已含守卫）；thresh_bias 触发全背景/全主体退化图由既有管线兜底
（budget 恒和为 T 的不变量）。

**R5 对 v2 固定网格的覆盖回归**：subject_gain 的 {0.35,0.55,0.65} 被 {0.30,0.60}+顶点
替代，0.65 类端点胜出（2 图）需顶点或补差轮到达；edge_frac 保留 v2 三点无此险。
缓解：验收标准 D.4-1 的 0.95 系数专门为该险设；若破线，把 subject_gain round-0
改为 v2 三点网格 + 顶点（+1 评估，仍在 40 内）。

**R6 顺序与交互效应变化**：band_radius 提前改变了后续维的 incumbent 起点，v2 的
最优轨迹不能逐点复现。round-1 条件化重探（本语料 6.5% 胜率，F3）是主要缓释；
E1 保证低敏维的腾挪不伤及交互收益（其 9 次 round-1 胜出全部保留）。

**R7 工程风险**：日志格式回归会断 tune_stats.py/ knob_stats.json 重建（D.2 约束 +
D.4 验证第 3 步覆盖）；无 `--tune-algo v2` 回退不开关合入；评估缓存键用位模式，
禁止对 f64 做容差合并（两个"几乎相等"的参数在 round(··T) 下可能合法地落在不同
预算格点，容差合并会漏评）。
