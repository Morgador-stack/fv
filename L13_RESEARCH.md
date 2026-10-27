# L1.3 vinyl 感知微调 — 深度研究文档（可执行版）

日期：2026-10-26 ｜ 状态：研究完成，零代码改动 ｜ 依据：STUDY.md §10 L1.3、M18_DESIGN.md、
`fv/crates/fv-gpu/src/diff/`（fwd/bwd/refine）、`fv/crates/fv-seg/src/{lib,plan}.rs`、
`fv/crates/fv-seg/src/bin/fvseg.rs`、`fv/crates/fv-fit/src/engine.rs`、notes-m1-scale.md

---

## 0. 结论速览（决策表）

| 决策点 | 结论 | 关键理由 |
|---|---|---|
| 主线路线 | **A.3 离线蒸馏**（先 C0 参数回归器，条件触发 C1 matte U-Net） | 唯一不需要"对贪心求梯度"也不需要百 GPU 时的路线；标签生成用的全是我们已有的、确定性已验证的管线 |
| 第零步（先于一切训练） | **MVE-0 headroom 探针**：纯 PlanParams/matte 变体扫描 | 若 matte/规划空间拧不出 ≥0.5dB，整个 L1.3 立即止损——一天内出结论，成本 ~4h 机时 |
| 训练框架 | **PyTorch @ 5090D 训 + ONNX 导出 + N1X `ort` 推理**（复用 L1.1 全链路） | candle 无 Vulkan 且训练弱；tch 无 WoA libtorch；burn-wgpu 是唯一 N1X 原生训练路径但成熟度风险高，**推迟**到 A 路线存活后再议 |
| 算力分工 | N1X = 标签工厂 + 评估机（Vulkan 管线已 40× 验证）；5090D = 训练 + 大预算终评（bundle-5090d 现成） | 两台各干最擅长的事，零新环境搭建 |
| 路线一（绕开拟合） | A.1b 软规划+可微小容量拟合替代 = **研究性副线**（PyTorch 镜像 M1.8 数学，非桥接）；A.1a 直接形状解码 = 独立 1 天 MVE，存活则另立 L1.4 | A.1a 若成立则 matte 不再需要——但只能覆盖小 N 基底，不能吃满 3000 预算 |
| 路线二（无梯度） | A.2a CEM on PlanParams = 标签/示波器工具（兼 L1.3-lite 交付物）；A.2b REINFORCE/ES 直接训网络 = **定量否决**（10^5–10^7 次管线评估） | 拟合单跑 5s–2.5h，网络权重维 10^6 级，乘起来不可行 |
| 赢的标准 | PSNR@300 预算受限档，对 base-matte 分层管线均值 **≥ +0.3dB** 且无域均值回退 | L1.2b 已证明主体占满图时分层≈盲拟合（Δ−0.03dB），L1.3 的生存空间在预算受限/结构化背景/预算路由 |

**第一步行动（MVE-0）**：给 fvseg 加 `--plan-json`（把 `plan_layers_with` 暴露进 CLI，约 30 行 + 测试，1–2h），然后 30 图 × ~20 候选 × 2 seed 扫描，N1X GPU 约 3–4h。判据见 §D.1。

> **主会话勘误（2026-10-26 晚）**：本研究进行期间，`--tune`（PlanParams 坐标搜索 CLI）已由主会话实现并验证（demo 图 13 评估 43s，band_radius 3→5 +0.2dB）——即本节"第一步"已落地为 `fvseg --tune`。MVE-0 可直接在 `--tune` 骨架上扩展（多候选值/多图扫描/双 seed），无需再写 `--plan-json`。

---

## A. 三路线深挖

### A.0 先把"黑箱"钉死：贪心到底哪里不可微（代码事实）

`fv-fit/src/engine.rs::fit_impl` 主循环每层做五件事（engine.rs:500–639）：

1. 误差网格 CDF 加权采样候选（`grid.sample_center`，随机、离散）；
2. i64 积分图精确评分（`evaluate`，阶跃函数——accept 边界处 delta 从 +1 跳到 −1）；
3. **argmax 接纳**（`better` 归约，max 不可导）；
4. 接纳后 `composite_ellipse` 改画布 + `sums.refresh_canvas` + `grid.rebuild`（画布是离散拓扑的分段常数函数，层数 N 本身是随机变量）；
5. 三振出局早停（`empty_streak`，终止时刻离散）。

贪心拟合因此是 **图像 → 形状栈的逐段常数映射**：输入 matte 的微小变化只在跨越某个接纳阈值时才改变输出栈，其余时候梯度恒为 0，跨越处梯度未定义。**这不是"梯度难算"而是"梯度几乎处处为零"**——这一事实直接否决了"对贪心做 straight-through"的天真方案，并把设计空间压缩为：绕开它（A.1）、把它当黑箱打分器（A.2）、把它蒸馏成监督信号（A.3）。

另一个常被忽略的代码事实：matte 的消费方式比想象中"钝"。`plan.rs::plan_layers_with` 只从 matte 提取三样东西——Otsu 阈值化出的二值 subject 掩码（+多数投票平滑）、Sobel 边界带（+膨胀）、以及主体面积占比 alpha（只进预算公式）。**matte 的连续灰度值域在 Otsu 之后只剩一个 bit**。所以"对椭圆重构友好的 matte"的可调自由度实际上 = {阈值位置, 平滑轮数, 带宽度, 预算三参数} ∪ 掩码形状本身。这正是 `PlanParams` 六个旋钮（`thresh_bias, smooth_rounds, band_radius, edge_frac, subject_gain, subject_floor`，plan.rs:141–167）的覆盖范围，也是 MVE-0 要量测的全部搜索空间。

> 勘误备案：plan.rs:139 注释提到"见 fvseg `--tune`"，但 fvseg.rs CLI 目前**并未实现** --tune，`plan_layers_with` 也没有任何调用方——PlanParams 旋钮已备好接口但从未被搜索过。MVE-0 的 30 行接线是它的第一次落地。

### A.1 路线一：绕开拟合（端到端可微）

#### A.1a 直接形状解码器（"那分割就不需要了"的正面回答）

**架构**：图像 → 编码器（小 U-Net 编码半部，~2M 参数）→ 特征 F；N 个可学习 slot query 与 F 交叉注意力 → 每 slot 八头回归 `(cx, cy, a, b, θ, r, g, b)`（sigmoid/softplus 压域）。损失 = M1.8 前向（β=σ((1−q)τ) 软合成）对图像的 MSE，梯度经 `backward` 链式回到网络权重。

**必须修正的一版设计**：一次性回归 N 个无序 slot 会撞上集合预测的多模态坍缩（同一图像有无穷多等价栈，回归均值给出糊椭圆）。改为**画家式多轮**（Learning to Paint, Zheng et al. 2019 的教训）：

```
canvas_0 = bg
for round r in 1..R:                     # R=4，每轮 N/R 层
    E_r = 残差图 (img − canvas_{r-1})²
    h_r = concat(F, conv(E_r))           # 误差条件化——解码器看"还差什么"
    stack_r = slot_decode(h_r)           # N/R 个椭圆的 8 参数
    canvas_r = render_soft(canvas_{r-1}, stack_r, τ)
loss = Σ_r γ^{R-r} · ||canvas_R − img||²   # 终端为主，中间轮打折
```

**梯度可行性有文献锚点**：Stylized Neural Painting（Zou et al. 2021）与 Learning to Paint 都用可微（软）光栅器训练笔画预测网络成功；我们的 M1.8 已把同类数学（sigmoid 覆盖 + over 合成）的梯度做到 FD rel≤5.5e-4。已知坑与对策（全部来自 M1.8 实测）：
- 高 τ 时 β 过渡带薄、梯度稀疏（M18 τ0=4 伤害实心栈、τ0=32 才单调收敛的教训反过来用）：**网络训练用固定中 τ（16–64），评测永远硬光栅**；
- θ 与像素单位不可比：loss 内对 θ 学习率 ×0.05（refine.rs:112–115 已验证）；
- 快照内存 O(N·H·W)：N≤64、256² 无压力。

**训练数据是免费的（vinyl-native 合成）**：用我们自己的引擎采样随机形状栈（bg 层 + subject 层两段式）→ 渲染出图，ground truth = 栈本身。先监督热启动（直接回归参数），再切渲染损失精调。无限量、完美标注、零域偏移。

**诚实边界**：MVE 规模只能到 N≈16–64。3000 层全栈解码超出当前文献能力（Paint Transformer 类方法也只做百笔级）。所以 A.1a 若成立，定位是**替换 B 链的"精修基础栈"**（现在 M1.8 精修的 300 实心层来自贪心，可换成网络直出 + M1.8 精修），matte 在该子问题里确实消亡——但管线其余部分仍需 matte。**它不是 L1.3 的替代品，是潜在的 L1.4。**

#### A.1b 软规划 + 小容量可微拟合替代（保 matte 的版本）

回答"输出 matte 后由可微的'软规划'直接产生层？"——可以，但不是复刻贪心，而是把"这层目标好不好拟合"变成损失：

```
m = U-Net(img)                              # 连续 matte 场 [0,1]
# 可微三掩码（Otsu 换软阈值）：
s_sub = σ((m − t)/ε);  s_bg = 1 − s_sub;  band = 可微 Sobel+软膨胀
for layer l ∈ {bg, subject}:                # 两层足够 MVE
    T_l = s_l ⊙ img + (1 − s_l) ⊙ canvas_l   # fvseg "不关心区填画布"技巧的软版
    # 内环：小容量可微拟合替代贪心（K 个椭圆，联合 Adam，T_inner 步）
    stack_l = init(K)                        # 如误差图 top-K 峰处圆形
    for t in 1..T_inner:                     # K=24, T_inner=40
        g = ∇_stack ||render_soft(stack_l) − T_l||²   # M1.8 数学，精确梯度
        stack_l ← Adam(stack_l, g)
    L += ||render_soft(stack_l) − T_l||²
∂L/∂(U-Net 权重) 反传穿过软阈值、内环全部 Adam 步（展开式反传）
```

数学全部现成：内环就是 `diff_refine` 的逐层复用；`fvseg` 的 patched_target 技巧（fvseg.rs:484–497）保证"不关心区域零残差"语义在软版下仍然成立。计算量：K=24 层 256² 前向 ≈ 24×256²×30flop ≈ 47 MFLOP，×40 步展开 ×3（反传）≈ 5.6 GFLOP/样本，加上 U-Net 本体 ~10 GFLOP——5090D 上每秒几百样本，可行。

**核心风险（也是它只是副线的原因）**：训练用"K=24 可微拟合"当代理，部署用"300–3000 层贪心+软相"——**代理到全管线的迁移差距未经验证**，是 A.1b 的 go/no-go 实验本身（§D.4）。

### A.2 路线二：无梯度优化（拟合当黑箱打分器）

#### A.2a CEM / 坐标搜索 on PlanParams（L1.3-lite，兼交付物）

```
# per-image CEM，6 维（PlanParams 全体），目标 = 管线终评 PSNR
μ, Σ ← 先验（默认参数 ± 合理域）
for gen in 1..8:
    P = sample(μ, Σ, 16)                    # clamp 进各自合法域
    for p in P: score(p) = fvseg_pipeline(img, plan_layers_with(matte, T, p), seed=7, budget=B)
    μ, Σ ← 拟合 top-4 elite
```

成本：8 代 × 16 = 128 次管线跑。小预算（total=200，samples=20k，--gpu）N1X 单跑 ~5–15s（L1.2 修复后 GPU≡CPU 逐位一致、40× 提速的实测口径）→ **单图 ~11–32 分钟**。产品路径不实用，但三个高价值用途：
1. **MVE-0 的示波器**：量测 headroom 上限（CEM 结果 ≈ 该图 matte/规划空间可达 PSNR 的乐观估计）；
2. **A.3 的标签增强器**：给每图找出近优 PlanParams 作为回归目标；
3. **全局默认值调优**（一次性）：50 图 × 96 跑 ≈ 16h N1X / ~4h 5090D，产出更好的 `PlanParams::default()`，零模型、零推理开销、立刻全体用户受益——**无论 L1.3 后续成败，这个交付物都值得单独做**。

#### A.2b REINFORCE / ES 直接训网络（定量否决）

- ES on 网络权重：小型 U-Net ~2×10^6 参数，OpenAI-ES 需 ≥ 参数维 × 10–100 倍评估（10^7–10^8 次管线跑），每次 5s（小预算）到 2.5h（全预算）——10^12 秒量级，荒谬。
- REINFORCE on 随机 matte 头：每步需对"matte 采样分布"求 log-prob 梯度，一步 = 若干次完整管线跑；收敛需 10^5 步级 → 10^5–10^6 次跑 → 5×10^5–10^6 秒（小预算）= 6–12 天满载 N1X，且 REINFORCE 方差随像素维爆炸。
- 唯一保留角色：若 A.3 排名训练停滞，可用 REINFORCE 对**6 维 PlanParams 头**做最后 1dB 的策略微调（动作空间小、方差可控）——备而不用。

### A.3 路线三：离线蒸馏（推荐主线）

**思想**：管线 PSNR 是确定性函数（同 seed 同参数 → GPU≡CPU 逐位一致，L1.2 已验证），那就离线生成 `(图, matte 候选, 管线PSNR)` 三元组，监督训练一个小网络在推理时直接预测"PSNR 友好"的输出。梯度问题被彻底绕开，代价是一次性标签成本。

**两级递进**：

**C0 — PlanParams 回归器（先做这个）**。假设：headroom 主要藏在预算路由（edge_frac/subject_gain/subject_floor）而非掩码形状——L1.2b 的证据间接支持（bg 层 16 形状 mse 0.6 的预算红利 vs 主体区无人认领的预算黑洞）。

```
特征 x = [图像统计(色彩直方图/边缘密度/前景占比) ⊕ base-matte 统计(面积/周长/Otsu 双峰间距)]  (~96 维)
标签 y* = argmax_p PSNR(pipeline(img, plan_layers_with(m_base, T, p)))   # 由 A.2a 网格/CEM 离线求出
模型：2 隐层 MLP（96→64→32→6），训练 = 秒级（N1X CPU torch 或纯 Rust 都行）
部署：fvseg 启动时前向一次 → plan_layers_with(m_base, T, ŷ)
```

**C1 — matte U-Net（条件触发）**。仅当 MVE-0 消融显示掩码形状本身贡献 ≥0.3dB 时启动。

```
输入 = RGB ⊕ base_matte（U2Net 输出，4 通道）
输出 = m̂（单通道）
标签 = 同图候选 matte 的 PSNR 排名（不是回归 PSNR——贪心噪声让绝对值不稳）
损失 = Σ_pairs softplus(−(s_i − s_j)(φ(m̂_i) − φ(m̂_j)))   # 成对排名（Bradley-Terry）
      + α · λ₁·||m̂ − m_best||₁                            # 拉向每图最优候选（正则锚）
      + β · TV(m̂)                                          # 平滑正则
```

候选 matte 来源（每图 9–16 个，多样性是关键）：{U2Net, MODNet, GT-alpha(有则)} × {原样, 膨胀 r∈{2,5}, 腐蚀 r=2, 高斯羽化 σ=2} ⊕ {对最优候选的 PlanParams 变体}。掩码形态学操作离线用脚本生成 PNG，fvseg `--matte` 直读——**引擎零改动**。

**样本量估算**：C0——6 维回归 + 96 特征，300 图 × 16 候选 = 4800 个 (img, p, score) 点，充裕。C1——排名训练每图自带 ~C(12,2)=66 个 pair，300 图 ≈ 2 万 pair，配合裁剪增强（每图 6 个 256² crop）≈ 1800 训练样本；按经验够训一个 <2M 参数的小网，但域泛化紧——若 MVE-2 每域胜出不足，扩到 1–3k 图（标签成本线性增长，见 §D.3）。

**PSNR 噪声问题的正面回答**：同 seed 下噪声为零（确定性管线）；跨 seed 方差未知 → MVE-0 顺带测 10 图 × 5 seed 的 σ_seed。若 σ_seed > 0.3dB：标签固定 seed=7（确定性），训练用排名损失（对单调变换不变，天然抗噪）；用 seed=8 复核排名一致性（Spearman ρ ≥ 0.8 才采信该图标签）。

### A.4 路线对比决策表

| | A.1a 直出形状 | A.1b 软规划+内环 | A.2a CEM 参数 | A.2b ES/REINFORCE | A.3 蒸馏 (C0/C1) |
|---|---|---|---|---|---|
| 需要 matte | 否 | 是 | 是 | 是 | 是 |
| 训练时跑全管线 | 否 | 否（代理） | 否（离线搜索） | 是（每次评估） | 离线一次性 |
| 新训练设施 | PyTorch 镜像渲染器 | 同左 | 无 | 无 | PyTorch（最普通） |
| 标签成本 | 0（自合成） | 0（自合成+真图评估） | 128 跑/图 | 10^5+ 跑 | 12–20 跑/图 |
| 主要风险 | 多模态坍缩；只能小 N | 代理≠全管线迁移 | 仅 6 维上限有限 | 定量不可行 | 迁移泛化；域偏移 |
| 定位 | L1.4 候选（1 天 MVE） | 研究副线 | 示波器+默认值交付物 | 否决 | **主线** |

---

## B. 训练框架与硬件分工

### B.1 Rust 侧三选项（逐一核实）

| 框架 | N1X (GB10 ARM Win) 现状 | 训练成熟度 | 判定 |
|---|---|---|---|
| **candle** | CPU 后端可编译（纯 Rust）；**无 Vulkan 后端**（官方仅 CUDA/Metal，issue #344 wgpu 方案未合并，维护者认为 Vulkan compute 性能不佳）；CUDA 后端经 cudarc 理论上可试（N1X 有 CUDA 13.4 工具链，vinylizer sm_121 已建过）但**零先例、中高风险** | 有 Var/Adam/手写反传（mnist 训练例），定位推理优先，训练原始 | 不选：训练弱 + 无 Vulkan |
| **burn** | **burn-wgpu 走 Vulkan——这是我们唯一在本机全链验证过的 GPU 路径**（M3/M1.8 都跑在 wgpu+Vulkan 上）；ARM Win 无阻碍 | `Autodiff<Wgpu>` 反传可用，MNIST/CIFAR 训练示例齐；但 f32 无 tensor core、性能一般、ops 覆盖/精度有坑 | **唯一保留的 N1X 原生训练选项**，推迟到 A.1b 存活后评估 |
| **tch** | 无 aarch64-windows libtorch 官方发行（PyTorch WoA 官方 wheel 自 2.7/2025-04 起仅 **CPU-only** Python 包，C++ libtorch 不随发） | x64 上成熟 | 不选：N1X 不可用 |

### B.2 Python PyTorch 桥（推荐）

- **5090D（x64, 32GB GDDR7, 21760 CUDA 核, ~105 FP32 TFLOPS, 全 CUDA/PyTorch 生态）**：一切训练。小 U-Net 320² 前向 ~3 GFLOP/样本，训练 ~9 GFLOP/样本，按 20 TFLOPS 有效利用率 ≈ 2000 样本/s → **100 万样本 ≈ 10 分钟级**——训练永远不是瓶颈，标签生成才是。注意 5090D 兼任游戏/注入实机，训练窗口需错峰。
- **N1X（GB10, 128GB 统一内存, 273GB/s, ~30 FP32 TFLOPS 峰值, Vulkan via wgpu 已验证, CUDA 训练生态为零）**：标签工厂（fvseg --gpu 小预算 5–15s/跑）+ 评估机 + 推理部署验证（ort CPU EP 184ms 已达标；DirectML EP 可选加速）。原 L1.3 构想"N1X 统一内存可训"修正为：**N1X 只训得起 C0 的 MLP（CPU torch WoA wheel 或纯 Rust，秒级）**；真正的网络训练上 5090D。
- **部署回路零新增**：训练产物导出 ONNX → 走 L1.1 已验证的 `ort` 2.0.0-rc.10 + onnxruntime 1.22 win-arm64 CPU EP（`fv-seg` 现成路径），bundle-5090d 里的 x64 dll 同时覆盖 5090D 侧。

### B.3 推荐数据流

```
N1X (Vulkan, 已有)                      5090D (CUDA, 新增)               N1X/任意 (已有)
─────────────────────                   ─────────────────────           ─────────────────
候选 matte 生成(PNG 变体)   ──┐         PyTorch 训练 C0/C1/A.1b   ──►   ONNX 导出
fvseg --gpu 标签跑 (seed固定) ─┴──►  (图,候选,PSNR).parquet ──►  (小时级, 非瓶颈)      │
MVE-0/评估 扫描                          A.1a/A.1b 合成数据+训练          ▼
                                                                     ort 推理进 fv-seg
                                                   大预算终评(3000) 也在 5090D(bundle 现成)
```

A.1b 的渲染器损失**不做跨语言桥**：在 PyTorch 里按 M18_DESIGN §2 逐式重实现（~80 行，β=σ((1−q)τ) 全程张量化），用 Rust 侧 `cpu_forward_reference` 导出的 golden vector（固定栈 + τ∈{4,64,512} 的画布 npz）做一致性测试（门槛 max|diff| < 1e-3）。两边公式同源、测试互锁，比 IPC 桥快两个量级且不污染已验证的 Rust 内核。

---

## C. 数据方案

### C.1 真实数据集（本地可获得的，已核实）

| 数据集 | 内容 | 获取 | 许可 | 用途 |
|---|---|---|---|---|
| **DUT-OMRON**（HF `nobg/DUT-OMRON`） | 5,168 张自然图 + 二值显著掩码，**已在 hf-mirror 核实存在**（119MB parquet，仅 test split） | `hf-mirror.com/datasets/nobg/DUT-OMRON` | CC-BY-NC-4.0（研究/内部工具 OK） | SOD 域主训 + eval 域 S1 |
| **DUTS** | 10,528 train / 5,019 test + 掩码 | saliencydetection.net 或 GitHub SODsurvey 汇总（gh-proxy 可过） | 学术 | SOD 域训练扩充 |
| **aisegmentcn matting_human_half** | 34,427 张 600×800 半身人像 + **alpha 精修 matte** | 官方 Baidu/MEGA 镜像（腾讯云社区文章存链接）；**注意 HF `fredguth/aisegmentcn-matting-human` 镜像是空壳（已核实仅卡片无数据）** | 原站免费商用条款 | 人像域（MODNet 域） |
| **P3M-10K** | 10,421 张高清人脸模糊人像 + alpha | GitHub JizhiziLi/P3M 官方（OneDrive/Baidu 链，可达性待验） | 学术 | 人像域备选 |
| ayylmao 328² 等本地图 | 涂装车 meme 等 | refs/ 内 | — | **只做报告，永不入训**（防过拟合旗舰图） |

优先级：DUT-OMRON（拿来即用）> DUTS（同域扩量）> aisegmentcn（人像域，下载有摩擦）> P3M（可达性未验）。域偏好说明：ayylmao 主体是**涂装车**（U2Net 域而非 MODNet 域），SOD 数据是正确主域，人像是第二域。

### C.2 合成数据（两条路）

1. **gen_demo 式扩展**（gen_demo.rs 现状：单一 256² 固定场景——渐变底 + 双环 + 噪声，**多样性≈0，直接用没意义**）。参数化扩展方案：背景 ∈ {线性/径向渐变, Perlin 噪声, Voronoi 色块, 高斯模糊照片}；主体 ∈ {环/盘组合, 随机 Bezier 粗线, **DUTS 图前景 cutout 贴回**（copy-paste 增广， matte 取数据集掩码做羽化）}；matte = 解析真值。cutout 路线实质上把 SOD 数据集变成了无限合成器，是多样性主力。
2. **vinyl-native 形状栈渲染**（A.1a/b 专属）：随机采样 JSON 形状栈（bg 段 + subject 段）→ 引擎硬光栅出图。零标注成本、完美域内、ground truth = 栈参数。另兼 C1 的**评估上界参照**：合成图上可算"贪心 vs GT 栈"差距，给出 matte 可达增益的理论天花板。

### C.3 样本量与成本估算（汇总）

| 用途 | 规模 | 标签成本（N1X GPU, 10s/跑） | 5090D 等效 |
|---|---|---|---|
| MVE-0 探针 | 30 图 × 20 候选 × 2 seed = 1200 跑 | ~3.3h | ~1h |
| C0 标签 | 300 图 × 16 候选 = 4800 跑 | ~13h | ~3–4h |
| C1 标签（最小） | 300 图 × 12 matte 变体 = 3600 跑 | ~10h | ~2.5–3h |
| C1 标签（扩量） | 1500 图 × 12 = 18000 跑 | ~50h（挂夜） | ~12–15h |
| 训练（C0 MLP / C1 U-Net / A.1b） | — | — | 分钟级 / 2–6h / 2–4h |

小预算标签（total=200, samples=20k）与全预算的相关性必须先验（§D.1 判据 3）：50 对 (img, 候选) 在 budget=200 与 1500 下 PSNR 的 Spearman ρ ≥ 0.6 才允许用小预算标签。

---

## D. MVE 设计（最小可行实验）

### D.0 前置代码步（唯一允许的改动，~2h）

fvseg 加 `--plan-json <p.json>`：读 PlanParams 六参数 → `plan_layers_with`。30 行 + 1 测试。这是 MVE-0 的运行前提（现状 CLI 只有默认参数路径）。候选 matte 的阈值/膨胀变体不需要引擎改动——离线脚本产出 PNG 后 `--matte` 直读。

### D.1 MVE-0：headroom 探针（第一步，1 天，含 ~4h 机时）

**协议**：30 图（10 DUT-OMRON + 10 DUTS-test + 5 aisegment/本地人像 + 5 gen_demo 扩展合成）。
- 消融 A（预算 headroom）：固定 base matte，PlanParams 拉丁超立方 12 点（budget 三参数为主轴）；
- 消融 B（掩码 headroom）：固定 PlanParams=默认，matte 变体 8 个（双模型 × 形态学 4 种）；
- seed 稳定性：10 图 × 5 seed；
- 预算相关性：50 对 budget=200 vs 1500。
全部 N1X `--gpu`，budget=200，samples=20k，seed=7。

**判据（go/no-go）**：
1. **总 headroom**（消融 A∪B 的 max−base，CEM 复核 3 图）：p95 ≥ **0.5dB** 且 ≥30% 图 ≥0.5dB → L1.3 继续；全体 <0.3dB → **止损**，结论"matte/规划空间拧不出水"写入 STUDY，资源转 M1.6；
2. 消融 A vs B 分解决定走 C0（A 主导）还是 C0+C1（B 也 ≥0.3dB）；
3. ρ ≥ 0.6 才用小预算标签，否则标签升 budget（成本 ×5，重估）；
4. σ_seed > 0.3dB → 标签锁 seed + 排名损失（A.3 已预案）。

### D.2 MVE-1：C0 PlanParams 回归器（2–3 天，~13h 机时可与开发并行挂机）

标签：300 图 × 16 候选（DUT-0 措施后）→ 每图 y* = 最优 p。训练：MLP 秒级。评估：30 张留出图 × 3 arm（blind / base-matte 默认 / C0 预测）× budget∈{200, 300} × seed=7。
**判据**：vs base-matte 默认均值 ≥ +0.25dB 且 blind 不落后超 0.1dB → C0 进产品默认管线；否则仅保留全局默认值调优交付物（A.2a 用途 3）。

### D.3 MVE-2：C1 matte U-Net（1 周，标签 ~10h N1X + 训练 2–6h 5090D）

仅当 D.1 判据 2 显示掩码 headroom ≥0.3dB。300 图（可扩）× 12 matte 候选排名训练。评估同 D.2 加第 4 arm（C1 matte）+ GT-alpha 上界参照（有 alpha 的图）。
**判据**：vs base-matte ≥ +0.3dB（30 图配对 t 检验，效应 0.3dB / σ≈0.3dB / n=30 → 功效 ~85%）；任一域均值回退 >0.1dB 即判域失败；N1X ort 推理增量 ≤0.5s/图。

### D.4 A-MVE：路线一的两个一日实验（并行 opportunistic）

- **A.1a**：合成形状栈 256²、N=32、R=4 轮误差条件化解码、τ=32 固定、监督热启动→渲染损失精调，5090D 2–6h。判据：同 N=32 下渲染 PSNR 超贪心 ≥1dB（合成域）→ 立项 L1.4；否则归档。
- **A.1b**：golden-vector 一致性（0.5 天）+ 两层软规划代理训练（合成 5090D 2–4h）→ 30 张真图过全管线验证迁移。判据：真图 PSNR vs base-matte ≥ +0.2dB → A.1b 升级为主线二期；代理涨真图不涨 → 记"代理-管线差距"负结果归档。

### D.5 算力预算汇总

| 阶段 | N1X | 5090D | 人力 |
|---|---|---|---|
| D.0+MVE-0 | 4h GPU | — | 1 天 |
| MVE-1 (C0) | 13h 挂机 | <1h | 2 天 |
| MVE-2 (C1) | 10–50h 挂机 | 3–7h | 4–5 天 |
| A-MVE ×2 | — | 5–10h | 2 天 |
| 终评（§E 全矩阵） | 小预算部分 ~6h | 3000 预算抽查 5 图 × 2 arm × 20min ≈ 3.5h | 1 天 |

---

## E. 评估协议

1. **统一裁判**：`eval_design`（i64 积分图、2026-10-26 勘误后口径），bg=auto 且与拟合口径一致（记分板勘误教训）；GPU 路径（已验 ≡CPU 逐位）。
2. **Arm 矩阵**：① blind（无 matte 全局管线）② base-matte（MODNet/U2Net 原始 matte + 默认 PlanParams）③ C0 ④ C1 ⑤ 全局调优默认值 ⑥ GT-alpha（上界参照，有标注图）⑦ per-image CEM（昂贵 oracle，抽 3 图）。
3. **域多样性**（防 ayylmao 过拟合）：S1=SOD 物体（DUT-OMRON/DUTS-test）、P1=人像（aisegment）、V1=合成 vinyl-native、H1=手持真图含 ayylmao（**仅报告，永不调参**）。每域 ≥10 图，训练/评估严格分图。
4. **两个预算档**：300（受限档，主 KPI——L1.2b 结论决定的主战场）与 3000（满额档，5090D 抽查 5 图验证结论不翻转）。
5. **公平性**：等**预算**不等层数（三振出局差异属算法性质）；固定 seeds {7,8} 双报；推理时间开销并入报告（N1X ort）。
6. **与 L1.2 持平结论的关系（正面回答）**：L1.2b 真图主体占满时分层≈盲拟合（24.94 vs 24.97），因此 L1.3 的胜利条件明确为三选一：(a) 预算 ≤300 档显著赢（路由红利在受限时最大）；(b) 结构化背景域大幅赢（合成演示 +0.36dB 场景的推广）；(c) 预算路由（C0）在全域小赢但零成本。若三条全不成立 → L1.3 终结结论："瓶颈在拟合算法而非 matte"，产出（数据集+评估矩阵+PlanParams 默认值）入库，资源转 M1.6 混合循环。

---

## F. 风险表与止损判据

| # | 风险 | 概率 | 信号（提前暴露于） | 止损/缓解 |
|---|---|---|---|---|
| 1 | **PSNR 噪声淹没信号** | 低 | MVE-0 seed 测试 | 同 seed 确定性（已证）；σ_seed>0.3dB → 锁 seed + 排名损失 |
| 2 | **matte 空间拧不出水**（headroom<0.3dB） | 中 | MVE-0 判据 1 | 直接止损，写负结果；保住全局默认值调优交付物 |
| 3 | 小预算标签与全预算不相关 | 中 | MVE-0 判据 3（ρ<0.6） | 标签升 budget=1500（成本 ×5，重排 MVE-1 优先级）；ρ 仍低 → 说明 headroom 本身不稳 → 止损 |
| 4 | C1 域泛化失败（SOD 训不覆盖人像/涂装） | 中高 | MVE-2 分域报告 | 域回退 >0.1dB 的域不部署该网（按域回退 base-matte）；扩数据/加 copy-paste 增广一轮后仍败 → C1 降级为 S1 域专用 |
| 5 | 代理-管线迁移差距（A.1b） | 中高 | A-MVE 判据 | 真图不涨即归档负结果（本身是有价值结论） |
| 6 | A.1a 多模态坍缩 | 中 | A-MVE（监督损失也下不去即架构问题） | 换 R 轮误差条件化/增熵正则；仍败归档 |
| 7 | 训练不收敛/离群 matte（网络输出退化为全 0/全 1） | 低 | MVE-2 训练曲线 + matte 可视化抽检 | 排名+L1 锚+TV 正则三件套；输出 clamp 后过 plan 的退化路径已有（constant_mattes_degrade_gracefully 测试保证不崩） |
| 8 | 网络利用管线怪癖刷分（如 Otsu bias 压到极端 + bg 估计路径学坏） | 低中 | MVE-2 逐图 worst-case 审计 | 预测参数 clamp 进 MVE-0 扫描过的凸域；worst-case < −0.3dB 的图回退默认 |
| 9 | 5090D 时间被游戏/注入挤占 | 中 | 排期 | N1X 可承担除大网络训练外的一切；C0 甚至可在 N1X CPU 训 |
| 10 | burn-wgpu（若走 N1X 原生训练）ops/精度坑 | —（推迟） | 仅 A.1b 存活后评估 | 先 PyTorch 验证数学，再考虑移植 |

**总止损原则**：任何阶段连续两个判据未达 → 停止加投入，把已有产出（数据、评估矩阵、PlanParams 默认值、负结果）写入 STUDY.md 后关闭 L1.3。

---

## G. 与 M1.8 / L1.2 的衔接

1. **M1.8 的角色**：A.3 主线不直接用它的梯度（贪心在前挡路），但 MVE-0/C0 的标签管线终点若包含"--refine 精修"，应保持 B 链配方一致（精修基础栈→resume 续拟）——旗舰 30.83dB 就是 B 链所出，A/B 对照必须同链。A.1a/b 则直接消费 M1.8 数学：PyTorch 镜像 + golden vector 互锁（§B.3），τ0=32、θ lr ×0.05、软评测/硬裁判分离三条实测教训全部继承。
2. **L1.2 的角色**：fvseg 管线 = L1.3 的打分器与部署宿主；"不关心区填画布"技巧被 A.1b 软化复用；全局软相修复保持。L1.2b 的持平结论不是坏消息而是**定位约束**：它把 L1.3 的取胜区间精确圈到了预算受限/结构化背景/预算路由三处，§E 协议据此设计。
3. **PlanParams → --plan-json → --tune**：plan.rs 已备好旋钮但无调用方（A.0 勘误）。D.0 的 30 行接线让 MVE-0 跑起来；若全局默认值调优胜出，`--tune` CLI（plan.rs 注释里的名字）顺势落地为正式小搜索入口。
4. **记分板衔接**：一切终评数字必须用 2026-10-26 勘误后裁判 + bg 口径一致（auto 拟合栈用 --bg auto 评）重新计；与历史旗舰（1497 层 30.28 / B 链 2177 层 30.83）对比时注明预算档差异。
5. **交接路径**：L1.3 任何存活产物（C0 预测头）作为 fvseg 启动时的可选前缀步骤接入，ONNX 走 `SegModel` 注册表同款模式（`FV_SEG_*` env 覆盖），不新增运行时依赖（ort 已在）。

---

## 附：执行清单（一句话版）

1. **今天**：D.0 `--plan-json` 接线（2h）→ MVE-0 扫描挂机（4h）→ 明早看判据 1–4。
2. headroom ≥0.5dB：MVE-1 C0（2–3 天）→ 达标则产品化默认值调优 + C0；掩码 headroom 也达标则 MVE-2 C1（1 周）。
3. 并行 opportunistic：A-MVE 两日实验（5090D 空闲窗口），A.1a 存活立 L1.4。
4. 全程记录到 STUDY.md L1.3 条目；任何止损点写明负结果与保留交付物。
