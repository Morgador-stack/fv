# 千层 refine 失配 · 判别实验包（REFINE_FORENSICS_PLAN）

日期：2026-10-28 ｜ 状态：设计定稿，未执行 ｜ 本文档只读+新增，不改任何源码
上游：STUDY.md §13–§15 排查链 ｜ 数学事实来源：M18_DESIGN.md §2

---

## 0. 排查链快照（已排除项，实验设计的前提）

| # | 假说 | 判决 | 证据工具/来源 |
|---|---|---|---|
| 1 | 软硬目标漂移 ∝ n（全画布 PSNR 级） | **否证** | `fv/crates/fv-gpu/examples/soft_hard_gap.rs`（零 GPU）：300 与 1000 层 soft-vs-hard 偏差同量级，不随 n 增长到 dB 级 |
| 2 | Adam 噪声放大 ∝ n | **否证** | `fv/crates/fv-fit/examples/jitter_bench.rs`（零 GPU）：同幅度参数抖动的硬画布损伤不 ∝ n |
| 3 | 窗口化实现 bug | **否证** | fc 链 r1000 经窗口化 refine **+1.05dB 成功**（同一路径、同一 n）；窗口层梯度与全量精确一致（fwd.rs `render_windowed` 契约） |
| 4 | 步长病（lr 过大） | **否证** | ¼lr 仍崩 |
| 5 | 快照量化噪声地板（H1） | **否证** | 主会话数值复核：u16×257 步长 1/257≈0.004 灰阶，噪声地板 ~59dB，SNR 充足 |
| 6 | "栈已收敛、无处可去" | **否证** | STUDY §15：polish 链 36.11→40.24dB——硬语义余量真实存在（~2dB），refine 拿不到它 |

**锁定病灶**：`B1 栈 × n≥1000 × 梯度方向系统性有害`。对照矩阵：

| 栈 | n | refine 结果 |
|---|---|---|
| B1 采样 s300 | 300 | ✓（逐层 τ 修复后 +0.17，28.711→28.884 accepted） |
| 旧采样器 fc_r1000 | 1000 | ✓ 窗口化 +1.05dB |
| B1 采样 r1000/r2000 | ≥1000 | ✗ 崩（guard 拒绝，零损伤但零收益） |
| polish 后 2997 层 B1 栈 | 2997 | ✗ 仍崩（40.24→30.8@10）→ 与"垃圾层"无关，方法级边界 |

逐层尺度 τ_l = τ·min(a,b)/8 已落地（细椭圆不可见问题已治）。**open 问题：千层密度下，软目标上的下降方向为何在硬裁判上系统性有害。**

剩余假说空间（本实验包的靶子）：
- **H-A 目标失配**：软目标（τ_l 有限带宽）的极小化方向 ≠ 硬裁判的下降方向；密度升高 → 边界带重叠加剧 → 方向腐化（尽管全画布 PSNR 差不 ∝n，**梯度级**失配可以 ∝n——假说 1 的否证只在全画布 PSNR 粒度）。
- **H-B 规模级实现分歧**：FD 测试只做过 6 层 40 参数；窗口梯度拼装、u16 解包、workgroup 归约在 n=1000 下的数值行为未对账。
- **H-C 优化器动力学**：埋没层（T≈0）梯度≈0 但 Adam 尺度不变（step≈lr 与 |g| 无关）→ 700 个埋没层被噪声推着做随机游走；或各层单独有益的移动联合起来互相打架（Jacobi 冲突）。
- **H-D 带宽旋钮失当**：B_REF=8 对千层密度不是敏感 workpoint（且 clamp 下限 4 在薄层+低 τ0 段吞掉调度）。

---

## 0.5 公共材料、工具与纪律（P0，所有实验的前置）

### P0-a 材料重建与冻结（需 GPU，约 10–15 分钟）

**现状**：链条中间产物在 `%TEMP%`（`flagship_v3.sh` 的 `T=/c/Users/17479/AppData/Local/Temp`），**已被清理**。bundle 里只有终品 `bundle-5090d/designs/fv3_full.json`（3000 层）。

**步骤**（严格照 `D:\WorkSpace\ZCWorkspace\forza-vinyl\bench\flagship_v3.sh` 前四条，当前 target-b1 二进制）：
1. `fvfit-gpu ayylmao.png --layers 300 --samples 1000000 --opt-res 328 --bg auto --out-prefix fv3f_s300`（B1 采样）
2. `--refine-json fv3f_s300.json --diff-iters 300 --diff-tau0 32 --diff-lr-geo 0.2 --diff-lr-col 0.8`
3. `fvfit-gpu --resume fv3f_s300r.json --layers 1000 ... --out-prefix fv3f_r1000`
4. `--refine-json fv3f_r1000.json --diff-iters 300 ...`（预期 guard 拒绝 → `fv3f_r1000.refined.json` ≡ 输入，记录 trail 作为"崩"的现场证据）
5. **全部产物复制到 `D:\WorkSpace\ZCWorkspace\forza-vinyl\bench\results\forensics\` 冻结**，此后所有实验只用这份材料。
6. 零 GPU 体检（进 P0-b 工具或手写 20 行 python）：对 fv3f_r1000.json 输出每层统计直方图——min(a,b) 分位、纵横比 a/b 分位、θ 分布、bbox 重叠计数。这份数据是 E-A 分层抽样的底稿，也验证 H-D 的 clamp 占比（b≈1.2px、τ0=32、B_REF=8 → tl=4.8，远离下限；B_REF=32 → tl=1.2 **被 clamp 吞**）。

**注意**：B1 链脚本不带 `--seed`，重建栈与原 fv3_r1000 **不逐位相同**——无所谓，只要此后所有实验用同一份冻结材料（同二进制同会话纪律，STUDY §14 教训）。

### P0-b 判据口径（全实验统一）

- **裁判**：`eval_design <img> <json> --bg auto`（i64 积分图精确口径）。refine 内部 `hard_psnr`（refine.rs:190，q<1 像素中心近似）只用于 trail，**两裁判可能在小数位分歧**——凡"崩"的 arm 必须用 eval_design 复核一次，排除裁判差假象。
- **GPU 纪律**：单进程串行（STUDY §13 nvlddmkm 三次蓝屏教训）；每实验开工前先跑 `probe` 小探针确认设备健康；solo device-lost 则等 5 分钟重试。
- **同二进制同会话**：同一实验的 A/B arm 必须同一构建连跑；跨天/跨构建只做方向性参考。

### P0-c 代码改动量记法

零 = 不改代码 ｜ S = <50 行 ｜ M = 50–150 行。所有改动集中在：
`fv/crates/fv-gpu/src/diff/refine.rs`、`fv/crates/fv-gpu/src/bin/fvfit-gpu.rs`、`fv/crates/fv-gpu/examples/`（新增 example 不算侵入）。

---

## §E-A 梯度对账（grad_audit）

**问题**：对实栈，GPU backward 给出的梯度方向到底错没错？错在**哪类层**（重叠细椭圆 / 特定 θ / 极端 a/b / 埋没层）？错在**哪一环**（反向内核 / 快照量化 / 软目标本身）？

**性质**：核心零 GPU（CPU f64 中心差分），一次 GPU backward 取实栈梯度。**这是唯一能落到"每层每参数"粒度的实验，是机制级的地量。**

### 原理

三方对账，把"方向有害"拆解到环：
1. `g_gpu`：实栈一次前向+反向的真实梯度（窗口拼装，同 refine.rs:87–110 的做法）；
2. `d_soft`：**CPU f64** 软目标（`DiffRenderer::cpu_forward_reference`，fwd.rs:344，与内核同 τ_l 公式）上的中心差分方向导数——软目标的"真梯度"；
3. `d_hard`：CPU 硬光栅（`fv_fit::raster::composite_ellipse`，引擎行 span 同语义，soft_hard_gap.rs 已示范）上的中心差分——硬裁判的"真下降方向"。

判读逻辑：
- `g_gpu ≈ d_soft` 而 `cos(d_soft, d_hard) < 0` → **H-A 目标失配**（实现无罪，软景观本身把人往沟里带）；
- `g_gpu ≉ d_soft`（六层测试过、千层失败）→ **H-B 规模级实现分歧**；
- 三方都健康但 refine 仍崩 → **H-C 优化器动力学**（Adam 尺度不变 + 联合步冲突）。

### 抽样准则（分层，不是纯 top-|g|）

从 fv3f_r1000（1000 层）按 P0-b 直方图抽 **24 层**：
| 层组 | 数量 | 抽法 | 回答什么 |
|---|---|---|---|
| G1 top-\|g\| | 8 | 按 GPU 梯度 8 维 L2 范数最大 | "最自信的方向"对不对（Adam 主要在吃它们） |
| G2 埋没层 | 4 | 可见度 T 均值 < 0.05 随机抽 | H-C：梯度是否被噪声/量化占据 |
| G3 半露层 | 4 | T ∈ [0.05, 0.5] 随机抽 | 部分遮挡下的方向质量 |
| G4 全露层 | 4 | T > 0.5 随机抽 | 对照组 |
| G5 极端形态 | 2+2 | a/b 最大 2 层 + bbox 重叠计数最大 2 层 | "哪类层方向错"的直接答卷 |

T 估计：随机采 2000 像素，f64 自顶向下扫一遍累计每层命中均值的乘积（O(n·采样数)，秒级）。每层记录标签：a、b、a/b、θ mod π、min(a,b)、重叠数、T 均值。

### 步骤

1. 新 example `fv-gpu/examples/grad_audit.rs`（约 200–250 行，M）：
   - 载入 JSON → ShapeParams（转换抄 fvfit-gpu.rs:99–108，含 θ 取负约定）；
   - `DiffRenderer::new(328,328,n)` → 逐窗口 `render_windowed`+`backward` 拼装 `g_gpu`（抄 refine.rs 窗口循环，约 25 行）；
   - 逐层逐参数中心差分：**ε 与 Adam 实际步长同量纲**——几何 ±0.25px、θ ±0.002 rad、颜色 ±2.0（Adam step≈lr≈0.2px/0.8 色阶，探针打在优化器实际作用的尺度上）；另抽 2 层做 ε/2 稳定性复核（0.125px），排除非线性假差；
   - 每次 FD 评估 = 一遍 f64 软渲染（或硬光栅）→ 全画布 MSE。**朴素实现 O(n·plane) ≈ 3–10s/次 × 384 次 ≈ 20–60 分钟单核，可接受**；推荐优化：只重合成受影响 bbox（polish 的 A 集合记账思路，下层画布与上方透射各预扫一次，约 40 行）→ 分钟级；
   - 输出 CSV：layer, 组别, 标签, ‖g_gpu‖, ‖d_soft‖, ‖d_hard‖, cos(g_gpu,d_soft), cos(g_gpu,d_hard), relerr(g_gpu,d_soft), 判读。
2. 跑一次（零 GPU 部分 + 一次 GPU backward，几秒）。
3. **Tier-2（可选，+30 行）**：再对 4 层做 "GPU-FD"——扰动后用 GPU 前向画布（`last_canvas`）算 MSE 的差分。三方变四方后可把分歧进一步拆到 {前向内核+量化} vs {反向内核}（repo 惯用的"回读对账"法，soft_lifecycle 战役同款）。

### 判读表

| 观测 | 结论 | 指向的修法 |
|---|---|---|
| G1–G5 全部：relerr(g, d_soft) ≤ 2% 且 cos(g, d_hard) > 0.5 | 梯度健康，崩在优化器动力学 | H-C：交错更新/只更可见层 |
| relerr ≤ 2% 但 cos(g, d_hard) < 0（G3/G5 最重） | **H-A 目标失配坐实** | 带宽/τ 调度（联动 E-D）；逐 iter 硬裁判守卫步 |
| relerr > 5% 且只出现在 n=1000（六层测试仍绿） | **H-B 实现分歧** | Tier-2 定位前向 vs 反向；修内核或快照 |
| 仅 G2（埋没层）梯度无主、方向随机 | H-C 噪声游走变体 | 梯度幅值门控：T 或 \|g\| 低于阈值冻结该层 |
| d_hard 本身方向混乱（G1 也是） | 硬景观本身病态（与 refine 无关） | 回 polish 路线，refine 只做稀疏段（现行配方已隐含此结论，可收案） |

**"哪类层方向错"的答卷**：按 G1–G5 分组统计 cos 均值 + 按标签（a/b 分位、θ 象限、min(a,b) 分位）回归，直接输出"方向腐化集中在【高纵横比 × 半遮挡 × θ≈走向】层"这类可执行结论。

### 成本

- **零 GPU 为主**（FD 全在 CPU f64）+ 一次 GPU backward（秒级）｜预计耗时：实现 M（2–4 小时）+ 单次运行 ≤ 30 分钟（含分析）。
- 副产品：grad_audit 转正为常驻回归工具（与 soft_hard_gap 并列）。

---

## §E-B 渐进高度（崩坏临界 n 与模式）

**问题**：同一个 B1 链条，refine 从哪一层数开始坏？渐变还是悬崖？

**性质**：**需 GPU，几分钟**；几乎零代码——性价比最高的一手数据。

### 原理

B1 链是贪心顺序放置：fv3f_r1000 的前 k 层 = 链条自身在 n=k 时的真实历史状态（resume 从 s300r 顺序补到 1000）。对前缀栈分别 refine，Δ(n) 曲线的形状区分两种病灶：
- **渐变**：Δ 随 n 单调恶化 → 密度相关机制（H-A 的梯度级失配 / H-C 的游走层数累积），与 soft_hard_gap 的"全画布不 ∝n"不矛盾——失配在方向不在幅值；
- **悬崖**：n≤800 全好、1000 突坏 → 某个结构性阈值被跨过（窗口数从 1 变 2？重叠网络相变？Adam v 矩阵动态？），嫌疑立刻收窄。

### 步骤

1. 零代码造前缀：python 截断 `fv3f_r1000.json` → `fv3f_p300/p500/p800.json`（p300 应与 fv3f_s300r 逐位相同——顺手做血统校验，不是则 P0 材料有问题，停下）。
2. 每个前缀跑 `--refine-json pX.json --diff-iters 20 --diff-tau0 32 --diff-lr-geo 0.2 --diff-lr-col 0.8`（与 s300 已验证配方完全一致，唯一变量是 n）。
3. 记录：trail 全程（渐变/突变/立即损）、accepted 与否、best_iter 位置。
4. 判据补充：把 4 条 Δ(n) + 已知的 r2000（guard 拒绝）画在同一条轴上。

### 判据

| 观测 | 结论 |
|---|---|
| Δ(300)>0, Δ(500)>0, Δ(800)<0, Δ(1000)≪0 单调恶化 | 密度渐进腐化 → E-A 分层重点放在 G2/G3（遮挡谱） |
| 800 仍 ≥0、1000 陡崩 | 悬崖 → 查窗口数（cap 漂移，fwd.rs:102 的 96% 容量探测）、首个越界层、Adam 动态 |
| 全前缀都轻微 <0（含 p300） | 重建材料/配方与原 s300 成功现场不一致 → 停，查二进制与 τ_l 代码版本 |
| p500/p800 崩但 p300 好 | 临界 n* ∈ (300,800)，再补 p400/p600 细化（每次 +2 分钟） |

### 成本与代码改动

- **需 GPU**：4 次 × 20 iter（n≤1000 ≤2 窗口，每 iter 毫秒-几十毫秒级）≈ **5–8 分钟**。
- 代码改动：**S**——CLI 缺 `--diff-eval-every`（RefineConfig.eval_every 已存在但 CLI 未暴露；fvfit-gpu.rs 加 1 个 arg + 1 行字段，约 6 行）。不加也行（默认 10 → trail 只有 10/20 两点），但"首步即损 vs 累积损"的读数会糊掉，建议加。

---

## §E-C 前缀隔离（高层毒化 vs 全栈交互）

**问题**：r1000 的崩，毒在**新增 700 层自己的梯度**，还是**致密上下文让前 300 层（已被证明可精修的好层）的梯度也变坏**，还是**联合更新互相打架**？

**性质**：**需 GPU，几分钟**；逻辑分辨力最强的单一实验。

### 原理

三臂对照，同一栈、同一配方、只变"哪些层可动"（render_windowed 天然支持任意快照窗口；缺的只是冻结逻辑）：

| 臂 | 可动层 | 冻结层 | 读数 |
|---|---|---|---|
| C0 | [0,1000) | 无 | 同会话阳性对照（预期：崩） |
| C1 | [0,300) | [300,1000) | 好层在致密上下文里还精修得动吗 → **全栈交互** |
| C2 | [300,1000) | [0,300) | 新层的梯度在位面上是否有毒 → **高层毒化** |

### 判读矩阵（2×2 完备）

| C1 | C2 | 结论 |
|---|---|---|
| ✓ 升 | ✗ 崩 | **高层毒化**：毒在新增层自身梯度 → E-A 的 G1 应在 [300,1000) 层看到 cos(g,d_hard)<0 → 修法：可见度/幅值门控、只精修未被覆盖层 |
| ✗ 崩 | ✓ 升 | **上下文毒化**：致密覆盖把好层梯度带坏 → E-A 的 G4（全露层）应健康而 G3 变坏 → 修法：链上 refine 只动底部窗口（refine@1000 只精修前 300 = 现行配方即可安全吃梯度红利） |
| ✓ | ✓ | **联合步冲突**：各自有益、合起来打架（Jacobi vs Gauss-Seidel）→ 修法：交错更新（每 iter 只更半数层）或逐窗口序贯 |
| ✗ | ✗ | 与可动集无关 → 崩在评估/收口环节 → 复核 eval_design 裁判差，转 E-A Tier-2 |

### 步骤

1. 代码：`RefineConfig` 加 `update_range: Option<(usize,usize)>`（refine.rs；窗口循环只派发相交窗口、Adam 循环跳过界外索引，约 25 行）+ CLI `--refine-lo/--refine-hi`（约 8 行）。**合计 S（<50 行）**。
2. 三臂连跑（同二进制同会话），20 iter，配方同 E-B。
3. 每臂产物用 eval_design 复核。

### 成本

- **需 GPU**：3 臂 × 20 iter ≈ **5–8 分钟**；代码 S（约 1 小时含烟测）。
- 与 E-B 的正交组合：E-B 的 p300（300 高度 × 300 可动）+ E-C 的 C1（1000 高度 × 300 可动）构成"高度 × 可动集"2×2 的一半，可额外回答"高度本身（不动 700 层、只当上下文）是否已足以毒化"——这正是 C1 单臂的读数，无需加跑。

---

## §E-D τ_l 敏感性（B_REF 扫描）

**问题**：把逐层归一化的唯一旋钮 B_REF（现硬编码 8，带宽 ∝ B_REF/τ）扫一遍，失效是否系统性平移？——直接检验 H-D/H-A 的带宽机制。

**性质**：**需 GPU，10–15 分钟**；代码改动是四个实验里最大的，故排最后。

### 原理

τ_l = clamp(τ·min(a,b)/B_REF, 4, 4096)，像素级过渡带宽 ≈ B_REF/τ。B_REF↑ → 软目标更糊（离硬裁判更远、景观更平滑）；B_REF↓ → 更逼近硬语义（方向更"真"但更病态、且 clamp 下限 4 开始吞薄层）。若失效是带宽驱动（H-A/H-D），崩坏幅度/临界 n* 应随 B_REF **单调平移**；若对 B_REF 不敏感，机制是结构性的（遮挡网络/H-C），带宽不是修复旋钮。

**已知陷阱（设计时就写进判据）**：B1 薄层 b≈1.2px 时 tl = 32×1.2/B_REF → B_REF=32 得 1.2 **被 clamp 到 4**——高档位在前几 iter 对薄层实际无效，读数要按"clamp 占比"分层（P0-b 直方图给出占比）。

### 步骤

1. 代码（**M，50–90 行，5 处**）：B_REF 从常量提为 uniform——
   - `fv/crates/fv-gpu/src/diff_forward.wgsl:47`（beta_at）
   - `fv/crates/fv-gpu/src/diff_backward.wgsl:50`（bbox 半径）与 `:84`（循环内）
   - `fv/crates/fv-gpu/src/diff/fwd.rs:369`（cpu_forward_reference，保持逐位同源）
   - 走 Uni 追加字段（注意 uniform struct 尺寸须 16 字节对齐：现 9×f32，加 b_ref 后补 pad 到 12×f32）+ `RefineConfig.b_ref` + CLI `--diff-bref`。
   - **偷懒备选**：改常量重编译 4 个二进制（零表面改动；B_REF 本就是操纵变量，方法学可接受，但记录各二进制哈希，且 cpu_forward_reference 必须同步改否则 E-A 对账失配）。
2. 四臂：B_REF ∈ {4, 16, 32} + {8 对照}，各 `--refine-json fv3f_r1000.json --diff-iters 20`（配方同上，唯一变量 B_REF）。
3. 每臂记录：trail、accepted、eval_design 复核；有精力再加 r500 前缀看 n* 平移（+4 分钟）。

### 判据

| 观测 | 结论 |
|---|---|
| 崩坏幅度随 B_REF 单调（越大越崩或越小越崩） | **带宽机制坐实** → B_REF 就是修复旋钮，往有利方向调 + 重跑 E-B 找新 n* |
| 四臂读数同量级崩 | 带宽不敏感 → 排除 H-D，机制在 H-B/H-C，回 E-A 判读表 |
| 某档位 flip 成 accepted | 直接拿到工程修复（记录配方进 STUDY）；并用 E-B 在新 B_REF 下复核是否只是把 n* 推远 |

### 成本

- **需 GPU**：4 臂 × 20 iter ≈ **10–15 分钟**（含重编译）；代码 M（1–2 小时）。
- 注意：若 E-A 已给出 relerr(g, d_soft)>5%（H-B 实现分歧），E-D 的读数被实现噪声污染——**E-D 必须排在 E-A 之后解读**，这正是它排最后的原因。

---

## §F 执行顺序与信息量/成本排序

### 排序（期望信息量 ÷ 总成本，含人力）

| 序 | 实验 | 信息量 | 成本 | 比 | 一句话理由 |
|---|---|---|---|---|---|
| 1 | **E-B 渐进高度** | 中高（n* 锚点 + 渐变/悬崖分型，后续所有实验引用） | 零代码(S−)+5 分钟 GPU | **最高** | 最便宜的地图：先知道"坏从哪开始、怎么坏" |
| 2 | **E-C 前缀隔离** | **最高**（三臂互斥判别，直接决定修法分支） | S + 8 分钟 GPU | 高 | 逻辑上一次砍掉一半假说空间 |
| 3 | **E-A 梯度对账** | 最高（机制级、每层每参数、永久工具） | M + 30 分钟（CPU 为主） | 高 | 定格机制的证据级实验；放在 1/2 之后可用其结论收窄抽样层组 |
| 4 | **E-D B_REF 扫描** | 中（敏感性平移，间接推断） | M + 15 分钟 GPU | 中 | 解读依赖 E-A 无罪判决；但若 flip 即白捡修复 |

### 执行流（含决策点）

```
P0 材料重建+冻结+直方图（GPU 10–15min，一次性）
 ├─ 血统校验失败 → 修 P0，不停下其余设计
 ▼
E-B（GPU 5–8min）
 ├─ 全前缀连 p300 都崩 → 材料/配方/代码版本问题，回 P0 复核（终止）
 ├─ 悬崖型 → 重点查窗口数/cap 漂移与首坏层
 └─ 渐变型 → 密度机制，继续
 ▼
E-C（代码 S；GPU 5–8min）
 ├─ C1✓C2✗ → 高层毒化 → E-A 重点 G1@新增层 + 门控修法原型
 ├─ C1✗    → 上下文毒化 → E-A 重点 G3/G4；修法=链上只精修底窗
 └─ C1✓C2✓ → 联合步冲突 → 修法=交错更新（E-A 可减配为抽查）
 ▼
E-A（代码 M；CPU 为主 30min）
 ├─ relerr(g,d_soft)>5% → H-B 实现分歧 → Tier-2 定位前向/反向，修内核后重跑 E-B/E-C 复测
 ├─ cos(g,d_hard)<0     → H-A 目标失配 → E-D 上马找带宽 workpoint；或改逐 iter 硬守卫
 └─ 全健康              → H-C 优化器动力学 → 门控/交错更新原型，跳过 E-D
 ▼
E-D（代码 M；GPU 10–15min）——仅在 E-A 判"实现无罪"时执行/解读
```

### 收案标准

两个独立实验指向同一机制类（目标失配 / 实现分歧 / 优化器动力学 / 结构遮挡），即收案写进 STUDY.md；E-A 的 CSV + E-B 的 Δ(n) 曲线 + E-C 三臂表作为证据链归档 `bench/results/forensics/`。

### 最低成本路径（若只许跑一个）

**E-B**：零代码、几分钟，产出的 n* 与崩坏模式决定其余一切实验的抽样与解读框架。

---

## 附：涉及文件索引（绝对路径）

| 文件 | 角色 |
|---|---|
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\STUDY.md` | 排查链 §13–15（上游事实） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\M18_DESIGN.md` | 梯度数学唯一事实来源 |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-gpu\src\diff_forward.wgsl` | τ_l beta_at（:47，B_REF=8 硬编码处 ①） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-gpu\src\diff_backward.wgsl` | τ_l bbox（:50 ②）+ 循环（:84 ③）；窗口梯度拼装语义 |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-gpu\src\diff\fwd.rs` | render_windowed（:254）、容量 96% 探测（:102）、cpu_forward_reference（:344，τ_l :369 ④） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-gpu\src\diff\bwd.rs` | backward 派发（一 workgroup 一层） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-gpu\src\diff\refine.rs` | Adam/τ 退火/窗口循环（:87–110）、guard（:157）、θ lr×0.05（:126）、内部裁判 hard_psnr（:190） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-gpu\src\bin\fvfit-gpu.rs` | refine CLI（参数面；E-B 需补 --diff-eval-every，E-C 需补 lo/hi，E-D 需补 --diff-bref） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-gpu\examples\soft_hard_gap.rs` | 判别工具范例（零 GPU） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\fv\crates\fv-fit\examples\jitter_bench.rs` | 判别工具 #2（零 GPU；注意实际在 fv-fit 非 fv-gpu） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\bench\flagship_v3.sh` | B1 链配方来源（REFINE="--diff-tau0 32 --diff-lr-geo 0.2 --diff-lr-col 0.8"） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\bundle-5090d\designs\fv3_full.json` | 现存唯一 fv3 链终品（3000 层；中间产物已散佚） |
| `D:\WorkSpace\ZCWorkspace\forza-vinyl\bench\results\forensics\` | （P0 新建）全部实验材料与证据归档地 |
