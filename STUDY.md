# Forza Vinyl 工具研究笔记

日期：2026-10-24 ｜ 参考仓库见 `refs/`（经 gh-proxy 克隆，浅层）
原则：**只学不抄**——vinylizer 无许可证（保留所有权利，只读思路）；forza-painter-fh6 / geometrize-gpu / ForzaDesigner6 均为 MIT（可参考、可注明出处的移植）。

---

## 1. FH6 内存布局（三方交叉验证一致）

来源：`vinylizer/src/inject/inject.cpp` ≡ `ForzaDesigner6/fd6/inject/patterns/fh6_patterns.json` ≡ `forza-painter-fh6/experiments/fh6_layer_dump_restore/DEVELOPMENT.md`

### LiveryGroup（涂装组对象）
| 字段 | 偏移 | 类型 | 说明 |
|---|---|---|---|
| layer_count | 0x5A | u16 | 模板层数（定位锚点） |
| layer_table | 0x78 | u64* | 指向"层指针数组"，数组元素为 u64，步长 8 字节 |

### Layer（每层独立堆分配；blob 实际 ≥0x140，稳定可写区 0x80）
| 字段 | 偏移 | 类型 | 写入规则 |
|---|---|---|---|
| position | 0x18 | f32×2 | y 取负（图像系→游戏系） |
| scale | 0x28 | f32×2 | 椭圆类 ÷63，矩形 ÷127 |
| rotation | 0x50 | f32 | 写 360°−angle |
| color | 0x74 | u8×4 | RGBA；基础形状(101/102) alpha 必须 255；**228 软椭圆的 alpha=不透明度** |
| mask | 0x78 | u8 | 必须 0 |
| shape_id | 0x7A | u8 | 101=矩形，102=椭圆/圆，228=软(渐变)椭圆；社区形状 109/114… |
| ⚠0x58 | | u32 | 疑似 vertex/shape count，游戏会重算，勿写 |
| ⚠0xA8 | | u64 | 形状资源指针，**会话局部**，跨会话/跨模板复制必崩，勿写 |

- 写入**即时生效**，无 commit 步骤（FD6 patterns 注记）
- 上限：涂装组 3000 层；模板需多留 4 个边界层（3000 模板≈2996 可画）
- JSON type → shape_id：1→101，16→102，228→228

## 2. 层表定位算法（u16 扫描法，三家一致）

1. `VirtualQueryEx` 遍历 MEM_COMMIT + MEM_PRIVATE + 可读可写区域（排除镜像/映射）
2. 区域内扫 u16 == 模板层数 → 候选 LiveryGroup = 命中地址 − 0x5A
3. 读 候选+0x78 处 u64 得表指针，须为用户态指针且指向私有可写内存
4. 逐层验证：表项指针合法（含 blob 不跨区域尾）+ shape_id∈{101,102} + mask==0 + pos/scale/color 可读
5. vinylizer 要求前 64 层**全部**通过（最严格）；FD6 抽 16 层 ≥8 过、每层 ≥3/5 字段可信

RE 替代工具（FD6）：`hw_breakpoint.py`（DebugActiveProcess + DR0 硬件断点抓"是谁在写这个地址"，≈Cheat Engine 同款）；`rtti_locator.py`。

## 3. 两代拟合算法

### A. 贪心 geometrize（forza-painter 系、geometrize-gpu）
循环 N 次：随机采样 K 个候选旋转椭圆 → 每个算最优颜色（区域均值）→ 增量误差评估（只重算 bbox 内像素）→ 取误差下降最多者叠加。参数量级：3000 形状 × 20万~100万候选采样。GPU 版要点：只做旋转椭圆、bbox 增量更新、批量候选并行（OpenCL/Vulkan compute，`shaders/eval_v4.comp`）。

### B. 可微渲染 + 梯度下降（vinylizer，`src/core/optimizer.cpp`）
- N 个**软椭圆(228)**×9 参数（cx,cy,rx,ry,angle,r,g,b,a）构成参数向量
- CUDA forward/backward 可微渲染 → 对全部参数求梯度
- 融合 GPU Adam（梯度裁剪 + 参数钳位 + STE 量化），cosine 学习率
- **循环重定位**（关键创新）：按梯度历史稳定性 + 每形状可见性（混合权重和 <2 判定"被埋没/太小"）找出废形状 → 搬到误差图 top-K 区域顶层 → 冻结其余做局部优化 → 重置 Adam 状态
- 保存历史最优快照（异步 pinned memory 拷贝）
- 预处理：importance map = 边缘(膨胀×3)×显著性(clamp 0.5~15)×暗部(×1.5)；调色板量化色

## 4. JSON schema（geometrize 兼容，注入器 interchange）

```json
{"shapes": [
  {"type": 16, "data": [x, y, w, h, angle], "color": [r, g, b, a]},
  {"type": 1,  "data": [x, y, w, h],        "color": [r, g, b, 255]}
]}
```
- `type`: 1=矩形, 16=旋转椭圆, 228=软椭圆；`color[3]` = 不透明度×255（228 专用）
- 拟合坐标系 → 游戏坐标：见 §1 写入规则

## 5. RE 方法论（摘自 bvzrays DEVELOPMENT.md，为 FM2023 复用）

1. 干净模板 → dump 每层 0x140 blob
2. 游戏内**只改一个属性**（如某层颜色/形状）→ 再 dump → 逐字节 diff
3. 同会话小范围写验证（copy-shape-from-layer 到 1..N 层）
4. 记录失败模式：完整 blob 恢复=崩；0xA8 跨会话=崩；实验写后不存档、重启游戏
5. 已知未知项：0x58 语义、0xA8 能否从干净会话形状库推导、脏标记字段

## 6. 自研架构草案（fv-* 项目）

**选型原则**：目标是"真正的高度优化"，语言只是手段，按组件热点特征选：CPU 热循环用 Rust（rayon + core::simd，机器码与 C++ 同级，工具链一体）；GPU 用 **Vulkan compute**（GB10 ARM Win 和 5090D x64 两台都有驱动，CUDA 在 ARM Windows 侧不可行）；JSON/CLI 等冷路径怎么简单怎么来。注入器的内存扫描是 I/O 型顺序读，不值得做英雄主义优化，SIMD u16 扫 + 多线程区域读取即可。UI 不做，CLI + 预览 PNG 起步。

**性能工程分层**（顺序：算法 > 数据布局 > 并行 > 微优化；每层以剖析数据为准，先建基准后动手）：
- 算法级
  - 贪心评分：候选椭圆为凸形，每扫描线恰一段 → **多张积分图（SAT：target、target²、canvas、canvas²、target·canvas）按行段求和**，把单候选评估从 O(bbox 面积) 降到 O(椭圆高度)；百万级候选采样才成为可能
  - 粗到细两级评估：1/4 分辨率全量筛 → top-K 候选全分辨率精评
  - 增量画布：接纳形状后只重算其 bbox；canvas 系 SAT 每次接纳全量重建（≈1M 像素级，3000 次可承受）
  - 候选生成：SOA 随机流 + 计数式 RNG（xoshiro/PCG），线程无关可重放
  - 可微路线：椭圆按 tile 分桶建 per-tile 活动列表，forward 每 tile 线性合成；反向梯度 subgroup 归约 + 融合 Adam 单 kernel 常驻 GPU，消灭 CPU↔GPU 往返（vinylizer 的 fused_adam_update 即此思路）
- 数据布局：目标/画布 SoA u8 平面；积分图独立 f32/i32 缓冲；GPU 参数常驻显存 + pinned staging + 异步双缓冲快照
- 并行：候选评分无共享跨线程/跨 workgroup；内循环 u8x16 差分绝对值累加（AVX2/NEON 双路径，编译期分发）
- 微优化（最后）：branchless 内循环、LTO/PGO、SPIR-V 特化常量、持久线程
- 基准对手：geometrize-gpu 实测 ~435ms/椭圆（i5 核显）/ ~37ms（AIMAX395+8060S）；M1 起建 criterion 基准，逐里程碑对比

```
forza-vinyl/
├── refs/                    # 参考仓库（只读）
├── STUDY.md                 # 本文件
└── fv/                      # Cargo workspace
    ├── crates/fv-fit/       # 拟合器（库+CLI）
    │   ├── preprocess/      # image 解码、Sobel/显著性、调色板量化（自实现）
    │   ├── greedy/          # 贪心后端：rayon 多线程 + SIMD(u8x16 误差累加)
    │   ├── diff/            # 可微后端：wgpu/ash Vulkan compute（参考 render_kernel.cu 思路）
    │   └── out/             # geometrize JSON + 预览 PNG（serde_json/image）
    ├── crates/fv-inject/    # Windows 注入器（windows-rs）
    │   ├── scan/            # §2 u16 扫描 + 严格校验
    │   ├── write/           # 逐字段写入 + 读旧值回滚 + 抽样人工确认
    │   └── layout/          # 层布局 = 数据文件（仿 FD6 patterns.json，游戏更新热改）
    └── crates/fv-common/    # Shape/Color 类型、坐标换算、JSON schema
```

**里程碑**：
1. **M1 ✅（2026-10-24 完成）**：贪心 CPU 拟合器 `fv` workspace（Rust，约 2000 行）
   - 模块：`fv-common`（Shape/JSON 交换格式）+ `fv-fit`（raster 闭式解光栅化 / rowsum 12-map 行前缀和 / candidate xoshiro128++ / engine 贪心主循环 / render 预览 / CLI）
   - 核心算法：候选评估 O(椭圆高度)（多张积分图按行段求和）+ **误差块加权中心采样**（贪心版"重定位"，75% 加权 + 25% 均匀探索，16px 块 CDF 反演）+ 确定性爬山精修 + 三振出局早停
   - 实测（GB10 ARM，10 核）：单候选评估 **982ns**（≈1M/s/线程）、rowsum 全量刷新 512µs/接纳、ellipse_rows 430ns、seg 查询 0.5ns
   - 真图（328² meme 图，透明底）：300 层 × 10 万采样 = **21s**（71ms/层，vs geometrize-gpu 独显 37ms/层），MSE 2437→220（PSNR 14.7dB@opt / 21dB@原图渲染）；渲染管线与引擎画布**逐位一致**（245.20==245.20）
   - **resume 断点续拟合**（--resume，前缀逐位保留）+ 规模曲线：100k 采样 271 层饱和 @24.2dB；1M 采样 500 层仍在降 @26.7dB——**贪心纯色椭圆吃不满 3000 层预算，需 M3**
2. **M2 ✅（2026-10-24 完成）**：自研注入器 `fv-inject`
   - 架构：MemIo trait（MockMem 可离线全链路测试 / WinMem 实机）+ 层布局数据文件化（`--dump-layout` 导出，游戏更新只改 JSON）+ 严格校验（任一字段可疑整体拒绝）+ 逐字段回滚 + dry-run/抽样人工确认/`--yes` 三段安全流程 + `--center` 画布居中
   - Mock 全链路 13 测试全绿（定位+诱饵排除、字节级写入、回滚恢复、坏指针零写入）；实机手册见 `INJECT.md`（5090D 上操作）
3. **M3 尖刀+集成 ✅（2026-10-24 完成，正式可微渲染留后续）**：Vulkan compute GPU 评估
   - 环境：wgpu 26 + Vulkan + SHADER_INT64 在 N1X（GB10/Blackwell）验证可用；注意 all-backends 枚举会挂死在 GL 路径，须限 Vulkan（5090D/Arc/Radeon 同理全覆盖）
   - 内核 `eval.wgsl`：单 invocation 闭式解行区间 + 12 map 行段求和 + i64 全程精确；与 CPU 交叉验证 **4966/5000 逐位一致**（其余 ±1px 边界差）
   - 引擎集成：`BatchEvaluator` trait（fv-fit 定义、fv-gpu 实现，无依赖环），GPU 路径单线程生成候选→整批评估→CPU 精修；每层接纳后 19MB 前缀和重传
   - 吞吐：单候选批 **248 万候选/秒**（=CPU 单线程 14.6×）；端到端真图 **300 层 × 100 万采样 = 49s**（mse→180 / PSNR 25.58dB / 300 层满额，渲染仍与记账逐位一致）vs CPU 同采样 ~377s = **7.7× 全任务加速**；`fvfit-gpu` 二进制与 fv-fit 同参数
   - 后续优化空间：候选生成上 GPU、maps 增量更新、subgroup 归约、可微渲染（vinylizer 式 Adam + 重定位）——预计再有一个数量级
4. **M4**：FM2023 RE 独立立项（方法论 §5，从 dump/diff 工具链开始）

**工程状态（2026-10-24 08:00）**：4 crate（fv-common/fv-fit/fv-inject/fv-gpu）约 3200 行，**49 测试全绿、clippy 0 警告**；双硬件约束满足（ARM NEON 本机全程开发验证，x64 5090D 待实机注入验证）。

---

## 8. M3-P1 遮挡感知精确抛光（2026-10-24，纯自研算法）

**动机**：贪心形状在放置时优化，其后被上层遮挡导致有效区域缩小、参数过时；贪心 ~700 层饱和后采样找不到改进。

**算法**（`fv-fit/src/polish.rs`，别人做不了——依赖我们的 i64 精确积分图）：
- 实心覆盖语义下每像素归属 topmost 形状：建 top 图（O(N·area)），自底向上遍历每形状
- **可见集 R_i = {p : top[p]==i}** 上做精确重拟合：颜色 = R_i 目标均值（闭式）；几何 = 扰动爬坡，候选误差用"受影响集 A = R_i ∪ 候选可见覆盖（top≤i 或背景）"精确计账
- 记账公式：delta(c,k) = C1 − H(c,k)；C1 = Σ_{R_i}[D(old)−b]（=删除的收益），H = Σ_{cov_aff}[D(k)−b]。只接受 H < C1，**总误差保证单调不增**
- 无效层（R_i 空 / 删除更优）直接删除；增量维护 under 画布 + bbox 局部 top 重刷
- 接入：`--polish N`（FitConfig.polish_sweeps）；resume 血统标记（src/changed）保证未动初始层输出逐位保真

**开发中抓到并修掉的两个真 bug**：
1. 初版记账只算 R_i，候选几何**扩张到别的形状地盘的像素没计费** → 真图 mse 179.95"抛光"后反升到 613。修法 = 上述 A 集合模型（这是解析推导出错被真图数据当场证伪的案例）。
2. resume 的 shapes_opt 只含新层，polish 看不到 661 个基础层 → 引擎账 62.1 vs polish 渲染 2458 的账本分裂。修法 = 全量账本 + src/changed 血统。

**实测（ayylmao 328²，统一裁判 eval_design）**：

| 阶段 | 层数 | 引擎 mse(opt448) | PSNR |
|---|---|---|---|
| 贪心 300 | 300 | 179.95 | 23.92 dB |
| +polish 4 | 297 | **114.11**（−36%） | **24.48 dB** |
| 贪心 701（2500 目标早停） | 701 | 129.50 | 25.18 dB |
| +polish 6 | 661 | **62.23**（−52%） | **25.96 dB** |
| +resume 续拟合（饱和仅+26层）+polish 4 | 687 | **55.75** | **26.14 dB** ← 全场新高 |

vs vinylizer 同机最佳（2500 层 23.80dB）：**+2.34dB，层数 1/3.7**。634/2016/841 次接受改进全部是真改进（最终渲染与引擎记账逐位一致闭环验证）。

**意义**：place→polish→add→polish 交替循环成立；M3-P2（GPU 可微渲染/228 软椭圆透明层）叠上去之前，纯 CPU 侧的质量潜力已再挖出一大截。

---

## 9. M3-P2 软椭圆（228）混合层（2026-10-24，纯自研）

**理论核心**：实心覆盖在纹理区是罚分（平色盖高频必涨误差），软层是混合语义（往均值拉=降方差）——这是突破 ~700 层饱和、填满 3000 层预算的钥匙。数学：o（不透明度）与 k（颜色）在误差中退化等价，**固定 o=1 只优化 k**，k* 与 delta 闭式可解；alpha 语义 = 游戏形状 228 的径向衰减（解析牛顿式）× o。

**实现**：`fv-fit/src/soft.rs`（CPU 混合合成器 + f64 参考评估器）+ `eval_soft.wgsl`（**两遍扫描** GPU 内核：第一遍 Σβ²/Σβg 求 k*，第二遍逐像素累加 Δ²−(βk−g)²——单遍巨数相减公式在贴合区会 f32 灾难性抵消）+ engine 软通道（GPU 批评估 → 快照试混 → 实测 delta>0 才接纳否则回滚——u8 舍入防虚账）+ render/JSON/注入器全链 228 支持（fv-inject 从 day 1 就带 228）。

**调试战役**（两个子代理并行合围 + 主线二分，全部落网）：
1. 软接纳分支缺 `sums.refresh_canvas + g.on_accept` → GPU 对 stale 画布评估，预测全废（子代理 W 用 5 个生命周期测试独立复现同一结论；replay 测试以 dump 现场证明 GPU/CPU/实测三方其实一致）；
2. 软内核 plane_len 误用实心的 h·(w+1)（平面寻址差 1 行/通道错位）——**所有测试都在 448 上做而引擎在 328 上跑**的覆盖漏洞；
3. WGSL 保留字 `target`、`i64` 字面量、delta=1 边界带（f32/f64 取整口径，非竞态）。
方法论沉淀：数据校验和回读（GPU 缓冲对账 CPU）+ 现场 dump 重放测试，是 GPU 管线分歧的通用二分法。

**真图战果（ayylmao 328²，统一裁判）**：

| 阶段 | 层数 | 引擎 mse | PSNR |
|---|---|---|---|
| 贪心+polish | 687 | 62.23 | 26.14 dB |
| +soft 1200 | 1497 | 63.30 | 27.50 dB |
| **+soft-refit（守卫式）** | 1497 | **61.03** | **27.58 dB** ← 全场新高 |

注：refit 引擎 mse 略优于 soft 后的 63.30（软层引擎 mse 比 polish 后高是正常的——软层的收益大量来自对原分辨率渲染的逐像素逼近，opt 分辨率记账与裁判渲染口径不同）。refit 收益温和（199/1200 接纳）的原因：软栈刚被贪心放置、每层 k 在放置时已对彼时画布最优，只有后续层造成的漂移可修；当画布被其他操作（语义分层、预算调整）扰动后 refit 价值上升。

3000 层预算从"贪心 700 层饱和"变为"可填充"（soft 层持续有效）。

---

## 10. 路线图 v2：双轨制（2026-10-24 定）

应用层全部共用（fv-common 格式 / fv-inject 注入 / render 预览 / CLI / engine 的画布-评估-接纳基础设施）；两轨只在"智能来源"上分叉。**接口契约：L1 产出逐像素 alpha matte + 语义层掩码 → M1 消费（逐层拟合与预算分配）。**

### M1 算法轨（无模型依赖）
- M1.1 ✅ 贪心 + O(高度) 积分图评分 + 误差加权采样（27 测试）
- M1.2 ✅ 遮挡感知精确抛光 polish（§8）
- M1.3 ✅ 228 软椭圆混合层 + GPU 两遍扫描内核（§9，27.50dB）
- M1.4 ✅ 软栈颜色重拟合（refit_soft_stack）：**接纳守卫式**前向坐标下降。**关键算法认知（真图击穿教训）**：混合语义下改 k_i 会以 (1−β_j) 衰减穿透后续所有层的混合链——与覆盖语义的"后层截断"不同，E_i 上的闭式改进不保证全局改进（真图实测：790 层无守卫重拟合，mse 63.30 → 169.69，差 2.7 倍）。因此闭式解只作提案生成器，每层提案后**全栈重混合实测总误差，变差即回滚**（重混合 O(Σ触达面积) 毫秒级），守卫保证单调不增。另一教训：β 小的区域闭式 k 会冲饱和钳位（blend 灰→绿 @β=0.3 需 k=433）——正确行为。默认开启，`--no-soft-refit` 关。
- M1.5 ✅ 自适应预算（--auto-soft）：soft_layers==0 时实心三振出局后剩余层预算自动给软层（layers 语义变总预算）；FH6 场景 `--layers 3000 --auto-soft` 一步填满。
- M1.6 混合循环与预算策略：place→polish→soft→polish 交替、按区域误差动态分配层预算、软/实比例自适应（部分由 auto-soft 覆盖）
- M1.7 ✅ 描边放置（stroke.rs）：Marching Squares（16-case + **鞍点中心消歧**）→ 端点哈希链化 → 等弧长重采样沿链放椭圆（长轴=切向、颜色=局部均值、长链优先、零 RNG）。实现中抓到两个经典坑：**三角在内 case 全组写错**（all-but-X 应等价 X-单独几何，初版全错位致每 3~5 段断链，靠"断点审讯"探针定位）；鞍点两拆分方向需四角均值消歧。圆场验证：单闭合链 133px（周长 125.7 的 1.06 倍）。待接入：field 用 L1 边界带或误差图（目前 demo 用 alpha/亮度场）。
- M1.8 ✅ GPU 可微渲染器 + 融合优化器（2026-10-26 深夜）：`fv-gpu/src/diff/`（mod 契约 + fwd 前向 + bwd 反向 + refine Adam 驱动，M18_DESIGN.md 为数学唯一事实来源）。前向双扫描单内核（自底向上合成写 Cbelow 快照 + 自顶向下透射写 Tabove）；反向一 workgroup 一层、禁 atomics 固定树归约、颜色梯度走 β·T 直通、几何走 −τβ(1−β)·dq/dparam 链。**验收：FD 中心差分 40 参数全部 rel≤5.5e-4**（门槛 1e-2）；前向 vs f64 参考 τ=4/64 max|diff|=0.0000、τ=512 与硬光栅零超差；符号方向测试绿。真图 ayylmao 300 实心层：**25.46→26.55dB（+1.09dB）2.2s/300 迭代**（≈7ms/迭代，官方裁判确认 26.54）。CLI：`fvfit-gpu --refine-json X.json <img> --diff-iters N [--diff-tau0/lr-geo/lr-col]`（混合栈自动拒绝，只精修纯 type16 栈）。开发中抓到 4 个数学/工程 bug：① 设计文档 dq/dcy 多外提负号（cx 两项同号可提、cy 混号不可提，FD 符号翻转暴露，文档已勘误）；② 反向 bbox 半径少算 sigmoid 外尾（应为 max(a,b)·√(1+30/τ)+1，τ=4 时带外伸 ~2.6× 半轴，砍尾致 2% 系统偏差）；③ 反向像素循环从小数 bbox 起步与前向 0.5-center 网格错开 frac(x0)（量化修复，靠"s0/s2 对齐吻合、s1/s3/s4 错位"的规律定位）；④ Adam 误写成梯度上升 + θ 与像素共用学习率（0.2rad=11.5° 单步毁全栈→θ 独立 lr×0.05）。τ 退火起点教训：τ0=4 软目标对高密度实心栈开局伤害过深（25.5→21），τ0=32 + lr 调小后单调收敛；**评测永远用硬光栅 i64/精确口径，可微渲染只出梯度**。
- **裁判 bbox 勘误（2026-10-26，波及全部历史 PSNR）**：eval_design 的旋转椭圆外接界误用 max(|rx·c|,|ry·s|)（正确 √(rx²c²+ry²s²)），45° 旋转层四角漏渲染 ~1.8dB。修复后与引擎行 span 光栅逐位对账一致（solid300 MSE 184.91 = 引擎内部值）。**历史数字全部低估，修正记分板（ayylmao 328²，bg 须与拟合一致——auto 拟合的栈用 --bg auto 评，否则跨实例假灾难如 9473 vs 61）**：旗舰 1497 层（297 实心+1200 软，guarded refit）**30.28dB**（mse 61.02 与引擎内部 61.03 吻合；原记 27.58）；300 实心 25.46；+M1.8 精修 26.54。vinylizer 的 23.80@2500 同为旧裁判所测（其真实值会上浮，但差距量级不变）；其产物未存档无法重测，备注于此。

### L1 模型轨
- L1.1 ✅ 语义分割接入（地基）：**MODNet 人像 matting**（HF `Xenova/modnet`，Apache-2.0，ONNX 25.9MB）经 Rust `ort` 2.0.0-rc.10（load-dynamic）+ onnxruntime 1.22 win-arm64 CPU EP 端到端跑通：328² 冷 209ms/热 184ms，matte 0-255 全域正常。坑：v1.20 dylib 与 ORT_API_VERSION 22 不配对（须 1.22）；gh-proxy 大文件断点续传（curl -C -）。产物：`fv-seg` crate（`segment(path)->Matte` + probe 示例，clippy 0）、dll 在 deps/ort、模型在 deps/models。
- L1.2 ✅ 语义分层管线（fvseg）：matte → plan_layers 三掩码 → **"当前渲染填充不关心区域"**的逐层拟合技巧（bg 层：主体区填 bg 色 → fit；subject 层：背景区填已渲染 bg 画布 → fit；edge 层：band 外填已渲染画布 → fit + soft）→ z 序拼接 bg→subject→edge。非本层区域与画布零残差 → 贪心自然不放形状，无需改引擎。CLI `fvseg <img> --matte m.png --total N --soft M`。端到端演示（合成 256²：渐变背景+彩色环主体，统一裁判）：分层 288 形状 **23.92dB** vs 盲拟合 257 形状 23.56dB（+0.36dB 且预算路由语义化；旧裁判数字）。**`--gpu` 已接通（2026-10-26）并修复质量差**：三层全走 fit_gpu/resume_gpu，Vulkan 探测失败自动回退 CPU。**根因不是 ±1 评分噪声，而是 GPU 分支用独立单流 RNG 采样——候选分布本身与 CPU 分块流不同**（bg/subject 层 mse 差 50 级、subject 提前停 59/86）。修复：GPU 分支复刻 CPU 分块 RNG 方案（候选集逐位同源）+ GPU 报告 top-4 用 CPU 精确评分复核 + 全负批复核最接近者防假三振。修复后同参数（demo 288 预算 seed 7）：**GPU 与 CPU 逐位一致**（290 形状、subject mse 144.0、edge 8+72、26.30dB 全同），耗时 CPU 501s vs GPU ~10-15s（**~40× 提速、零质量差**；评测时若有其他 GPU 进程并行会翻倍）。CPU 501s 里 edge 层占 498s。
- L1.1b ✅ 模型注册表（SegModel）：`SegModel::{Modnet, U2Net}` + `segment_by()/segment_image_by()`，按模型走各自预处理（U2Net：320² 固定、ImageNet mean/std、输出 min-max 归一；MODNet：512 短边÷32、[-1,1]、含 sigmoid）。**u2net.onnx 177MB**（frankminors123/U2Net_ONNX，Apache-2.0，hf-mirror 下载）补位通用显著性——非人像图不再被 MODNet 硬扛。`FV_SEG_U2NET` 环境变量覆盖路径；旧 `segment()` 默认 MODNet、按路径后缀自动识别（兼容）。视觉验证（ayylmao 328²，主体实为**一辆涂装车**，外星人是车身图案）：U2Net matte=紧车侧剪影（mean 77），MODNet=松含地面亮带（mean 139）——两模型都没抠错，U2Net 对非人像明显更准。probe 示例 `examples/u2net_probe.rs` 并跑双模型出对比图。fv-seg 13 测试绿、clippy 0。
- L1.2b 真图端到端配对实验（2026-10-26，ayylmao 328²，统一 bg 030303 + 统一裁判）：fvseg 新增 `--model modnet|u2net`（接 L1.1b 注册表，非人像必选 u2net）。首轮配对暴露结构性缺陷：**分层 540 形状 22.88dB vs 盲拟合 904 形状 24.97dB（盲赢 2.1dB）**——根因不是分割质量，而是软层只画 3px 边界带，实心三振留下的主体内部残差无人认领，盲拟合的 auto-soft 把 699 软层铺进主体内部。**修复：fvseg 重构为终段全局软相**（bg→subject 实心→edge 实心→soft 全域按残差放置；resume 层数=实心数 → 实心循环即跳、直接进引擎软相；引擎侧补了 resume 层数满时 grid 未重建/GPU 未 on_ready 的坑）。修复后配对：**分层 940 形状 24.94dB vs 盲拟合 904 形状 24.97dB（持平，Δ-0.03dB）**。诚实结论：语义路由在近纯色背景上省预算（bg 仅 16 形状 mse 0.6），但主体占满画面的图无路由红利；合成演示（结构化背景）的 +0.36dB 优势场景=复杂背景/预算受限/需要掩码保护的图。另：引擎软相入口无条件 grid.rebuild+on_ready（修 polish 后陈旧 + resume 即跳两类隐患）。
- **A/B 管线判定（2026-10-26 凌晨，新旗舰 30.83dB）**：全管线对照（ayylmao 328²，1M 采样，预算 3000 auto-soft，统一裁判 bg auto）——**B 链（M1.8 精修基础栈 26.97@300 → resume 续拟）：2177 形状 30.83dB（mse 53.77）**；A 链（盲跑同参）：2141 形状 30.21dB（mse 62.03）；旧旗舰（带 polish，无精修）：1497 形状 30.28dB。**M1.8 基础栈优势 +1.22dB 经 ~1850 层贪心续拟后仍保留 +0.62dB——可微精修的价值在管线级成立**；A 略低于旧旗舰系旧旗舰带 polish 而 A 未开。新旗舰已入 bundle（fvB_full.json，逐位复核 30.83）。注意 N1X 上 1M 采样 ×3000 预算全管线 ≈ 2.5h（GPU 96% 满载），5090D 应缩到 ~15-20min。
- M1.8 补充（同日）：fvseg --gpu 质量差根因修正认知：**不是 ±1 评分噪声而是 GPU 分支独立单流 RNG——候选分布与 CPU 分块流不同**；修复为候选集逐位同源 + GPU top-4 CPU 复核 + 全负批复核防假三振后，**GPU ≡ CPU 逐位一致（290 形状/mse/PSNR 全同），~40× 提速**。
- M1.7 接入落地（同日）：fvseg `--strokes`——edge 层改画家式描边（field=边界带强度场、颜色=原图局部均值、250 笔瞬置 vs 贪心 36s+）。真图配对：描边版 1111 形状 24.55dB vs 贪心 edge 版 940 形状 24.94dB——**风格化代价 0.4dB**（旗标文档如实标注）；两模式共存可选。
- L1.3-lite ✅（2026-10-26 晚，轻负载）：PlanParams 参数化规划（thresh_bias/smooth_rounds/band_radius/edge_frac/subject_gain/subject_floor，默认=原行为零破坏）+ fvseg `--tune` 坐标搜索（13 次评估 ×43s 小预算 GPU 短脉冲）。demo 图验证：**band_radius 3→5 唯一胜出 +0.2dB**——"对椭圆重构友好"的规划参数空间确实存在且可搜索。这是 L1.3 神经版的先声：神经微调要赢的基线就是这组手调参数。方法说明：搜索用小预算（--total 240 --samples 5000），最优参数再放大重跑才公平。
- **5090D 预铺包（2026-10-26 凌晨）**：`forza-vinyl/bundle-5090d/`——**ARM64 本机交叉编译 x86_64-pc-windows-msvc 一次通过**（rustup target add + VS ARM64 工具链原生交叉链接，2m06s 全依赖图；PE32+ 已验证）。内容：四 exe（fv-inject/fvfit-gpu/fvseg/eval_design）+ 三份设计 JSON（26.97/30.28/24.94dB，bg 口径已核对）+ modnet/u2net.onnx + **onnxruntime x64 dll**（gh-proxy 大文件限速 10KB/s 的坑 → npmmirror 的 `onnxruntime-node` npm 包 6MB/s 秒下，内含 napi-v6/win32/x64/onnxruntime.dll）+ INJECT.md + README（含 5090D 快速开始四连命令）。**WoA x64 模拟实机验证**：eval_design x64 在 N1X 上跑出与原生逐位一致的 PSNR（跨架构一致性）；**fvseg x64 全链路烟测通过**（ORT 推理 + U2Net + 分层拟合 + 软相，模拟 2558s——真机原生速度，467 形状 23.68dB 正常产出）。设备窗口纪律：到机即测，零闲置。
- L1.3 （研究项）vinyl 感知微调：把 M1.8 的可微渲染器放进损失环（matte → 拟合 → 可微重建 → loss），微调小型 U-Net 输出"对椭圆重构友好"的 matte。文献无先例；依赖 M1.8 就绪；N1X 统一内存可训

### 历史里程碑对照
旧 M1(拟合)/M2(注入)/M3(GPU) → 归入共享底座与 M1.1–M1.3（全部完成）；旧 M4(FM2023 RE) 保持独立立项，不占轨。

---

## 7. vinylizer 同机实测（2026-10-24，GB10 ARM + CUDA 13.4）

**构建**：源码 refs/vinylizer + OpenCV 4.11 自编（ARM64，`-DCPU_BASELINE=NEON -DCPU_DISPATCH=`——**Windows-on-ARM 不上报 DOTPROD/FP16，必须压基线否则运行时 fatal**）+ Qt 6.11.3 ARM64（aqtinstall）+ CUDA sm_121（nvcc 13.4 原生支持）。VS 生成器的 CUDA 集成对 VS18 失效，须 vcvars+`Ninja`。安装的 OpenCVConfig.cmake 需两处补丁（ARM64 探测 + 平铺 lib 布局），**每次 `cmake --install` 会覆盖补丁**。构建脚本：`build_vinylizer.bat` / `build_vinylizer_cli.bat`；依赖在 deps/（opencv+qt，约 4GB）。

**同台对比**（ayylmao 328²，统一裁判 `fv-fit --example eval_design`——228 软椭圆 alpha 按其源码牛顿法 LUT 复刻）：

| 方案 | 实际层数 | 耗时 | PSNR |
|---|---|---|---|
| vinylizer c3 @300 | 300 | **2.3s** | 20.27 dB |
| vinylizer c10 @300 | 300 | 7s | 19.94 dB |
| vinylizer c3 @2500 | 2500 | **16s** | 23.80 dB |
| vinylizer c10 @2500 | 2500 | 55s | 23.50 dB |
| **fvfit-gpu @300**（1M 采样） | 300 | 49s | **23.92 dB** |
| **fvfit-gpu @2500**（1M 采样） | 701（三振早停） | ~115s | **25.18 dB** |

**结论**：
- 我们赢**质量/每层密度**：300 层即打平他们 2500 层（23.92 vs 23.80）；701 层 25.18dB 全场最高。
- 他们赢**速度**：2500 层 16s vs 我们 701 层 ~115s（可微渲染的迭代式复杂度 vs 我们的采样式，结构不同）。
- 他们的价值路线：层数多时持续有效（我们贪心 1M 采样 ~700 层饱和）；我们的价值路线：少层数高保真 + M3 上可微渲染补满预算。
- **GB10 鲁棒性问题（实测）**：显存模式启发式非确定性段错误（2500c3 一次崩/重跑好——统一内存的空闲显存上报波动大）；GUI 主线程画布冻结（后台线程优化时可视化不刷新）；cycles=10 质量反低于 c3（best-snapshot/重定位怪癖）。
- 值得偷师：228 软椭圆的解析 alpha S 曲线（`alpha_228_newton`，常量 C1/C2/C3）——适合我们做半透明层扩展；K-Means 调色板预处理。

**GUI 实测过程**（computer-use + PowerShell 合成输入 + 视觉分析）：启动→加载图→设参数→开始生成全程可驱动；CUDA 优化确认在跑（GPU 占用+CPU 时间增长），可视化冻结但 CLI 日志证实迭代正常（mse 1997→~270@600 iter）。CUA helper 曾在原生文件对话框上无响应，PowerShell SendInput 兜底成功。

**风险**：ToS/封号（内存写在线游戏）；游戏更新→偏移漂移（layout 数据文件缓解）；注入必须在能跑游戏的 x64 Windows 上。

## 12. 多图基准（2026-10-26 晚，FH 社区涂装真实语料）

语料：forza-painter 内嵌 1400 张社区涂装模板（单色剪影 on 透明，游戏内再上色）。**直接拟合退化**（bg=字形均值=目标，mse=0）→ `flatten` 工具智能压平（字形亮度 ≥128 → 黑底，否则白底；fv-fit examples/flatten.rs）。精选 12 类各一，`bench/bench.sh [layers] [samples] [solid|soft]`，统一裁判 --bg auto，GPU 路径，全 12 图 35s（轻负载）。

| 图 | solid@300 | soft@300 | Δ |
|---|---|---|---|
| primitives（基元剪影） | 43.90 | 46.04 | +2.14 |
| paint_splats（泼溅） | 41.09 | 41.09 | 0（双双三振早停） |
| gradient_shapes（渐变） | 37.31 | **42.64** | **+5.33** |
| lower_letters（小写字母） | 36.44 | **41.20** | **+4.76** |
| upper_letters（大写字母） | 35.21 | 39.04 | +3.83 |
| stripes（条纹） | 22.59 | 26.90 | +4.31 |
| flames（火焰） | 23.86 | 25.82 | +1.96 |
| community_vinyls_1 | 22.99 | 25.85 | +2.86 |
| tribal（部落） | 19.40 | 20.53 | +1.13 |
| nature（自然插画） | 18.29 | 18.29 | 0.00 |
| racing_icons（赛车图标） | 17.50 | 18.22 | +0.72 |
| community_vinyls_3（细密插画） | 14.75 | 14.96 | +0.21 |
| **均值** | **27.78** | **30.05** | **+2.27** |

结论：**软椭圆（228）增益高度 regime 相关**——平滑渐变与细结构（字母/条纹边缘）+4~5dB；已饱和细密插画 +0~0.2（实心已榨干、软层无处着墨）。ayylmao 单图结论（软层大赢）不可外推到插画类。**后续所有质量主张以此 12 图基准为准**（ayylmao 另列"复杂照片类"参考）。低分难图（community_3 14.75/nature 18.29/racing 17.50）= 下一步算法攻坚的靶子（高频细节超 300 层椭圆的表达极限，候选：分层分辨率、M1.8 全栈精修、L1 语义路由）。

## 13. B1 尺度-结构感知采样（2026-10-26 深夜，R2 提案落地）

**根因（M19_PROPOSALS §A）**：三难图（community_3 14.75 / racing 17.50 / nature 18.29）的失败不是预算不足——是 sqrt-面积均匀采样对 2-10px 细结构近乎全盲（理想细长椭圆命中 ~2.6×10⁻⁸/候选），且误差 76-94% 集中在墨迹边界 ±2px 带。solid@300 几乎是墨宽中位数的单调函数。

**实现（fv-fit）**：`orient.rs`（一次预处理：chamfer 距离变换量墨宽 p95 → thin spec max=min(原,p95×4)；Sobel 法向场 + 3 轮置信传播进平坦墨迹内部）+ `candidate::sample_structured`（对数均匀长半轴 + 截断 Pareto(1)∈[1,12] 长宽比 + 走向 ±5° 高斯/低置信回退均匀，thin/orig spec 各 50%）+ engine 四个采样点全接入（solid GPU/CPU + soft GPU/CPU，GPU≡CPU 同源机制不变）。`FitConfig.structured` 默认开，`--no-structured` 回旧行为。理论命中率 10⁻⁸→2×10⁻²（6 个数量级）。37 测试绿。

**⚠️ 机器稳定性（10-26 第 3 次蓝屏）**：旗舰链终段 + B1 bench 双 GPU 进程并发 ~2 分钟触发 nvlddmkm 153 风暴 → 重启（与前两次同签名 0x119）。**纪律：本机 GPU 任务严格串行**。重启后 solo 也曾 device-lost 两次（21:57/22:03），小探针恢复后重试成功——驱动 621.01 状态不稳，旗舰终段若再崩则 B1 验证转 CPU、终段转 5090D。

**B1 验证（bench A/B）**：待 GPU 空闲后 12 图 solid 对比（B1 on/off）+ ayylmao/gradient 回归门禁（Δ>−0.1dB）。结果见 §12 表更新。

### B1 验证战报（2026-10-26 深夜，12 图 bench A/B + 照片门禁）

| 图 | 旧 solid@300 | **B1 solid@300** | Δ |
|---|---|---|---|
| stripes | 22.59 | **44.14** | **+21.6** |
| tribal | 19.40 | **36.79** | +17.4 |
| flames | 23.86 | **37.92** | +14.1 |
| community_vinyls_1 | 22.99 | **43.60** | +20.6 |
| lower/upper_letters | 36.44/35.21 | **∞/∞**（82/131 形状 MSE=0 完美重构） | +∞ |
| primitives / paint_splats | 43.90/41.09 | **∞/∞**（118/268 形状完美重构） | +∞ |
| gradient_shapes | 37.31 | 41.10 | +3.79 |
| nature | 18.29 | **24.27** | **+5.98** |
| racing_icons | 17.50 | **25.34** | **+7.84** |
| community_vinyls_3 | 14.75 | **19.19** | **+4.44** |
| **ayylmao 照片门禁** | 25.76 | **28.71** | **+2.95（非但不回退，大涨）** |

**结论**：B1 全图无一回退，远超 R2 预测（+2.5~6/难图）——因为"社区涂装本来就是椭圆艺术"，采样盲区一除即达完美重构（5/12 图 MSE=0）。照片类（ayylmao）同样受益（车身细条纹=细结构）。**B1 设为新默认**（`--no-structured` 可回退）。R2 的预测模型修正：vinyl-art 域的"理想椭圆"密度比自然照片高 2-3 个数量级。

### MVE-0 双域判决（L1.3 go/no-go，post-B1 复测）

- **vinyl 域（30 图，fvseg --tune 全管线坐标搜索，budget 240）：GO**——headroom 均值 **+0.994dB**、p50 0.372、**p95 4.07dB**、40% 图 ≥0.5dB（双门槛 p95≥0.5 且 ≥30% 全过）。样本亮点：primitives_30 +8.71、upper_letters_1 +4.07、lower_letters_5 +2.33。B1 之后规划参数仍有 ~1dB 可拧。
- **照片域（DUT-OMRON 10 图）：止损**——headroom 均值 0.161、max 0.331，全体 <0.5dB。真实照片的 matte/PlanParams 空间拧不出水；照片类直接 B1 盲拟合即可（L1.2b 持平 + B1 +2.95 已覆盖）。
- **L1.3 立项范围收敛**：神经微调（C0 PlanParams 回归器 → C1 matte U-Net）只做 vinyl-art 域（= 工具目标用户图）；照片域关闭。语料/协议见 L13_RESEARCH.md §D/E（S1 域改用本地 1400 张 vinyl 语料）。

### 新旗舰时间线（ayylmao，统一裁判 bg auto）

| 版本 | 配方 | 形状 | PSNR |
|---|---|---|---|
| 旧旗舰 | 300 solid + polish + 1200 soft + guarded refit | 1497 | 30.28 |
| B 链 | refine@300 → resume 3000 auto-soft | 2177 | 30.83 |
| **fc 链（B3 交替）** | refine@300/1000/2000 → 3000 auto-soft | 2091 | **31.16** |
| v3 链（B1 采样全链） | 同上但 B1 采样 | ? | 运行中 |

### 旗舰终榜（2026-10-27 00:20 更新）

| 版本 | 配方 | 形状 | PSNR |
|---|---|---|---|
| 旧旗舰（裁判修正后） | 297 solid + 1200 soft + refit | 1497 | 30.28 |
| B 链 | refine@300 → 3000 auto-soft | 2177 | 30.83 |
| fc 链（B3 交替） | refine@300/1000/2000 → auto-soft | 2091 | 31.16 |
| **fv3 链（B1 采样全链）** | B1 solid 300 + refine@300/1000 + 3000 solid（软层三振即弃） | **3000** | **36.11（MSE 15.93 / MAE 1.03）** |

**B1 在全链上放大**：solid300 单段 +2.95 → 全链 +4.95dB。3000 层全实心即达 36.11（软层无用武之地——B1 实心已把残差压到软层无法再降的程度）。注意 fv3 链的 r2000 段 refine 因驱动 buffer 限制跳过（见 §13 坑②），补上可能还有余量。

**待办**：diff 快照 u8 量化（支持 >1300 层 refine）；B1+soft 组合的 12 图复测；B1 版 x64 bundle 刷新。

### diff 快照 v2 量化（2026-10-27 凌晨）

**动机**：崩后驱动 buffer 上限 1.29GB，v1 快照（f32）只够 ~1000 层，fv3 链被迫跳过 r2000 精修。

**最终方案（v2.2，两次迭代后的定版）**：cbelow 每像素 2×u32（**u16 定点 ×257**，8B/px）+ tabove **f32 精确**（4B/px）→ 12B/px·层，容量≈驱动 limit÷12B。超出容量的栈走**窗口化 refine**：全栈渲染、仅存 `[lo,hi)` 窗口快照（fwd 内核窗口守卫）——窗口层梯度与全量**精确一致**（不是近似）。容量探测 = device.limits().max_buffer_size×96%。

**两次踩坑（量化设计的教科书案例）**：① u8 RGB 打包（4B/px）→ FD 表面合格（3.4%）但真实栈 refine 崩（−1.6dB 不恢复）：**近收敛区 |c−Cbelow| 小于量化步 → 梯度被噪声劫持**，5 层 FD 测不出（ vivid 色差大）；② tabove u8 → 深层小透射率（T~1e-3）被碾成 0.004 台阶 → 梯度系统性过权。**教训：快照量化必须按"信号动态范围"选位宽，不是按"值域"**——cbelow 信号动态 ~1/257 起步（u16 恰好），tabove 动态 1e-8（必须 f32/f16）。v2.2 FD 回到 5.5e-4（v1 水平）。

**v2.2 验证**：FD/前向/符号三测全绿；**旧采样器栈 refine 完美复现**（25.71→26.69 accepted，与 B 链 v1 行为一致）。**新发现（算法层，与 v2 无关）**：refine 对 **B1 采样的细长椭圆栈失效**（28.71→27.1 崩、guard 拒绝零损伤；τ0=96/半步长同样崩）——归因：τ 软渲染边界带宽度绝对于像素（≈4·min(a,b)/τ），B1 细椭圆（b≈1.2px）在 τ=32 下带宽 0.15px 软目标近乎不可见 → 景观失真。**修法方向：逐层尺度归一化 τ_i = τ0·min(a,b)/2**（带宽恒为短轴的固定比例），未实现。另：驱动 limit 还在漂移（1.29GB→1.0GB，容量 1920→800 层）——窗口化路径正好兜底。**v3/v4 链的 36.11 从未依赖 refine**（其 refine 全被 guard 拒绝，链路靠 B1 采样本身）。

### B1+soft 组合复测（2026-10-27 凌晨）

12 图 @300 预算 `--auto-soft`（B1 采样）：与 B1-solid 相比，多数图软层未启用（实心未三振、数值逐位一致），两图显著增益：**stripes 44.14→55.42（+11.3dB，227 实心+73 软）**、community_1 43.60→45.22（+1.6）。**结论：软层与 B1 正交互补**（B1 攻细结构、软层攻渐变过渡），无回退；"软层被 B1 淘汰"的猜想证伪。

### 隔离实验终案（v2.2 无罪 + B1 栈 refine 失效归因）

对照矩阵：旧采样器栈（--no-structured 造）+ v2.2 refine = 25.71→**26.69 accepted** ✓；B1 栈 + 任何 τ/lr 组合 refine = 崩（guard 拒绝零损伤）。v3/v4 链的 refine 全程被 guard 拒绝 → **36.11 纯粹来自 B1 采样**。待办：逐层尺度归一化 τ（带宽∝短轴）。

## 14. C0 标签工厂与 L1.3 转向（2026-10-27 凌晨）

**基础设施**（全部落地）：`labels_postb1.csv`（25-30 条逐图最优 PlanParams，--tune 日志解析）+ `features.csv`（imgstats 例程：fg 分位/墨宽 chamfer 分位/边界密度/走向各向异性，bg 用**中位数**——均值被 >50% 墨量图污染）+ fvseg `--plan-params` 六参直通（模型评估通道）。

**C0-v0/v1（线性最小二乘 7 特征→6 参数，留出验证，同二进制干净对照）**：v0（post-B1 标签前夜版本）+0.016dB；v1（post-B1 标签）−0.04dB（nature_3 +1.27 命中、community_1 −1.64 误推）——**线性模型到顶**，标签空间交互重、噪声大。

**重大方法论教训（二进制混淆）**：首轮"评估"曾得 +13dB 假象——MVE-0 旧标签产自 pre-B1 二进制，与新二进制跨版本比较无意义。**所有 A/B 必须同二进制同会话**。

**MVE-0 post-B1 重跑（判决反转）**：headroom 不降反升——均值 0.99→**3.16dB**、p50 1.58、p95 10.76（tribal_33 +10.8、upper_letters_5 +8.5）。B1 抬高基线后规划路由**更重要**。

**L1.3-C0 转向（产品结论）**：post-B1 的 `--tune` 每图 **~22 秒**（实测 tears_38：32.5→36.7dB，+4.2dB/22s）即可拿到逐图最优参数——**运行时逐图 tune 直接击败任何回归器**（C0 的前提"搜索太贵"在 B1 时代失效）。C0 神经版降级为研究备选；工程主线 = `--tune` 作为智能默认（生产放大预算后时间线性增长，5090D 上再缩 3-5×）。C1（matte U-Net）同理重新审视：tune 的 3dB 余量里多少来自 matte 空间、多少来自 PlanParams——留给后续消融。

**refine 密度边界补充**：逐层尺度 τ（τ_l=τ·min(a,b)/8，带宽恒为短轴比例）修复了 300 层 B1 栈精修（28.711→28.884 accepted +0.17）但 ≥1000 层仍失效（1000/2000 两段均被 guard 拒绝）——细椭圆不可见问题已治，**千层密度下的软硬目标失配是独立问题**（open）。旗舰 36.11 纯靠 B1 采样 + 300 层精修红利。

### tune v2（2026-10-27 清晨）：扩网格 + 二轮坐标下降

候选网格扩宽（thresh_bias ±16/±32、smooth 0/2/3、band 1/5/7、edge 0.10/0.35/0.45、gain 0.35/0.55/0.65、floor 0.20/0.40）+ 两轮坐标下降（旋钮交互：第一轮定大局、第二轮精修），每图 ~32 评估 ≈ 59s。

**全量 30 图 headroom：均值 4.88dB**（v1 网格 3.16 → v2 4.88），p50 2.63、p95 20.4、max 25.0。样本：upper_letters_5 +13.58、tribal_33 +12.33、tears_38 +6.49（32.50→38.99）。**"逐图 tune 作为智能默认"的工程结论进一步强化**：一分钟搜索 ≈ 5dB 级增益，任何回归器短期无法企及；5090D 上再缩 3-5×。后续吃余量方向：更细网格（连续化）+ CEM + 早停跳过不敏感旋钮。

### 视觉验收（2026-10-27 清晨，visdiff 工具转正）

手工 PNG 解码器两次翻车（行序/位深假设）后改走 image crate：`visdiff` 例程（src vs 预览 → 预览+×8 差异热图竖排）。**旗舰 36.11 视觉结论**：整车 95%+ 面积残差 ≤8 灰阶（×8 放大仍最深蓝），可见误差仅三处——车身轮廓 1px 抗锯齿带（预期）、引擎盖人像脸部高细节区（下阶段"局部密度自适应"的靶子）、贴画边缘；黑背景零残差。PSNR 数字与肉眼观感一致，36.11 可信。`visdiff` 转正为常规质检工具（每次冲榜后必看）。

### git 发布状态（2026-10-27 06:35）

- commit 1（3045e4e）已入库：全部源码+文档+基准（当时扫描 incomplete 放行）
- commit 2（排查链文档+5 个 example 路径守卫）**被 Mimosa L3 拦截**：其规则把"CLI 参数→read_design/save"一律判 path-traversal 高危，不认可 `..` 分量校验守卫。工作树改动完好未丢。下次会话解法（择一）：①扫描器 allowlist 配置；②examples 输入改 stdin/环境变量；③守卫函数换成其认可的标准 sanitizer 模式。

### NEXT_IDEAS.md 落地（2026-10-27 晨，头脑风暴代理产出）

20 点子三档（近战 8 / 中期 6 / 远期 6）。要点：
- **下一冲刺三件事**：① tune v3（候选连续化+CEM+不敏感旋钮早停——确定性收割已证 4.88dB headroom）；② refine 判别包（**E1：对旗舰栈跑 M3-P1 polish（精确硬语义、零梯度）——polish≈0 ⇒ 已收敛假说成立、refine 之“崩”是 guard 对“无处可去”的正确拒绝+梯度推入劣区的并存；polish 有增益 ⇒ 数值性失效**，八成实验零 GPU）；③ fv-bench 裁判库化 + **发布前 license 清洗（refs/ 含无证 vinylizer 源码——已被 .gitignore 挡住未入库，但 STUDY 引用其源码常量处发布前复核一次）**。
- refine 失配新假说 H1（量化噪声地板）经主会话复核**数值有误**（cbelow u16×257 实际步长 1/257≈0.004 灰阶，噪声地板 ~59dB，SNR 充足）——但 E1 polish 判别独立成立，采纳其实验、修正其算术。
- **FM2023 迁移率**：方法论 ~100% / 工具壳 ~90%（layout 数据驱动已备）/ schema 先验 60-70% / 数值 10-20% → 代码复用 ~70%，1.5-3 周 part-time，三道 go/no-go 门（G0 社区调研零成本）。

### CODE_REVIEW.md 落地（全码库评审代理，8258 行通读）

Top 发现（详见文档 §A-G，20 条 action 按 S/M/L 排序）：
1. **行为级 bug★**：fv-gpu `on_ready` 失败后 `eval_batch` 静默全 None → 引擎三振早退 → 无 Vulkan 机器上 fvfit-gpu 写近空 JSON 且 **exit 0**（lib.rs:113-132）——应 fail-fast 或真回退 CPU。
2. **架构★**：fvseg 编排困在 bin（--tune/--plan-params 不可测）——layered_e2e 已复刻整份编排现漂移；库化（M）。
3. **数值雷★**：fv-inject u16 扫描 chunk **接缝漏检**（scan.rs:157，1 字节重叠即修）；inf 坐标可从 JSON 一路写进游戏内存（is_finite 双闸）；soft 三函数不查 is_finite；alpha_228 常量三拷贝（引擎牛顿 vs 裁判 LUT 不一致风险）。
4. fit_impl 542 行巨函数；CPU/GPU 候选循环手工双拷贝无回归锁。
测试缺口 Top5：fvseg tune 零覆盖 / GPU 候选同源 Mock / accept-guard 回滚 / inject protect 边界 / 非有限输入。

## 15. E1 判别实验·爆炸性结论（2026-10-27 深夜）：polish 链 36.11 → 40.24dB

**实验**（NEXT_IDEAS 冲刺第二件，纯 CPU）：对旗舰 fv3_full（3000 层 B1 栈，36.11dB）跑 M3-P1 polish——
| 轮 | 操作 | 层数 | MSE | PSNR | 增量 | 耗时 |
|---|---|---|---|---|---|---|
| 0 | 基线 | 3000 | 15.93 | 36.11 | — | — |
| 1 | polish×1 | 2955 | 9.96 | **38.15** | +2.04 | **1.9s** |
| 2 | 补满+polish×2 | 2989 | 7.64 | **39.30** | +1.15 | ~40s |
| 3 | 补满+polish×3 | 2998 | 6.52 | **39.99** | +0.69 | ~40s |
| 4 | 补满+polish×3 | 2997 | 6.15 | **40.24** | +0.25 | ~40s |

增益几何衰减（比率 ~0.6→0.35）→ 平台。**新旗舰 40.24dB / MAE 0.74（亚像素）**。

**三重意义**：
1. **"已收敛"假说死亡**（refine 失配之谜第五案）：栈上一直躺着 2dB 硬语义余量——B/fc/v3 全部历史链从未跑过 polish，白扔。
2. **refine vs polish 二分法定案**：polish 后的 2997 层栈上 refine 探针仍崩（40.24→30.8@10，guard 拒绝）——梯度法在致密栈的失败与"垃圾层"无关，是方法级边界。**配方改写：B1 采样 + refine@300（稀疏段梯度红利）→ resume → polish 循环（致密段离散记账）**。两者各司其职。
3. **成本收益惊人**：+4.13dB 总增益中 polish 贡献 ~3.2dB，全部 CPU 秒级-分钟级——比任何 GPU 大招都便宜。

**复现**：`fvfit-gpu <img> --resume e{n}.json --layers 3000 --polish {n} --samples 1M --bg auto` 循环（e0=fv3_full）。

**待办**：视觉验收（40dB 的 visdiff）；polish 轮数自适应（增量 <0.1dB 停）；12 图基准上跑 polish 循环验证泛化。

## 16. GitHub 发布闭环（2026-10-29）：api.github.com 通道全量同步完成

**仓库**：https://github.com/Morgador-stack/fv —— 本地 HEAD（0a9f4ed，232 文件）↔ 远端 231 文件逐 blob-SHA 对账一致。

**通道定型**（CQUPT 网络下 github.com:443 push 被 SNI 阻断，api.github.com 可达）：
1. 首建：`tools/gh_upload.py`（Git Data API：blob→tree→commit→ref，240 文件全量）。
2. 增量：细粒度 PAT 无 tree/commit/ref 写权限（Git Data 写端点 403）→ 改 **Contents API 逐文件 PUT**（每文件一个 commit），`tools/sync_finish.py` 收官：
   - 先 GET commits/main + trees?recursive=1 对账（blob SHA 级），只传缺失/内容不同者；
   - 上传前排除 `.github/workflows/*`（需 PAT 的 **Workflows** 权限，非 Administration——ci.yml 仍缺，待 token 加权后补传或网页端手传）；
   - AUDIT_REPORT.md 含 PAT 前缀字面量触发 **secret scanning push protection**（422）→ 本地脱敏后上传通过。

**教训两则**：
- `core.autocrlf=true` 下工作区文件裸读 ≠ HEAD blob（CRLF/LF 转换）——同步必须 `git cat-file blob <sha>` 直读，不能 open 工作区文件（曾致 3 个文件误报"SHA 漂移"被防护拦下）。
- 系统重装后 harness 的 Bash 工具退化成 cmd.exe（MinGit 不在 PATH，MinGit 亦无 bash.exe）——POSIX 命令一律 `D:\Tools\MinGit\usr\bin\sh.exe -c` 或全路径 git.exe + Python 单行/脚本文件；多行 `python -c` 在 cmd 下不可靠，写成脚本文件跑。

**工具链**：`tools/sync_finish.py`（对账+补传一体，`--dry-run` 预览；token 走 `.token`，已 gitignore）。后续任何"同步到 GitHub"直接重跑该脚本即可。

