# fv 工作区代码评审（只读架构与质量评审）

- 范围：`fv/Cargo.toml` + `fv/crates/{fv-common, fv-fit, fv-gpu, fv-seg, fv-inject}`
- 方式：纯源码阅读，零构建零运行。所有结论附 `file:line` 证据（路径相对 `fv/crates/`，`ENGINE` 指 `fv-fit/src/engine.rs`）。
- 体量实测：库代码 8,258 行（fv-common 149 / fv-fit 3,711 / fv-gpu 1,720 / fv-seg 1,605 / fv-inject 1,073）+ WGSL 内核 264 行 + 测试/示例/基准 3,254 行；`#[test]` 共 **94** 个（2 / 51 / 15 / 13 / 13）。
- 总评：这是一个**质量显著高于业余水准**的工作区——几何真源唯一（raster）、i64 精确评分、确定性全序 reduce、血统账本保真、MockMem 全链路离线测试都是教科书级的做法。主要债务集中在四处：**① fv-gpu 失败静默降级会产出近空结果（行为级 bug）；② fvseg 的编排/调参逻辑困在 bin 里，测试只能重写一份；③ `fit_impl` 542 行巨函数 + 同一采样循环的 CPU/GPU 双拷贝；④ 35 处 `Result<_, String>` 无类型化错误**。

---

## §A 架构：模块边界与耦合

### A.1 实际依赖图

```text
fv-common (shape/json, 149 行)
   ↑            ↑
fv-fit ──────── fv-inject        （fv-inject 只依赖 fv-common，隔离最好）
   ↑
fv-gpu
   ↑
fv-seg  （依赖 fv-common + fv-fit + fv-gpu + ort）
```

依赖方向本身健康：不存在环；fv-inject 与拟合栈完全解耦（只通过 geometrize JSON 交换）；trait 反转设计正确——`ENGINE:19` 定义 `BatchEvaluator`，fv-gpu 实现它，引擎对 GPU 无任何依赖。

### A.2 fv-gpu → fv-fit 的类型依赖：方向对，但寄生对象选错了宿主

fv-gpu 库层从 fv-fit 拿三类东西（`fv-gpu/src/lib.rs:5-7`）：

1. `engine::{BatchEvaluator, EvalResults}` —— **trait 反转，方向正确**，这是依赖倒置的标准形态；
2. `raster::EllipseParams` —— 几何词汇类型，**放错了宿主**；
3. `rowsum::RowSums`（经 `planes_i64()` 取 12 张 i64 平面）—— 数据布局契约，**同样放错宿主**。

问题：`EllipseParams` 是全工作区的公共词汇（fv-fit 光栅、fv-gpu 两个内核、fv-seg 编排、注入器坐标换算全都围绕椭圆参数），却住在 fv-fit 里；`RowSums::planes_i64()` 的 `map_slot*3+ch` 布局是 fv-gpu WGSL 内核（`eval.wgsl:17-19`）的寻址契约，却由拟合 crate 私有决定。后果是 fv-gpu 为了两个 struct 拖进 fv-fit 的全部重量（rayon、image，以及——见 §B.1——clap）。

**改法（S）**：把 `EllipseParams`、`RowSpan`/`spans_*` 纯几何件、`RowSums` 的平面布局描述移入 fv-common（或新建 `fv-core`），fv-fit 与 fv-gpu 同时依赖它。`BatchEvaluator` trait 留在 fv-fit 是合理的（它是引擎的扩展点，不是 GPU 的）。

### A.3 fvseg 736 行 bin：库能力被困的最典型案例

`fv-seg/src/bin/fvseg.rs` 里困着三块可复用资产：

- `run_pipeline`（fvseg.rs:388-578）：三层语义编排（背景/主体/边界 + 终段全局软相），这是 STUDY.md L1.2b "重构为终段全局软相" 的核心算法成果；
- `--tune` 坐标搜索（fvseg.rs:226-317）：两轮坐标下降 + 6 旋钮网格，通用调参器；
- `--plan-params` 直通评估通道（fvseg.rs:184-223）：C0 标签工厂的评估入口。

**直接证据（测试被迫重写）**：`fv-seg/tests/layered_e2e.rs:5-7` 开头自述 *"编排逻辑在此独立实现（bin 无法被集成测试引用）"*，然后复制了 `patched`（layered_e2e.rs:48-62 ≡ fvseg.rs:635-648）、`psnr_db`（layered_e2e.rs:128-145 ≡ fvseg.rs:652-669）和整个 `layered` 编排。两份实现已经在细枝上漂移（测试的 `psnr_db` 用 `3.0*n` 做分母写法不同）。bin 一改，测试不会红——这是最危险的测试形态。

**改法（M）**：在 fv-seg 增加 `pipeline` 模块（`run_pipeline`/`PipelineResult`/`LayerStat`/`psnr_db`/`patched_target`）和 `tune` 模块（坐标搜索器，泛型化 `eval: impl Fn(&PlanParams) -> Result<f64>`）；bin 只留参数解析与 IO。layered_e2e 改测真实现，漂移窗口关闭。examples/外部工具（如批量调参）即刻可复用。

### A.4 其他边界观察

- **裁判工具困在 examples**：`fv-fit/examples/eval_design.rs` 是 STUDY.md 全篇引用的"官方裁判"（所有 dB 数字出自它），却是个无测试的 example，还自带第三份 alpha_228 实现（见 §E.3）。应升格为 `[[bin]]` 并给它加 golden 测试（S）。
- **fv-seg 的遗留模型嗅探**：`fv-seg/src/lib.rs:131-135, 145-149` 用 `model_path.ends_with("u2net.onnx")` 推断模型，而 `SegModel` 枚举已是正式分发方式。字符串嗅探留着只会让新模型接入者掉坑（S，标记 deprecated 或删除）。
- **fv-inject 是边界设计的正面样本**：`MemIo` trait（mem.rs:10-19）+ `MockMem` + `#[cfg(windows)] win`，全链路可离线测试；布局数据文件化（layout.rs）应对游戏更新。无需改动。
- **workspace 定义**（fv/Cargo.toml）：5 成员、无 `[workspace.dependencies]`——`image 0.25` 在 3 个 crate、`clap 4` 在 4 个 crate、`serde/serde_json` 在 2 个 crate 各自声明（含 feature 差异：fv-fit 的 image 关 default-features，fv-gpu 的不关）。建议收编进 workspace 依赖表（S）。

---

## §B API 气味

### B.1 `fv_fit::cli` 是 pub 模块：clap 进入库 API 面

`fv-fit/src/lib.rs:12` 声明 `pub mod cli`，于是 `fv_fit::clap` 类型（`Args`）成为库公共 API（fvfit-gpu.rs:10 确实复用了 `Args` 与 `run_with`——这个复用本身是好的）。代价：所有把 fv-fit 当库用的下游（fv-gpu、fv-seg）传递依赖 clap；`Args` 的任何字段变更都是库 API 破坏性变更。**改法（S）**：`cli` 移到 `src/bin/` 共享模块或 `#[cfg(feature = "cli")]` 门控；`run_with(args, gpu)` 的可注入设计保留。

### B.2 巨函数 `fit_impl`：542 行，7 个阶段内联

`ENGINE:335-876`。一个函数里串了：缩放与 SoA 平面化（344-368）→ resume 初始形状合成（370-390）→ 前缀和/初始误差（392-406）→ 贪心主循环（504-653，本身 150 行）→ 抛光同步（655-695）→ 软椭圆相（697-835，140 行）→ 软栈重拟合与输出（837-876）。闭包 `to_full`（428-466）和 `write_outputs`（468-502）已是被捕获环境钉死在原地的"伪函数"——它们依赖 9 个局部变量，这正是无法抽出的信号。

**改法（M）**：引入 `struct FitCtx { w, h, n, scale, inv_scale, sums, grid, canvas, target, guide, spec, cfg, … }`，按阶段拆为 `prepare()` / `solid_phase()` / `polish_phase()` / `soft_phase()` / `refit_and_output()`。血统账本（`shapes_opt/srcs/changed`，ENGINE:420-422）抽成独立 struct（它已有一对一同步点 ENGINE:667-673，值得封装）。拆分是纯机械移动，行为不变，但要靠现有确定性测试（resume.rs / soft.rs）护航。

### B.3 CPU/GPU 双拷贝的候选生成循环：同一段代码写了两遍，靠注释维持逐位一致

- 实心相：GPU 路径 ENGINE:519-541（分块 RNG、`seed ^ (layer<<32) ^ chunk*0x100000001B3`、75/25 中心混采、structured 二选一）与 CPU 路径 ENGINE:575-616 是**手工复刻**，ENGINE:520 注释明说 *"候选集与 CPU 路径逐位同源：复刻 CPU 的分块 RNG 方案"*；
- 软相：GPU ENGINE:728-741 与 CPU ENGINE:753-772 同样成对；
- 第三份近亲：fvseg 的 `init_gpu_evaluator`（fvseg.rs:622-632）又复刻了 `fit_impl` 的分辨率换算公式（ENGINE:346-351）——引擎改 `max(4)` 就静默失配。

**改法（S）**：抽 `fn gen_candidates(layer_seed: u64, per_thread: usize, threads: usize, grid, guide/spec, structured) -> Vec<EllipseParams>`，CPU 路径直接消费同一 Vec（CPU 的 rayon 并行只是把"生成+评估" fusion 了；拆开后多一次 20k×20B 的内存写，相对评估成本可忽略，且换来结构不变式）。分辨率换算抽 `fn fit_dims(ow, oh, opt_max_dim) -> (w, h, scale)` 供 fvseg 复用。

### B.4 `DiffRenderer` 的 `pub(crate)` 字段：够用，但结构在报警

fwd.rs:27-50：16 个字段中 **12 个 `pub(crate)`**（device/queue/params/五种 buf/w/h/cap/n/snap/last_canvas/bwd），bwd.rs:53-66 直接引用内部 buffer 组 bind group。`pub(crate)` 本身是正确选择（bwd/refine 在 crate 内，不该升 pub），mod.rs:37-45 的契约注释也到位。问题在**字段混合了三种角色**：GPU 资源（device/buf）、会话状态（n/snap/last_canvas）、懒缓存（bwd）。backward() 的合法性完全依赖"必须在 render 之后"这一隐式时序（bwd.rs:15-20 仅靠 Err 字符串把关）。**改法（S）**：`n/snap` 合并为 `Option<RenderState>`，`backward()` 在 `None` 时返回类型化错误——把时序契约从文档挪进类型。

### B.5 重复样板清单（均可机械消除）

| 重复 | 位置 | 改法 |
|---|---|---|
| soft.rs 扫描线循环 **3 拷贝** | blend_composite:63-98 / eval_soft:128-168 / for_each_touched:213-242（`yext` 行三处同构，见 soft.rs:60/123/210） | 抽 `fn visit_ellipse(e,w,h,f: FnMut(x,y,u,v))`（S） |
| 爬山扰动集 2 拷贝 | ENGINE:235-270 `refine` vs polish.rs:289-342（6 选 1 扰动 + σ 衰减 0.6 完全同款） | 抽 `fn perturb(rng, e, sig) -> EllipseParams`（S） |
| WGSL 扫描线 + alpha228 各 2-3 份 | eval.wgsl / eval_soft.wgsl（两遍循环 69-107 与 121-157 内部又重复一遍）/ soft.rs | WGSL 侧抽 fn（S） |
| bind group 助手 3 套 | lib.rs:164-188 `buffer_entry/uniform_entry`、fwd.rs:390-414 `storage_ro/storage_rw/ent`、bwd.rs:121-158 `uni/ro/rw/ent` | 合并到 `fv-gpu/src/wgpu_util.rs`（S） |
| map_async 回读样板 3 份 | lib.rs:462-480 / fwd.rs:317-331 / bwd.rs:95-109 | 抽 `read_back(device, queue, buf, bytes) -> Result<Vec<u8>,_>`（S） |
| 六进制 bg 解析 2 份 | fv-fit cli.rs:89-94 vs fvseg.rs:674-681 | 移入 fv-common（S） |
| default_prefix/append_suffix | cli.rs:181-191 vs fvseg.rs:727-736（+ ENGINE:487-494 手拼） | 移入 fv-common（S） |

### B.6 其他 API 观察点

- `FitConfig` 已 15 个字段（ENGINE:33-62），每加一个旋钮（structured/auto_soft/soft_refit…）都在膨胀，且 e2e/soft/polish 测试用**完整结构体字面量**构造（e2e.rs:48-63 等 5 处）——每加字段要改 6+ 处。改法（S）：测试改用 `..FitConfig::default()` 只覆写关注项（部分已这么做），长期可引入 builder。
- `fit/fit_gpu/resume/resume_gpu` 四入口（ENGINE:273-332）是 `fit_impl(.., Option<&mut dyn BatchEvaluator>)` 的 2×2 展开——可收敛为 2 个（`fit`/`resume` 各带 `Option<gpu>`），但公开面不大，低优先。
- `GpuEvaluator::evaluate` 返回的 `CandidateResult.area`（lib.rs:12-17）只有交叉验证测试用，engine 路径忽略——无害，标注即可。
- `fv-fit/src/render.rs:51` 文档腐烂：`render_shapes` doc 说 *"SoftEllipse 的 alpha 渐变 M1 暂按实心处理"*，但 23-32 行明明走了 `blend_composite` 混合。改 doc（S）。
- `RowSums::seg` 只带 `debug_assert`（rowsum.rs:139）：release 下越界索引仍会 panic（slice indexing），但错误信息会退化为裸索引越界。可接受。

---

## §C 错误处理

### C.1 【最严重】fv-gpu 失败静默降级 = 近空结果 + 退出码 0

调用链：`GpuEvaluator::on_ready` 初始化失败 → `eprintln!` + `self.inner = None`（lib.rs:113-118）；此后 `eval_batch` 对每批返回 `vec![None; n]`（lib.rs:130-132，eval 失败同样 138-142）。引擎侧收到"全 None 候选池"→ `empty_streak` 三连 → **提前终止**（ENGINE:618-627）。

净效果：在无 Vulkan / 无 SHADER_INT64 的机器上，`fvfit-gpu` 会安静地跑完 3 层采样、写出≈0 个形状的 JSON、`完成：N 个形状`、**exit 0**。这不是"回退 CPU"——fvseg 的 `--gpu` 显式预初始化路径（fvseg.rs:167-177）才是正确做法：失败警告后走 CPU。fvfit-gpu.rs:54-55 用 `GpuEvaluator::default()` + `run_with`，把降级决策留给了最坏的实现。

**改法（M）**：`BatchEvaluator` 增加 `fn healthy(&self) -> bool`（或 `on_ready -> bool`）；引擎在 GPU 路径首层发现评估器失活时 `panic!`/返回错误（更倾向：`GpuEvaluator::eval_batch` 失败时直接返回 `Err` 并让 `run_with` 中止——静默正确性崩塌不可接受）。至少 fvfit-gpu 应像 fvseg 一样预初始化并硬失败。

### C.2 String 错误的规模与代价

`Result<_, String>` 共 **35 处**，分布 11 个文件（fv-gpu lib/fwd/bwd/bin、fv-inject convert/layout/win、fv-seg lib/plan/bin、fv-fit cli）。代价具体化：

- **不可编程分支**：fv-inject 的 `write_layer` 已经意识到结构化错误的价值——`Err((usize, String))`（write_process.rs:50，"已完成字段数"），但 `(usize, String)` 同样无法 match；上层 main.rs 只能整串打印。
- **调用方被迫 format! 套 format!**：fvfit-gpu.rs:72-171 一个闭包里 12 次 `map_err(|e| format!(...))`，原始错误类型信息（io/ort/serde/wgpu）全部熔毁成文本。
- **wgpu 错误二次熔毁**：`device.poll(...).map_err(|e| e.to_string())`（lib.rs:467、fwd.rs:322-324、bwd.rs:100-102）——wgpu 的 `PollError` 本可区分 device-lost/timeout。

**改法（M，按收益排序）**：
1. **fv-inject** 最该先做（MemIo 链是安全关键）：
   ```rust
   #[derive(thiserror::Error, Debug)]
   pub enum InjectError {
       #[error("布局文件: {0}")] Layout(#[from] std::io::Error),
       #[error("不支持的形状 type {kind}")] UnsupportedShape { kind: u32 },
       #[error("layer[{index}] 字段[{field}] 写失败: {addr:#x}")] MemWrite { index: usize, field: &'static str, addr: u64 },
       #[error("层表定位失败（层数 {count}）")] LocateFailed { count: usize },
   }
   ```
   MemIo trait 本身的 `bool` 返回保留（热路径，WinAPI 就是 bool），错误类型化发生在 scan/convert/write_process 的 API 边界。
2. **fv-gpu**：`enum GpuError { NoAdapter, NoInt64Feature, BufferAlloc{..}, MapFailed(String), Wgpu(String) }`——`GpuEvaluator::new` 的调用方（fvseg init_gpu_evaluator、fvfit-gpu）需要区分"没显卡"与"驱动崩了"。
3. fv-seg/fv-fit 跟进即可，优先级低（错误消息本身写得很好，中文指引完整）。

### C.3 库代码向 stderr 打日志

库层 `eprintln!` 共 31 处（非测试）。其中三类：① fv-gpu 错误吞没（§C.1，必须改）；② **引擎的 polish/refit 报告**（ENGINE:685-688、852-855）——`PolishReport`/`SoftRefitReport` 结构体都已存在并返回，却又直接打印，库消费者（如未来 GUI/wasm）无法静音；③ CLI 进度条（合理，但在 `cli::run_with` 里）。**改法（S）**：② 删 eprintln，报告经 `progress` 回调或结构体上抛。

---

## §D 测试缺口 Top 5

94 个测试覆盖了：光栅化 vs 暴力逐像素全等（raster.rs:291-327）、前缀和 vs naive（rowsum.rs:179-216）、确定性（多层）、resume 前缀逐位保真、软层 e2e、GPU↔CPU 数值交叉（有卡才跑）、MockMem 全链路含诱饵与回滚。**没测但最该测**：

1. **fvseg `--tune` / `--plan-params` / `run_pipeline`：零覆盖**（fvseg.rs:184-317, 388-578）。唯一相关测试 layered_e2e 测的是**复制品**（§A.3）。要测的具体性质：tune 的确定性（同输入两轮搜索路径一致——坐标下降依赖 eval 顺序）、`run_pipeline` 软相目标构造（`patched_target` 的 canvas 填充语义）、GPU/CPU 后端产出 PSNR 差界。前置条件是 §A.3 的库化。
2. **engine GPU 路径的候选生成与 CPU 逐位同源**（ENGINE:519-541 vs 575-616）。目前这条"复刻"不变式没有任何无 GPU 环境下的测试：用一个 **Mock BatchEvaluator**（记录收到的 `cands`，返回全 None 或注入固定值）即可断言"GPU 分支喂进 eval_batch 的候选序列 == CPU 分支会评估的序列"（同 layer/seed 下）。这是 STUDY.md 记载过的 1dB 级质量坑（GPU 曾用独立采样流），回归风险真实存在。附带可测 VERIFY_K 复核与全负批复核分支（ENGINE:546-573）——纯 CPU 逻辑，现在完全无覆盖。
3. **refine/抛光的 accept-guard 回滚路径（引擎层）**：soft.rs 单元测了 `refit_soft_stack` 的守卫回滚（soft.rs:293-327 + 测试），但引擎软相的"快照试混→实测 delta≤0→回滚 bbox"（ENGINE:810-833）没有直接测试——e2e 只断言最终 MSE 降。同理 **polish 后的血统同步**（ENGINE:667-673：srcs/changed 按 survivors 重写）无 e2e 断言："resume 输入经 polish 改动后，输出中被改的初始层不再逐位等于输入、未改的仍然逐位等于输入"这一契约只有一半被 resume.rs 覆盖（scale==1 且 polish_sweeps==0）。
4. **fv-inject `protect()` 边界与注入中止路径**：mock_e2e.rs:146-181 测了"中间字段（color）写保护→回滚"，未测：a) **最后一个字段（mask）失败**——回滚须还原因 5 个已写字段；b) `inject_all` 的中止条件 `fail >= 10 && ok == 0`（write_process.rs:112-115）——构造 10 个坏指针层验证不会白写 1000 层；c) protect 区间**恰好与字段端点重合/只覆盖半个字段**（is_protected 的半开区间语义 mem.rs:83-86）；d) scan 的 **u16 跨 chunk 边界**（见 §E.1——现在这个测试会失败，先修再加回归测试）。
5. **非常规输入的引擎健壮性**：极窄图（1×N 触发 `max(4)` 放大，ENGINE:349-351）、全透明图（target==bg→total_err==0→`sample_center` 均匀分支 ENGINE:149-151）、`opt_max_dim` > 原图（scale==1 的另一侧）、**含非有限坐标的 resume JSON**（§E.5——当前会被静默接受并原样透传到输出）。另：`render_shapes`（拟合引擎口径）与 eval_design（裁判口径）的一致性无测试——两者已发生过 bbox 勘误事件（STUDY.md §裁判 bbox 勘误）。

---

## §E 数值雷（对照 STUDY.md 已修四坑的同类排查）

已修四坑回顾：① eval_soft 单遍巨数相减 f32 灾难性抵消→两遍扫描；② 软内核 `plane_len` 误用 `h*(w+1)`；③ WGSL 保留字/i64 字面量/delta=1 边界带；④ u8 舍入虚账→快照实测接纳守卫。以下是**尚未引爆的同型雷**：

1. **【边界类，同 plane_len 型】fv-inject u16 扫描的 chunk 接缝漏检**。scan.rs:147-168：`off += got` 步进，内层 `for i in 0..buf.len()-1` 只在 chunk 内配对相邻字节——**跨 64MiB chunk 边界的 u16 命中永远检不到**（`region_chunk=64MB`，layout.rs:84）。概率低（每边界 2 字节窗口）但这是确定性算法的完整性漏洞，且完全无提示。**修（S）**：下一 chunk 从 `off + got - 1` 起（回退 1 字节重叠），或扫描时保留上 chunk 尾字节拼接。
2. **【f32/实现定义类，同 delta=1 边界带型】soft.rs 三函数不查 is_finite，靠 Rust as 饱和语义"侥幸安全"**。raster.rs:51-62 显式拒绝非有限参数；而 blend_composite（soft.rs:50-51）、eval_soft（:113-114）、for_each_touched（:200-201）只挡 `a <= 0.0`——**NaN 比较为 false，直接放行**，随后 `ia=1/NaN=NaN` 一路传播，最终因 `NaN.max(0.0)=0`、`NaN as u32=0`（Rust 饱和转换）退化成空循环。CPU 侧侥幸；但 WGSL 侧（eval.wgsl:44-45 的 `i32(ceil(cy - yext - 1.0))`）对超出 i32 范围的 f32→i32 转换是 **implementation-defined**，与 CPU 饱和语义不保证一致——离屏候选在 CPU=空、GPU=未定义行为，正是"边界带 ±1"坑的放大版。**修（S）**：三函数入口补与 raster.rs 相同的 is_finite 守卫；WGSL 侧 y0/y1 用 f32 clamp 后再转 i32。
3. **【常量三拷贝，裁判漂移型】alpha_228 的 C1/C2/C3 存在三份实现**：soft.rs:19-36（f64 牛顿）、eval_soft.wgsl:21-32（f32 牛顿）、eval_design.rs:11-30（**f32 牛顿 + LUT 4096 量化**）。裁判用的是 LUT 插值而引擎用直接求解——同一形状在两边的有效 α 可差一个 LUT 台阶（rn*4095 取整），148 软层的栈累计误差可至 0.0xdB 量级。历史上已经因 bbox 公式不同低见过 1.8dB。**修（S）**：eval_design 改调 `fv_fit::soft::alpha_228`（升 f64），常量只留 soft.rs 一份；WGSL那份加注释锚定来源。
4. **【同型 plane_len 复发温床】`Params.plane_len` 一字段两义**：实心内核=`h*(w+1)`（lib.rs:283-293、eval.wgsl:9），软内核=`w*h`（lib.rs:525-535、eval_soft.wgsl:14），eval_soft 里靠 526-534 的注释钉住。这正是已修坑②的形态——字段名没变、语义随管线切换。**修（S）**：拆成两个 uniform `stride`/`solid_plane_len` 与 `soft_plane_len`（或软内核直接用 `w*h` 派生，删掉复用），让 WGSL struct 与 Rust struct 一一对应无二义。
5. **【外部输入未验证型】非有限坐标可沿 JSON→引擎→注入器直达游戏内存**：`shapes_from_design`（cli.rs:111-148）校验 type/data.len，**不校验数值有限性**；serde_json 可解析 `1e40`（f32 溢为 inf）。inf 形状在引擎里被 raster.rs 安全地渲染为空、被 `to_full` 逐位原样保留（ENGINE:441-454）输出回 JSON，最后 `convert::prepare`（convert.rs:39-57）不做任何检查，把 `[inf, -inf]` 写进游戏层坐标。**修（S）**：两处（shapes_from_design 与 prepare）加 `is_finite` 全字段校验并报错。
6. **【量纲混用残余】`refine.rs` 的 Adam 对 θ 用 `lr_g*0.05` 补丁**（refine.rs:125-129）——像素与弧度共享一个优化器，已用系数救火（STUDY.md M1.8 坑④）。若后续给参数加归一化（a/b 用对数空间、θ 用 sin/cos 分量）可根治；当前补丁可用，仅登记。
7. **【确认无雷项】**（排除以免误改）：`spans_bbox` 的 `y+1`（raster.rs:126 注释成立）；rowsum 12 平面 i64 上界（每行 ≤65025·w，远低于 i64）；ErrorGrid u64 块和 ≤3·65025·n≈4e10，f64 换算无精度损失；polish 的 `m·Σo²` ≈ 4e10 in i64 安全；`360.0 - angle.rem_euclid(360.0)`（convert.rs:53）产 (0,360] 无奇异。

---

## §F 发布卫生

### F.1 panic 路径盘点（库代码，非测试）

| 位置 | 形态 | 评估 |
|---|---|---|
| ENGINE:307-319 resume 两个 `assert!` | 契约违反即 panic（doc 已声明 ENGINE:294-300） | **合理**，且 CLI 层已前置校验，panic 实际不可达 |
| rowsum.rs:87,109 / raster.rs:140 `assert_eq/debug_assert` | 布局契约 | 合理；release 下 debug_assert 消失但 slice 索引仍兜底 |
| stroke.rs:88 `unreachable!("4bit 全覆盖")` | 4-bit match 全覆盖 | 合理，可改 `_ => {}` 消除（S，可选） |
| stroke.rs:118 `pts.last().unwrap()` | 链构造保证 ≥2 点 | 安全但脆弱（chain_segments 重构时易踩），建议 `pts[pts.len()-1]` |
| stroke.rs:221-222 `partial_cmp().unwrap()` ×2 | 链长恒有限（网格中点） | **NaN 不可达的前提是输入 field/u8**——成立；但 `let _ = PI`（eval_design.rs:168）式的死代码同文件出现两处 |
| fv-gpu lib.rs:85 `as_mut().unwrap()` | 上面 `if let` 刚检查过 | 重借用写法问题，`if let Some(inner) = self.inner.as_mut() { inner.upload..; return Ok(()) }` 即消除 |
| fv-seg lib.rs:34 `.expect(ancestors)` | 编译期 CARGO_MANIFEST_DIR | 恒真，安全 |
| fv-inject mem.rs:72-128 MockMem 若干 unwrap | 测试基建 | 建议给 `poke/add_block` 标 `#[doc(hidden)]` 或注明 test-only |

### F.2 doc comment 覆盖度

pub 项/文档行比：fv-common 8/23、fv-fit 51/249、fv-seg 26/194（**最佳**，连私有 `dilate_radius` 都有契约 doc）、fv-gpu 18/74、fv-inject 35/78。具体缺口：`fv-gpu/src/lib.rs:12-17 CandidateResult` 四字段无 doc（`ok`/`area` 语义要读内核才知道）；fwd.rs `DiffRenderer::render` 系列 doc 良好但 `capacity()` 未说明"截容后调用方必须做什么"（refine.rs:87-91 有、类型上没有）；fv-inject `mem.rs:22-47` 读原语无 doc。建议 crate 级 `#![warn(missing_docs)]` 先挂 fv-seg（已达标）与 fv-common，逐步推广（S）。

### F.3 feature gate：一个都没有

- fv-seg 无条件依赖 fv-gpu（Cargo.toml:15）→ 只想要 `plan_layers`/Otsu（纯图像处理）的下游拖进 wgpu + pollster；同理无条件依赖 fv-fit（rayon/image）。
- fv-fit 库暴露 clap（§B.1）。
- fv-gpu 的 wgpu 是其存在意义，不需门控；但 **fv-gpu 在 GPU 不可用时的行为**（§C.1）比门控更紧迫。
- **建议（S）**：fv-seg 拆 `default = ["fit"]`，`fit` 拉 fv-fit/fv-gpu，纯 plan 用户 `--no-default-features`；fv-fit `cli` feature 包住 `pub mod cli`。

### F.4 元数据

workspace 无 `rust-version`（edition 2024 要求 ≥1.85，未声明则旧工具链报错不友好）；无 `repository`/`description`/`readme`；`[profile.release] lto="thin" + codegen-units=1` 合理。共享依赖应收编 `[workspace.dependencies]`（§A.4）。`publish` 未禁——若无意发 crates.io，建议 workspace 级 `publish = false` 防误发（内部 path 依赖反而会阻止发布，但显式更稳）。

---

## §G 优先级 action 清单

工作量：S = 小时级 / M = 一两天 / L = 一周级。★ = 建议本轮就做。

| # | action | 工作量 | 依据 |
|---|---|---|---|
| 1★ | **修 GPU 静默降级**：`on_ready`/`eval_batch` 失败上抛（或 fvfit-gpu 预初始化硬失败），禁止 exit 0 的近空结果 | M | §C.1，行为级 bug |
| 2★ | **fvseg 库化**：`run_pipeline`/tune/psnr_db/patched_target 移入 fv-seg 库，bin 变薄；layered_e2e 改测真实现 | M | §A.3 / §D.1 |
| 3★ | **scan.rs chunk 接缝 1 字节重叠**修复 + 跨界回归测试 | S | §E.1 |
| 4★ | **非有限坐标双闸**：shapes_from_design + convert::prepare 补 is_finite 校验 | S | §E.5 |
| 5 | soft.rs 三函数补 is_finite 守卫；WGSL y0/y1 先 clamp 后转 i32 | S | §E.2 |
| 6 | `Params.plane_len` 拆义（solid/soft 各自 uniform） | S | §E.4 |
| 7 | eval_design 升格 `[[bin]]`，改调 soft::alpha_228 消灭 LUT 三拷贝，加 golden 测试 | S | §A.4 / §E.3 |
| 8 | Mock BatchEvaluator：锁死"GPU 分支候选生成 ≡ CPU 分支"逐位同源 + VERIFY_K 分支覆盖 | M | §D.2 |
| 9 | `fit_impl` 拆阶段（FitCtx + 血统 struct），机械移动、确定性测试护航 | M | §B.2 |
| 10 | 候选生成循环合一（gen_candidates + fit_dims 供 fvseg 复用） | S | §B.3 |
| 11 | fv-inject thiserror（InjectError），fv-gpu GpuError；wgpu 错误不再 to_string | M | §C.2 |
| 12 | 引擎内 eprintln 报告改结构化上抛（polish/soft-refit） | S | §C.3 |
| 13 | `EllipseParams`/`RowSpan`/RowSums 布局移入 fv-common（或 fv-core），fv-gpu 摆脱 fv-fit | S | §A.2 |
| 14 | fv-fit `pub mod cli` 加 feature 门控；fv-seg 拆 `fit` feature | S | §B.1 / §F.3 |
| 15 | soft.rs 扫描线 visit_ellipse 合一；refine/polish 扰动集合一；wgpu bind/readback 助手合一 | S | §B.5 |
| 16 | fv-inject protect 边界/末字段回滚/inject_all 中止补测 | S | §D.4 |
| 17 | workspace：`[workspace.dependencies]` 收编 + `rust-version` + `publish=false` | S | §A.4 / §F.4 |
| 18 | DiffRenderer 时序契约类型化（Option\<RenderState\>） | S | §B.4 |
| 19 | doc 补齐（fv-gpu CandidateResult、fv-inject 原语）+ missing_docs lint 分级上线 | S | §F.2 |
| 20 | （远期）refine 参数空间归一化根治 θ 学习率补丁；FitConfig builder 化 | L | §E.6 / §B.6 |

**一句话收束**：这个工作区的算法内核（raster/rowsum/engine 数值口径）是可信的——它的风险不在数学，而在**边界之外**：失败传播（#1）、bin 围墙（#2）、输入验证（#4/#5）和接缝（#3）。先做 1-4★，其余按表推进。
