# fv-bench 设计：裁判库化（把散落的 PSNR 评估逻辑统一成一个 crate）

- 状态：设计稿（只读评审产出，未动任何源码）
- 依据：CODE_REVIEW.md §A.3/A.4/B.5/E.3、STUDY.md:200-201（裁判 bbox 勘误 + bg 跨实例假灾难）、现行源码逐文件核实（行号均为核实时的真实行号，路径相对 `fv/crates/`）
- 工作量记法：S = 小时级 / M = 一两天 / L = 一周级（与 CODE_REVIEW §G 一致）

---

## 0. 现状盘点（设计前的事实核对）

任务背景引用的 CODE_REVIEW 问题清单**部分已过时**——评审落地过程中已消灭两处拷贝。设计必须基于当前真实状态：

| # | 问题 | CODE_REVIEW 时的状态 | 当前状态（已核实） |
|---|---|---|---|
| 1 | alpha_228 三拷贝（soft.rs f64 / eval_soft.wgsl f32 / eval_design LUT） | 三份 | **两份**：eval_design 已改调 `fv_fit::soft::alpha_228`（eval_design.rs:12-14，LUT 已删，但 :5 的头注释仍写"LUT 4096"——文档腐烂）；eval_soft.wgsl:22-33 的 f32 内联常数**仍在且无锚定注释** |
| 2 | psnr_db 在 fvseg bin 和 layered_e2e 各一份 | 两份 | **已消除**：库化落地后 layered_e2e.rs:9 直调 `fv_seg::pipeline::psnr_db`。但 PSNR 实现仍有 **5 处**（见下） |
| 3 | resolve_bg 在 fv-fit cli 和 fvseg bin 两处 | 两份 | 形式上收敛为 `fv_fit::cli::resolve_bg`（cli.rs:69-95）与 `fv_seg::pipeline::resolve_bg`（pipeline.rs:430-457）各一份——但两份**语义已分叉**（见 #4），另有两份内联变体 |
| 4 | eval_design 独立渲染/PSNR/alpha | 是 | **仍是最核心未解决问题**：eval_design.rs 自带 f32 float 参考渲染器（:66-150）+ 内联 PSNR/MAE（:153-160）+ 内联 bg 均值（:31-49），无任何测试 |

**当前 PSNR 实现全量清单（5 处，3 种口径）**：

| 位置 | 口径 | 备注 |
|---|---|---|
| `fv-seg/src/pipeline.rs:409-426` `psnr_db(render, orig, bg)` | u8 图像对；透明原图像素换 bg；mse≤0 → inf | 库化后已是唯一"图像对"真源（fvseg bin、layered_e2e 共用） |
| `fv-fit/examples/eval_design.rs:153-160` | **f32/f64 float 参考**：SoA 平面、逐形状合成不落 u8 取整 | 官方记分板口径（STUDY 全部 dB 数字出自它）；比引擎 u8 栈渲染**偏高**（无逐层取整损耗） |
| `fv-fit/examples/psnr.rs:12-25,35-49` | u8 图像对；源图被 resize 到预览尺寸 | 内联 bg 均值第 3 份 |
| `fv-fit/examples/jitter_bench.rs:36` | 闭包 psnr(&[u8],&[u8])，画布对画布 | 抖动损伤实验 |
| `fv-fit/examples/b2_band.rs:46` | 闭包 psnr(&[Shape])，经 render_shapes | B2 原型 |

**bg auto 均值实现全量清单（4 处，2 种语义）**：

| 位置 | auto 语义 |
|---|---|
| `fv-fit/src/cli.rs:69-95` | 全图不透明（a≥128）像素均值（拟合/引擎目标构造语义） |
| `fv-seg/src/pipeline.rs:430-457` | **掩码感知三级退化**：干净背景区 → 背景掩码区 → 全图（fvseg 编排策略） |
| `fv-fit/examples/eval_design.rs:31-49` | 全图不透明均值（f32 版内联，与 cli.rs 同语义不同代码） |
| `fv-fit/examples/psnr.rs:12-25` | 全图不透明均值（u8 版内联） |

**两次真实事故（§D 的教训来源，STUDY.md:200-201 原文）**：
1. **bbox 1.8dB**：eval_design 旋转椭圆外接界误用 `max(|rx·c|,|ry·s|)`（正确 `√(rx²c²+ry²s²)`），45° 旋转层四角漏渲染 ~1.8dB，全部历史 PSNR 被低估。现行 eval_design.rs:100-103 已改正确公式——但这是一份**无测试守护的手抄公式**，与引擎 raster 的解析行 span 是两套几何。
2. **bg 跨实例假灾难**：auto 拟合的栈用 hex bg 评，MSE 9473 vs 61——口径不配对不是数值误差，是**语义错误**。

**顺手发现的潜伏 bug**：eval_design.rs:81 `let ang = if s.data.len() >= 4 { s.data[4] as f64 } else { 0.0 }`——rect（type=1）的 data 恰 4 项，`len() >= 4` 为真后索引 `data[4]` **越界 panic**（应为 `>= 5`）。现有 eval 从未评过含 rect 的 design 所以没炸，但这是"无测试的裁判"的直接罪证，迁移时必须修。

---

## §A fv-bench crate 骨架

### A.1 位置与依赖方向

新 crate：`fv/crates/fv-bench`（package `fv-bench`，lib `fv_bench`），加入 `fv/Cargo.toml` members。

```text
fv-common（shape/json + bg 原语〔新增〕）
   ↑              ↑
fv-fit ────── fv-bench（judge / render_ref / alpha / bg 门面）
   ↑  ↑           ↑              ↑
fv-gpu │           │              └─ fv-fit [dev-dependencies]：
       │           │                 examples（eval_design/psnr/…）复用，命令不变
       └─ fv-seg ──┘
          （可选：pipeline::psnr_db 变委托）
```

**依赖规则（硬约束）**：
- `fv-bench` 依赖 `fv-common` + `fv-fit`（+ `image`，与 fv-fit 同 feature：`default-features = false, features = ["png", "jpeg"]`）。
- **反向禁止**：`fv-common`、`fv-fit` 的 `[dependencies]` 永不出现 `fv-bench`。理由：bg 原语是拟合期也需要的底层件，必须住在依赖图底部（fv-common），bench 只是它的**门面消费者**；否则 fv-fit 为拿 bg 反向拖进裁判，层次倒挂。
- `fv-bench` **不依赖 `fv-gpu`**。GPU↔CPU 数值交叉验证留在 fv-gpu 自己的 tests（已有，需显卡）；bench 是纯 CPU 的确定性裁判，拉 wgpu 会毁掉"任何机器可跑 golden 测试"的性质。
- fv-seg → fv-bench 是允许的方向（bench 在 fit 之上、seg 之侧）；只用于 psnr_db 委托（§C 步骤 2），不把编排逻辑挪进 bench。
- `fv-fit` 对 `fv-bench` 的引用**仅限 `[dev-dependencies]`**（Cargo 允许 dev-dep 与普通 dep 成环，不影响库分层）——examples 留在 fv-fit 下，`cargo run --release --example eval_design` 的肌肉记忆与 STUDY.md 全部命令行零变化。

### A.2 模块划分

```text
fv-bench/src/
├── lib.rs         # crate doc：口径总纲（哪个函数是记分板口径、哪个是引擎口径）
├── alpha.rs       # 228 语义门面：re-export soft::alpha_228 + kind/opacity 规范化 + 形状校验
├── bg.rs          # BgSpec → Bg（带来源凭证）；fv_common::bg 原语的裁判侧门面
├── render_ref.rs  # float 参考渲染器（f64 合成，无逐层 u8 取整）+ design → Vec<Shape> 规范化
└── judge.rs       # Judge builder（type-state）+ JudgeReport + 各口径 PSNR/MAE 函数
tests/
├── parity.rs      # golden：render_ref vs render_shapes 的逐位/有界对账（§B.3）
└── golden_psnr.rs # 固定 design 固件 → 冻结的 PSNR 数值（防口径静默漂移）
```

### A.3 公共 API 签名

```rust
// ===== alpha.rs —— 228 语义的唯一门面（真源仍是 fv-fit/src/soft.rs，见 §B）=====

/// 真源 re-export：bench 内任何代码不得本地重写 α 公式（惯例 + 评审检查项）。
pub use fv_fit::soft::{alpha_228, C1_228, C2_228, C3_228};

/// design JSON 形状 → 规范化 Shape（kind 白名单 1/16/228；逐 kind 校验 data 长度
/// rect=4 / 椭圆=5；非法即 Err——顺带消灭 eval_design.rs:81 的 rect 索引 panic）。
pub fn normalize_design(d: &fv_common::json::DesignJson) -> Result<Vec<fv_common::shape::Shape>, String>;

/// SoftEllipse 的有效不透明度（color[3]/255）。基础形状恒 1.0。
pub fn opacity_of(s: &fv_common::shape::Shape) -> f64;

// ===== bg.rs —— 背景口径（原语下沉 fv-common，见 §B.4；此处是裁判侧门面）=====

/// 背景规格：显式 hex 或 auto。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgSpec {
    /// 全图不透明（a≥128）像素逐通道均值（与 fv-fit 目标构造同规则）。
    Auto,
    /// 显式 RRGGBB。
    Hex([u8; 3]),
}

/// 解析结果：颜色 + **来源凭证**（§D：bg 配错是语义事故，必须留痕）。
#[derive(Debug, Clone, Copy)]
pub struct Bg {
    pub rgb: [u8; 3],
    pub spec: BgSpec,
    /// Auto 时记录参与均值的像素数（0 = 全透明图退化为黑，报告中可见）。
    pub auto_samples: u64,
}

pub fn resolve_bg(spec: BgSpec, img: &image::RgbaImage) -> Bg;

/// CLI 字符串（"auto" | "RRGGBB" | "#rrggbb"）→ BgSpec。
pub fn parse_bg_spec(s: &str) -> Result<BgSpec, String>;

// ===== render_ref.rs —— float 参考渲染（裁判口径的"理想画布"）=====
// 原则：**bench 零几何**。实心形状直接消费 raster::ellipse_rows 的解析行区间
// （引擎同一几何真源，无第二份 bbox 公式）；软椭圆消费 soft.rs 的扫描访客
// （CODE_REVIEW §B.5 提议的 visit_ellipse 合一，bench 顺路成为其首个消费者）。

/// 规范化形状列表 → f64 SoA 画布（3 平面）。实心：按 ellipse_rows span 整面覆盖
/// （f64 存整数色，无损）；228：f64 全程合成 `c = c·(1−α) + color·α`，α =
/// alpha_228(rn)·opacity，**不做逐层 u8 取整**——与引擎 render_shapes 的
/// u8 逐层取整形成一对有界孪生（对账契约见 §B.3）。
pub fn render_design_f64(
    w: u32, h: u32,
    shapes: &[fv_common::shape::Shape],
    bg: [u8; 3],
) -> Vec<f64>;

// ===== judge.rs —— 评审入口（口径参数化见 §D）=====

/// 两裸画布 PSNR（f64 平面，float 参考口径）。n=0 → inf。
pub fn psnr_f64(target: &[f64], render: &[f64]) -> f64;

/// 两裸画布 PSNR + MAE（u8 平面，jitter_bench/b2_band 的画布对画布口径）。
pub fn psnr_mae_u8(a: &[u8], b: &[u8]) -> (f64, f64);

/// 渲染图 vs 原图（透明原图像素按 bg 替换后逐像素 3 通道）。
/// = `fv_seg::pipeline::psnr_db` 的真源（后者变薄委托，签名不变）。
pub fn psnr_rgba(render: &image::RgbaImage, orig: &image::RgbaImage, bg: [u8; 3]) -> f64;

/// 评审报告：数字 + 全部口径凭证（§D：报告必须能自证口径）。
#[derive(Debug, Clone)]
pub struct JudgeReport {
    pub psnr: f64,
    pub mse: f64,
    pub mae: f64,
    pub bg: Bg,
    pub dims: (u32, u32),
    pub shapes: usize,
    /// 每 kind 计数 [rect, ellipse, soft]。
    pub per_kind: [usize; 3],
}

/// 目标构造：源图 + bg 规则（透明像素换 bg）→ f64 目标平面。
pub fn target_planes(img: &image::RgbaImage, bg: [u8; 3]) -> Vec<f64>;

/// builder（type-state：bg 未设前无法评审，§D.2）。
pub struct JudgeBuilder<TargetSet>(/* … */);
pub struct JudgeBuilder<BgSet>(/* … */);

impl JudgeBuilder<TargetSet> {
    /// 唯一入口：目标图定死，bg 必须显式给一次（不提供 Default bg）。
    pub fn target(img: &image::RgbaImage) -> Self;
    pub fn bg(self, spec: BgSpec) -> JudgeBuilder<BgSet>;
}

impl JudgeBuilder<BgSet> {
    /// 引擎口径复核：报告同时给出 u8 引擎渲染的 PSNR（默认关，开启多一次 render_shapes）。
    pub fn with_engine_check(self, yes: bool) -> Self;
    /// 评审：FloatRef 口径（记分板口径，默认）。
    pub fn judge(self, shapes: &[fv_common::shape::Shape]) -> JudgeReport;
    /// 评审 design JSON（内部 normalize_design，报告错误原样上抛）。
    pub fn judge_design(self, d: &fv_common::json::DesignJson) -> Result<JudgeReport, String>;
}
```

`render_shapes`（引擎 u8 口径）**不搬**——它留在 `fv_fit::render`（引擎真源），bench 通过 `fv_fit::render::render_shapes` 调用做 engine check。渲染逻辑不进 bench，bench 只定义"怎么比"。

---

## §B 统一原则

### B.1 alpha_228：只留 soft.rs 一份（已过半，收尾清单）

真源 = `fv-fit/src/soft.rs:19-36`（C1/C2/C3 常量 + f64 牛顿 5 轮）。收尾三件事：

1. **eval_design 侧已消灭**（:12-14 委托），但 :5 头注释仍描述已删除的"LUT 4096"实现——步骤 2 重写该文件时顺带修，避免下一个读代码的人以为存在第二实现。
2. **WGSL 侧锚定**：见 B.2（两方案评估）。
3. **纪律条款写进 crate doc**：fv-bench 内出现第二份 α 公式（含"f64 转 f32 再手算一遍"的变体）视为评审阻断项。bench 的角色是**调用** `fv_fit::soft::alpha_228`，不是替它做数値——render_ref 全程 f64（现行 eval_design 在包装函数里截成 f32，属无谓的精度损失，重写时去掉）。

### B.2 228 语义的 WGSL 侧锚定：两方案评估

现状：`fv-gpu/src/eval_soft.wgsl:22-33` 以 f32 内联硬编码牛顿式（`fa = 1.0 − 0.9132a − 0.1875a² + 0.1007a³`，即 `−C3·a³` 展开后的手写变形；`fpa` 同理），无任何指向 soft.rs 的注释。

**方案甲：注释契约 + 常量奇偶测试（推荐）**
- eval_soft.wgsl 的 `alpha228` 上加锚定注释：来源 = `fv/crates/fv-fit/src/soft.rs C1_228/C2_228/C3_228`，改动义务 = 两处同改；
- fv-gpu 增加 `tests/shader_constant_parity.rs`：
  1. `include_str!("../src/eval_soft.wgsl")` 文本中提取三个浮点常数，与 `fv_fit::soft::{C1_228, C2_228, C3_228}` 数值比对（文本解析即可，不需正则严格性——找不到即红）；
  2. 用 Rust 复刻 WGSL 的 f32 公式（测试内的 f32 逐行对照实现），在 rn ∈ [0,1) 采 4096 点与 `alpha_228`（f64）比对，断言最大偏差 < 1e-5（f32 牛顿 vs f64 牛顿的实际量级 ~1e-7，留 100× 余量）；
  3. 扫描 `fv-gpu/src/*.wgsl` 全目录，断言 `0.9132` 只出现在 eval_soft.wgsl——**防新内核再抄一份**（diff 系内核未来若需要 β，会撞上这条测试，被迫改走 include/上传路径）。
- 优点：零构建复杂度（工作区现有零 build.rs，保持）；WGSL 保持可直接阅读；漂移必红；测试 3 顺带防"新增拷贝"这一最可能的复发形态。
- 缺点：本质仍是两份文本，靠测试锁同——不是结构上的单一来源。

**方案乙：build.rs 代码生成**
- 常数唯一存于 soft.rs（或共享 spec 文件），fv-gpu 加 build.rs 把常数代入 WGSL 模板生成到 OUT_DIR，运行时 `include_str!(concat!(env!("OUT_DIR"), "/eval_soft.wgsl"))`。
- 优点：真单一来源，编译期保证。
- 缺点：为一个三元组给 fv-gpu 引入 build.rs + OUT_DIR 链路；WGSL 源文件不再是"你看到的就是跑的"（评审/调试都要先脑内跑一遍 codegen）；WGSL 无 `#include`，要么生成整文件（模板-生成物两份文本，问题只换了个地方）要么上 naga 模块拼接（过度工程）。

**结论：选甲。** 甲用一条 `include_str!` 测试把"漂移必红 + 新拷贝必红"两个实际风险都关掉，成本约 30 行测试；乙把成本花在构建链路上，换来的"编译期保证"对这个风险量级（注释级常数、5 轮牛顿对 f32 舍入本就不敏感）不成比例。升级路径保留：若未来出现第 3 个需要 α 的内核或常数进入 uniform 契约，再上乙。

### B.3 渲染对账：bbox 教训的根治 = bench 零几何

1.8dB 事故的根源不是"公式写错"，是**同一几何事实存在两份手写实现**（裁判 bbox vs 引擎行 span），且裁判那份无测试。根治方式按优先级：

1. **消灭拷贝而非约束公式**：render_ref 实心路径直接消费 `fv_fit::raster::ellipse_rows`（解析行区间，边界判据 `u²/a²+v²/b² ≤ 1`，raster.rs:1-20 已有模块级数学契约 + raster.rs:171-194 的暴力参考测试锁等价）；软路径消费 soft.rs 的扫描访客（把 CODE_REVIEW §B.5 提议的 `visit_ellipse` 合一作为前置件，回调签名给到 `(x, y, rn)`，render_ref 在回调里做 f64 合成）。**bench 内不出现任何 bbox/区间/判据公式**——1.8dB 类事故在结构上不可能复发，而不是被测试"逮住"。
2. **golden 对账测试**（fv-bench/tests/parity.rs）：
   - 纯实心 design（rect/椭圆混合）：`render_design_f64` 取整后与 `render_shapes` 画布**逐位相等**（实心覆盖 + 整数色在 f64 中无取整损耗；边界判据两边一致——实心 `≤ 1.0`、软层 `< 1.0`，已逐处核实：raster.rs:188 vs eval_design.rs:124，soft.rs:87 vs eval_design.rs:132——逐位相等是可达且应当的硬标准，即 STUDY.md:201 "solid300 MSE 184.91 = 引擎内部值"对账的永久化）；
   - 含 228 的 design：两口径 PSNR 差 ≤ 冻结阈值（合成固件上实测定标，预期 ≤0.05dB；软层逐像素逐层 u8 取整 ±0.5 的累积上界）。
3. 这条测试同时是"裁判 bbox 勘误"级事故的回归网：任何一边几何改动导致四角漏渲染，逐位对账立刻红。

### B.4 bg：原语下沉 fv-common，语义分叉显式化

4 份实现、2 种语义的正确解法不是"挑一个当真源"，而是**分层**：

- **fv-common 新增 `bg` 模块**（纯原语，无策略）：
  ```rust
  /// hex/auto 公共解析（RRGGBB 校验两处现行实现完全一致，唯一真源）。
  pub fn parse_hex(spec: &str) -> Result<[u8; 3], String>;
  /// 谓词式均值：sel(i)=true 且 a≥128 的像素逐通道均值四舍五入（u64 累加，
  /// (sum+cnt/2)/cnt——与现行两处实现逐位同规则）。
  pub fn opaque_mean(img: &image::RgbaImage, sel: impl Fn(usize) -> bool) -> Option<[u8; 3]>;
  ```
  fv-common 需新增 `image` 依赖（default-features = false, 仅解码不需要 feature——类型使用）。这是唯一的底层依赖变化；CODE_REVIEW §B.5 本就建议"六进制 bg 解析移入 fv-common"，此处顺路完成。
- **策略留在各自所有者**：`fv_fit::cli::resolve_bg`（全图均值）与 `fv_seg::pipeline::resolve_bg`（掩码三级退化）签名、语义、错误文案均不变，内部改调 `fv_common::bg` 原语——两份语义是**有意的策略差异**（fvseg 的干净背景区比全图更准），保留；消灭的是两份相同的累加/解析代码。
- **bench 门面**：`fv_bench::bg::resolve_bg` 组装 `Bg`（颜色 + 凭证），裁判与拟合天然同规则（同一条 fv-common 原语），§D.2 的"bg 配对"约束由此获得实现基础。

---

## §C 迁移路径（3 步，每步可独立提交、测试不红）

前置说明：CODE_REVIEW action #2（fvseg 库化）与 #7 的前半（eval_design 改调 soft::alpha_228）**已落地**，故本路径从"半成品"出发，不重复已完成工作。

### 步骤 1：bg 原语下沉 + fv-bench 骨架 + psnr_rgba 真源（S）

**动的文件**：
| 文件 | 动作 |
|---|---|
| `fv/Cargo.toml` | members += `crates/fv-bench` |
| `fv/crates/fv-common/Cargo.toml` | + image（no default features） |
| `fv/crates/fv-common/src/lib.rs`、新增 `src/bg.rs` | `parse_hex` + `opaque_mean`（含从 cli.rs 平移来的 4 个既有测试） |
| `fv/crates/fv-fit/src/cli.rs` | `resolve_bg` 内部改调 fv_common::bg（签名/语义/错误文案不变；测试原样绿） |
| `fv/crates/fv-seg/src/pipeline.rs` | `resolve_bg` 同上（三级退化策略保留，只换累加实现） |
| `fv/crates/fv-bench/`（新建） | Cargo.toml、lib.rs、bg.rs、judge.rs（本步仅 `psnr_rgba`/`psnr_f64`/`psnr_mae_u8` + 单测） |
| `fv/crates/fv-seg/src/pipeline.rs`（psnr_db） | 函数体改为一行委托 `fv_bench::judge::psnr_rgba`（签名不变，layered_e2e 立即验证等价） |

**不红保证**：所有既有 pub 签名不动；bg 测试随实现平移；psnr_db 委托前后数值逐位一致（同公式同累加序）。此步结束后：bg 4 份 → 1 份原语 + 2 份策略壳；PSNR "图像对"口径 5 份 → 1 份真源 + 4 个消费者。

### 步骤 2：render_ref + Judge builder 落地，eval_design/psnr 变薄壳（M）

**动的文件**：
| 文件 | 动作 |
|---|---|
| `fv/crates/fv-bench/src/render_ref.rs` | 新增：f64 参考渲染（实心消费 ellipse_rows、软消费 visit_ellipse，bench 零几何）+ `normalize_design`（kind/data-len 校验，**顺带修 eval_design.rs:81 的 rect panic**） |
| `fv/crates/fv-fit/src/soft.rs` | 暴露 `visit_ellipse` 扫描访客 pub（CODE_REVIEW §B.5 的三拷贝合一提案，render_ref 的 f64 合成回调是其首个消费者）；`raster::ellipse_rows` 已是 pub，实心路径直接用，无需改动 |
| `fv/crates/fv-bench/src/judge.rs` | 补全 JudgeBuilder type-state + JudgeReport + target_planes |
| `fv/crates/fv-bench/tests/parity.rs`、`golden_psnr.rs` | §B.3 对账 + 固件冻结数值（首版数值来自迁移时 eval_design 对固件的实测输出，冻结后任何口径漂移必红） |
| `fv/crates/fv-fit/examples/eval_design.rs` | 重写为 ~40 行薄壳：解析参数 → `fv_bench` builder → 打印 JudgeReport；`--save` 走 render_shapes 或 render_ref（参数指明口径）；修头注释 LUT 腐烂 |
| `fv/crates/fv-fit/Cargo.toml` | [dev-dependencies] += fv-bench |
| `fv/crates/fv-fit/examples/psnr.rs` | 重写为 fv_bench::judge 调用（bg 内联删除） |

**不红保证**：eval_design 重写后对 STUDY.md 固件（如 ayylmao + 旗舰 JSON）输出与旧版一致（f64 化后允许 ±0.01dB 级差异，记入提交说明；bbox/合成序不变故实心部分应逐位同）。golden 冻结在本步完成，形成回归锚点。

### 步骤 3：WGSL 锚定 + 尾部 examples 收编 + 文档收口（S）

**动的文件**：
| 文件 | 动作 |
|---|---|
| `fv/crates/fv-gpu/src/eval_soft.wgsl` | alpha228 上加锚定注释（§B.2 甲方案） |
| `fv/crates/fv-gpu/tests/shader_constant_parity.rs`（新增） | 常量提取比对 + f32 公式数值奇偶 + 全 wgsl 目录扫描防新拷贝 |
| `fv/crates/fv-fit/examples/jitter_bench.rs` | 本地 psnr 闭包 → `fv_bench::judge::psnr_mae_u8` |
| `fv/crates/fv-fit/examples/b2_band.rs` | 同上（经 render_shapes 的口径用 engine check 组合表达） |
| `fv/crates/fv-fit/src/render.rs:51` | 顺手修 doc 腐烂（"M1 暂按实心处理"与实际 blend 行为不符，CODE_REVIEW §B.6 已登记） |
| `CODE_REVIEW.md` | §E.3 / §A.4 行动项标记落地；登记 fv-bench 为后续裁判唯一入口 |

**不红保证**：全部为消费端替换与新增测试；fv-gpu 测试仅文本+数值断言，无 GPU 依赖。

依赖顺序：1 → 2 → 3 严格串行（2 依赖 1 的 bg 原语；3 的锚定测试不依赖 2 但放最后以免与 2 的 golden 定标互相干扰）。

---

## §D judge API 语义：口径参数化与"1.8dB 教训"的 API 化

### D.1 三条口径轴

| 轴 | 取值 | 语义 | 默认 |
|---|---|---|---|
| **bg 来源** | `Auto` / `Hex([u8;3])` | 目标图与画布的底色规则；Auto 与 fv-fit 目标构造同一条 fv-common 原语 | 无默认——**必填**（见 D.2） |
| **alpha 处理** | `FloatRef` / `EngineU8` | FloatRef：f64 α、无逐层取整（记分板口径，理论上限）；EngineU8：`render_shapes` 逐层 u8 取整（注入游戏实见口径） | FloatRef（历史数字连续性） |
| **bbox 语义** | 唯一 | bench 零几何：实心走 `raster::ellipse_rows`、软走 `soft::visit_ellipse`，几何真源只在 fv-fit——**不参数化**：给 bbox 留旋钮等于给 1.8dB 事故留门票 | （唯一） |

引擎的第三种既有口径（fit 期 i64 精确 MSE @ 448 工作分辨率）**不进 judge**——它是优化器内部信号，不是质量度量；JudgeReport 不产出它，避免第三口径混入记分板。

### D.2 事故 → API 约束的映射（选型：轻量 type-state builder + 凭证化报告 + golden 测试；拒绝纯文档约束）

| 事故 | 根因 | 约束手段（为何这样选） |
|---|---|---|
| bg 假灾难（9473 vs 61） | 跨实例 bg 不配对，且无任何机制提醒 | **builder type-state**：`JudgeBuilder<TargetSet>` 上不设 bg 就没有 `judge` 方法——"必须显式选 bg"从文档变成编译错误。`Bg` 携带来源凭证（spec + auto_samples），`JudgeReport.bg` 强制留痕，报告打印含 bg 行——事后审计"这 30.28dB 是什么口径"不再靠翻 STUDY。CLI 层（eval_design 薄壳）在 `--bg` 与 auto 均值不一致时打一行 stderr 提示（不阻止，允许有意跨实例对比）。**不选**完整 newtype（`Bg<Fit>`/`Bg<Judge>` 相等性语义复杂，超过收益） |
| bbox 1.8dB | 裁判私有第二份几何公式，无测试 | **消灭拷贝**而非约束使用：共享助手 + 逐位 parity 测试（§B.3）。不给 bbox 做 type-state——bbox 不是调用方该选的参数，把它做成可选项本身就是错误设计 |
| LUT/常数漂移 | 三份独立实现 | **单一函数 + 奇偶测试**（§B.1/B.2）；API 层面以 `pub use` 门面表达"唯一入口"，评审检查项兜底 |
| eval_design rect panic | 无测试的裁判 + 手写形状解析 | **`normalize_design` 集中校验**（kind 白名单 + 逐 kind data 长度），builder 的 `judge_design` 是唯一 JSON 入口 |

选型理由小结：三条历史事故没有一条是"调用方参数选错"，全是**实现拷贝漂移**与**必填项缺失**。故约束的主形态是（a）让拷贝不存在，（b）让必填项成为编译错误，（c）让每次评审自带口径凭证。（d）golden 冻结数值作最终防线——任何静态手段漏掉的口径变化都会撞上冻结数值。文档契约只作为辅助（crate doc 的口径总纲），永不作为唯一防线——CODE_REVIEW 里 render.rs:51 的 doc 腐烂证明纯文档防线必然失效。

### D.3 报告与打印契约

- `JudgeReport` 打印行保持与现行 eval_design 输出**字段兼容**（`shapes=… dims=… MSE=… PSNR=… dB MAE=…`），追加 `bg=` 与口径标记——STUDY.md 的数字引用与脚本不断裂。
- `with_engine_check(true)` 时追加一行 `engine_u8=… dB (Δ=…)`——FloatRef 与 EngineU8 的差值本身是"逐层取整损耗"的观测窗，软层占比大的 design 上这个 Δ 有诊断价值。

---

## §E 工作量评估

| 步骤 | 内容 | 量 | 说明 |
|---|---|---|---|
| 1 | bg 原语下沉 fv-common + fv-bench 骨架 + psnr 委托 | **S**（2-4h） | 纯平移 + 委托，既有测试护航；风险最低 |
| 2 | render_ref + Judge builder + golden/parity 测试 + eval_design/psnr 薄壳化 | **M**（1-2 天） | 大头在 golden 定标（跑固件、冻结数值、f64 化差异核对）与 builder 类型设计；visit_ellipse 合一为前置件（~2h，本身即 §B.5 的独立收益项） |
| 3 | WGSL 锚定测试 + 尾部 examples 收编 + doc 收口 | **S**（2-4h） | 锚定测试 ~30 行；examples 各 ~10 行改动 |

**合计 ≈ 3-4 人日**（M 为上限口径）。无 L 项：本设计刻意不做引擎改造、不做 GPU 侧重构、不引入 build.rs——bench 的全部价值在"收拢与锁死"，不在新能力。

风险与缓解：
- **f64 化导致的数字漂移**（eval_design 现为 f32 合成）：步骤 2 用固件实测量化差异并写入提交说明；若需绝对连续可保留 f32 路径为 `FloatRef32`（不推荐——口径越多越乱，优先一次切净）。
- **fv-fit dev-dep fv-bench 的环**：Cargo 合法（dev-dep 环不参与构建环检查）；若团队接受不了观感，备选是把 eval_design/psnr 移到 `fv-bench/examples`，代价是命令改为 `-p fv-bench --example eval_design`（STUDY.md 命令记录需更新）。
- **golden 固件的存放**：合成固件代码化（如 layered_e2e 的 synth() 风格），不存二进制 PNG 进库；冻结数值以字符串字面量入测试。

---

## 附：与 CODE_REVIEW action 清单的关系

- #7（eval_design 升格 + 消灭 LUT 三拷贝 + golden 测试）：本设计步骤 2/3 以 fv-bench 形态完成（比升格 [[bin]] 更进一步——库化使 golden 可被其他工具消费）。
- #2（fvseg 库化）：已落地，本设计步骤 1 的 psnr 委托是其收尾。
- §B.5"六进制 bg 解析移入 fv-common"：步骤 1 完成。
- 新增登记项：eval_design.rs:81 rect 索引 panic（步骤 2 顺带修复，建议单独在提交说明中标注，因其本身值一个 hotfix）。
