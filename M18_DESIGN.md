# M1.8 GPU 可微渲染器设计（diff-refine）— 契约 v1

目标：对贪心放置的**实心层栈**做联合梯度精修（8 参数/层同时优化），修贪心短视；
τ 退火逼近硬光栅；**质量评测永远用引擎 i64 精确渲染**（可微渲染只用于求梯度）。

## 1. 范围（v1）

- 只精修实心层（容量 ≤ 2048 层），opacity=1。
- 软层 228 不参与（已有 guarded refit）；z 序冻结（不可微）。
- 参数 per layer（`diff::ShapeParams`，8×f32=32B）：
  `cx, cy, a(局部x半轴px), b(局部y半轴px), theta(rad), r, g, bl(0-255)`

## 2. 数学（唯一事实来源，两侧公式必须同源）

### 覆盖与合成
- 旋转坐标：`x' = (px−cx)·cosθ − (py−cy)·sinθ`，`y' = (px−cx)·sinθ + (py−cy)·cosθ`
- `q = x'²/a² + y'²/b²`，`β = σ((1−q)·τ)`（mod.rs `coverage()`；q=1 边界 β=0.5）
- 合成（over，自底向上 i=0..n−1，与引擎画布同向）：
  `C ← β·c_i + (1−β)·C`，初始 `C = bg`
- 等价闭式：`C_final = Σ_i β_i c_i T_i + bg·T_all`，
  其中 `T_i = Π_{j>i}(1−β_j)`（层 i 上方透射），`T_all = T_0`。

### 梯度（MSE 损失，E = mean over 3·H·W of (C−gt)²）
- 残差 `R(p,ch) = 2·(C(p,ch) − gt(p,ch)) / (3·H·W)`
- **颜色**（解析，无几何链）：`∂E/∂c_i,ch = Σ_p R·β_i·T_i`
- **几何**：β 标量作用于三通道，`∂E/∂β_i(p) = Σ_ch R·T_i·(c_i,ch − Cbelow_i,ch(p))`
  其中 `Cbelow_i(p)` = 前向循环处理层 i **之前**的画布（下层合成色）。
  β 链：`dβ/dq = −τ·β·(1−β)`
  q 链（在层 i 局部坐标处）：
  - `dq/dcx = −(2x'·cosθ/a² + 2y'·sinθ/b²)`
  - `dq/dcy = 2x'·sinθ/a² − 2y'·cosθ/b²`（v1 勘误：无外提负号——∂x'/∂cy=+s 与 ∂y'/∂cy=−c 混号）
  - `dq/da = −2x'²/a³`，`dq/db = −2y'²/b³`
  - `dq/dθ = 2x'y'·(1/b² − 1/a²)`
  每像素贡献：`s(p) = ∂E/∂β_i(p) · dβ/dq`，随后
  `g_param = Σ_p s(p)·(dq/dparam)`。
  注意颜色梯度与几何梯度**独立**（颜色走 β 路径之外的直通项）。

### 快照（backward 依赖的两块大缓冲）
- `Cbelow[i]`：H×W×3 f32，层 i 合成前画布
- `Tabove[i]`：H×W×1 f32，层 i 上方透射（自顶向下扫：T=1 起步，先记录再乘 (1−β_i)）
两块都在前向 dispatch 中写（两次扫或一次双写，实现自选），容量 max_layers×w×h 分配。

## 3. 缓冲与内核契约

- 全部 f32；快照平面长 = `w*h`（**不是** h·(w+1)——eval_soft 的教训）
- 缓冲按容量全量分配（wgpu 拒绝零尺寸/变尺寸重绑）
- WGSL 标识符避开保留字 `target`（用 `tgt`）；i64 字面量写 `i64(0)`（本模块用不到）
- 后端限 Vulkan（沿用 GpuEvaluator 实践，all-backups 枚举挂 GL）
- 禁 atomics；归约固定树（确定性是测试前提）
- params uniform（8×f32）：`[w, h, n, tau, bg.r, bg.g, bg.bl, pad]`

### fwd.rs（Agent D1）
```rust
pub struct DiffRenderer { /* device/queue/管线/缓冲, pub(crate) 字段 */ }
impl DiffRenderer {
    pub fn new(w: u32, h: u32, max_layers: usize) -> Result<Self, String>;
    /// 渲染 + 写快照 + 存 target；返回 canvas（f32×3 平面交错 RGB）。
    pub fn render(&mut self, shapes: &[ShapeParams], tau: f32, bg: [f32; 3], target: &[f32]) -> Result<Vec<f32>, String>;
    /// 纯 Rust 参考（f64 内累积、f32 出），测试基准；bwd 的 FD 测试复用。
    pub fn cpu_forward_reference(shapes: &[ShapeParams], tau: f32, bg: [f32; 3], w: u32, h: u32) -> Vec<f32>;
    // pub(crate) 访问器：device(), queue(), w(), h(), shapes_buf(), cbelow_buf(),
    //                    tabove_buf(), canvas_buf(), target_buf(), params 流程自管
    // grads 缓冲也在 new() 分配：max_layers×8 f32，STORAGE|COPY_SRC
}
```
测试门槛：GPU vs cpu_forward_reference，τ∈{4, 64}，6 层小场景（48×32，含旋转/细长轴），
max |diff| < 0.5（0-255 标度）；τ=512 时与硬椭圆（β∈{0,1}，band 像素 ≤2% 除外）一致。

### bwd.rs（Agent D2）
```rust
impl DiffRenderer {
    /// 梯度 of MSE wrt 每层 8 参数；行主序 n×8。必须在 render() 之后调用。
    pub fn backward(&mut self) -> Result<Vec<f32>, String>;
}
```
测试门槛：FD（中心差分）对 CPU 参考前向（cpu_forward_reference），5 层 40×40 场景，
τ=4：每参数 rel err < 1e-2 或 abs < 1e-4（f32 求和与 f64 FD 的现实容忍度）；
另需一测：单层纯平移梯度方向正确（画布上目标在层右侧 → g_cx < 0 应推层右移，验证符号约定）。

## 4. 优化器与集成（主会话实现，Agent D3 职责由我合并）

- Adam（CPU，n×8 规模微不足道）：β1=0.9 β2=0.999 ε=1e-8
- 学习率初值：几何 0.35px、颜色 1.2（255 标度），随 τ 增长衰减 ×0.97/iter
- τ 日程：τ_k = 4·1.3^k，clip 192；每 iter 前向+反向一次
- 裁剪：a,b ∈ [1.5, 512] px；θ 不裁；颜色 [0,255]
- 收口：迭代完跑一次引擎 i64 精确渲染评测（eval_design 同裁判），**劣化则回滚到迭代前**
- 集成点：fvfit `--diff-refine <iters>`（实心贪心+polish 之后、soft 之前）；lineage src="diff"

## 5. 验收

1. fwd/bwd 两个测试门槛全绿（上文数值）
2. ayylmao 328²、300 实心层、50 iter：eval_design PSNR 提升 ≥ +0.3dB（或给出失败分析）
3. 性能：300 层 328² 一次前向+反向 < 30ms（N1X Vulkan），50 iter < 3s
4. 全 workspace `cargo test` 绿 + `clippy` 0

## 6. 备选思路（Agent D4 竞标评审）

- 颜色 Gauss-Newton（每层闭式，几何交给 polish）——更便宜但修不了贪心几何短视
- 分块坐标 Adam（每 iter 只更新 1/k 层）——省快照内存，代价迭代数×k
- τ 直接取大 + 依赖 β 稀疏带 —— 梯度消失风险
D4 从数学正确性（尤其 §2 符号/链式）、失败模式（层塌缩、梯度消失、z 序冻结的影响）、
以上备选的性价比三方面输出书面评审到 forza-vinyl/M18_REVIEW_D4.md，并给 go/no-go。
