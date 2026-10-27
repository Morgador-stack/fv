//! M1.8 可微渲染：契约模块。
//!
//! 分工：`fwd`（前向软光栅 + 快照）与 `bwd`（反向梯度）分文件实现，
//! 本模块只钉死共享契约，两边不得改动对方的公式常数。
//!
//! 契约要点（详见 forza-vinyl/M18_DESIGN.md）：
//! - 所有像素/参数缓冲 f32；快照平面长 = w*h（不是 h*(w+1)，eval_soft 的教训）
//! - 覆盖函数 β = σ((1−q)·τ)，q = x'²/a² + y'²/b²（旋转坐标）；τ 全局统一
//! - 确定性：内核禁 atomics，归约用固定树
//! - 颜色 0-255 f32 标度；几何 px；θ rad；z 序冻结不参与梯度
//! - 缓冲一律按容量分配（wgpu 拒绝零尺寸绑定）；标识符避开 `target`（WGSL 保留字，用 tgt）

/// 每层 8 参数，32B（WGSL storage 布局安全）。
/// a=旋转前局部 x 半轴，b=局部 y 半轴，bl=蓝通道（避免与半轴 b 重名）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeParams {
    pub cx: f32,
    pub cy: f32,
    pub a: f32,
    pub b: f32,
    pub theta: f32,
    pub r: f32,
    pub g: f32,
    pub bl: f32,
}

pub const SHAPE_FLOATS: usize = 8;

/// 平滑覆盖（CPU 参考实现与内核公式必须逐位同源）。
#[inline]
pub fn coverage(q: f32, tau: f32) -> f32 {
    let z = (1.0 - q) * tau;
    1.0 / (1.0 + (-z).exp())
}

/// 前向模块（Agent D1 实现）：`DiffRenderer` 拥有设备与全部缓冲，
/// bwd 侧通过 pub(crate) 访问器复用，不再自建设备。
pub mod fwd;

/// 反向模块（Agent D2 实现）：为 `DiffRenderer` 追加 `backward` impl 块。
pub mod bwd;

/// 优化器驱动（主会话实现）：Adam + τ 退火 + 硬光栅收口。
pub mod refine;
