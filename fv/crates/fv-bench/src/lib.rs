//! fv-bench：Forza vinyl 的裁判库——PSNR/MAE 评估与背景/α 语义的统一门面。
//!
//! ## 口径总纲（哪个函数是什么口径）
//! - 记分板口径（float 参考）：[`judge::psnr_f64`]——f64 平面、无逐层 u8 取整。
//! - 画布对画布口径：[`judge::psnr_mae_u8`]——u8 平面（jitter/b2 类实验口径）。
//! - 图像对口径：[`judge::psnr_rgba`]——渲染图 vs 原图，透明原图像素按 bg 替换；
//!   此公式为唯一真源（`fv_seg::pipeline::psnr_db` 与之逐位同公式，后续变薄委托）。
//!
//! ## 纪律条款（评审阻断项，FV_BENCH_DESIGN §B.1）
//! - α 228 语义唯一真源 = `fv_fit::soft::alpha_228`；本 crate 只 re-export
//!   （[`alpha`] 模块），出现第二份 α 公式（含"f64 转 f32 再手算一遍"的变体）即阻断。
//! - bg 解析/累加唯一真源 = `fv_common::bg` 原语；[`bg`] 模块只是组装凭证的门面。
//! - 本 crate 不依赖 fv-gpu：保持"任何机器可跑 golden 测试"的确定性。
//!
//! ## 迁移状态（FV_BENCH_DESIGN §C）
//! 步骤 1（本步）：bg 原语下沉 fv-common + bench 骨架 + psnr_rgba 真源。
//! 步骤 2/3（未做）：render_ref、JudgeBuilder、WGSL 锚定、examples 收编。

pub mod alpha;
pub mod bg;
pub mod judge;
