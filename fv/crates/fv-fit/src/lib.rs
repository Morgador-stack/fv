//! fv-fit：图片 → vinyl 形状拟合器（M1：贪心 + 积分图 O(高度) 评分）。
//!
//! 模块分工（M1）：
//! - `raster`：旋转椭圆的逐行区间光栅化与画布合成（唯一的几何真源）
//! - `rowsum`：每行前缀和（积分图），支撑 O(高度) 的候选评分
//! - `candidate`：候选形状采样（xoshiro128++ 计数式 RNG）
//! - `engine`：贪心主循环（多线程评分 + 爬山精修 + 增量画布 + checkpoint）
//! - `render`：最终形状 → 预览 PNG
//! - `cli`：命令行入口

pub mod candidate;
pub mod cli;
pub mod engine;
pub mod orient;
pub mod polish;
pub mod raster;
pub mod render;
pub mod rowsum;
pub mod soft;
pub mod stroke;
