//! fv-inject：把 geometrize 设计 JSON 写入运行中 Forza 游戏的 vinyl 图层（M2）。
//!
//! 架构：所有扫描/校验/写入逻辑面向 [`mem::MemIo`] trait 实现，
//! 可用 [`mem::MockMem`] 全链路离线测试；Windows 实机走 [`win`]（WinAPI）。
//!
//! 安全设计（继承 vinylizer 的严格模式）：
//! - 定位要求模板**全部采样层**通过严格校验（shape_id ∈ {矩形,椭圆}、mask==0）
//! - 每层写入前读旧值，失败逐字段回滚
//! - 默认 dry-run + 抽样人工确认后才真写

pub mod convert;
pub mod layout;
pub mod mem;
pub mod scan;
#[cfg(windows)]
pub mod win;
pub mod write_process;
