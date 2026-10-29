//! 228 软椭圆 α 语义的唯一门面：真源 re-export。
//!
//! 真源 = `fv-fit/src/soft.rs`（C1/C2/C3 常量 + f64 牛顿 5 轮）。
//! 本模块**只 re-export**，不复制常量、不重写公式——bench 的角色是调用真源，
//! 不是替它做数值（FV_BENCH_DESIGN §B.1 纪律条款）。

pub use fv_fit::soft::{alpha_228, C1_228, C2_228, C3_228};

#[cfg(test)]
mod tests {
    use super::*;

    /// 门面锚定：re-export 必须原样指向真源——常量数值 + 端点/单调语义。
    /// （若有人把 re-export 换成本地重实现并改动常数，此处必红。）
    #[test]
    fn facade_points_at_true_source() {
        assert_eq!(C1_228, 0.9132);
        assert_eq!(C2_228, 0.1875);
        assert_eq!(C3_228, -0.1007);
        assert!((alpha_228(0.0) - 1.0).abs() < 1e-6);
        assert_eq!(alpha_228(1.0), 0.0);
        assert!(alpha_228(0.25) > alpha_228(0.75) && alpha_228(0.75) > 0.0);
    }
}
