//! 背景口径的裁判侧门面：BgSpec → Bg（颜色 + 来源凭证）。
//!
//! 解析/累加真源 = `fv_common::bg` 原语（拟合与裁判天然同规则）；
//! 本模块不重写任何累加或解析逻辑，只组装"颜色 + 凭证"——
//! bg 配错是语义事故（跨实例 9473 vs 61 的假灾难），报告必须能自证口径。

use image::RgbaImage;

/// 背景规格：显式 hex 或 auto。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgSpec {
    /// 全图不透明（a≥128）像素逐通道均值（与 fv-fit 目标构造同规则）。
    Auto,
    /// 显式 RRGGBB。
    Hex([u8; 3]),
}

/// 解析结果：颜色 + **来源凭证**（§D：必须留痕，报告可见）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bg {
    pub rgb: [u8; 3],
    pub spec: BgSpec,
    /// Auto 时记录参与均值的像素数（0 = 全透明图退化为黑，报告中可见）。
    pub auto_samples: u64,
}

/// CLI 字符串（"auto" | "RRGGBB" | "#rrggbb"）→ BgSpec。
pub fn parse_bg_spec(s: &str) -> Result<BgSpec, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(BgSpec::Auto);
    }
    Ok(BgSpec::Hex(fv_common::bg::parse_hex(s)?))
}

/// 解析背景：颜色 + 凭证一次成型。
/// Auto 的 `auto_samples` 即参与均值的不透明像素数；Hex 时为 0。
pub fn resolve_bg(spec: BgSpec, img: &RgbaImage) -> Bg {
    match spec {
        BgSpec::Auto => {
            let (rgb, auto_samples) =
                fv_common::bg::opaque_mean(img, |_| true).unwrap_or(([0, 0, 0], 0));
            Bg { rgb, spec, auto_samples }
        }
        BgSpec::Hex(rgb) => Bg { rgb, spec, auto_samples: 0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixed_img() -> RgbaImage {
        // 2×1：一个不透明 (10,20,30)，一个透明 (255,255,255)
        let mut img = RgbaImage::new(2, 1);
        img.get_pixel_mut(0, 0).0 = [10, 20, 30, 255];
        img.get_pixel_mut(1, 0).0 = [255, 255, 255, 0];
        img
    }

    #[test]
    fn auto_carries_credential() {
        let bg = resolve_bg(BgSpec::Auto, &mixed_img());
        assert_eq!(bg.rgb, [10, 20, 30]);
        assert_eq!(bg.auto_samples, 1);
        assert_eq!(bg.spec, BgSpec::Auto);
    }

    #[test]
    fn auto_all_transparent_is_black_with_zero_samples() {
        let mut img = RgbaImage::new(1, 1);
        img.get_pixel_mut(0, 0).0 = [9, 9, 9, 0];
        let bg = resolve_bg(BgSpec::Auto, &img);
        assert_eq!(bg.rgb, [0, 0, 0]);
        assert_eq!(bg.auto_samples, 0);
    }

    #[test]
    fn hex_passthrough_no_samples() {
        let bg = resolve_bg(BgSpec::Hex([255, 128, 1]), &mixed_img());
        assert_eq!(bg.rgb, [255, 128, 1]);
        assert_eq!(bg.auto_samples, 0);
    }

    #[test]
    fn parse_spec_variants() {
        assert_eq!(parse_bg_spec("auto").unwrap(), BgSpec::Auto);
        assert_eq!(parse_bg_spec("AUTO").unwrap(), BgSpec::Auto);
        assert_eq!(parse_bg_spec("#ff8001").unwrap(), BgSpec::Hex([255, 128, 1]));
        assert!(parse_bg_spec("12345").is_err());
    }

    #[test]
    fn auto_matches_fv_common_resolve() {
        // 门面与拟合侧 resolve 同规则（同一 fv-common 原语）
        let img = mixed_img();
        let bg = resolve_bg(BgSpec::Auto, &img);
        assert_eq!(bg.rgb, fv_common::bg::resolve(&img, "auto").unwrap());
    }
}
