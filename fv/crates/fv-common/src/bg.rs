//! 背景色原语：hex 解析与不透明均值（纯原语，无策略）。
//!
//! 住在依赖图底部——拟合（`fv-fit`）与裁判（`fv-bench`）都必须消费同一份
//! 累加/解析实现；策略（何时用哪条选择器）留在各自所有者
//! （fv-fit::cli 全图均值 / fv-seg::pipeline 掩码三级退化，FV_BENCH_DESIGN §B.4）。

/// 解析 6 位十六进制 RRGGBB（大小写均可，可带 `#` 前缀）。
/// 错误文案是既有 CLI 契约（fv-fit / fv-seg 同文案），改动需同步核对两侧。
pub fn parse_hex(spec: &str) -> Result<[u8; 3], String> {
    let s = spec.trim_start_matches('#');
    if s.len() != 6 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("非法背景色 \"{spec}\"：应为 auto 或 RRGGBB"));
    }
    let v = u32::from_str_radix(s, 16).map_err(|e| e.to_string())?;
    Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

/// 谓词式不透明均值：`sel(i)=true` 且 a≥128 的像素逐通道均值四舍五入
/// （u64 累加，`(sum+cnt/2)/cnt`）。返回 (均值，参与像素数)；
/// 无参与像素 → None。
pub fn opaque_mean(img: &image::RgbaImage, sel: impl Fn(usize) -> bool) -> Option<([u8; 3], u64)> {
    let (mut sum, mut cnt) = ([0u64; 3], 0u64);
    for (i, px) in img.pixels().enumerate() {
        if sel(i) && px.0[3] >= 128 {
            for (s, v) in sum.iter_mut().zip(px.0.iter().take(3)) {
                *s += u64::from(*v);
            }
            cnt += 1;
        }
    }
    if cnt == 0 {
        return None;
    }
    let mean: [u8; 3] = core::array::from_fn(|ch| ((sum[ch] + cnt / 2) / cnt).clamp(0, 255) as u8);
    Some((mean, cnt))
}

/// 背景色解析（拟合/引擎目标构造语义）：`auto` → 全图不透明像素（a≥128）逐通道
/// 均值四舍五入（无不透明像素则黑）；否则 [`parse_hex`]。
/// `fv_fit::cli::resolve_bg` 的语义真源（后者为薄壳，签名/错误文案不变）。
pub fn resolve(img: &image::RgbaImage, spec: &str) -> Result<[u8; 3], String> {
    if spec.eq_ignore_ascii_case("auto") {
        return Ok(opaque_mean(img, |_| true).map_or([0, 0, 0], |(m, _)| m));
    }
    parse_hex(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_with(alpha: u8, rgb: [u8; 3]) -> image::RgbaImage {
        let mut img = image::RgbaImage::new(2, 2);
        for px in img.pixels_mut() {
            *px = image::Rgba([rgb[0], rgb[1], rgb[2], alpha]);
        }
        img
    }

    // ---- 从 fv-fit cli.rs 平移的 4 个既有用例（语义原样）----

    #[test]
    fn bg_auto_mean_rounding() {
        // 4 像素：两个不透明 (10,20,30)/(11,21,31)，两个透明 (255,255,255)
        let mut img = image::RgbaImage::new(2, 2);
        let vals = [[10, 20, 30, 255], [11, 21, 31, 255], [255, 255, 255, 0], [1, 2, 3, 10]];
        for (i, px) in img.pixels_mut().enumerate() {
            *px = image::Rgba(vals[i]);
        }
        assert_eq!(resolve(&img, "auto").unwrap(), [11, 21, 31]);
    }

    #[test]
    fn bg_auto_all_transparent_is_black() {
        assert_eq!(resolve(&img_with(0, [9, 9, 9]), "auto").unwrap(), [0, 0, 0]);
    }

    #[test]
    fn bg_hex_variants() {
        assert_eq!(parse_hex("FF8001").unwrap(), [255, 128, 1]);
        assert_eq!(parse_hex("#ff8001").unwrap(), [255, 128, 1]);
        // resolve 走同一真源
        assert_eq!(resolve(&img_with(255, [0; 3]), "#FF8001").unwrap(), [255, 128, 1]);
    }

    #[test]
    fn bg_hex_rejects_garbage() {
        for bad in ["12345", "zzz888", "", "1234567", "12 456"] {
            assert!(parse_hex(bad).is_err(), "应拒绝 {bad}");
            assert!(resolve(&img_with(255, [0; 3]), bad).is_err(), "应拒绝 {bad}");
        }
    }

    // ---- 原语级新增覆盖 ----

    #[test]
    fn auto_alpha_threshold_is_128() {
        // a=127 不算不透明，a=128 算（阈值语义的显式锚定）
        let mut img = image::RgbaImage::new(2, 1);
        img.get_pixel_mut(0, 0).0 = [10, 10, 10, 127];
        img.get_pixel_mut(1, 0).0 = [30, 30, 30, 128];
        assert_eq!(resolve(&img, "auto").unwrap(), [30, 30, 30]);
    }

    #[test]
    fn auto_case_insensitive() {
        assert_eq!(resolve(&img_with(255, [7, 7, 7]), "Auto").unwrap(), [7, 7, 7]);
    }

    #[test]
    fn opaque_mean_counts_samples_and_respects_selector() {
        let mut img = image::RgbaImage::new(2, 1);
        img.get_pixel_mut(0, 0).0 = [10, 20, 30, 255];
        img.get_pixel_mut(1, 0).0 = [30, 40, 50, 255];
        let (mean, cnt) = opaque_mean(&img, |i| i == 0).unwrap();
        assert_eq!(mean, [10, 20, 30]);
        assert_eq!(cnt, 1);
        let (mean_all, cnt_all) = opaque_mean(&img, |_| true).unwrap();
        assert_eq!(mean_all, [20, 30, 40]);
        assert_eq!(cnt_all, 2);
    }

    #[test]
    fn opaque_mean_none_when_no_selected_opaque_pixel() {
        let img = img_with(0, [9, 9, 9]);
        assert!(opaque_mean(&img, |_| true).is_none());
    }

    #[test]
    fn hex_error_message_is_stable() {
        // 错误文案是既有 CLI 契约，冻结防漂移
        assert_eq!(
            parse_hex("12345").unwrap_err(),
            "非法背景色 \"12345\"：应为 auto 或 RRGGBB"
        );
    }
}
