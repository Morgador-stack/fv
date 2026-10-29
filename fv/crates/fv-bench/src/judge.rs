//! 评审入口：各口径的 PSNR/MAE（口径总纲见 crate doc）。
//!
//! - [`psnr_f64`]：float 参考口径（记分板）。
//! - [`psnr_mae_u8`]：u8 画布对画布（PSNR + MAE）。
//! - [`psnr_rgba`]：图像对口径唯一真源——与 `fv_seg::pipeline::psnr_db` 逐位同公式
//!   （i64 累加、zip 截断语义），后续 fv-seg 变薄委托；改动任一侧都必须保持
//!   两侧数值逐位一致（golden 测试锁定）。

/// 由平方误差和与样本数得 PSNR（255 满幅口径）。n=0 或 mse≤0 → inf。
fn psnr_from_sse(sse: f64, n: usize) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    let mse = sse / n as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// 两裸画布 PSNR（f64 平面，float 参考口径）。n=0 → inf。
pub fn psnr_f64(target: &[f64], render: &[f64]) -> f64 {
    assert_eq!(target.len(), render.len(), "psnr_f64: 两画布长度必须一致");
    let sse: f64 = target.iter().zip(render).map(|(t, r)| (t - r) * (t - r)).sum();
    psnr_from_sse(sse, target.len())
}

/// 两裸画布 PSNR + MAE（u8 平面；逐元素比较，布局无关，调用方保证同布局）。
/// n=0 → (inf, 0)。长度不一致 panic（评审不得静默错位比较）。
pub fn psnr_mae_u8(a: &[u8], b: &[u8]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "psnr_mae_u8: 两画布长度必须一致");
    let mut sse = 0u64;
    let mut abs_sum = 0u64;
    for (x, y) in a.iter().zip(b) {
        let d = i64::from(*x) - i64::from(*y);
        sse += (d * d) as u64;
        abs_sum += d.unsigned_abs();
    }
    let n = a.len();
    let mae = if n == 0 { 0.0 } else { abs_sum as f64 / n as f64 };
    (psnr_from_sse(sse as f64, n), mae)
}

/// 渲染图 vs 原图：透明（a<128）原图像素按 bg 替换后逐像素 3 通道 MSE → PSNR。
/// **图像对口径的唯一真源**：函数体与 `fv_seg::pipeline::psnr_db` 逐位同公式
/// （含两图尺寸不齐时 `zip` 截断到较小者的既有语义）。
pub fn psnr_rgba(render: &image::RgbaImage, orig: &image::RgbaImage, bg: [u8; 3]) -> f64 {
    let mut se = 0i64;
    let mut n = 0usize;
    for (o, r) in orig.pixels().zip(render.pixels()) {
        let c = if o.0[3] < 128 { bg } else { [o.0[0], o.0[1], o.0[2]] };
        for (cv, rv) in c.iter().zip(r.0.iter().take(3)) {
            let d = i64::from(*cv) - i64::from(*rv);
            se += d * d;
        }
        n += 1;
    }
    let mse = se as f64 / (3.0 * n.max(1) as f64);
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_length_mismatch_panics() {
        let result = std::panic::catch_unwind(|| psnr_mae_u8(&[0u8; 3], &[0u8; 4]));
        assert!(result.is_err(), "长度不一致必须 panic");
        let result = std::panic::catch_unwind(|| psnr_f64(&[0.0; 3], &[0.0; 4]));
        assert!(result.is_err(), "长度不一致必须 panic");
    }

    #[test]
    fn rgba_empty_pixels_do_not_panic() {
        // n.max(1) 守卫：0 像素对（两空图）→ inf，不除零
        let a = image::RgbaImage::new(0, 0);
        assert_eq!(psnr_rgba(&a, &a, [0, 0, 0]), f64::INFINITY);
    }
}
