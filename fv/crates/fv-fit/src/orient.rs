//! B1 结构导引场：一次预处理，量出墨迹宽度分布与局部走向（M19_PROPOSALS §B1）。
//!
//! 背景：sqrt-面积均匀采样对 2–10px 细结构近乎全盲（理想细长椭圆命中 ~10⁻⁸/候选）。
//! 本模块从目标平面（SoA u8，alpha<128 已按 bg 折算）提取：
//! - `angle`：Sobel 梯度的法向（= 墨迹走向），[0,π)
//! - `conf`：Sobel 幅值（低对比区回退均匀角度）
//! - `thin`：窄 spec，max_radius = min(原 max, 墨宽 p95 × 4)

use crate::candidate::CandidateSpec;

/// 每 pix 一格的走向场 + 采样 spec（`orig` 保留原 spec 做 50/50 混采，照片类不回退）。
pub struct StructureGuide {
    pub w: u32,
    pub h: u32,
    pub angle: Vec<f32>,
    pub conf: Vec<f32>,
    pub conf_gate: f32,
    pub thin: CandidateSpec,
    pub orig: CandidateSpec,
}

/// 从目标平面（SoA 3n）+ 背景色构建。`orig` 为引擎当前的采样 spec。
pub fn build_guide(
    target: &[u8],
    bg: [u8; 3],
    w: u32,
    h: u32,
    orig: &CandidateSpec,
) -> StructureGuide {
    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    assert_eq!(target.len(), n * 3);

    // 灰度（三通道均值）与背景灰
    let mut gray = vec![0f32; n];
    let bg_gray = (f32::from(bg[0]) + f32::from(bg[1]) + f32::from(bg[2])) / 3.0;
    for p in 0..n {
        gray[p] = (f32::from(target[p])
            + f32::from(target[n + p])
            + f32::from(target[2 * n + p]))
            / 3.0;
    }

    // ---- 墨迹距离变换（两遍 chamfer，1 / √2 权重）→ 宽度 = 2d ----
    let inf = f32::INFINITY;
    let mut dist = vec![inf; n];
    let fg: Vec<bool> = gray.iter().map(|&g| (g - bg_gray).abs() > 60.0).collect();
    for (p, &f) in fg.iter().enumerate() {
        if f {
            dist[p] = 0.0;
        }
    }
    let s2 = 2f32.sqrt();
    let nb = |x: i64, y: i64, d: &[f32]| -> f32 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            f32::INFINITY
        } else {
            d[y as usize * w + x as usize]
        }
    };
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let p = (y as usize) * w + x as usize;
            let d = dist[p];
            dist[p] = d
                .min(nb(x - 1, y, &dist) + 1.0)
                .min(nb(x - 1, y - 1, &dist) + s2)
                .min(nb(x, y - 1, &dist) + 1.0)
                .min(nb(x + 1, y - 1, &dist) + s2);
        }
    }
    for y in (0..h as i64).rev() {
        for x in (0..w as i64).rev() {
            let p = (y as usize) * w + x as usize;
            let d = dist[p];
            dist[p] = d
                .min(nb(x + 1, y, &dist) + 1.0)
                .min(nb(x + 1, y + 1, &dist) + s2)
                .min(nb(x, y + 1, &dist) + 1.0)
                .min(nb(x - 1, y + 1, &dist) + s2);
        }
    }
    // 墨宽 p95（fg 像素的 2d）
    let mut widths: Vec<f32> = (0..n)
        .filter(|&p| fg[p] && dist[p].is_finite())
        .map(|p| dist[p] * 2.0)
        .collect();
    widths.sort_by(|a, b| a.total_cmp(b));
    let w_p95 = if widths.is_empty() {
        orig.max_radius
    } else {
        widths[((widths.len() as f32 * 0.95) as usize).min(widths.len() - 1)]
    };

    // ---- Sobel 法向场 ----
    let g = |x: i64, y: i64| -> f32 {
        gray[(y.clamp(0, h as i64 - 1) as usize) * w + (x.clamp(0, w as i64 - 1) as usize)]
    };
    let mut angle = vec![0f32; n];
    let mut conf = vec![0f32; n];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let tl = g(x - 1, y - 1);
            let tc = g(x, y - 1);
            let tr = g(x + 1, y - 1);
            let ml = g(x - 1, y);
            let mr = g(x + 1, y);
            let bl = g(x - 1, y + 1);
            let bc = g(x, y + 1);
            let br = g(x + 1, y + 1);
            let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
            let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
            let p = (y as usize) * w + x as usize;
            conf[p] = (gx * gx + gy * gy).sqrt();
            // 墨迹走向 = 梯度法向
            let th = gy.atan2(gx) + core::f32::consts::FRAC_PI_2;
            angle[p] = th.rem_euclid(core::f32::consts::PI);
        }
    }
    let conf_gate = 80.0;

    // ---- 走向传播（3 轮 3×3 最大置信邻取）：墨迹内部是平的（Sobel 无响应），
    //      把边缘的角度/置信向内扩散，细结构中心才拿得到走向 ----
    let (mut angle, mut conf) = (angle, conf);
    for _ in 0..3 {
        let mut na = angle.clone();
        let mut nc = conf.clone();
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let (mut bc, mut ba) = (conf[p], angle[p]);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (xx, yy) = ((x as i64 + dx).clamp(0, w as i64 - 1) as usize,
                                        (y as i64 + dy).clamp(0, h as i64 - 1) as usize);
                        let q = yy * w + xx;
                        if conf[q] > bc {
                            bc = conf[q];
                            ba = angle[q];
                        }
                    }
                }
                nc[p] = bc;
                na[p] = ba;
            }
        }
        angle = na;
        conf = nc;
    }

    let thin_max = (w_p95 * 4.0).min(orig.max_radius).max(orig.min_radius * 2.0);
    StructureGuide {
        w: w as u32,
        h: h as u32,
        angle,
        conf,
        conf_gate,
        thin: CandidateSpec {
            min_radius: orig.min_radius,
            max_radius: thin_max,
        },
        orig: CandidateSpec {
            min_radius: orig.min_radius,
            max_radius: orig.max_radius,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stripe_target(w: u32, h: u32, x0: u32, x1: u32) -> (Vec<u8>, [u8; 3]) {
        let n = (w * h) as usize;
        let mut t = vec![0u8; n * 3];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) as usize;
                let v = if x >= x0 && x < x1 { 255u8 } else { 0u8 };
                for ch in 0..3 {
                    t[ch * n + p] = v;
                }
            }
        }
        (t, [0, 0, 0])
    }

    #[test]
    fn vertical_stripe_guide() {
        // 6px 宽竖条：p95 宽 ≈ 6 → thin_max ≈ 24；走向 ≈ 竖直（π/2 附近，条内边缘处）
        let (t, bg) = stripe_target(64, 64, 29, 35);
        let orig = CandidateSpec { min_radius: 1.2, max_radius: 40.0 };
        let guide = build_guide(&t, bg, 64, 64, &orig);
        assert!(guide.thin.max_radius < orig.max_radius, "窄 spec 应收紧");
        assert!(guide.thin.max_radius >= orig.min_radius * 2.0);
        // 条内部某像素（x=32,y=32，左右都是边缘高置信）
        let p = 32 * 64 + 32;
        assert!(guide.conf[p] > guide.conf_gate);
        let d = (guide.angle[p] - core::f32::consts::FRAC_PI_2)
            .abs()
            .min((guide.angle[p] + core::f32::consts::FRAC_PI_2).abs());
        assert!(d < 0.3, "竖条走向应近竖直，得 {}", guide.angle[p]);
    }

    #[test]
    fn flat_image_falls_back() {
        // 全平图：无 fg、无梯度——thin_max 退回原 spec，conf 全低于 gate
        let n = 32 * 32;
        let t = vec![128u8; n * 3];
        let orig = CandidateSpec { min_radius: 1.2, max_radius: 20.0 };
        let guide = build_guide(&t, [128, 128, 128], 32, 32, &orig);
        assert_eq!(guide.thin.max_radius, orig.max_radius);
        assert!(guide.conf.iter().all(|&c| c <= guide.conf_gate));
    }
}
