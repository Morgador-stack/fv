//! M1.8 优化器驱动：Adam + τ 退火 + 硬光栅精确收口。
//!
//! 与 fwd/bwd 的契约：`DiffRenderer::render(shapes, tau, bg, target)` 返回画布并
//! 写快照，随后 `backward()` 返回 n×8 行主序 MSE 梯度（顺序同 ShapeParams 字段序）。
//! 收口裁定用本地硬覆盖（q<1，像素中心）合成——与引擎行 span 光栅同语义；
//! CLI 层最终裁定仍以 fv-fit eval_design（i64 积分图口径）为准。

use super::fwd::DiffRenderer;
use super::ShapeParams;
use super::SHAPE_FLOATS;

/// 精修配置（M18_DESIGN §4 默认值的物化）。
#[derive(Debug, Clone)]
pub struct RefineConfig {
    pub iters: usize,
    pub lr_geo: f32,
    pub lr_col: f32,
    pub lr_decay: f32,
    pub tau0: f32,
    pub tau_growth: f32,
    pub tau_max: f32,
    /// 每隔多少 iter 用硬覆盖精确评一次 PSNR（0 = 只评首末）。
    pub eval_every: usize,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            iters: 50,
            lr_geo: 0.35,
            lr_col: 1.2,
            lr_decay: 0.97,
            tau0: 4.0,
            tau_growth: 1.3,
            tau_max: 192.0,
            eval_every: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefineReport {
    pub psnr_start: f64,
    pub psnr_best: f64,
    pub best_iter: usize,
    pub accepted: bool,
    pub tau_final: f32,
    pub trail: Vec<(usize, f64)>,
}

/// 联合精修实心层栈。target 为 0-255 f32 RGB 平面交错（w*h*3）。
/// 返回（最优形状集, 报告）；accepted=false 时返回原始输入。
pub fn diff_refine(
    shapes: Vec<ShapeParams>,
    w: u32,
    h: u32,
    bg: [f32; 3],
    target: &[f32],
    cfg: &RefineConfig,
) -> Result<(Vec<ShapeParams>, RefineReport), String> {
    let n = shapes.len();
    if n == 0 {
        return Ok((
            shapes,
            RefineReport {
                psnr_start: f64::INFINITY,
                psnr_best: f64::INFINITY,
                best_iter: 0,
                accepted: false,
                tau_final: cfg.tau0,
                trail: Vec::new(),
            },
        ));
    }
    if target.len() != (w as usize) * (h as usize) * 3 {
        return Err(format!(
            "target len {} != w*h*3 {}",
            target.len(),
            (w as usize) * (h as usize) * 3
        ));
    }

    let mut r = DiffRenderer::new(w, h, n)?;
    let cap = r.capacity();
    // v2：大栈分窗口（每迭代逐窗口取梯度，窗口内梯度与全量精确一致——
    // 全栈渲染、仅存窗口快照）。窗口大小 = 渲染器容量。
    let windows: Vec<(usize, usize)> = if n <= cap {
        vec![(0, n)]
    } else {
        (0..n).step_by(cap).map(|s| (s, (s + cap).min(n))).collect()
    };
    let psnr_start = hard_psnr(&shapes, w, h, bg, target);
    let mut best = (psnr_start, shapes.clone());
    let mut trail = vec![(0usize, psnr_start)];

    // Adam 矩阵：n×8，几何 0..5（cx cy a b θ），颜色 5..8。
    let np = n * SHAPE_FLOATS;
    let mut m = vec![0f32; np];
    let mut v = vec![0f32; np];

    let mut cur = shapes;
    let mut full_grads = vec![0f32; np];
    for k in 0..cfg.iters {
        let tau = (cfg.tau0 * cfg.tau_growth.powi(k as i32)).min(cfg.tau_max);
        for &(lo, hi) in &windows {
            r.render_windowed(&cur, tau, bg, target, (lo, hi))?;
            let grads = r.backward()?;
            full_grads[lo * SHAPE_FLOATS..hi * SHAPE_FLOATS]
                .copy_from_slice(&grads[lo * SHAPE_FLOATS..hi * SHAPE_FLOATS]);
        }
        let grads = &full_grads;

        let lr_g = cfg.lr_geo * cfg.lr_decay.powi(k as i32);
        let lr_c = cfg.lr_col * cfg.lr_decay.powi(k as i32);
        let t = k as f32 + 1.0;
        let bc1 = 1.0 - 0.9f32.powf(t);
        let bc2 = 1.0 - 0.999f32.powf(t);
        for (i, val) in cur.iter_mut().enumerate() {
            for p in 0..SHAPE_FLOATS {
                let idx = i * SHAPE_FLOATS + p;
                let g = grads[idx];
                m[idx] = 0.9 * m[idx] + 0.1 * g;
                v[idx] = 0.999 * v[idx] + 0.001 * g * g;
                let step = (m[idx] / bc1) / ((v[idx] / bc2).sqrt() + 1e-8);
                // 单位不可比：cx/cy/a/b 是像素，θ 是弧度（0.2 rad = 11.5°，会毁掉整栈）
                let lr = match p {
                    0..=3 => lr_g,
                    4 => lr_g * 0.05,
                    _ => lr_c,
                };
                let slot = match p {
                    0 => &mut val.cx,
                    1 => &mut val.cy,
                    2 => &mut val.a,
                    3 => &mut val.b,
                    4 => &mut val.theta,
                    5 => &mut val.r,
                    6 => &mut val.g,
                    _ => &mut val.bl,
                };
                *slot -= lr * step;
            }
            clip_shape(val, w, h);
        }

        if (cfg.eval_every > 0 && (k + 1) % cfg.eval_every == 0) || k + 1 == cfg.iters {
            let p = hard_psnr(&cur, w, h, bg, target);
            trail.push((k + 1, p));
            if p > best.0 {
                best = (p, cur.clone());
            }
        }
    }

    let tau_final = (cfg.tau0 * cfg.tau_growth.powi(cfg.iters.saturating_sub(1) as i32))
        .min(cfg.tau_max);
    let accepted = best.0 > psnr_start;
    Ok((
        best.1,
        RefineReport {
            psnr_start,
            psnr_best: best.0,
            best_iter: trail
                .iter()
                .copied()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(k, _)| k)
                .unwrap_or(0),
            accepted,
            tau_final,
            trail,
        },
    ))
}

/// 裁剪（M18_DESIGN §4）：半轴下限防塌缩，颜色 [0,255]，中心允许适度离屏。
fn clip_shape(s: &mut ShapeParams, w: u32, h: u32) {
    let lim_x = w as f32 + 64.0;
    let lim_y = h as f32 + 64.0;
    s.cx = s.cx.clamp(-64.0, lim_x);
    s.cy = s.cy.clamp(-64.0, lim_y);
    s.a = s.a.clamp(1.5, 512.0);
    s.b = s.b.clamp(1.5, 512.0);
    s.r = s.r.clamp(0.0, 255.0);
    s.g = s.g.clamp(0.0, 255.0);
    s.bl = s.bl.clamp(0.0, 255.0);
}

/// 硬覆盖合成 PSNR（引擎语义近似：q<1 像素中心、over 合成、bg 起点）。
fn hard_psnr(shapes: &[ShapeParams], w: u32, h: u32, bg: [f32; 3], target: &[f32]) -> f64 {
    let (w, h) = (w as usize, h as usize);
    let mut canvas = vec![0f64; w * h * 3];
    for p in 0..w * h {
        canvas[3 * p] = bg[0] as f64;
        canvas[3 * p + 1] = bg[1] as f64;
        canvas[3 * p + 2] = bg[2] as f64;
    }
    for s in shapes {
        let (cos, sin) = (s.theta.cos(), s.theta.sin());
        let rad = s.a.max(s.b) + 1.0;
        let x0 = (s.cx - rad).floor().max(0.0) as usize;
        let x1 = (s.cx + rad).ceil().min(w as f32) as usize;
        let y0 = (s.cy - rad).floor().max(0.0) as usize;
        let y1 = (s.cy + rad).ceil().min(h as f32) as usize;
        let col = [s.r as f64, s.g as f64, s.bl as f64];
        for y in y0..y1 {
            for x in x0..x1 {
                let dx = x as f32 + 0.5 - s.cx;
                let dy = y as f32 + 0.5 - s.cy;
                let xp = dx * cos - dy * sin;
                let yp = dx * sin + dy * cos;
                let q = xp * xp / (s.a * s.a) + yp * yp / (s.b * s.b);
                if q < 1.0 {
                    let i = (y * w + x) * 3;
                    canvas[i..i + 3].copy_from_slice(&col);
                }
            }
        }
    }
    let mut se = 0f64;
    for (c, t) in canvas.iter().zip(target.iter()) {
        let d = c - f64::from(*t);
        se += d * d;
    }
    let mse = se / (w * h * 3) as f64;
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
    fn clip_bounds() {
        let mut s = ShapeParams {
            cx: -1e9,
            cy: 1e9,
            a: 0.0,
            b: 1e9,
            theta: 0.0,
            r: -5.0,
            g: 300.0,
            bl: 128.0,
        };
        clip_shape(&mut s, 328, 328);
        assert!(s.a >= 1.5 && s.b <= 512.0 && s.cx >= -64.0 && s.r >= 0.0 && s.g <= 255.0);
    }

    #[test]
    fn hard_psnr_perfect() {
        let w = 8u32;
        let target = vec![100f32; (w * w * 3) as usize];
        let s = ShapeParams {
            cx: 4.0,
            cy: 4.0,
            a: 8.0,
            b: 8.0,
            theta: 0.0,
            r: 100.0,
            g: 100.0,
            bl: 100.0,
        };
        let p = hard_psnr(&[s], w, w, [100.0; 3], &target);
        assert!(p > 60.0, "full-cover same-color should be near-perfect, got {p}");
    }
}
