//! 228 软椭圆（FH6 游戏形状语义）：混合合成、候选评估、软栈颜色重拟合。
//!
//! ## 游戏侧渐变（形状 228 的径向衰减，从 FH6 渲染行为反推的解析式）
//! 归一化半径 rn ∈ [0,1) 的覆盖度 a 满足 `1 − C1·a − C2·a² − C3·a³ = rn`
//! （牛顿解，中心 1 → 边缘 0）。有效 α = a(rn) × 不透明度。
//!
//! ## 混合候选的闭式最优
//! 候选 (几何 E, 颜色 k) 在当前画布 C 上混合：new(p) = C + β·(k − C)，β = a(rn)·o。
//! **o 与 k 在误差中退化等价**（β·o 线性进入），故固定 o=1（alpha 字节 255）只优化 k：
//! ```text
//! e(p)   = β·k − g(p)，g = T − (1−β)·C
//! k*_ch  = Σβg_ch / Σβ²
//! delta  = ΣΔ² − err(k*)，Δ = T − C；用取整后的 k 带回公式得精确 delta
//! ```
//! 全程 f64（CPU 参考）——软层是统计性改进，无需 i64 精确。

use crate::raster::EllipseParams;

pub const C1_228: f64 = 0.9132;
pub const C2_228: f64 = 0.1875;
pub const C3_228: f64 = -0.1007;

/// 覆盖度 a(rn)，牛顿 5 轮（与游戏语义一致；rn ≥ 1 → 0）。
#[inline]
pub fn alpha_228(rn: f64) -> f64 {
    if rn >= 1.0 {
        return 0.0;
    }
    let mut a = 1.0 - rn;
    for _ in 0..5 {
        let fa = 1.0 - C1_228 * a - C2_228 * a * a - C3_228 * a * a * a;
        let fpa = -(C1_228 + 2.0 * C2_228 * a + 3.0 * C3_228 * a * a);
        a -= (fa - rn) / fpa;
    }
    a.clamp(0.0, 1.0)
}

/// 把软椭圆按混合语义合成到画布（SoA u8，3 平面）。α = β·opacity。
pub fn blend_composite(
    canvas: &mut [u8],
    w: u32,
    h: u32,
    e: &EllipseParams,
    color: [u8; 3],
    opacity: f64,
) {
    let n = (w * h) as usize;
    let (cx, cy) = (f64::from(e.cx), f64::from(e.cy));
    let (a, b) = (f64::from(e.rx), f64::from(e.ry));
    if !e.cx.is_finite() || !e.cy.is_finite() || !e.rx.is_finite() || !e.ry.is_finite() {
        return;
    }
    if a <= 0.0 || b <= 0.0 {
        return;
    }
    let th = f64::from(e.angle_rad);
    let (c, s) = (th.cos(), th.sin());
    let ia = 1.0 / (a * a);
    let ib = 1.0 / (b * b);
    let big_a = c * c * ia + s * s * ib;
    let big_b = c * s * (ia - ib);
    let big_c = s * s * ia + c * c * ib;
    let yext = (a * a * s * s + b * b * c * c).sqrt();
    let y0 = (cy - yext - 1.0).floor().max(0.0) as u32;
    let y1 = ((cy + yext + 1.0).ceil() as u32).min(h);
    for y in y0..y1 {
        let dy = f64::from(y) + 0.5 - cy;
        let q = big_b * big_b * dy * dy - big_a * (dy * dy * big_c - 1.0);
        if q < 0.0 {
            continue;
        }
        let hw = q.sqrt() / big_a;
        let pm = -big_b * dy / big_a;
        let x0f = (cx - 0.5 + pm - hw).ceil();
        let x1f = (cx - 0.5 + pm + hw).floor() + 1.0;
        if x1f <= 0.0 || x0f >= f64::from(w) {
            continue;
        }
        let x0 = x0f.max(0.0) as u32;
        let x1 = (x1f as u32).min(w);
        let row = y as usize * w as usize;
        for x in x0..x1 {
            let dx = f64::from(x) + 0.5 - cx;
            let u = c * dx + s * dy;
            let v = -s * dx + c * dy;
            let r2 = u * u * ia + v * v * ib;
            if r2 >= 1.0 {
                continue;
            }
            let beta = alpha_228(r2.sqrt()) * opacity;
            if beta <= 0.0 {
                continue;
            }
            let p = row + x as usize;
            for ch in 0..3 {
                let cv = f64::from(canvas[ch * n + p]);
                let nv = cv + beta * (f64::from(color[ch]) - cv);
                canvas[ch * n + p] = nv.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// 软候选评估（f64 参考）：返回 (最优整数色, 精确 delta)。
/// delta 为把取整 k 带回误差公式的精确值（非连续近似）。
pub fn eval_soft(
    target: &[u8],
    canvas: &[u8],
    w: u32,
    h: u32,
    e: &EllipseParams,
) -> Option<([u8; 3], i64)> {
    let n = (w * h) as usize;
    let (cx, cy) = (f64::from(e.cx), f64::from(e.cy));
    let (a, b) = (f64::from(e.rx), f64::from(e.ry));
    if !e.cx.is_finite() || !e.cy.is_finite() || !e.rx.is_finite() || !e.ry.is_finite() {
        return None; // inf/NaN 候选（如 JSON 1e999）不得进评分
    }
    if a <= 0.0 || b <= 0.0 {
        return None;
    }
    let th = f64::from(e.angle_rad);
    let (c, s) = (th.cos(), th.sin());
    let ia = 1.0 / (a * a);
    let ib = 1.0 / (b * b);
    let big_a = c * c * ia + s * s * ib;
    let big_b = c * s * (ia - ib);
    let big_c = s * s * ia + c * c * ib;
    let yext = (a * a * s * s + b * b * c * c).sqrt();
    let y0 = (cy - yext - 1.0).floor().max(0.0) as u32;
    let y1 = ((cy + yext + 1.0).ceil() as u32).min(h);

    let (mut sb2, mut sbg, mut sg2, mut sd2) = (0f64, [0f64; 3], [0f64; 3], [0f64; 3]);
    for y in y0..y1 {
        let dy = f64::from(y) + 0.5 - cy;
        let q = big_b * big_b * dy * dy - big_a * (dy * dy * big_c - 1.0);
        if q < 0.0 {
            continue;
        }
        let hw = q.sqrt() / big_a;
        let pm = -big_b * dy / big_a;
        let x0f = (cx - 0.5 + pm - hw).ceil();
        let x1f = (cx - 0.5 + pm + hw).floor() + 1.0;
        if x1f <= 0.0 || x0f >= f64::from(w) {
            continue;
        }
        let x0 = x0f.max(0.0) as u32;
        let x1 = (x1f as u32).min(w);
        let row = y as usize * w as usize;
        for x in x0..x1 {
            let dx = f64::from(x) + 0.5 - cx;
            let u = c * dx + s * dy;
            let v = -s * dx + c * dy;
            let r2 = u * u * ia + v * v * ib;
            if r2 >= 1.0 {
                continue;
            }
            let beta = alpha_228(r2.sqrt());
            if beta <= 0.0 {
                continue;
            }
            let p = row + x as usize;
            sb2 += beta * beta;
            for ch in 0..3 {
                let t = f64::from(target[ch * n + p]);
                let cv = f64::from(canvas[ch * n + p]);
                let g = t - (1.0 - beta) * cv;
                let d = t - cv;
                sbg[ch] += beta * g;
                sg2[ch] += g * g;
                sd2[ch] += d * d;
            }
        }
    }
    if sb2 <= 0.0 {
        return None;
    }
    let k: [u8; 3] = core::array::from_fn(|ch| (sbg[ch] / sb2).round().clamp(0.0, 255.0) as u8);
    let kk = k.map(f64::from);
    let mut err = 0f64;
    let mut cur = 0f64;
    for ch in 0..3 {
        err += kk[ch] * kk[ch] * sb2 - 2.0 * kk[ch] * sbg[ch] + sg2[ch];
        cur += sd2[ch];
    }
    let delta = (cur - err).round() as i64;
    if delta <= 0 {
        return None;
    }
    Some((k, delta))
}

/// 软栈重拟合报告。
#[derive(Debug, Clone)]
pub struct SoftRefitReport {
    pub layers: usize,
    pub refits: usize,
    pub mse_before: f64,
    pub mse_after: f64,
}

/// 遍历软椭圆的触达像素（r2 < 1），回调 (像素索引, β)。
fn for_each_touched(e: &EllipseParams, w: u32, h: u32, mut f: impl FnMut(usize, f64)) {
    let (cx, cy) = (f64::from(e.cx), f64::from(e.cy));
    let (a, b) = (f64::from(e.rx), f64::from(e.ry));
    if a <= 0.0 || b <= 0.0 {
        return;
    }
    let th = f64::from(e.angle_rad);
    let (c, s) = (th.cos(), th.sin());
    let ia = 1.0 / (a * a);
    let ib = 1.0 / (b * b);
    let big_a = c * c * ia + s * s * ib;
    let big_b = c * s * (ia - ib);
    let big_c = s * s * ia + c * c * ib;
    let yext = (a * a * s * s + b * b * c * c).sqrt();
    let y0 = (cy - yext - 1.0).floor().max(0.0) as u32;
    let y1 = ((cy + yext + 1.0).ceil() as u32).min(h);
    for y in y0..y1 {
        let dy = f64::from(y) + 0.5 - cy;
        let q = big_b * big_b * dy * dy - big_a * (dy * dy * big_c - 1.0);
        if q < 0.0 {
            continue;
        }
        let hw = q.sqrt() / big_a;
        let pm = -big_b * dy / big_a;
        let x0f = (cx - 0.5 + pm - hw).ceil();
        let x1f = (cx - 0.5 + pm + hw).floor() + 1.0;
        if x1f <= 0.0 || x0f >= f64::from(w) {
            continue;
        }
        let x0 = x0f.max(0.0) as u32;
        let x1 = (x1f as u32).min(w);
        let row = y as usize * w as usize;
        for x in x0..x1 {
            let dx = f64::from(x) + 0.5 - cx;
            let u = c * dx + s * dy;
            let v = -s * dx + c * dy;
            let r2 = u * u * ia + v * v * ib;
            if r2 >= 1.0 {
                continue;
            }
            let beta = alpha_228(r2.sqrt());
            if beta > 0.0 {
                f(row + x as usize, beta);
            }
        }
    }
}

/// 软栈颜色重拟合（M1.4）：**接纳守卫式**前向坐标下降。
///
/// 提案：每层 i 在其有效集 E_i = {β_i>0 ∧ 无更晚软层触达} 上用闭式解重求 k——
/// 但混合语义下改 k_i 会以 (1−β_j) 衰减穿透后续所有层的混合链（与覆盖语义不同，
/// 后层只是"遮蔽"而非"截断"），E_i 上的改进不保证全局改进。因此闭式解只作提案，
/// **每层提案后全栈重混合实测总误差，变差即回滚**（重混合 O(Σ触达面积)，毫秒级）。
/// 返回最终画布；总误差单调不增（守卫保证，非数值近似）。
pub fn refit_soft_stack(
    target: &[u8],
    w: u32,
    h: u32,
    solid_canvas: &[u8],
    softs: &mut [(EllipseParams, [u8; 3])],
) -> (SoftRefitReport, Vec<u8>) {
    let n = (w * h) as usize;
    // 存量清洗：非有限参数的软层原地剔除（保留序）
    let k_count = softs.len();
    let mut k = 0usize;
    for i in 0..k_count {
        let (e, _) = &softs[i];
        let ok = e.cx.is_finite() && e.cy.is_finite() && e.rx.is_finite() && e.ry.is_finite();
        if ok {
            softs.swap(i, k);
            k += 1;
        }
    }
    let softs = &mut softs[..k];

    let fold = |stack: &[(EllipseParams, [u8; 3])]| -> Vec<u8> {
        let mut c = solid_canvas.to_vec();
        for (e, k) in stack {
            blend_composite(&mut c, w, h, e, *k, 1.0);
        }
        c
    };
    let mse_of = |canvas: &[u8]| {
        let mut se = 0i64;
        for i in 0..target.len() {
            let d = i64::from(target[i]) - i64::from(canvas[i]);
            se += d * d;
        }
        se as f64 / target.len() as f64
    };

    let start = fold(softs);
    let mse_before = mse_of(&start);

    // 1. 反向收集各层提案集 E_i（像素 + β）
    let mut claimed = vec![false; n];
    let mut eff: Vec<Vec<(u32, f64)>> = vec![Vec::new(); k_count];
    for j in (0..k_count).rev() {
        for_each_touched(&softs[j].0, w, h, |p, beta| {
            if !claimed[p] {
                eff[j].push((p as u32, beta));
            }
            claimed[p] = true;
        });
    }

    // 2. 守卫式下降：提案 → 全栈重混合实测 → 差即回滚
    let mut canvas = start;
    let mut cur_mse = mse_before;
    let mut refits = 0usize;
    for i in 0..k_count {
        let mut sb2 = 0f64;
        let mut sbg = [0f64; 3];
        for &(p, beta) in &eff[i] {
            let p = p as usize;
            sb2 += beta * beta;
            for ch in 0..3 {
                let t = f64::from(target[ch * n + p]);
                let cv = f64::from(canvas[ch * n + p]);
                sbg[ch] += beta * (t - (1.0 - beta) * cv);
            }
        }
        if sb2 <= 0.0 {
            continue;
        }
        let new_k: [u8; 3] =
            core::array::from_fn(|ch| (sbg[ch] / sb2).round().clamp(0.0, 255.0) as u8);
        let old_k = softs[i].1;
        if new_k == old_k {
            continue;
        }
        softs[i].1 = new_k;
        let trial = fold(softs);
        let trial_mse = mse_of(&trial);
        if trial_mse < cur_mse {
            canvas = trial;
            cur_mse = trial_mse;
            refits += 1;
        } else {
            softs[i].1 = old_k;
        }
    }
    let mse_after = cur_mse;
    (
        SoftRefitReport {
            layers: k_count,
            refits,
            mse_before,
            mse_after,
        },
        canvas,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_monotone_and_ends() {
        assert!((alpha_228(0.0) - 1.0).abs() < 1e-6);
        assert_eq!(alpha_228(1.0), 0.0);
        let mut prev = 1.0;
        for i in 1..100 {
            let a = alpha_228(i as f64 / 100.0);
            assert!(a < prev && a >= 0.0, "a({i})={a}");
            prev = a;
        }
    }

    #[test]
    fn blend_pulls_toward_color() {
        let (w, h) = (16u32, 16u32);
        let mut canvas = vec![0u8; 16 * 16 * 3];
        for ch in 0..3 {
            canvas[ch * 256..(ch + 1) * 256].fill(100);
        }
        let e = EllipseParams {
            cx: 8.0,
            cy: 8.0,
            rx: 5.0,
            ry: 5.0,
            angle_rad: 0.0,
        };
        blend_composite(&mut canvas, w, h, &e, [200, 200, 200], 1.0);
        // 中心 β≈1 → 接近 200；边缘更接近 100
        let center = canvas[8 * 16 + 8];
        assert!(center > 160, "center={center}");
        assert!(canvas[0] == 100, "角落不该被波及");
    }

    #[test]
    fn eval_matches_blend_simulation() {
        // 随机画布上：闭式 delta 与"真混合后逐像素重算"一致（取整误差 ≤ 每像素 1）
        let (w, h) = (48u32, 48u32);
        let n = (w * h) as usize;
        let mut seed = 0xACEu64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        let target: Vec<u8> = (0..n * 3).map(|_| next()).collect();
        let mut canvas: Vec<u8> = (0..n * 3).map(|_| next()).collect();
        let e = EllipseParams {
            cx: 24.0,
            cy: 24.0,
            rx: 14.0,
            ry: 9.0,
            angle_rad: 0.7,
        };
        let (k, delta) = eval_soft(&target, &canvas, w, h, &e).expect("应有正收益");

        let cur: f64 = (0..n * 3)
            .map(|i| (f64::from(target[i]) - f64::from(canvas[i])).powi(2))
            .sum();
        blend_composite(&mut canvas, w, h, &e, k, 1.0);
        let new: f64 = (0..n * 3)
            .map(|i| (f64::from(target[i]) - f64::from(canvas[i])).powi(2))
            .sum();
        let actual = (cur - new).round() as i64;
        assert!(
            (actual - delta).abs() <= (n as i64), // u8 取整 + 舍入漂移界
            "闭式 {delta} vs 实际 {actual}"
        );
    }

    #[test]
    fn refit_recovers_ring_color() {
        // 实心画布 = 灰 100；target = 全绿；soft0 大椭圆（原色灰，放置后被 soft1 部分盖住）；
        // soft1 小椭圆已绿。refit 应把 soft0 的有效环重着色为绿。
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let target = {
            let mut t = vec![0u8; n * 3];
            for ch in 0..3 {
                t[ch * n..(ch + 1) * n].fill(if ch == 1 { 200 } else { 0 });
            }
            t
        };
        let solid = {
            let mut c = vec![0u8; n * 3];
            for ch in 0..3 {
                c[ch * n..(ch + 1) * n].fill(100);
            }
            c
        };
        let big = EllipseParams {
            cx: 32.0,
            cy: 32.0,
            rx: 30.0,
            ry: 30.0,
            angle_rad: 0.0,
        };
        let small = EllipseParams {
            cx: 32.0,
            cy: 32.0,
            rx: 12.0,
            ry: 12.0,
            angle_rad: 0.0,
        };
        let mut softs = vec![(big, [100, 100, 100]), (small, [0, 200, 0])];
        let (rep, final_canvas) = refit_soft_stack(&target, w, h, &solid, &mut softs);
        assert!(rep.refits >= 1, "应重着色 soft0");
        // 环区 β 普遍偏小 → 闭式解把 k 推向饱和（blend 灰→绿需要超饱和色）
        assert_eq!(softs[0].1, [0, 255, 0], "环区应重着色为饱和绿");
        assert!(
            rep.mse_after < rep.mse_before,
            "{} -> {}",
            rep.mse_before,
            rep.mse_after
        );
        // 最终画布：soft0 环区应被拉绿（β 小处钳位饱和也只能部分逼近）
        let p = 32 * w as usize + 8; // 环内、small 之外
        assert!(f64::from(final_canvas[n + p]) > 115.0, "环区应被拉向绿");
    }

    #[test]
    fn refit_deterministic() {
        let (w, h) = (48u32, 48u32);
        let n = (w * h) as usize;
        let mut seed = 0x7u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        let target: Vec<u8> = (0..n * 3).map(|_| next()).collect();
        let solid: Vec<u8> = (0..n * 3).map(|_| next()).collect();
        let mk = |i: usize| {
            (
                EllipseParams {
                    cx: 10.0 + 7.0 * i as f32,
                    cy: 20.0 + 5.0 * i as f32,
                    rx: 8.0 + 3.0 * i as f32,
                    ry: 6.0 + 2.0 * i as f32,
                    angle_rad: 0.3 * i as f32,
                },
                [40 + i as u8 * 10, 80, 120],
            )
        };
        let run = || {
            let mut softs = (0..5).map(mk).collect::<Vec<_>>();
            let (rep, canvas) = refit_soft_stack(&target, w, h, &solid, &mut softs);
            (rep.refits, rep.mse_after.to_bits(), canvas)
        };
        let (r1, m1, c1) = run();
        let (r2, m2, c2) = run();
        assert_eq!(r1, r2);
        assert_eq!(m1, m2);
        assert_eq!(c1, c2);
    }
}
