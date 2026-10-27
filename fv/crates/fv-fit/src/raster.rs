//! 旋转椭圆的逐行区间光栅化与画布合成——全项目唯一的几何真源。
//!
//! ## 数学（实现必须遵守，内层用 f64）
//! 椭圆中心 (cx,cy)，半轴 a=rx, b=ry，逆时针旋转角 θ（弧度，`angle_rad`）。
//! 像素 (x,y) 的**像素中心** (x+0.5, y+0.5) 在椭圆内部当且仅当：
//!
//! ```text
//! u =  c·px + s·dy,  v = -s·px + c·dy        （px = x+0.5-cx, dy = y+0.5-cy, c=cosθ, s=sinθ）
//! u²/a² + v²/b² ≤ 1
//! ```
//!
//! 展开后是 px 的二次方程 `A·px² + B·px + C ≤ 0`：
//! ```text
//! A = c²/a² + s²/b²
//! B = 2·dy·c·s·(1/a² − 1/b²)
//! C = dy²·(s²/a² + c²/b²) − 1
//! ```
//! 每个扫描行 y 解出 px∈[p0,p1]，换算整数区间（像素中心制）：
//! `x0 = ceil(cx − 0.5 + p0)`，`x1_excl = floor(cx − 0.5 + p1) + 1`，
//! 再与 [0, w) 求交。判别式 < 0 或 a/b ≤ 0 → 该行无区间。
//!
//! ## 画布布局
//! 画布为 3 平面 SoA 的 u8：`canvas[ch·(w·h) + y·w + x]`，len == 3·w·h。

/// 椭圆几何参数（浮点，像素单位；cx/cy 可在图外，由调用方裁剪）。
#[derive(Debug, Clone, Copy)]
pub struct EllipseParams {
    pub cx: f32,
    pub cy: f32,
    pub rx: f32,
    pub ry: f32,
    /// 逆时针弧度（与 Shape::angle_deg 换算：`to_radians()`）。
    pub angle_rad: f32,
}

/// 一行的整数区间：x ∈ [x0, x1)（半开），已与图像边界求交，保证 x0 < x1 ≤ w。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSpan {
    pub y: u32,
    pub x0: u32,
    pub x1: u32,
}

/// 求椭圆在 (w,h) 图内的所有行区间，按 y 升序；空椭圆/完全在图外 → 空 Vec。
///
/// 逐行解二次方程（闭式解）：p_mid = −B/2A，半宽 hw = √D/2A，
/// p0 = p_mid − hw，p1 = p_mid + hw（中点±半宽形式，数值上比 (−B±√D)/2A 更稳）。
/// 整数化用像素中心制：x0 = ceil(cx−0.5+p0)（含边界像素），x1_excl = floor(cx−0.5+p1)+1。
pub fn ellipse_rows(e: &EllipseParams, w: u32, h: u32) -> Vec<RowSpan> {
    // 退化与非有限参数 → 空（规范只定义 a/b ≤ 0，NaN/Inf 一并按空处理更安全）。
    if w == 0
        || h == 0
        || !(e.cx.is_finite()
            && e.cy.is_finite()
            && e.rx.is_finite()
            && e.ry.is_finite()
            && e.angle_rad.is_finite())
        || e.rx <= 0.0
        || e.ry <= 0.0
    {
        return Vec::new();
    }

    // 内层全部 f64。
    let cx = e.cx as f64;
    let cy = e.cy as f64;
    let a = e.rx as f64;
    let b = e.ry as f64;
    let th = e.angle_rad as f64;
    let c = th.cos();
    let s = th.sin();
    let inv_a2 = 1.0 / (a * a);
    let inv_b2 = 1.0 / (b * b);

    // 行间常量：A（px² 系数，恒 > 0）、B 的 dy 系数、C 的 dy² 系数。
    let aa = c * c * inv_a2 + s * s * inv_b2;
    let b_fac = 2.0 * c * s * (inv_a2 - inv_b2);
    let c_fac = s * s * inv_a2 + c * c * inv_b2;
    let two_a = 2.0 * aa;

    // 椭圆在 y 方向的半延展 = √(a²s² + b²c²)，用于估算容量（避免大图小椭圆的过度分配）。
    let y_half = (a * a * s * s + b * b * c * c).sqrt();
    let est_rows = (2.0 * y_half + 2.0).min(h as f64).max(0.0) as usize;
    let mut out = Vec::with_capacity(est_rows);

    let wi = w as i64;
    for y in 0..h {
        let dy = y as f64 + 0.5 - cy;
        let bq = b_fac * dy; // B
        let cq = c_fac * dy * dy - 1.0; // C
        let disc = bq * bq - 4.0 * aa * cq;
        if disc < 0.0 {
            continue; // 该行与椭圆不相交
        }
        let hw = disc.sqrt() / two_a;
        let mid = -bq / two_a;
        let p0 = mid - hw;
        let p1 = mid + hw;

        // 像素中心制整数化（含边界），再与 [0, w) 求交。
        // f64→i64 的 as 转换饱和，极端 cx 下也能安全钳位。
        let lo = (cx - 0.5 + p0).ceil() as i64;
        let hi = (cx - 0.5 + p1).floor() as i64 + 1;
        let x0 = lo.max(0);
        let x1 = hi.min(wi);
        if x0 < x1 {
            out.push(RowSpan {
                y,
                x0: x0 as u32,
                x1: x1 as u32,
            });
        }
    }
    out
}

/// 所有区间覆盖的像素数（Σ x1−x0）。
pub fn spans_pixel_count(spans: &[RowSpan]) -> u64 {
    spans.iter().map(|s| (s.x1 - s.x0) as u64).sum()
}

/// 所有区间的包围盒 (x0, y0, x1_excl, y1_excl)；无区间 → None。
pub fn spans_bbox(spans: &[RowSpan]) -> Option<(u32, u32, u32, u32)> {
    let first = *spans.first()?;
    let (mut x0, mut y0, mut x1) = (first.x0, first.y, first.x1);
    let mut y1 = first.y + 1; // y < h ≤ u32::MAX，+1 不溢出
    for s in &spans[1..] {
        x0 = x0.min(s.x0);
        y0 = y0.min(s.y);
        x1 = x1.max(s.x1);
        y1 = y1.max(s.y + 1);
    }
    Some((x0, y0, x1, y1))
}

/// 把实心椭圆合成到画布上（逐区间直接覆盖 color，无混合）。
/// `canvas` 布局见模块文档；color 为 RGB。
pub fn composite_ellipse(canvas: &mut [u8], w: u32, h: u32, e: &EllipseParams, color: [u8; 3]) {
    let n = (w as usize) * (h as usize);
    debug_assert_eq!(canvas.len(), n * 3, "canvas 必须是 3 平面 SoA，len == 3·w·h");
    let spans = ellipse_rows(e, w, h);
    let wu = w as usize;
    for sp in &spans {
        let row_off = sp.y as usize * wu;
        let (x0, x1) = (sp.x0 as usize, sp.x1 as usize);
        for (ch, &cv) in color.iter().enumerate() {
            let base = ch * n + row_off + x0;
            canvas[base..base + (x1 - x0)].fill(cv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Rng;

    const W: u32 = 96;
    const H: u32 = 80;

    fn ell(cx: f32, cy: f32, rx: f32, ry: f32, angle_rad: f32) -> EllipseParams {
        EllipseParams {
            cx,
            cy,
            rx,
            ry,
            angle_rad,
        }
    }

    /// 独立的逐像素暴力参考：对每个像素用模块文档的中心点判据（f64）。
    fn brute_rows(e: &EllipseParams, w: u32, h: u32) -> Vec<Vec<u32>> {
        let mut rows = vec![Vec::<u32>::new(); h as usize];
        if !(e.rx.is_finite() && e.ry.is_finite() && e.rx > 0.0 && e.ry > 0.0) {
            return rows;
        }
        let (cx, cy) = (e.cx as f64, e.cy as f64);
        let (a, b) = (e.rx as f64, e.ry as f64);
        let th = e.angle_rad as f64;
        let c = th.cos();
        let s = th.sin();
        for y in 0..h {
            let dy = y as f64 + 0.5 - cy;
            for x in 0..w {
                let px = x as f64 + 0.5 - cx;
                let u = c * px + s * dy;
                let v = -s * px + c * dy;
                if u * u / (a * a) + v * v / (b * b) <= 1.0 {
                    rows[y as usize].push(x);
                }
            }
        }
        rows
    }

    /// 把升序 x 列表折叠成连续 run（first, last）闭区间对。
    fn runs_of(xs: &[u32]) -> Vec<(u32, u32)> {
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for &x in xs {
            match runs.last_mut() {
                Some(last) if last.1 + 1 == x => last.1 = x,
                _ => runs.push((x, x)),
            }
        }
        runs
    }

    /// 每行合并相邻/重叠区间 → 不相交 (x0, x1_excl) 列表。
    fn merged_rows(spans: &[RowSpan], h: u32) -> Vec<Vec<(u32, u32)>> {
        let mut rows = vec![Vec::<(u32, u32)>::new(); h as usize];
        for s in spans {
            rows[s.y as usize].push((s.x0, s.x1));
        }
        for r in rows.iter_mut() {
            r.sort_unstable();
            let mut merged: Vec<(u32, u32)> = Vec::with_capacity(r.len());
            for &(x0, x1) in r.iter() {
                match merged.last_mut() {
                    Some(last) if x0 <= last.1 => last.1 = last.1.max(x1),
                    _ => merged.push((x0, x1)),
                }
            }
            *r = merged;
        }
        rows
    }

    fn random_case(rng: &mut Rng) -> EllipseParams {
        // 中心允许落在图外（±60px 余量），半轴/角度随机，刻意避开 0/π/2 的精确轴向角。
        let cx = rng.next_f32() * (W as f32 + 120.0) - 60.0;
        let cy = rng.next_f32() * (H as f32 + 120.0) - 60.0;
        let rx = 1.0 + rng.next_f32() * 55.0;
        let ry = 1.0 + rng.next_f32() * 55.0;
        let ang = rng.next_f32() * core::f32::consts::PI;
        ell(cx, cy, rx, ry, ang)
    }

    // ---- 1. 手算基准（轴对齐、含边界像素、精确 dyadic 参数） ----

    #[test]
    fn axis_aligned_hand_computed() {
        // cx=cy=2.5, rx=ry=1, θ=0, 5×5。像素中心落在单位圆内（含边界）：
        // y=1/3：dy=∓1 → 仅 px=0（x=2）；y=2：dy=0 → px∈[−1,1]（x∈[1,4)）。
        let spans = ellipse_rows(&ell(2.5, 2.5, 1.0, 1.0, 0.0), 5, 5);
        assert_eq!(
            spans,
            vec![
                RowSpan { y: 1, x0: 2, x1: 3 },
                RowSpan { y: 2, x0: 1, x1: 4 },
                RowSpan { y: 3, x0: 2, x1: 3 },
            ]
        );
        assert_eq!(spans_pixel_count(&spans), 5);
        assert_eq!(spans_bbox(&spans), Some((1, 1, 4, 4)));
        assert_eq!(spans_bbox(&[]), None);
        assert_eq!(spans_pixel_count(&[]), 0);
    }

    // ---- 2. 面积 ≈ π·rx·ry（半径 ≥ 8，误差 < 3%） ----

    #[test]
    fn pixel_count_approximates_ellipse_area() {
        // 独立大画布保证整幅在图内；(rx, ry, angle)，半径全部 ≥ 8。
        let (aw, ah) = (160u32, 160u32);
        let cases: [(f32, f32, f32); 7] = [
            (8.0, 8.0, 0.0),
            (8.0, 8.0, 0.7),
            (9.0, 12.0, 0.0),
            (12.0, 9.0, 1.1),
            (20.0, 30.0, 0.0),
            (30.0, 20.0, 2.0),
            (10.0, 25.0, -0.8),
        ];
        for &(rx, ry, ang) in &cases {
            let e = ell(80.5, 80.5, rx, ry, ang);
            let spans = ellipse_rows(&e, aw, ah);
            assert!(!spans.is_empty());
            let n = spans_pixel_count(&spans) as f64;
            let want = core::f64::consts::PI * (rx as f64) * (ry as f64);
            let rel = ((n - want) / want).abs();
            assert!(
                rel < 0.03,
                "rx={rx} ry={ry} ang={ang}: count={n} want={want:.3} rel={rel:.4}"
            );
        }
    }

    // ---- 3. 与逐像素暴力法完全一致（随机 30 + 定向边界 7） ----

    #[test]
    fn matches_per_pixel_bruteforce_exactly() {
        let mut cases: Vec<EllipseParams> = Vec::new();
        let mut rng = Rng::new(0xFACE);
        for _ in 0..30 {
            cases.push(random_case(&mut rng));
        }
        // 定向：完全图外（左上 / 右侧）、大椭圆被裁剪、轴对齐 + 通用半径、π/2、极小。
        cases.push(ell(-80.0, -70.0, 20.0, 20.0, 0.3));
        cases.push(ell(W as f32 + 80.0, H as f32 * 0.5, 25.0, 25.0, 1.0));
        cases.push(ell(0.0, 0.0, 60.0, 60.0, 0.0));
        cases.push(ell(0.0, H as f32, 40.0, 40.0, 0.7));
        cases.push(ell(48.0, 40.0, 23.7, 17.3, 0.0));
        cases.push(ell(48.5, 40.5, 33.1, 8.9, core::f32::consts::FRAC_PI_2));
        cases.push(ell(10.0, 10.0, 1.2, 0.9, 2.6));

        for e in &cases {
            let spans = ellipse_rows(e, W, H);
            let brute = brute_rows(e, W, H);
            let merged = merged_rows(&spans, H);
            for y in 0..H as usize {
                let runs = runs_of(&brute[y]); // 暴力行必然连续 → 应恰一段
                let got = &merged[y];
                assert_eq!(
                    runs.len(),
                    got.len(),
                    "e={e:?} y={y} brute_runs={runs:?} spans={got:?}"
                );
                for (r, m) in runs.iter().zip(got.iter()) {
                    assert_eq!(
                        (r.0, r.1 + 1),
                        (m.0, m.1),
                        "e={e:?} y={y} brute={r:?} span={m:?}"
                    );
                }
            }
        }
    }

    // ---- 4. 区间性质：y 严格升序（每行至多一段）、范围合法、合并后每行 ≤ 1 段 ----

    #[test]
    fn span_ordering_and_bounds() {
        let mut rng = Rng::new(7);
        for _ in 0..50 {
            let e = random_case(&mut rng);
            let spans = ellipse_rows(&e, W, H);
            for (i, s) in spans.iter().enumerate() {
                assert!(s.y < H, "y 越界: {s:?}");
                assert!(s.x0 < s.x1 && s.x1 <= W, "x 范围非法: {s:?}");
                if i > 0 {
                    assert!(
                        spans[i - 1].y < s.y,
                        "y 必须严格升序（同 y 至多一段）: {:?}",
                        &spans[i - 1..=i]
                    );
                }
            }
            for (y, r) in merged_rows(&spans, H).iter().enumerate() {
                assert!(r.len() <= 1, "行 {y} 合并后仍有 {} 段: {r:?}", r.len());
            }
        }
    }

    // ---- 5. 退化与裁剪 ----

    #[test]
    fn degenerate_and_full_cover() {
        // 半轴为 0 / 负 → 空。
        assert!(ellipse_rows(&ell(50.0, 40.0, 0.0, 10.0, 0.5), W, H).is_empty());
        assert!(ellipse_rows(&ell(50.0, 40.0, 10.0, 0.0, 0.5), W, H).is_empty());
        assert!(ellipse_rows(&ell(50.0, 40.0, -3.0, 10.0, 0.5), W, H).is_empty());
        // 完全在图外（近处 + 极远处）→ 空。
        assert!(ellipse_rows(&ell(-100.0, -100.0, 20.0, 20.0, 0.0), W, H).is_empty());
        assert!(ellipse_rows(&ell(1e9, 1e9, 5.0, 5.0, 0.0), W, H).is_empty());
        assert!(ellipse_rows(&ell(-30.0, H as f32 * 0.5, 25.0, 25.0, 0.3), W, H).is_empty());
        // 覆盖全图的大椭圆 → 每行 [0, W)，共 H 行。
        let e = ell(
            W as f32 * 0.5,
            H as f32 * 0.5,
            W as f32 * 3.0,
            H as f32 * 3.0,
            0.42,
        );
        let spans = ellipse_rows(&e, W, H);
        assert_eq!(spans.len(), H as usize);
        for (i, s) in spans.iter().enumerate() {
            assert_eq!(s, &RowSpan { y: i as u32, x0: 0, x1: W });
        }
        assert_eq!(spans_pixel_count(&spans), (W as u64) * (H as u64));
        assert_eq!(spans_bbox(&spans), Some((0, 0, W, H)));
        // 零尺寸画布 → 空。
        assert!(ellipse_rows(&ell(0.0, 0.0, 10.0, 10.0, 0.0), 0, 0).is_empty());
    }

    // ---- 6. composite：span 内 == color、span 外不变（全像素 + 中心/边缘采样） ----

    #[test]
    fn composite_overwrites_exactly_spans() {
        let (w, h) = (40u32, 30u32);
        let base = [200u8, 150, 100];
        let color = [10u8, 220, 30];
        let e = ell(20.3, 15.7, 12.0, 8.0, 0.9);
        let n = (w * h) as usize;

        let mut canvas = vec![0u8; n * 3];
        for ch in 0..3 {
            canvas[ch * n..(ch + 1) * n].fill(base[ch]);
        }
        let spans = ellipse_rows(&e, w, h);
        assert!(!spans.is_empty());
        composite_ellipse(&mut canvas, w, h, &e, color);

        // 全像素逐点核对（SoA 通道布局一并验证）。
        let mut colored = 0u64;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let in_span = spans.iter().any(|s| s.y == y && s.x0 <= x && x < s.x1);
                if in_span {
                    colored += 1;
                }
                for ch in 0..3 {
                    assert_eq!(
                        canvas[ch * n + idx],
                        if in_span { color[ch] } else { base[ch] },
                        "ch={ch} ({x},{y}) in_span={in_span}"
                    );
                }
            }
        }
        assert_eq!(colored, spans_pixel_count(&spans));

        // 中心采样：椭圆中心所在像素必为 color。
        let (cxp, cyp) = (e.cx.floor() as u32, e.cy.floor() as u32);
        for ch in 0..3 {
            assert_eq!(canvas[ch * n + (cyp * w + cxp) as usize], color[ch]);
        }
        // 边缘采样：中心行 span 的 x1−1（内）与 x1（外，若未触边）。
        let mid = spans.iter().find(|s| s.y == cyp).expect("中心行必有区间");
        for ch in 0..3 {
            assert_eq!(canvas[ch * n + (cyp * w + (mid.x1 - 1)) as usize], color[ch]);
            if mid.x1 < w {
                assert_eq!(canvas[ch * n + (cyp * w + mid.x1) as usize], base[ch]);
            }
        }
        // 远离椭圆的角落不变。
        for (x, y) in [(0u32, 0u32), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
            let idx = (y * w + x) as usize;
            for ch in 0..3 {
                assert_eq!(canvas[ch * n + idx], base[ch], "corner ({x},{y}) ch={ch}");
            }
        }
    }

    // ---- 7. 性能量级（不设硬阈值，只输出量级供人工确认） ----

    #[test]
    fn perf_ellipse_rows_magnitude() {
        let (w, h) = (448u32, 448u32);
        let mut rng = Rng::new(0xBEEF);
        let calls = 2000u32;
        let mut total_px = 0u64;
        let mut total_spans = 0u64;
        let t = std::time::Instant::now();
        for _ in 0..calls {
            let e = ell(
                rng.next_f32() * w as f32,
                rng.next_f32() * h as f32,
                2.0 + rng.next_f32() * 120.0,
                2.0 + rng.next_f32() * 120.0,
                rng.next_f32() * core::f32::consts::PI,
            );
            let spans = ellipse_rows(&e, w, h);
            total_spans += spans.len() as u64;
            total_px += spans_pixel_count(&spans);
        }
        let dt = t.elapsed();
        let per_call_us = dt.as_micros() as f64 / calls as f64;
        eprintln!(
            "perf: {calls} × ellipse_rows(448×448) = {dt:?} ({per_call_us:.1} µs/call, {:.2} Mpx/s), avg spans/call = {:.0}",
            total_px as f64 / dt.as_secs_f64() / 1e6,
            total_spans as f64 / calls as f64,
        );
        assert!(total_px > 0 && total_spans > 0);
    }
}
