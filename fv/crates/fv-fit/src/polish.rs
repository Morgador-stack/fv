//! 遮挡感知精确抛光（M3-P1）。
//!
//! 贪心放置的形状是在"当时没有上层遮挡"时优化的；后续形状覆盖后，其有效区域
//! 缩小为**实际可见集** R_i = {p : top[p] == i}（实心覆盖语义下 = 覆盖 p 的最高层）。
//! 本模块自底向上遍历每个形状，在 R_i 上做**精确**（i64）重拟合：
//!
//! - 颜色：R_i 上目标均值（闭式最优）
//! - 几何：对当前几何做小扰动爬坡，候选误差用跨 R_i 的精确和评估
//! - 删除：若"整个形状消失"误差更低（含 R_i 为空的完全遮挡层），删除之
//!
//! 数学：设 under = 下方形状叠合画布，b(p) = Σ_ch (under−target)²（候选不覆盖 p
//! 时的误差），D(p;k) = Σ_ch (k−target)²。替换形状 i 为 (geo,k) 后
//! ```text
//! new_err = Σ_{R_i∩cov} D(p;k) + Σ_{R_i\cov} b(p)
//!         = [Σ_cov D − Σ_cov b] + Σ_{R_i} b = H + Btot
//! ```
//! H 与 Btot 均可用前缀和/逐像素和精确计算；old_err 同理。全程 i64、确定性。
//!
//! 接纳后增量维护：under（合成新形状）、top（并集 bbox 局部重刷）。

use crate::candidate::Rng;
use crate::raster::{self, EllipseParams};

/// 抛光报告。
#[derive(Debug, Clone)]
pub struct PolishReport {
    pub sweeps_run: usize,
    pub accepted: usize,
    pub deleted: usize,
    pub mse_before: f64,
    pub mse_after: f64,
}

#[derive(Clone, Copy)]
struct S {
    e: EllipseParams,
    color: [u8; 3],
    dead: bool,
    bbox: Bbox,
    /// 血统：u32::MAX = 本轮新增；否则 = resume 初始形状的下标（输出保真用）
    src: u32,
    /// 被抛光改动过（几何/颜色）
    changed: bool,
}

type Bbox = (u32, u32, u32, u32); // 半开 (x0,y0,x1,y1)

fn bbox_of(e: &EllipseParams, w: u32, h: u32) -> Bbox {
    raster::spans_bbox(&raster::ellipse_rows(e, w, h)).unwrap_or((0, 0, 0, 0))
}

fn bboxes_intersect(a: Bbox, b: Bbox) -> bool {
    a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
}

fn union(a: Bbox, b: Bbox) -> Bbox {
    (
        a.0.min(b.0),
        a.1.min(b.1),
        a.2.max(b.2),
        a.3.max(b.3),
    )
}

/// top[p] = 覆盖 p 的最高层索引（u32::MAX = 背景）。range = Some(bbox) 时只刷局部。
fn repaint_top(sh: &[S], top: &mut [u32], w: u32, h: u32, range: Option<Bbox>) {
    let (bx0, by0, bx1, by1) = range.unwrap_or((0, 0, w, h));
    for y in by0..by1 {
        let row = y as usize * w as usize;
        for x in bx0..bx1 {
            top[row + x as usize] = u32::MAX;
        }
    }
    for (idx, s) in sh.iter().enumerate() {
        if s.dead {
            continue;
        }
        if let Some(r) = range
            && !bboxes_intersect(s.bbox, r)
        {
            continue;
        }
        for span in raster::ellipse_rows(&s.e, w, h) {
            if span.y < by0 || span.y >= by1 {
                continue;
            }
            let x0 = span.x0.max(bx0);
            let x1 = span.x1.min(bx1);
            let row = span.y as usize * w as usize;
            for x in x0..x1 {
                top[row + x as usize] = idx as u32;
            }
        }
    }
}

/// 由形状叠合出最终画布（跳过 dead）并返回对 target 的精确 MSE。
fn render_mse(sh: &[S], target: &[u8], w: u32, h: u32, bg: [u8; 3]) -> f64 {
    let n = (w * h) as usize;
    let mut canvas = vec![0u8; n * 3];
    for ch in 0..3 {
        canvas[ch * n..(ch + 1) * n].fill(bg[ch]);
    }
    for s in sh {
        if s.dead {
            continue;
        }
        raster::composite_ellipse(&mut canvas, w, h, &s.e, s.color);
    }
    let mut se = 0i64;
    for p in 0..n * 3 {
        let d = target[p] as i64 - canvas[p] as i64;
        se += d * d;
    }
    se as f64 / (n * 3) as f64
}

/// 在候选 spans ∩ R_i（top==idx）上累加 (m_c, S1c[3], S2c[3], Bc)。
struct CovStats {
    m: i64,
    s1: [i64; 3],
    s2: [i64; 3],
    b: i64,
}

fn accumulate_cov(
    spans: &[raster::RowSpan],
    top: &[u32],
    idx: u32,
    target: &[u8],
    under: &[u8],
    w: u32,
    n: usize,
) -> CovStats {
    let mut st = CovStats {
        m: 0,
        s1: [0; 3],
        s2: [0; 3],
        b: 0,
    };
    for span in spans {
        let row = span.y as usize * w as usize;
        for x in span.x0..span.x1 {
            let p = row + x as usize;
            // 候选实际可见（会改变最终画布）的像素：本层是 top，或其上方无任何形状
            //（top==MAX 即全裸背景）。top>idx 处候选被上层遮住，不产生任何影响。
            let t_p = top[p];
            if t_p != u32::MAX && t_p > idx {
                continue;
            }
            st.m += 1;
            for ch in 0..3 {
                let t = target[ch * n + p] as i64;
                let u = under[ch * n + p] as i64;
                st.s1[ch] += t;
                st.s2[ch] += t * t;
                st.b += (u - t) * (u - t);
            }
        }
    }
    st
}

/// 抛光结果：幸存形状（z 序不变）+ 血统/改动标记（供调用方做输出保真）。
pub struct PolishedShape {
    pub e: EllipseParams,
    pub color: [u8; 3],
    pub src: u32,
    pub changed: bool,
}

/// 抛光主入口。`shapes` 为 opt 分辨率坐标（z 序自底向上），就地修改（删除 dead）。
#[allow(clippy::too_many_arguments)]
pub fn polish_opt(
    target: &[u8],
    w: u32,
    h: u32,
    shapes: &mut Vec<(EllipseParams, [u8; 3])>,
    bg: [u8; 3],
    sweeps: usize,
    iters: usize,
    seed: u64,
) -> (PolishReport, Vec<PolishedShape>) {
    let n = (w * h) as usize;
    let long_edge = w.max(h) as f32;
    let mut sh: Vec<S> = shapes
        .iter()
        .enumerate()
        .map(|(j, (e, c))| S {
            e: *e,
            color: *c,
            dead: false,
            bbox: bbox_of(e, w, h),
            src: j as u32,
            changed: false,
        })
        .collect();
    let mse_before = render_mse(&sh, target, w, h, bg);

    let mut top = vec![u32::MAX; n];
    let mut under = vec![0u8; n * 3];
    let mut accepted_total = 0usize;
    let mut used = 0usize;

    for sweep in 0..sweeps.max(1) {
        repaint_top(&sh, &mut top, w, h, None);
        for ch in 0..3 {
            under[ch * n..(ch + 1) * n].fill(bg[ch]);
        }
        let mut accepted = 0usize;

        for i in 0..sh.len() {
            if sh[i].dead {
                continue;
            }
            let idx = i as u32;

            // ---- R_i：当前形状 spans 上 top==idx 的连续段 ----
            let old_spans = raster::ellipse_rows(&sh[i].e, w, h);
            let mut runs: Vec<raster::RowSpan> = Vec::new();
            for span in &old_spans {
                let row = span.y as usize * w as usize;
                let mut x = span.x0;
                while x < span.x1 {
                    if top[row + x as usize] == idx {
                        let x0 = x;
                        while x < span.x1 && top[row + x as usize] == idx {
                            x += 1;
                        }
                        runs.push(raster::RowSpan { y: span.y, x0, x1: x });
                    } else {
                        x += 1;
                    }
                }
            }

            // 完全遮挡 → 直接删（免费省一层）
            if runs.is_empty() {
                sh[i].dead = true;
                repaint_top(&sh, &mut top, w, h, Some(sh[i].bbox));
                continue;
            }

            // ---- R_i 全量统计 ----
            let (mut m, mut s1, mut s2, mut btot) = (0i64, [0i64; 3], [0i64; 3], 0i64);
            for run in &runs {
                let row = run.y as usize * w as usize;
                for x in run.x0..run.x1 {
                    let p = row + x as usize;
                    m += 1;
                    for ch in 0..3 {
                        let t = target[ch * n + p] as i64;
                        let u = under[ch * n + p] as i64;
                        s1[ch] += t;
                        s2[ch] += t * t;
                        btot += (u - t) * (u - t);
                    }
                }
            }
            let old = sh[i].color;
            let o = [old[0] as i64, old[1] as i64, old[2] as i64];
            let old_err = m * (o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
                - 2 * (o[0] * s1[0] + o[1] * s1[1] + o[2] * s1[2])
                + (s2[0] + s2[1] + s2[2]);

            // ---- 候选评估（C1/H 记账） ----
            // 受影响集 A = R_i ∪ 候选可见覆盖（top≤idx 或背景）。
            // delta(c,k) = C1 − H(c,k)，其中
            //   C1   = Σ_{R_i}[D(old) − b] = old_err − btot（删除候选的 delta，H=0）
            //   H    = Σ_{cov_aff}[D(k) − b] = Σ_cov D(k) − Bc
            // 只接受 H < 当前最优（初始 = C1，即删除兜底）。
            let c1 = old_err - btot;
            let mut best_h = c1;
            let mut best: Option<(EllipseParams, [u8; 3])> = None;
            let mut best_is_delete = c1 > 0;

            // 几何爬坡（与 engine::refine 同款扰动集）
            let mut rng = Rng::new(
                seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15)
                    ^ ((sweep as u64 + 1).wrapping_mul(0x5EED5)),
            );
            let mut cur = sh[i].e;
            let mut cur_h = 0i64; // 当前几何的 H（爬坡链上单调降）
            let (mut sig_pos, mut sig_rad, mut sig_ang) = (
                long_edge * 0.02,
                0.18f32,
                10.0f32,
            );
            for _ in 0..iters.max(1) {
                let mut improved = false;
                for _ in 0..6 {
                    let mut g2 = cur;
                    let pick = rng.next_u32() % 6;
                    let sign = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
                    match pick {
                        0 => g2.cx += sign * sig_pos,
                        1 => g2.cy += sign * sig_pos,
                        2 => g2.rx = (g2.rx * (1.0 + sign * sig_rad)).max(0.5),
                        3 => g2.ry = (g2.ry * (1.0 + sign * sig_rad)).max(0.5),
                        4 => g2.angle_rad += sign * sig_ang.to_radians(),
                        _ => {
                            g2.rx = (g2.rx * (1.0 + sign * sig_rad)).max(0.5);
                            g2.ry = (g2.ry * (1.0 - sign * sig_rad)).max(0.5);
                        }
                    }
                    let spans2 = raster::ellipse_rows(&g2, w, h);
                    if spans2.is_empty() {
                        continue;
                    }
                    let st = accumulate_cov(&spans2, &top, idx, target, &under, w, n);
                    if st.m == 0 {
                        continue;
                    }
                    // 最优整数色 = 可见覆盖的目标均值
                    let k = [
                        ((st.s1[0] + st.m / 2) / st.m).clamp(0, 255) as u8,
                        ((st.s1[1] + st.m / 2) / st.m).clamp(0, 255) as u8,
                        ((st.s1[2] + st.m / 2) / st.m).clamp(0, 255) as u8,
                    ];
                    let kk = [k[0] as i64, k[1] as i64, k[2] as i64];
                    let cov_d = st.m * (kk[0] * kk[0] + kk[1] * kk[1] + kk[2] * kk[2])
                        - 2 * (kk[0] * st.s1[0] + kk[1] * st.s1[1] + kk[2] * st.s1[2])
                        + (st.s2[0] + st.s2[1] + st.s2[2]);
                    let h_c = cov_d - st.b;
                    if h_c < cur_h {
                        cur = g2;
                        cur_h = h_c;
                        improved = true;
                        if h_c < best_h {
                            best_h = h_c;
                            best = Some((g2, k));
                            best_is_delete = false;
                        }
                        break;
                    }
                }
                if !improved {
                    sig_pos *= 0.6;
                    sig_rad *= 0.6;
                    sig_ang *= 0.6;
                }
            }

            // ---- 决策与增量维护 ----
            if best_is_delete {
                sh[i].dead = true;
                repaint_top(&sh, &mut top, w, h, Some(sh[i].bbox));
                accepted += 1;
            } else if let Some((g, k)) = best {
                let new_bbox = bbox_of(&g, w, h);
                let ub = union(sh[i].bbox, new_bbox);
                sh[i].e = g;
                sh[i].color = k;
                sh[i].changed = true;
                sh[i].bbox = new_bbox;
                raster::composite_ellipse(&mut under, w, h, &g, k);
                repaint_top(&sh, &mut top, w, h, Some(ub));
                accepted += 1;
            } else {
                raster::composite_ellipse(&mut under, w, h, &sh[i].e, sh[i].color);
            }
        }

        used += 1;
        accepted_total += accepted;
        sh.retain(|s| !s.dead);
        if accepted == 0 {
            break;
        }
    }

    let deleted = shapes.len() - sh.len();
    let mse_after = render_mse(&sh, target, w, h, bg);
    shapes.clear();
    let survivors: Vec<PolishedShape> = sh
        .into_iter()
        .map(|s| PolishedShape {
            e: s.e,
            color: s.color,
            src: s.src,
            changed: s.changed,
        })
        .collect();
    shapes.extend(survivors.iter().map(|s| (s.e, s.color)));
    (
        PolishReport {
            sweeps_run: used,
            accepted: accepted_total,
            deleted,
            mse_before,
            mse_after,
        },
        survivors,
    )
}
