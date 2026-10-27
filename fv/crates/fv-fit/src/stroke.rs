//! M1.7 描边放置：沿强度场等值线链式放置小实心椭圆。
//!
//! 这是"画家式"边缘画法——贪心采样学不会的沿轮廓排笔。流程：
//! 1. Marching Squares（16-case）提取 field≥iso 的边界线段
//! 2. 线段链化为折线（端点哈希贪心连接），短链丢弃
//! 3. 沿链按弧长步进取点：椭圆中心=链点、长轴=切向、颜色=参考图局部均值
//!
//! 全程确定性（链按 (y,x) 排序，无 RNG）。

use crate::raster::EllipseParams;

#[derive(Debug, Clone)]
pub struct StrokeLayer {
    pub shapes: Vec<(EllipseParams, [u8; 3])>,
}

/// 一条链化的折线（像素坐标点序列）。
struct Chain {
    pts: Vec<(f32, f32)>,
}

/// Marching Squares：返回二值边界线段（(p0,p1) 端点，亚像素中点插值）。
/// 段方向约定：左侧为 ≥iso 区域（用于链化方向一致性，本实现不依赖方向）。
fn marching_squares(field: &[u8], w: u32, h: u32, iso: u8) -> Vec<((f32, f32), (f32, f32))> {
    let mut segs = Vec::new();
    let at = |x: u32, y: u32| field[(y * w + x) as usize] >= iso;
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            // 角点顺序：TL TR BR BL
            let tl = at(x, y);
            let tr = at(x + 1, y);
            let br = at(x + 1, y + 1);
            let bl = at(x, y + 1);
            let code = (tl as u8) << 3 | (tr as u8) << 2 | (br as u8) << 1 | bl as u8;
            // 单元边中点
            let (fx, fy) = (x as f32, y as f32);
            let top = (fx + 0.5, fy);
            let right = (fx + 1.0, fy + 0.5);
            let bottom = (fx + 0.5, fy + 1.0);
            let left = (fx, fy + 0.5);
            let mut push = |a: (f32, f32), b: (f32, f32)| segs.push((a, b));
            // 位序：TL<<3 | TR<<2 | BR<<1 | BL。链化不依赖方向，只需覆盖一致。
            // 鞍点（对角双角）用四角均值消歧：中心在内 → 内部连通（弧绕两个外角）；
            // 中心在外 → 两个楔形互不连通（弧各绕一个内角）。方向选错会断链。
            let center_inside = {
                let tl_v = u16::from(field[(y * w + x) as usize]);
                let tr_v = u16::from(field[(y * w + x + 1) as usize]);
                let br_v = u16::from(field[((y + 1) * w + x + 1) as usize]);
                let bl_v = u16::from(field[((y + 1) * w + x) as usize]);
                (tl_v + tr_v + br_v + bl_v) / 4 >= u16::from(iso)
            };
            match code {
                0 | 15 => {}
                // 单角在内
                8 => push(left, top),     // TL
                4 => push(top, right),    // TR
                2 => push(right, bottom), // BR
                1 => push(bottom, left),  // BL
                // 两角相邻
                0b1100 => push(left, right), // TL+TR（上行）
                0b0110 => push(top, bottom), // TR+BR（右列）
                0b0011 => push(left, right), // BR+BL（下行）
                0b1001 => push(top, bottom), // TL+BL（左列）
                // 鞍点：中心在内 → 弧绕两个外角；中心在外 → 各绕一个内角
                0b0101 => {
                    if center_inside {
                        push(left, top);
                        push(right, bottom);
                    } else {
                        push(top, right);
                        push(bottom, left);
                    }
                }
                0b1010 => {
                    if center_inside {
                        push(top, right);
                        push(bottom, left);
                    } else {
                        push(left, top);
                        push(right, bottom);
                    }
                }
                // 三角在内（= 单角在外）：几何等价于"外的那个角单独在内"
                0b1110 => push(bottom, left), // all-but-BL ≡ BL 单独
                0b1101 => push(right, bottom), // all-but-BR ≡ BR 单独
                0b1011 => push(top, right),   // all-but-TR ≡ TR 单独
                0b0111 => push(left, top),    // all-but-TL ≡ TL 单独
                _ => unreachable!("4bit 全覆盖"),
            }
        }
    }
    segs
}

/// 线段链化：端点哈希（量化到 0.5px 格）贪心连接。
fn chain_segments(segs: Vec<((f32, f32), (f32, f32))>, tol2: f32) -> Vec<Chain> {
    use std::collections::HashMap;
    // 端点 → 线段端索引
    let mut endpoint_map: HashMap<(i32, i32), Vec<(usize, u8)>> = HashMap::new();
    let key = |p: (f32, f32)| -> (i32, i32) {
        ((p.0 * 2.0).round() as i32, (p.1 * 2.0).round() as i32)
    };
    for (i, (a, b)) in segs.iter().enumerate() {
        endpoint_map.entry(key(*a)).or_default().push((i, 0));
        endpoint_map.entry(key(*b)).or_default().push((i, 1));
    }
    let mut used = vec![false; segs.len()];
    let mut chains = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        // 从 start 双向生长
        let mut pts = vec![segs[start].0, segs[start].1];
        // 向尾端生长
        loop {
            let tail = *pts.last().unwrap();
            let mut next: Option<(usize, u8)> = None;
            if let Some(cands) = endpoint_map.get(&key(tail)) {
                for &(i, end) in cands {
                    if used[i] {
                        continue;
                    }
                    let other = if end == 0 { segs[i].1 } else { segs[i].0 };
                    if dist2(tail, other) <= tol2 {
                        next = Some((i, end));
                        break;
                    }
                }
            }
            match next {
                Some((i, end)) => {
                    used[i] = true;
                    pts.push(if end == 0 { segs[i].1 } else { segs[i].0 });
                }
                None => break,
            }
        }
        // 向头端生长
        loop {
            let head = pts[0];
            let mut next: Option<(usize, u8)> = None;
            if let Some(cands) = endpoint_map.get(&key(head)) {
                for &(i, end) in cands {
                    if used[i] {
                        continue;
                    }
                    let other = if end == 0 { segs[i].1 } else { segs[i].0 };
                    if dist2(head, other) <= tol2 {
                        next = Some((i, end));
                        break;
                    }
                }
            }
            match next {
                Some((i, end)) => {
                    used[i] = true;
                    pts.insert(0, if end == 0 { segs[i].1 } else { segs[i].0 });
                }
                None => break,
            }
        }
        chains.push(Chain { pts });
    }
    chains
}

fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    dx * dx + dy * dy
}

fn chain_length(c: &Chain) -> f32 {
    c.pts.windows(2).map(|w| dist2(w[0], w[1]).sqrt()).sum()
}

/// 参考图（3 平面 SoA）在 (x,y) 的 3×3 均值颜色。
fn local_mean(target: &[u8], w: u32, h: u32, x: i32, y: i32) -> [u8; 3] {
    let n = (w * h) as usize;
    let (mut sum, mut cnt) = ([0u32; 3], 0u32);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (xx, yy) = (x + dx, y + dy);
            if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                continue;
            }
            let p = (yy as u32 * w + xx as u32) as usize;
            for ch in 0..3 {
                sum[ch] += target[ch * n + p] as u32;
            }
            cnt += 1;
        }
    }
    if cnt == 0 {
        return [0, 0, 0];
    }
    core::array::from_fn(|ch| ((sum[ch] + cnt / 2) / cnt).clamp(0, 255) as u8)
}

/// 主入口：沿 field≥iso 的等值线放置椭圆（预算 max_strokes，长链优先）。
#[allow(clippy::too_many_arguments)]
pub fn contour_strokes(
    field: &[u8],
    target: &[u8],
    w: u32,
    h: u32,
    iso: u8,
    max_strokes: usize,
    thickness: f32,
    length: f32,
    spacing: f32,
) -> StrokeLayer {
    let segs = marching_squares(field, w, h, iso);
    let mut chains = chain_segments(segs, 1.5 * 1.5);
    chains.retain(|c| c.pts.len() >= 2 && chain_length(c) >= 8.0);
    // 确定性排序：链长降序，同长按首点 (y,x)
    chains.sort_by(|a, b| {
        chain_length(b)
            .partial_cmp(&chain_length(a))
            .unwrap()
            .then((a.pts[0].1, a.pts[0].0).partial_cmp(&(b.pts[0].1, b.pts[0].0)).unwrap())
    });

    let mut shapes = Vec::with_capacity(max_strokes);
    'outer: for c in &chains {
        // 等弧长重采样：n_steps 个落笔点均匀分布于链弧长
        let pts = &c.pts;
        let total = chain_length(c);
        let n_steps = ((total / spacing).floor() as usize).max(1);
        for s in 0..n_steps {
            if shapes.len() >= max_strokes {
                break 'outer;
            }
            let target_s = total * (s as f32 + 0.5) / n_steps as f32;
            // 找到弧长位置
            let (mut walked, mut cx, mut cy, mut tx, mut ty) =
                (0f32, pts[0].0, pts[0].1, 1f32, 0f32);
            for i in 0..pts.len() - 1 {
                let l = dist2(pts[i], pts[i + 1]).sqrt();
                if walked + l >= target_s {
                    let t = if l > 0.0 { (target_s - walked) / l } else { 0.0 };
                    cx = pts[i].0 + (pts[i + 1].0 - pts[i].0) * t;
                    cy = pts[i].1 + (pts[i + 1].1 - pts[i].1) * t;
                    // 切向：中心差分窗口（前一点/后一点）
                    let (pa, pb) = (
                        pts[i.saturating_sub(1)],
                        pts[(i + 1).min(pts.len() - 1)],
                    );
                    tx = pb.0 - pa.0;
                    ty = pb.1 - pa.1;
                    break;
                }
                walked += l;
            }
            let norm = (tx * tx + ty * ty).sqrt();
            let (tx, ty) = if norm > 1e-6 {
                (tx / norm, ty / norm)
            } else {
                (1.0, 0.0)
            };
            let angle_rad = ty.atan2(tx);
            let color = local_mean(target, w, h, cx.round() as i32, cy.round() as i32);
            shapes.push((
                EllipseParams {
                    cx,
                    cy,
                    rx: length,
                    ry: thickness,
                    angle_rad,
                },
                color,
            ));
        }
    }
    StrokeLayer { shapes }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_field(w: u32, h: u32, cx: f32, cy: f32, r: f32) -> Vec<u8> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                if d < r {
                    255
                } else {
                    0
                }
            })
            .collect()
    }

    fn flat_target(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut t = vec![0u8; n * 3];
        for ch in 0..3 {
            t[ch * n..(ch + 1) * n].fill(rgb[ch]);
        }
        t
    }

    #[test]
    fn circle_contour_single_closed_chain() {
        let (w, h) = (64u32, 64u32);
        let field = circle_field(w, h, 32.0, 32.0, 20.0);
        let segs = marching_squares(&field, w, h, 128);
        assert!(!segs.is_empty());
        let chains = chain_segments(segs, 1.5 * 1.5);
        let long: Vec<_> = chains.into_iter().filter(|c| chain_length(c) >= 8.0).collect();
        assert_eq!(long.len(), 1, "圆的等值线应闭合成一条链");
        let len = chain_length(&long[0]);
        let expect = 2.0 * std::f32::consts::PI * 20.0;
        assert!(
            (len - expect).abs() / expect < 0.12,
            "链长 {len} vs 周长 {expect}"
        );
    }

    #[test]
    fn strokes_on_circle_edge() {
        let (w, h) = (64u32, 64u32);
        let field = circle_field(w, h, 32.0, 32.0, 20.0);
        let target = flat_target(w, h, [10, 200, 30]);
        let layer = contour_strokes(&field, &target, w, h, 128, 800, 1.8, 3.5, 2.2);
        assert!(!layer.shapes.is_empty());
        assert!(layer.shapes.len() <= 800);
        // 全部椭圆中心距圆边 < 2.5px
        for (e, c) in &layer.shapes {
            let d = ((e.cx - 32.0).powi(2) + (e.cy - 32.0).powi(2)).sqrt();
            assert!((d - 20.0).abs() < 2.5, "中心距边 {d}");
            assert_eq!(c, &[10, 200, 30]);
        }
        // 数量 ≈ 周长/步距
        let expect = (2.0 * std::f32::consts::PI * 20.0 / 2.2) as usize;
        assert!(
            (layer.shapes.len() as i32 - expect as i32).abs() < expect as i32 / 4,
            "{} vs {expect}",
            layer.shapes.len()
        );
    }

    #[test]
    fn empty_and_full_fields() {
        let (w, h) = (32u32, 32u32);
        let t = flat_target(w, h, [0, 0, 0]);
        assert!(contour_strokes(&vec![0u8; (w * h) as usize], &t, w, h, 128, 100, 1.5, 3.0, 2.0)
            .shapes
            .is_empty());
        assert!(contour_strokes(&vec![255u8; (w * h) as usize], &t, w, h, 128, 100, 1.5, 3.0, 2.0)
            .shapes
            .is_empty());
    }

    #[test]
    fn two_circles_long_first() {
        let (w, h) = (96u32, 96u32);
        let mut field = vec![0u8; (w * h) as usize];
        for (cx, cy, r) in [(30.0, 30.0, 20.0), (74.0, 70.0, 8.0)] {
            for i in 0..w * h {
                let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() < r {
                    field[i as usize] = 255;
                }
            }
        }
        let t = flat_target(w, h, [128, 128, 128]);
        // 只给 5 笔：应全落在大圆（长链优先）
        let layer = contour_strokes(&field, &t, w, h, 128, 5, 1.5, 3.0, 2.0);
        assert_eq!(layer.shapes.len(), 5);
        for (e, _) in &layer.shapes {
            let d = ((e.cx - 30.0).powi(2) + (e.cy - 30.0).powi(2)).sqrt();
            assert!((d - 20.0).abs() < 2.5, "应在长链（大圆）上，距 {d}");
        }
    }
}
