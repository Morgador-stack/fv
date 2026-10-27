//! 每行前缀和（积分图）：支撑候选椭圆 O(高度) 误差评分。
//!
//! ## 12 张 map
//! 对每个通道 ch ∈ {0,1,2} 各四张（i64，值为对应量的行前缀和）：
//! - `T`  : Σ target            （最优颜色 = T/|R|）
//! - `T2` : Σ target²
//! - `TC` : Σ target·canvas     （画布变化后需刷新）
//! - `C2` : Σ canvas²           （画布变化后需刷新）
//!
//! 候选评分只需：new_err − cur_err
//! ```text
//! new_err(R, color) = Σ_ch [ T2 − 2·color_ch·T + |R|·color_ch² ]
//! cur_err(R)        = Σ_ch [ T2 − 2·TC + C2 ]
//! ```
//! |R| 由 `raster::spans_pixel_count` 提供。
//!
//! ## 布局
//! `data` 为 map-major：map 顺序 (T,T2,TC,C2)×ch，共 12 张；
//! 每张 h 行、每行 (w+1) 个 i64（行首哨兵 0），行内连续。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapId {
    T,
    T2,
    TC,
    C2,
}

fn map_slot(map: MapId) -> usize {
    match map {
        MapId::T => 0,
        MapId::T2 => 1,
        MapId::TC => 2,
        MapId::C2 => 3,
    }
}

/// 填充一张 plane（行前缀和）：`val(pixel)` 返回该像素进入前缀的值。
fn fill_plane(
    data: &mut [i64],
    plane_base: usize,
    stride: usize,
    w: usize,
    h: usize,
    val: impl Fn(usize) -> i64,
) {
    for y in 0..h {
        let row = plane_base + y * stride;
        let mut acc = 0i64;
        data[row] = 0; // 行首哨兵
        for x in 0..w {
            acc += val(y * w + x);
            data[row + x + 1] = acc;
        }
    }
}

pub struct RowSums {
    w: u32,
    h: u32,
    /// 12 张行前缀和，plane-major：plane = map_slot*3 + ch，
    /// 每张 h 行 × (w+1) i64，行首哨兵 0。
    data: Vec<i64>,
    /// set_target 时保存的 target 副本（refresh_canvas 计算 TC 需要）
    target: Vec<u8>,
}

impl RowSums {
    /// 分配 12 张 map 的存储（不填充）。
    pub fn new(w: u32, h: u32) -> Self {
        let stride = w as usize + 1;
        Self {
            w,
            h,
            data: vec![0i64; 4 * 3 * h as usize * stride],
            target: Vec::new(),
        }
    }

    fn plane_stride(&self) -> usize {
        self.w as usize + 1
    }

    /// 由目标图填充 T/T2（3 平面 SoA u8，len == 3·w·h）。只需调用一次。
    pub fn set_target(&mut self, target: &[u8]) {
        let n = (self.w * self.h) as usize;
        assert_eq!(target.len(), n * 3, "target 必须是 3 平面 SoA");
        self.target = target.to_vec();
        let stride = self.plane_stride();
        let (w, h) = (self.w as usize, self.h as usize);
        let plane_len = h * stride;
        let data = &mut self.data;
        for ch in 0..3 {
            let src = &target[ch * n..(ch + 1) * n];
            fill_plane(data, (map_slot(MapId::T) * 3 + ch) * plane_len, stride, w, h, |p| {
                src[p] as i64
            });
            fill_plane(data, (map_slot(MapId::T2) * 3 + ch) * plane_len, stride, w, h, |p| {
                let v = src[p] as i64;
                v * v
            });
        }
    }

    /// 由当前画布填充 TC/C2（3 平面 SoA u8）。每接纳一个形状后调用。
    pub fn refresh_canvas(&mut self, canvas: &[u8]) {
        let n = (self.w * self.h) as usize;
        assert_eq!(canvas.len(), n * 3, "canvas 必须是 3 平面 SoA");
        assert!(
            !self.target.is_empty(),
            "必须先 set_target 再 refresh_canvas"
        );
        let stride = self.plane_stride();
        let (w, h) = (self.w as usize, self.h as usize);
        let plane_len = h * stride;
        let Self { data, target, .. } = self;
        for ch in 0..3 {
            let t = &target[ch * n..(ch + 1) * n];
            let c = &canvas[ch * n..(ch + 1) * n];
            fill_plane(data, (map_slot(MapId::TC) * 3 + ch) * plane_len, stride, w, h, |p| {
                t[p] as i64 * c[p] as i64
            });
            fill_plane(data, (map_slot(MapId::C2) * 3 + ch) * plane_len, stride, w, h, |p| {
                let v = c[p] as i64;
                v * v
            });
        }
    }

    /// 12 张 map 的原始 i64 平面数据（map-major：plane = map_slot*3+ch，每张 h 行 × (w+1)）。
    /// 供 GPU 评估器上传显存用。
    pub fn planes_i64(&self) -> &[i64] {
        &self.data
    }

    /// 行段和：map 在通道 ch、行 y 上 [x0, x1) 的和（半开，x0 ≤ x1 ≤ w）。
    #[inline]
    pub fn seg(&self, map: MapId, ch: usize, y: u32, x0: u32, x1: u32) -> i64 {
        debug_assert!(ch < 3 && y < self.h && x0 <= x1 && x1 <= self.w);
        let stride = self.plane_stride();
        let plane = map_slot(map) * 3 + ch;
        let row = plane * self.h as usize * stride + y as usize * stride;
        self.data[row + x1 as usize] - self.data[row + x0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 37;
    const H: u32 = 23;

    fn fixture() -> (RowSums, Vec<u8>, Vec<u8>) {
        let n = (W * H) as usize;
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        let target: Vec<u8> = (0..n * 3).map(|_| next()).collect();
        let canvas: Vec<u8> = (0..n * 3).map(|_| next()).collect();
        let mut rs = RowSums::new(W, H);
        rs.set_target(&target);
        rs.refresh_canvas(&canvas);
        (rs, target, canvas)
    }

    fn naive_seg(data: &[u8], ch: usize, y: u32, x0: u32, x1: u32, quad: impl Fn(i64) -> i64) -> i64 {
        let mut s = 0i64;
        for x in x0..x1 {
            let v = data[ch * (W * H) as usize + y as usize * W as usize + x as usize] as i64;
            s += quad(v);
        }
        s
    }

    #[test]
    fn all_maps_exact_vs_naive() {
        let (rs, target, canvas) = fixture();
        let maps = [
            (MapId::T, 0usize),
            (MapId::T2, 1),
            (MapId::TC, 2),
            (MapId::C2, 3),
        ];
        let mut probe = 0xACE1u32;
        let mut next = || {
            probe = probe.wrapping_mul(1664525).wrapping_add(1013904223);
            probe >> 16
        };
        for _ in 0..2000 {
            let ch = (next() % 3) as usize;
            let y = next() % H;
            let x0 = next() % (W + 1);
            let x1 = x0 + next() % (W + 1 - x0);
            for (map, slot) in maps {
                let got = rs.seg(map, ch, y, x0, x1);
                let want = match slot {
                    0 => naive_seg(&target, ch, y, x0, x1, |v| v),
                    1 => naive_seg(&target, ch, y, x0, x1, |v| v * v),
                    _ => {
                        // TC / C2
                        let mut s = 0i64;
                        for x in x0..x1 {
                            let t = target[ch * (W * H) as usize + y as usize * W as usize + x as usize] as i64;
                            let c = canvas[ch * (W * H) as usize + y as usize * W as usize + x as usize] as i64;
                            s += if slot == 2 { t * c } else { c * c };
                        }
                        s
                    }
                };
                assert_eq!(got, want, "map={map:?} ch={ch} y={y} [{x0},{x1})");
            }
        }
    }

    #[test]
    fn seg_boundaries_and_additivity() {
        let (rs, _, _) = fixture();
        let ch = 1;
        let y = 5;
        assert_eq!(rs.seg(MapId::T, ch, y, 3, 3), 0);
        let full = rs.seg(MapId::T2, ch, y, 0, W);
        let split = rs.seg(MapId::T2, ch, y, 0, 11) + rs.seg(MapId::T2, ch, y, 11, W);
        assert_eq!(full, split);
    }

    #[test]
    fn refresh_after_canvas_change() {
        let n = (W * H) as usize;
        let target = vec![100u8; n * 3];
        let mut canvas = vec![50u8; n * 3];
        let mut rs = RowSums::new(W, H);
        rs.set_target(&target);
        rs.refresh_canvas(&canvas);
        // 改画布一个像素 → TC/C2 变化
        canvas[0] = 200;
        rs.refresh_canvas(&canvas);
        assert_eq!(rs.seg(MapId::TC, 0, 0, 0, 1), 100 * 200);
        assert_eq!(rs.seg(MapId::C2, 0, 0, 0, 1), 200 * 200);
        assert_eq!(rs.seg(MapId::C2, 0, 0, 1, 2), 50 * 50);
    }

    #[test]
    fn prefix_rows_monotonic_for_t() {
        let (rs, _, _) = fixture();
        for y in 0..H {
            for ch in 0..3 {
                for x in 1..=W {
                    assert!(
                        rs.seg(MapId::T, ch, y, x - 1, x) >= 0,
                        "u8 前缀和的增量必须非负"
                    );
                }
            }
        }
    }
}
