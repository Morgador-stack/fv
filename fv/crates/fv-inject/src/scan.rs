//! 层表定位：u16 扫描法（与 vinylizer/FD6/bvzrays 三方一致的策略，严格模式）。
//!
//! 1. 遍历可扫描区域，分块找 u16 == 模板层数
//! 2. 候选 LiveryGroup = 命中地址 − layer_count_offset
//! 3. 读 layer_table_offset 处 u64 表指针，校验可读
//! 4. 严格校验：前 sample_layers 层全部通过（指针合法、shape_id ∈ {rect, ellipse}、mask==0、
//!    pos/scale/color 可读且 blob 不越区域边界）
//! 5. 全过 → 命中。任何一个候选字段可疑即整体拒绝（宁可失败不可写错）。

use crate::layout::Layout;
use crate::mem::{MemIo, read_u64, read_u8};

/// 单层读取快照（抽样确认/统计用）。
#[derive(Debug, Clone)]
pub struct LayerSnapshot {
    pub index: usize,
    pub ptr: u64,
    pub pos: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
    pub color: [u8; 4],
    pub shape_id: u8,
    pub mask: u8,
}

#[derive(Debug, Clone)]
pub struct Located {
    pub group_addr: u64,
    pub table_addr: u64,
    pub layer_count: usize,
}

fn is_user_ptr(v: u64) -> bool {
    (0x1_0000..=0x0000_7FFF_FFFF_FFFF).contains(&v)
}

/// 指针必须落在某块可读区域内，且目标 blob 完整在区域内。
fn ptr_in_readable_region(regions: &[crate::mem::Region], ptr: u64, need: usize) -> bool {
    if !is_user_ptr(ptr) {
        return false;
    }
    regions
        .iter()
        .any(|r| ptr >= r.base && ptr + need as u64 <= r.base + r.size as u64)
}

/// 读一层快照；任何字段不可读 → None。
pub fn snapshot_layer(m: &dyn MemIo, lay: &Layout, ptr: u64) -> Option<LayerSnapshot> {
    let l = &lay.layer;
    let pos = read_f32x2(m, ptr + l.position_offset)?;
    let scale = read_f32x2(m, ptr + l.scale_offset)?;
    let mut rot_b = [0u8; 4];
    if !m.read_exact(ptr + l.rotation_offset, &mut rot_b) {
        return None;
    }
    let mut color = [0u8; 4];
    if !m.read_exact(ptr + l.color_offset, &mut color) {
        return None;
    }
    let shape_id = read_u8(m, ptr + l.shape_id_offset)?;
    let mask = read_u8(m, ptr + l.mask_offset)?;
    Some(LayerSnapshot {
        index: 0,
        ptr,
        pos,
        scale,
        rotation: f32::from_le_bytes(rot_b),
        color,
        shape_id,
        mask,
    })
}

fn read_f32x2(m: &dyn MemIo, addr: u64) -> Option<[f32; 2]> {
    let mut b = [0u8; 8];
    if !m.read_exact(addr, &mut b) {
        return None;
    }
    Some([
        f32::from_le_bytes(b[0..4].try_into().unwrap()),
        f32::from_le_bytes(b[4..8].try_into().unwrap()),
    ])
}

/// 严格校验一层：shape_id ∈ {rect, ellipse} 且 mask == 0。
fn strict_ok(lay: &Layout, s: &LayerSnapshot) -> bool {
    (s.shape_id == lay.shape_ids.rect || s.shape_id == lay.shape_ids.ellipse) && s.mask == 0
}

/// 统计有效层分布（确认模板健康度）。
pub fn count_layers(m: &dyn MemIo, lay: &Layout, table: u64, count: usize) -> (usize, usize, usize) {
    let mut ok = 0;
    let mut rects = 0;
    let mut ellipses = 0;
    for i in 0..count {
        let Some(ptr) = read_u64(m, table + (i as u64) * 8) else {
            continue;
        };
        let Some(s) = snapshot_layer(m, lay, ptr) else {
            continue;
        };
        if !strict_ok(lay, &s) {
            continue;
        }
        ok += 1;
        if s.shape_id == lay.shape_ids.rect {
            rects += 1;
        } else {
            ellipses += 1;
        }
    }
    (ok, rects, ellipses)
}

/// 抽样 n 层做人工确认（均匀取下标）。
pub fn sample_layers(
    m: &dyn MemIo,
    lay: &Layout,
    table: u64,
    count: usize,
    n: usize,
) -> Vec<LayerSnapshot> {
    let mut out = Vec::new();
    if count == 0 || n == 0 {
        return out;
    }
    let step = (count / n).max(1);
    let mut i = 0usize;
    while i < count && out.len() < n {
        if let Some(ptr) = read_u64(m, table + (i as u64) * 8)
            && let Some(mut s) = snapshot_layer(m, lay, ptr) {
                s.index = i;
                out.push(s);
            }
        i += step;
    }
    out
}

/// 主定位入口：全内存 u16 扫描 + 严格校验。
pub fn locate_layer_table(m: &dyn MemIo, lay: &Layout, layer_count: usize) -> Option<Located> {
    let regions = m.regions();
    let chunk = lay.scan.region_chunk;
    let n_check = layer_count.min(lay.scan.sample_layers);
    let pattern = layer_count as u16;

    for region in &regions {
        let mut off = 0usize;
        while off < region.size {
            let len = chunk.min(region.size - off);
            let mut buf = vec![0u8; len];
            let got = m.read_partial(region.base + off as u64, &mut buf);
            if got < 2 {
                break;
            }
            buf.truncate(got);
            for i in 0..buf.len().saturating_sub(1) {
                if u16::from_le_bytes([buf[i], buf[i + 1]]) != pattern {
                    continue;
                }
                let hit = region.base + off as u64 + i as u64;
                let Some(found) = try_candidate(m, lay, &regions, hit, layer_count, n_check) else {
                    continue;
                };
                return Some(found);
            }
            // 1 字节重叠推进：u16 模式跨块边界（末字节+下块首字节）原实现漏检
            // （CODE_REVIEW §E；扫描语义不变，仅多查一次跨界对）
            off += got.saturating_sub(1).max(1);
        }
    }
    None
}

fn try_candidate(
    m: &dyn MemIo,
    lay: &Layout,
    regions: &[crate::mem::Region],
    hit: u64,
    layer_count: usize,
    n_check: usize,
) -> Option<Located> {
    let group = hit.checked_sub(lay.livery_group.layer_count_offset)?;
    let table = read_u64(m, group + lay.livery_group.layer_table_offset)?;
    if !ptr_in_readable_region(regions, table, layer_count * 8) {
        return None;
    }
    // 严格校验：全部采样层必须通过
    for i in 0..n_check {
        let ptr = read_u64(m, table + (i as u64) * 8)?;
        if !ptr_in_readable_region(regions, ptr, lay.layer.blob_size as usize) {
            return None;
        }
        let s = snapshot_layer(m, lay, ptr)?;
        if !strict_ok(lay, &s) {
            return None;
        }
    }
    Some(Located {
        group_addr: group,
        table_addr: table,
        layer_count,
    })
}
