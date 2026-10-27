//! Mock 全链路 e2e：构造假地址空间（含诱饵）→ 定位 → 统计/抽样 → 写入 → 字节级验证 → 回滚验证。

use fv_inject::convert::{self, CanvasOffset};
use fv_inject::layout::Layout;
use fv_inject::mem::{MemIo, MockMem};
use fv_inject::{scan, write_process};
use fv_common::json::ShapeJson;

const GROUP: u64 = 0x5000_0000;
const TABLE: u64 = 0x6000_0000;
const LAYERS: u64 = 0x7000_0000;
const N: usize = 64;

/// 构造一个"健康的球模板"地址空间 + 两个诱饵。
fn build_mem() -> MockMem {
    let mut m = MockMem::default();
    let lay = Layout::default();

    // LiveryGroup：count@0x5A=64，table@0x78
    let mut group = vec![0u8; 0x200];
    group[0x5A..0x5C].copy_from_slice(&(N as u16).to_le_bytes());
    group[0x78..0x80].copy_from_slice(&TABLE.to_le_bytes());
    m.add_block(GROUP, group);

    // 层表：64 个 u64 指针
    let mut table = Vec::with_capacity(N * 8);
    for i in 0..N {
        table.extend_from_slice(&(LAYERS + (i as u64) * 0x200).to_le_bytes());
    }
    m.add_block(TABLE, table);

    // 每层 blob：椭圆 102 / mask 0 / 白色 / 单位缩放
    for i in 0..N {
        let mut blob = vec![0u8; 0x200];
        blob[lay.layer.shape_id_offset as usize] = lay.shape_ids.ellipse;
        // mask @0x78 保持 0
        blob[lay.layer.position_offset as usize..lay.layer.position_offset as usize + 8]
            .copy_from_slice(&[0.0f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat());
        blob[lay.layer.scale_offset as usize..lay.layer.scale_offset as usize + 8]
            .copy_from_slice(&[1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat());
        blob[lay.layer.color_offset as usize..lay.layer.color_offset as usize + 4]
            .copy_from_slice(&[255, 255, 255, 255]);
        m.add_block(LAYERS + (i as u64) * 0x200, blob);
    }

    // 诱饵 1：u16==64 命中，但表指针指向垃圾
    let mut d1 = vec![0u8; 0x200];
    d1[0x5A..0x5C].copy_from_slice(&(N as u16).to_le_bytes());
    d1[0x78..0x80].copy_from_slice(&0x1u64.to_le_bytes()); // 非法指针
    m.add_block(0x8000_0000, d1);

    // 诱饵 2：完整结构，但某层 shape_id 是社区形状（非基础）
    let mut d2g = vec![0u8; 0x200];
    d2g[0x5A..0x5C].copy_from_slice(&(N as u16).to_le_bytes());
    d2g[0x78..0x80].copy_from_slice(&0x8100_0000u64.to_le_bytes());
    m.add_block(0x8080_0000, d2g);
    let mut d2t = Vec::new();
    for _ in 0..N {
        d2t.extend_from_slice(&0x8200_0000u64.to_le_bytes());
    }
    m.add_block(0x8100_0000, d2t);
    let mut d2l = vec![0u8; 0x200];
    d2l[lay.layer.shape_id_offset as usize] = 114; // 社区形状，严格模式拒绝
    m.add_block(0x8200_0000, d2l);

    m
}

fn f32_at(m: &MockMem, addr: u64) -> f32 {
    let mut b = [0u8; 4];
    m.read_exact(addr, &mut b);
    f32::from_le_bytes(b)
}

#[test]
fn locate_finds_real_table_and_rejects_decoys() {
    let m = build_mem();
    let lay = Layout::default();
    let found = scan::locate_layer_table(&m, &lay, N).expect("应命中真实层表");
    assert_eq!(found.group_addr, GROUP);
    assert_eq!(found.table_addr, TABLE);
}

#[test]
fn locate_fails_on_wrong_count() {
    let m = build_mem();
    let lay = Layout::default();
    assert!(scan::locate_layer_table(&m, &lay, N + 1).is_none());
}

#[test]
fn stats_and_sampling() {
    let m = build_mem();
    let lay = Layout::default();
    let (ok, rects, ellipses) = scan::count_layers(&m, &lay, TABLE, N);
    assert_eq!((ok, rects, ellipses), (N, 0, N));
    let s = scan::sample_layers(&m, &lay, TABLE, N, 6);
    assert_eq!(s.len(), 6);
    assert!(s.iter().all(|x| x.shape_id == 102 && x.mask == 0));
    assert_eq!(s[0].index, 0);
}

#[test]
fn locate_finds_pattern_straddling_chunk_seam() {
    // 回归（CODE_REVIEW §E）：u16 模式骑在扫描块边界（末字节+下块首字节），
    // 旧实现 off += got 漏检。构造 region_chunk=0x100、count 落在
    // GROUP+0x1FF 处（第一块末字节 + 第二块首字节）。
    let mut m = MockMem::default();
    let mut lay = Layout::default();
    lay.scan.region_chunk = 0x100;

    // 健康模板，但 count 写在 0x1FF（跨界对：块1 的 0xFF + 块2 的 0x100）。
    // try_candidate 以 count 地址为锚：表指针须在 hit+(0x78−0x5A)=0x21D。
    let mut group = vec![0u8; 0x400];
    group[0x1FF..0x201].copy_from_slice(&(N as u16).to_le_bytes());
    group[0x21D..0x225].copy_from_slice(&TABLE.to_le_bytes());
    m.add_block(GROUP, group);
    let mut table = Vec::new();
    for i in 0..N {
        table.extend_from_slice(&(LAYERS + (i as u64) * 0x200).to_le_bytes());
    }
    m.add_block(TABLE, table);
    let l = Layout::default();
    for i in 0..N {
        let mut blob = vec![0u8; 0x200];
        blob[l.layer.shape_id_offset as usize] = l.shape_ids.ellipse;
        blob[l.layer.scale_offset as usize..l.layer.scale_offset as usize + 8]
            .copy_from_slice(&[1.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat());
        blob[l.layer.color_offset as usize..l.layer.color_offset as usize + 4]
            .copy_from_slice(&[255, 255, 255, 255]);
        m.add_block(LAYERS + (i as u64) * 0x200, blob);
    }
    let found =
        scan::locate_layer_table(&m, &lay, N).expect("跨界 u16 必须命中（接缝修复回归）");
    // count 锚定语义：group_base = hit − 0x5A
    assert_eq!(found.group_addr, GROUP + 0x1FF - 0x5A);
}

#[test]
fn inject_writes_exact_bytes() {
    let mut m = build_mem();
    let lay = Layout::default();
    let shapes: Vec<ShapeJson> = (0..10)
        .map(|i| ShapeJson {
            kind: 16,
            data: vec![100.0 + i as f32, 200.0, 63.0, 126.0, 30.0],
            color: [10, 20, 30, 255],
        })
        .collect();
    let fields: Vec<_> = shapes
        .iter()
        .map(|s| convert::prepare(s, &lay, CanvasOffset::NONE).unwrap())
        .collect();

    let (ok, fail) = write_process::inject_all(&mut m, &lay, TABLE, N, &fields);
    assert_eq!((ok, fail), (N, 0), "10 写入 + 54 清空全部成功");

    // 逐字节验证 layer[0]（写入的）
    let p0 = LAYERS;
    let mut b = [0u8; 4];
    m.read_exact(p0 + lay.layer.position_offset, &mut b);
    assert_eq!(f32::from_le_bytes(b), 100.0);
    m.read_exact(p0 + lay.layer.position_offset + 4, &mut b);
    assert_eq!(f32::from_le_bytes(b), -200.0, "y 取负");
    m.read_exact(p0 + lay.layer.scale_offset, &mut b);
    assert_eq!(f32::from_le_bytes(b), 1.0);
    m.read_exact(p0 + lay.layer.rotation_offset, &mut b);
    assert!((f32::from_le_bytes(b) - 330.0).abs() < 1e-3);
    let mut c = [0u8; 4];
    m.read_exact(p0 + lay.layer.color_offset, &mut c);
    assert_eq!(c, [10, 20, 30, 255]);
    m.read_exact(p0 + lay.layer.shape_id_offset, &mut b[..1]);
    assert_eq!(b[0], 102);

    // layer[63]（被清空的）：缩放 0、矩形
    let p63 = LAYERS + 63 * 0x200;
    assert_eq!(f32_at(&m, p63 + lay.layer.scale_offset), 0.0);
    m.read_exact(p63 + lay.layer.shape_id_offset, &mut b[..1]);
    assert_eq!(b[0], 101);
}

#[test]
fn rollback_on_partial_write_failure() {
    let mut m = build_mem();
    let lay = Layout::default();
    let fields = convert::prepare(
        &ShapeJson {
            kind: 16,
            data: vec![111.0, 222.0, 63.0, 63.0, 0.0],
            color: [1, 2, 3, 255],
        },
        &lay,
        CanvasOffset::NONE,
    )
    .unwrap();

    // layer[5] 的 color 字段写保护：pos/scale/rot 先写成功，color 失败 → 触发回滚
    let p5 = LAYERS + 5 * 0x200;
    m.protect(p5 + lay.layer.color_offset, 4);

    let r = write_process::write_layer(&mut m, &lay, TABLE, 5, &fields);
    assert!(r.is_err(), "写保护层必须失败");
    assert_eq!(r.unwrap_err().0, 3, "pos/scale/rot 已写、color 失败");

    // 回滚验证：三个字段都恢复旧值（0.0/0.0、1.0/1.0、0.0）
    assert_eq!(f32_at(&m, p5 + lay.layer.position_offset), 0.0);
    assert_eq!(f32_at(&m, p5 + lay.layer.position_offset + 4), 0.0);
    assert_eq!(f32_at(&m, p5 + lay.layer.scale_offset), 1.0);
    assert_eq!(f32_at(&m, p5 + lay.layer.scale_offset + 4), 1.0);
    assert_eq!(f32_at(&m, p5 + lay.layer.rotation_offset), 0.0);

    // 坏指针层：读旧值阶段即失败，零字节写入
    m.poke(TABLE + 6 * 8, &0x9000_0000u64.to_le_bytes());
    let r2 = write_process::write_layer(&mut m, &lay, TABLE, 6, &fields);
    assert!(r2.is_err());
    assert_eq!(r2.unwrap_err().0, 0, "未写任何字段");
}
