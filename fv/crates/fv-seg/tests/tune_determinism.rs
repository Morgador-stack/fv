//! `--tune` 确定性：同种子双跑逐位一致（CPU 路径、微预算）。
//! tune_search 已入库（pipeline.rs），这是它入库前做不到的测试。

use fv_common::shape::{Shape, ShapeKind};
use fv_seg::pipeline::tune_search;
use fv_seg::{plan_layers, Matte};

const W: u32 = 96;
const H: u32 = 96;
const TOTAL: usize = 12;
const SAMPLES: usize = 120;
const SEED: u64 = 7;
const SOFT: usize = 2;

fn synth() -> (image::RgbaImage, Matte) {
    let mut img = image::RgbaImage::new(W, H);
    let mut matte = vec![0u8; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let t = x as f32 / W as f32;
            let (dx, dy) = (x as f32 - 48.0, y as f32 - 48.0);
            let d2 = dx * dx + dy * dy;
            let mut px = [40.0 + 160.0 * t, 60.0 + 120.0 * t, 100.0];
            if d2 <= 30.0 * 30.0 {
                px = [200.0 + 40.0 * (d2.sqrt() / 30.0).sin(), 90.0, 60.0];
                matte[(y * W + x) as usize] = 255;
            }
            img.put_pixel(x, y, image::Rgba([px[0] as u8, px[1] as u8, px[2] as u8, 255]));
        }
    }
    (img, Matte { width: W, height: H, data: matte })
}

fn fingerprint(s: &Shape) -> String {
    format!(
        "{:?}|{:?}|{:?}",
        [s.cx, s.cy, s.rx, s.ry, s.angle_deg],
        s.color,
        s.kind
    )
}

#[test]
fn tune_search_is_deterministic() {
    let (img, matte) = synth();
    let _ = plan_layers(&matte, TOTAL).expect("plan"); // 尺寸健全性
    let bg = [30u8, 60, 90];

    let mut g1 = None;
    let (r1, p1, _) = tune_search(&img, &matte, TOTAL, SAMPLES, SEED, bg, SOFT, false, &mut g1)
        .expect("tune run 1");
    let mut g2 = None;
    let (r2, p2, _) = tune_search(&img, &matte, TOTAL, SAMPLES, SEED, bg, SOFT, false, &mut g2)
        .expect("tune run 2");

    assert_eq!(p1, p2, "两次搜索应收敛到同一参数");
    assert_eq!(r1.psnr.to_bits(), r2.psnr.to_bits(), "PSNR 逐位一致");
    assert_eq!(r1.shapes.len(), r2.shapes.len());
    for (a, b) in r1.shapes.iter().zip(r2.shapes.iter()) {
        assert_eq!(fingerprint(a), fingerprint(b), "形状逐位一致");
    }
    // 软段（若启用）全部 228
    let tail = &r1.shapes[r1.shapes.len().saturating_sub(SOFT.min(r1.shapes.len()))..];
    assert!(tail.iter().all(|s| s.kind == ShapeKind::SoftEllipse));
}
