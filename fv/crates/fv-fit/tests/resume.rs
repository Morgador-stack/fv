//! 断点续拟合（resume）测试：前缀保留、误差改进/单调、确定性。
//!
//! 图用 192×144 双向渐变 + 强噪声（避免过早收敛）；opt_max_dim=192 == 图长边
//! → scale==1.0，orig↔opt 坐标缩放往返无损（前缀逐位一致的前提，与引擎
//! "初始形状原样保留" 的契约双重保险）。

use fv_common::shape::{Shape, ShapeKind};
use fv_fit::engine::{self, FitConfig};

fn gradient_noise_image() -> image::RgbaImage {
    let (w, h) = (192u32, 144u32);
    let mut img = image::RgbaImage::new(w, h);
    // 确定性纹理噪声（LCG，贴近真实图片特性，防平滑图过早收敛）
    let mut noise = 0xABCDu32;
    let mut nz = || {
        noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
        ((noise >> 27) as i32) - 15 // [-15, 16]
    };
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / w as f32;
            let s = y as f32 / h as f32;
            let c = [
                (t * 255.0) as i32,
                (s * 200.0) as i32,
                ((t + s) * 0.5 * 255.0) as i32,
            ];
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    (c[0] + nz()).clamp(0, 255) as u8,
                    (c[1] + nz()).clamp(0, 255) as u8,
                    (c[2] + nz()).clamp(0, 255) as u8,
                    255,
                ]),
            );
        }
    }
    img
}

fn cfg(layers: usize) -> FitConfig {
    FitConfig {
        layers,
        samples_per_shape: 3000,
        opt_max_dim: 192,
        seed: 0x5EED100,
        refine_iters: 16,
        checkpoints: vec![],
        out_prefix: None,
        bg: [0, 0, 0],
        polish_sweeps: 0,
        soft_layers: 0,
        soft_refit: false,
    auto_soft: false,
        structured: true,
    }
}

/// 逐位一致（f32 按 to_bits；位相同 → JSON 序列化亦相同）。
fn assert_same(a: &Shape, b: &Shape, idx: usize) {
    assert_eq!(a.kind, b.kind, "shape[{idx}] kind");
    assert_eq!(a.cx.to_bits(), b.cx.to_bits(), "shape[{idx}] cx");
    assert_eq!(a.cy.to_bits(), b.cy.to_bits(), "shape[{idx}] cy");
    assert_eq!(a.rx.to_bits(), b.rx.to_bits(), "shape[{idx}] rx");
    assert_eq!(a.ry.to_bits(), b.ry.to_bits(), "shape[{idx}] ry");
    assert_eq!(
        a.angle_deg.to_bits(),
        b.angle_deg.to_bits(),
        "shape[{idx}] angle_deg"
    );
    assert_eq!(a.color, b.color, "shape[{idx}] color");
}

#[test]
fn resume_improves_and_preserves_prefix() {
    let img = gradient_noise_image();

    // 第一段：拟合 100 层
    let cfg1 = cfg(100);
    let mut mses1: Vec<f64> = Vec::new();
    let mut p1 = |_, m| mses1.push(m);
    let r1 = engine::fit(&img, &cfg1, &mut p1);
    let k = r1.shapes.len();
    assert!(k >= 80, "第一段过早收敛：仅 {k} 层");
    assert!(r1.final_mse < mses1[0], "fit 未改进误差");

    // 续拟合到 200 层
    let cfg2 = cfg(200);
    let mut mses2: Vec<f64> = Vec::new();
    let mut p2 = |_, m| mses2.push(m);
    let r2 = engine::resume(&img, &cfg2, &r1.shapes, &mut p2);

    // (a) 前 k 个形状与 initial 完全一致（逐位，即 JSON 数值级相同）
    assert!(r2.shapes.len() >= k, "层数回退 {} < {k}", r2.shapes.len());
    for (i, (a, b)) in r1.shapes.iter().zip(r2.shapes.iter()).enumerate() {
        assert_same(a, b, i);
    }

    // resume 起点应≈第一段终点：scale==1 下前缀画布精确还原（角度 deg↔rad
    // 往返的 1ulp 扰动至多翻转个别边界像素，给小容差）
    assert!(
        (mses2[0] - r1.final_mse).abs() < 20.0,
        "resume 起点 {} vs fit 终点 {}",
        mses2[0],
        r1.final_mse
    );

    // resume 确实继续加了层
    assert!(r2.shapes.len() > k, "resume 未加层: {k} -> {}", r2.shapes.len());

    // (b) 终点误差 < resume 起点误差（起点 = progress 首次回调值）
    assert!(
        r2.final_mse < mses2[0],
        "final {} !< 起点 {}",
        r2.final_mse,
        mses2[0]
    );

    // (c) 误差单调不增（整数 i64 换算，容差 1e-9）
    for w in mses2.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "mse 回升: {} -> {}", w[0], w[1]);
    }
}

#[test]
fn resume_deterministic() {
    let img = gradient_noise_image();

    // 同参数第一段（fit 本身确定性）→ 固定 initial
    let mut p = |_, _| {};
    let r1 = engine::fit(&img, &cfg(40), &mut p);
    assert!(!r1.shapes.is_empty());

    // 同参两次 resume：结果逐位一致
    let run = || {
        let mut p = |_, _| {};
        engine::resume(&img, &cfg(120), &r1.shapes, &mut p)
    };
    let a = run();
    let b = run();

    assert_eq!(a.shapes.len(), b.shapes.len());
    assert_eq!(
        a.final_mse.to_bits(),
        b.final_mse.to_bits(),
        "final_mse {} vs {}",
        a.final_mse,
        b.final_mse
    );
    for (i, (x, y)) in a.shapes.iter().zip(b.shapes.iter()).enumerate() {
        assert_same(x, y, i);
    }
    assert!(a.shapes.len() > r1.shapes.len(), "resume 未加层");
}

// ---- 契约拒绝路径（panic 带清晰消息） ----

#[test]
#[should_panic(expected = "只接受 RotatedEllipse")]
fn resume_rejects_rect_shape() {
    let img = gradient_noise_image();
    let bad = Shape {
        kind: ShapeKind::Rect,
        cx: 5.0,
        cy: 5.0,
        rx: 3.0,
        ry: 3.0,
        angle_deg: 0.0,
        color: [1, 2, 3, 255],
    };
    let mut p = |_, _| {};
    engine::resume(&img, &cfg(10), std::slice::from_ref(&bad), &mut p);
}

#[test]
#[should_panic(expected = "超过 cfg.layers")]
fn resume_rejects_too_many_initial() {
    let img = gradient_noise_image();
    let mut p = |_, _| {};
    let s = Shape::ellipse(5.0, 5.0, 3.0, 3.0, 0.0, [1, 2, 3]);
    engine::resume(&img, &cfg(1), &[s, s], &mut p);
}
