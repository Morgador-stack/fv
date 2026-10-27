//! 软椭圆（228）通道 e2e：实心饱和后软层继续降误差；JSON 含 228；确定性。

use fv_fit::engine::{self, FitConfig};
use fv_common::shape::ShapeKind;

fn noisy_image(w: u32, h: u32) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(w, h);
    let mut noise = 0xC0FFu32;
    let mut nz = || {
        noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
        ((noise >> 28) as i32) - 8
    };
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / w as f32;
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    ((t * 255.0) as i32 + nz()).clamp(0, 255) as u8,
                    (60 + nz()).clamp(0, 255) as u8,
                    ((255.0 - t * 255.0) as i32 + nz()).clamp(0, 255) as u8,
                    255,
                ]),
            );
        }
    }
    img
}

fn cfg(layers: usize, polish: usize, soft: usize) -> FitConfig {
    FitConfig {
        layers,
        samples_per_shape: 1500,
        opt_max_dim: 160,
        seed: 0x50F7,
        refine_iters: 12,
        checkpoints: vec![],
        out_prefix: None,
        bg: [0, 0, 0],
        polish_sweeps: polish,
        soft_layers: soft,
        soft_refit: true,
    auto_soft: false,
        structured: true,
    }
}

#[test]
fn soft_layers_improve_after_saturation() {
    let img = noisy_image(160, 120);
    let mut p = |_, _| {};
    let base = engine::fit(&img, &cfg(60, 2, 0), &mut p);
    let with_soft = engine::fit(&img, &cfg(60, 2, 40), &mut p);
    assert!(
        with_soft.final_mse < base.final_mse,
        "soft {} vs base {}",
        with_soft.final_mse,
        base.final_mse
    );
    let soft_count = with_soft
        .shapes
        .iter()
        .filter(|s| s.kind == ShapeKind::SoftEllipse)
        .count();
    assert!(soft_count >= 20, "软层数 {soft_count}");
    for s in with_soft.shapes.iter().filter(|s| s.kind == ShapeKind::SoftEllipse) {
        assert_eq!(s.color[3], 255, "软层 alpha=255（opacity 退化进颜色）");
    }
}

#[test]
fn soft_deterministic() {
    let img = noisy_image(96, 72);
    let run = || {
        let mut p = |_, _| {};
        let r = engine::fit(&img, &cfg(30, 1, 15), &mut p);
        (r.final_mse.to_bits(), r.shapes.len())
    };
    assert_eq!(run(), run());
}

#[test]
fn auto_soft_fills_remaining_budget() {
    let img = noisy_image(160, 120);
    let mut p = |_, _| {};
    // 实心-only：60 层预算会早停（噪声图饱和快）
    let solid_only = engine::fit(&img, &cfg(60, 1, 0), &mut p);
    // auto-soft：总预算 60，实心饱和后软层接管
    let auto = engine::fit(
        &img,
        &FitConfig {
            auto_soft: true,
        structured: true,
            ..cfg(60, 1, 0)
        },
        &mut p,
    );
    assert!(
        auto.shapes.len() > solid_only.shapes.len(),
        "auto {} vs solid-only {}",
        auto.shapes.len(),
        solid_only.shapes.len()
    );
    assert!(auto.final_mse < solid_only.final_mse);
    // 总数不超过总预算
    assert!(auto.shapes.len() <= 60);
    let soft = auto
        .shapes
        .iter()
        .filter(|s| s.kind == ShapeKind::SoftEllipse)
        .count();
    assert!(soft > 0, "应有软层接管剩余预算");
}
