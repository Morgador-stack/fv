//! 端到端测试：合成图 → 拟合 → 产物校验（确定性、质量、文件）。

use fv_fit::engine::{self, FitConfig};
use image::GenericImageView;

fn synthetic_image(w: u32, h: u32) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(w, h);
    // 确定性纹理噪声（避免纯平滑图过早收敛，贴近真实图片特性）
    let mut noise = 0x5EEDu32;
    let mut nz = || {
        noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
        ((noise >> 28) as i32) - 7 // [-7, 8]
    };
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / w as f32;
            let mut c = [
                (t * 255.0) as i32,
                40,
                (255.0 - t * 255.0) as i32,
            ];
            let dx = x as f32 - w as f32 * 0.5;
            let dy = y as f32 - h as f32 * 0.5;
            let r = (dx * dx + dy * dy).sqrt();
            if r < h as f32 * 0.18 {
                c = [220, 30, 30];
            } else if (r - h as f32 * 0.32).abs() < 3.0 {
                c = [240, 240, 240];
            }
            let n = nz();
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    (c[0] + n).clamp(0, 255) as u8,
                    (c[1] + nz()).clamp(0, 255) as u8,
                    (c[2] + nz()).clamp(0, 255) as u8,
                    255,
                ]),
            );
            let _ = n;
        }
    }
    img
}

fn small_cfg(layers: usize, samples: usize, seed: u64, prefix: Option<std::path::PathBuf>) -> FitConfig {
    FitConfig {
        layers,
        samples_per_shape: samples,
        opt_max_dim: 192,
        seed,
        refine_iters: 16,
        checkpoints: vec![],
        out_prefix: prefix,
        bg: [0, 0, 0],
        polish_sweeps: 0,
        soft_layers: 0,
        soft_refit: false,
    auto_soft: false,
        structured: true,
    }
}

#[test]
fn e2e_fit_quality_and_outputs() {
    let img = synthetic_image(192, 144);
    let dir = std::env::temp_dir().join(format!("fv-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("out");

    let cfg = FitConfig {
        checkpoints: vec![100, 200],
        out_prefix: Some(prefix.clone()),
        ..small_cfg(200, 3000, 7, None)
    };
    let mut mses: Vec<f64> = Vec::new();
    let mut progress = |_, mse| mses.push(mse);
    let result = engine::fit(&img, &cfg, &mut progress);

    assert!(result.shapes.len() >= 60, "层数 {}", result.shapes.len());
    assert!(
        result.final_mse < mses[0] * 0.5,
        "mse {} -> {}",
        mses[0],
        result.final_mse
    );
    // 误差单调不增（整数 i64 换算，容差 1e-9）
    for w in mses.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "mse 回升: {} -> {}", w[0], w[1]);
    }
    // 产物：终稿 + checkpoint
    let j = std::path::PathBuf::from(format!("{}.json", prefix.display()));
    assert!(j.exists(), "缺 {j:?}");
    let design = fv_common::json::read_design(&j).unwrap();
    assert_eq!(design.shapes.len(), result.shapes.len());
    for s in &design.shapes {
        assert_eq!(s.kind, 16);
        assert_eq!(s.data.len(), 5);
        assert_eq!(s.color[3], 255);
    }
    for cp in [100, 200] {
        let jc = std::path::PathBuf::from(format!("{}.{}.json", prefix.display(), cp));
        // 收敛早退时，超过实际层数的 checkpoint 按定义不会产生
        if cp <= result.shapes.len() {
            assert!(jc.exists(), "缺 checkpoint {jc:?}");
        }
    }
    let prev = std::path::PathBuf::from(format!("{}.preview.png", prefix.display()));
    assert!(prev.exists());
    let back = image::open(&prev).unwrap();
    assert_eq!(back.dimensions(), (192, 144));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn e2e_determinism() {
    let img = synthetic_image(96, 72);
    let run = || {
        let cfg = small_cfg(60, 2000, 123, None);
        let mut p = |_, _| {};
        let r = engine::fit(&img, &cfg, &mut p);
        (r.final_mse, r.shapes.len())
    };
    assert_eq!(run(), run());
}

#[test]
fn e2e_transparent_region_untouched() {
    // 左半不透明（纯红），右半全透明（bg=绿）：形状应集中在左半
    let mut img = image::RgbaImage::new(128, 64);
    for y in 0..64 {
        for x in 0..128 {
            let px = if x < 64 { [200, 0, 0, 255] } else { [0, 0, 0, 0] };
            img.put_pixel(x, y, image::Rgba(px));
        }
    }
    let cfg = FitConfig {
        bg: [0, 200, 0],
        ..small_cfg(40, 1500, 42, None)
    };
    let mut p = |_, _| {};
    let r = engine::fit(&img, &cfg, &mut p);
    let left = r.shapes.iter().filter(|s| s.cx < 64.0).count();
    assert!(left * 2 > r.shapes.len(), "左半形状 {left}/{}", r.shapes.len());
}
