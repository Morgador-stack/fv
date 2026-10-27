//! 前向软光栅 GPU vs CPU 交叉验证（M18_DESIGN §3 验收门槛 1）。

use fv_gpu::diff::fwd::DiffRenderer;
use fv_gpu::diff::ShapeParams;

const W: u32 = 48;
const H: u32 = 32;
const BG: [f32; 3] = [30.0, 60.0, 90.0];

fn scene() -> Vec<ShapeParams> {
    vec![
        // 居中圆
        ShapeParams { cx: 24.0, cy: 16.0, a: 8.0, b: 8.0, theta: 0.0, r: 200.0, g: 40.0, bl: 40.0 },
        // 45° 旋转椭圆，压在圆上方
        ShapeParams { cx: 28.0, cy: 14.0, a: 14.0, b: 4.0, theta: 0.785, r: 250.0, g: 220.0, bl: 90.0 },
        // 细长 20:1
        ShapeParams { cx: 20.0, cy: 24.0, a: 20.0, b: 1.0, theta: -0.3, r: 30.0, g: 240.0, bl: 30.0 },
        // 离屏大层（部分入画）
        ShapeParams { cx: -10.0, cy: -10.0, a: 30.0, b: 22.0, theta: 1.2, r: 90.0, g: 90.0, bl: 200.0 },
        // 80° 细旋转
        ShapeParams { cx: 36.0, cy: 26.0, a: 9.0, b: 2.0, theta: 1.4, r: 255.0, g: 255.0, bl: 255.0 },
        // 全覆盖大层
        ShapeParams { cx: 24.0, cy: 16.0, a: 100.0, b: 100.0, theta: 0.3, r: 10.0, g: 130.0, bl: 180.0 },
    ]
}

fn target_solid() -> Vec<f32> {
    // 常数目标即可：前向只依赖形状，不依赖 target 内容
    vec![120.0; (W * H * 3) as usize]
}

#[test]
fn gpu_matches_cpu_reference() {
    let shapes = scene();
    let target = target_solid();
    let mut r = match DiffRenderer::new(W, H, shapes.len()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("跳过（无可用 GPU）：{e}");
            return;
        }
    };
    for tau in [4.0f32, 64.0] {
        let gpu = r.render(&shapes, tau, BG, &target).expect("render");
        let cpu = DiffRenderer::cpu_forward_reference(&shapes, tau, BG, W, H);
        let mut max_diff = 0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((g - *c as f32).abs());
        }
        assert!(
            max_diff < 0.5,
            "tau={tau}: GPU vs CPU 参考 max|diff| = {max_diff} ≥ 0.5"
        );
        eprintln!("tau={tau}: max|diff| = {max_diff:.4}");
    }
}

#[test]
fn tau_512_approximates_hard_raster() {
    let shapes = scene();
    let target = target_solid();
    let mut r = match DiffRenderer::new(W, H, shapes.len()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("跳过（无可用 GPU）：{e}");
            return;
        }
    };
    let gpu = r.render(&shapes, 512.0, BG, &target).expect("render");
    // 硬光栅 CPU 参考（q<1 像素中心）
    let mut hard = vec![0f32; (W * H * 3) as usize];
    for y in 0..H {
        for x in 0..W {
            let px = f64::from(x) + 0.5;
            let py = f64::from(y) + 0.5;
            let mut col = [f64::from(BG[0]), f64::from(BG[1]), f64::from(BG[2])];
            for s in &shapes {
                let (sin, cos) = f64::from(s.theta).sin_cos();
                let dx = px - f64::from(s.cx);
                let dy = py - f64::from(s.cy);
                let xp = dx * cos - dy * sin;
                let yp = dx * sin + dy * cos;
                let q = xp * xp / f64::from(s.a * s.a) + yp * yp / f64::from(s.b * s.b);
                if q < 1.0 {
                    col = [f64::from(s.r), f64::from(s.g), f64::from(s.bl)];
                }
            }
            let i = ((y * W + x) * 3) as usize;
            hard[i..i + 3].copy_from_slice(&[col[0] as f32, col[1] as f32, col[2] as f32]);
        }
    }
    let n_px = (W * H) as usize;
    let mut over = 0usize;
    let mut max_band_diff = 0f32;
    for p in 0..n_px {
        let mut worst = 0f32;
        for ch in 0..3 {
            worst = worst.max((gpu[3 * p + ch] - hard[3 * p + ch]).abs());
        }
        if worst > 1.5 {
            over += 1;
        }
        max_band_diff = max_band_diff.max(worst);
    }
    // 边带像素（|q−1|·τ 小处 sigmoid 与硬覆盖分歧）允许 ≤ 2%
    assert!(
        (over as f64) / n_px as f64 <= 0.02,
        "τ=512 边带超差点 {over}/{n_px} > 2%（max_diff={max_band_diff}）"
    );
    eprintln!("τ=512: 超差点 {over}/{n_px}（max|diff|={max_band_diff:.1}，边带半覆盖正常）");
}
