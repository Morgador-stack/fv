//! 零 GPU 判别实验：软渲染（逐层尺度 τ）vs 硬光栅画布的偏差随层数的变化。
//! 假说 H_density：偏差随 n 增长到 dB 级 → 千层 refine 失配=软硬目标漂移；
//! 反证（300 与 1000 偏差同量级）→ 嫌疑转向窗口化实现 bug。
//! `soft_hard_gap <design.json> [tau0]` —— 纯 CPU（f64 参考 + 行 span 光栅）。
use fv_common::json;
use fv_gpu::diff::fwd::DiffRenderer;
use fv_gpu::diff::ShapeParams;



fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tau: f32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(32.0);
    let design = json::read_design_bytes(&std::fs::read(&args[1]).expect("read json")).unwrap();
    let shapes: Vec<ShapeParams> = design
        .shapes
        .iter()
        .map(|s| ShapeParams {
            cx: s.data[0],
            cy: s.data[1],
            a: s.data[2],
            b: s.data[3],
            theta: -s.data[4].to_radians(), // JSON 引擎角 → diff 角（STUDY 勘误约定）
            r: s.color[0] as f32,
            g: s.color[1] as f32,
            bl: s.color[2] as f32,
        })
        .collect();
    let n = shapes.len();
    let (w, h) = (328u32, 328u32);
    let plane = (w * h) as usize;

    // 硬光栅（引擎行 span，与裁判一致）
    let mut hard = vec![0u8; plane * 3];
    for ch in 0..3 {
        hard[ch * plane..(ch + 1) * plane].fill(32); // bg auto ≈ 深灰（近似，两画布同底可比）
    }
    for s in &design.shapes {
        let e = fv_fit::raster::EllipseParams {
            cx: s.data[0],
            cy: s.data[1],
            rx: s.data[2],
            ry: s.data[3],
            angle_rad: s.data[4].to_radians(),
        };
        fv_fit::raster::composite_ellipse(&mut hard, w, h, &e, [s.color[0], s.color[1], s.color[2]]);
    }

    // 软渲染（f64 CPU 参考，含逐层 τ）——bg 也用 32 保持同底
    let soft = DiffRenderer::cpu_forward_reference(&shapes, tau, [32.0; 3], w, h);

    let mut se = 0f64;
    for p in 0..plane {
        for ch in 0..3 {
            let d = soft[3 * p + ch] - f64::from(hard[ch * plane + p]);
            se += d * d;
        }
    }
    let mse = se / (3.0 * plane as f64);
    let psnr = 10.0 * (255.0 * 255.0 / mse).log10();
    println!("n={n} tau0={tau}: soft-vs-hard PSNR = {psnr:.2} dB (mse={mse:.1})");
}
