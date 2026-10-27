//! 零 GPU 判别实验 #2：n 层 × 同幅度参数抖动 → 硬光栅画布损伤是否 ∝ n。
//! 若 0.2px 级抖动在 n=2000 就打出 ~5dB 损伤 → "Adam 噪声放大"假说足以解释
//! 千层 refine 失配（无需窗口 bug）；若损伤很小 → 窗口化实现是头号嫌疑。
//! `jitter_bench <design.json> [amp=0.2] [trials=5]` —— 纯 CPU。
use fv_common::json;



fn main() {
    let args: Vec<String> = std::env::args().collect();
    let amp: f32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0.2);
    let trials: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    let design = json::read_design_bytes(&std::fs::read(&args[1]).expect("read json")).unwrap();
    let to_shape = |s: &fv_common::json::ShapeJson| fv_common::shape::Shape::ellipse(
        s.data[0], s.data[1], s.data[2], s.data[3],
        if s.data.len() >= 5 { s.data[4] } else { 0.0 },
        [s.color[0], s.color[1], s.color[2]],
    );
    let (w, h) = (328u32, 328u32);
    let bg = [32u8; 3];

    let render = |shapes: &[fv_common::shape::Shape]| -> Vec<u8> {
        let mut cv = vec![0u8; (w * h) as usize * 3];
        for ch in 0..3 {
            cv[ch * (w * h) as usize..(ch + 1) * (w * h) as usize].fill(bg[ch]);
        }
        for s in shapes {
            let e = fv_fit::raster::EllipseParams {
                cx: s.cx, cy: s.cy, rx: s.rx, ry: s.ry,
                angle_rad: s.angle_deg.to_radians(),
            };
            fv_fit::raster::composite_ellipse(&mut cv, w, h, &e, [s.color[0], s.color[1], s.color[2]]);
        }
        cv
    };
    let psnr = |a: &[u8], b: &[u8]| -> f64 {
        let mut se = 0i64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = i64::from(*x) - i64::from(*y);
            se += d * d;
        }
        10.0 * (255.0 * 255.0 / (se as f64 / a.len() as f64)).log10()
    };

    let full: Vec<fv_common::shape::Shape> = design.shapes.iter().map(&to_shape).collect();
    let base_full = render(&full);
    for &n in &[300usize, 1000, 2000, full.len().min(3000)] {
        if n > full.len() { continue; }
        let subset = &full[..n];
        let base = render(subset);
        let mut worst = f64::INFINITY;
        let mut sum = 0f64;
        for t in 0..trials {
            // xoshiro 简版：确定性抖动
            let mut s = (0x9E3779B97F4A7C15u64 ^ (t as u64) ^ (n as u64).wrapping_mul(0x100000001B3)).wrapping_add(0x1234);
            let mut jit = subset.to_vec();
            for sh in &mut jit {
                let next = |s: &mut u64| {
                    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((*s >> 33) as f32 / 2147483648.0) - 1.0
                };
                sh.cx += next(&mut s) * amp;
                sh.cy += next(&mut s) * amp;
                sh.rx = (sh.rx + next(&mut s) * amp).max(1.0);
                sh.ry = (sh.ry + next(&mut s) * amp).max(1.0);
                sh.angle_deg += next(&mut s) * amp * 2.87; // 0.2px 档配 ~0.01rad
            }
            let p = psnr(&base, &render(&jit));
            worst = worst.min(p);
            sum += p;
        }
        let _ = &base_full;
        println!("n={n:<5} amp={amp}px: jitter-vs-base PSNR mean={:.2} worst={:.2} dB", sum / trials as f64, worst);
    }
}
