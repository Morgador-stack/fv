//! B2 误差边带相原型（M19_PROPOSALS §B2）：实心栈跑完后，取误差场等值线
//! 沿链放描边椭圆（M1.7 引擎），攻击"76-94% 误差集中在墨迹边界 ±2px 带"。
//! `b2_band <flat.png> <design.json> <budget> <out_prefix>` → 前后 PSNR + JSON/预览。
use fv_common::json;
use fv_common::shape::Shape;
use std::path::PathBuf;



fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: b2_band <flat.png> <design.json> <budget> <out_prefix>");
        std::process::exit(2);
    }
    let img = image::load_from_memory(&std::fs::read(&args[1]).expect("read img")).unwrap().to_rgba8();
    let (w, h) = img.dimensions();
    let n = (w * h) as usize;
    let budget: usize = args[3].parse().unwrap();
    let out = PathBuf::from(&args[4]);

    // bg = 灰度中位数（imgstats 教训：均值被高墨量图污染）
    let mut grays: Vec<u8> = img
        .pixels()
        .map(|p| p.0[0] / 3 + p.0[1] / 3 + p.0[2] / 3)
        .collect();
    grays.sort_unstable();
    let bg_gray = grays[n / 2];
    let bg = [bg_gray; 3];

    // target 平面 SoA（不透明像素；flatten 产物全不透明）
    let mut target = vec![0u8; n * 3];
    for (p, px) in target.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let q = img.get_pixel(p as u32 % w, p as u32 / w);
        let c = if q.0[3] < 128 { bg } else { [q.0[0], q.0[1], q.0[2]] };
        *px = c;
    }

    let mut shapes: Vec<Shape> = json::read_design_bytes(&std::fs::read(&args[2]).expect("read json")).unwrap()
        .shapes.iter().map(|s| Shape::ellipse(
            s.data[0], s.data[1], s.data[2], s.data[3],
            if s.data.len() >= 5 { s.data[4] } else { 0.0 },
            [s.color[0], s.color[1], s.color[2]],
        )).collect();

    let psnr = |shapes: &[Shape]| -> f64 {
        let canvas = fv_fit::render::render_shapes(w, h, shapes, bg);
        let mut se = 0i64;
        for (t, c) in target.as_chunks::<3>().0.iter().zip(canvas.pixels()) {
            for (tv, cv) in t.iter().zip(c.0.iter()) {
                let d = i64::from(*tv) - i64::from(*cv);
                se += d * d;
            }
        }
        let mse = se as f64 / (3.0 * n as f64);
        10.0 * (255.0 * 255.0 / mse).log10()
    };
    let before = psnr(&shapes);
    let base_len = shapes.len();

    // 残差场 = max 通道 |target − canvas|（描边该找"错得最狠"的等值线）
    let canvas = fv_fit::render::render_shapes(w, h, &shapes, bg);
    let mut field = vec![0u8; n];
    for p in 0..n {
        let cp = canvas.get_pixel(p as u32 % w, p as u32 / w);
        let d0 = (target[p] as i32 - cp.0[0] as i32).abs();
        let d1 = (target[n + p] as i32 - cp.0[1] as i32).abs();
        let d2 = (target[2 * n + p] as i32 - cp.0[2] as i32).abs();
        field[p] = d0.max(d1).max(d2).min(255) as u8;
    }

    let strokes = fv_fit::stroke::contour_strokes(
        &field, &target, w, h, 48, budget, 1.4, 12.0, 2.0,
    );
    for (e, c) in strokes.shapes {
        shapes.push(Shape::ellipse(e.cx, e.cy, e.rx, e.ry, e.angle_rad.to_degrees(), c));
    }
    let after = psnr(&shapes);

    json::write_design(&out.with_extension("b2.json"), &shapes).unwrap();
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(fv_fit::render::render_shapes(w, h, &shapes, bg))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    std::fs::write(out.with_extension("b2.preview.png"), png).unwrap();
    println!(
        "b2_band: {base_len} + {} strokes | iso=48 thick=1.4 | PSNR {before:.2} -> {after:.2} dB",
        shapes.len() - base_len
    );
}
