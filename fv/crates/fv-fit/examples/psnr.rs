//! 对比源图与拟合预览图的 PSNR/MAE：`cargo run --release --example psnr -- <source> <preview>`

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (src_path, prev_path) = (&args[1], &args[2]);
    let src = image::open(src_path).unwrap().to_rgba8();
    let prev = image::open(prev_path).unwrap().to_rgba8();
    let (w, h) = prev.dimensions();
    let src = image::imageops::resize(&src, w, h, image::imageops::FilterType::Triangle);

    // 与引擎一致：透明像素按不透明均值 bg 填充后再比较
    let (mut sum, mut cnt) = ([0u64; 3], 0u64);
    for px in src.pixels() {
        if px.0[3] >= 128 {
            for (s, v) in sum.iter_mut().zip(px.0.iter().take(3)) {
                *s += *v as u64;
            }
            cnt += 1;
        }
    }
    let bg: [u8; 3] = if cnt > 0 {
        core::array::from_fn(|ch| ((sum[ch] + cnt / 2) / cnt).clamp(0, 255) as u8)
    } else {
        [0, 0, 0]
    };
    let src_flat = image::RgbaImage::from_fn(w, h, |x, y| {
        let p = src.get_pixel(x, y);
        if p.0[3] >= 128 {
            *p
        } else {
            image::Rgba([bg[0], bg[1], bg[2], 255])
        }
    });

    let (mut se, mut ae, mut n) = (0u64, 0u64, 0u64);
    for (a, b) in src_flat.pixels().zip(prev.pixels()) {
        for c in 0..3 {
            let d = a.0[c] as i64 - b.0[c] as i64;
            se += (d * d) as u64;
            ae += d.unsigned_abs();
        }
        n += 3;
    }
    let mse = se as f64 / n as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    println!(
        "dims={w}x{h}  MSE={mse:.2}  PSNR={psnr:.2} dB  MAE={:.2}",
        ae as f64 / n as f64
    );
}
