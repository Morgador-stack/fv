//! RGBA → 实底压平（剪影模板资产 → 可拟合的实心目标图）。
//! `flatten.exe in.png out.png [RRGGBB|auto]`（默认 auto：字形亮度均值 ≥128
//! → 黑底，否则白底——剪影常见白字/白形，直接压白底会退化为全白 mse=0）。


fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: flatten <in.png> <out.png> [RRGGBB|auto]");
        std::process::exit(2);
    }
    let img = image::load_from_memory(&std::fs::read(&args[1]).expect("read input")).expect("decode").to_rgba8();
    let bg: [u8; 3] = match args.get(3).map(|s| s.as_str()) {
        Some("auto") | None => {
            let (mut sum, mut cnt) = (0u64, 0u64);
            for px in img.pixels() {
                if px.0[3] >= 128 {
                    sum += u64::from(px.0[0]) + u64::from(px.0[1]) + u64::from(px.0[2]);
                    cnt += 1;
                }
            }
            if cnt == 0 {
                [255, 255, 255]
            } else if sum / (3 * cnt) >= 128 {
                [0, 0, 0]
            } else {
                [255, 255, 255]
            }
        }
        Some(hex) => {
            let v = u32::from_str_radix(hex.trim_start_matches('#'), 16).expect("RRGGBB");
            [(v >> 16) as u8, ((v >> 8) & 0xFF) as u8, (v & 0xFF) as u8]
        }
    };
    let (w, h) = img.dimensions();
    let mut out = image::RgbImage::new(w, h);
    for (p, px) in out.pixels_mut().zip(img.pixels()) {
        let a = f64::from(px.0[3]) / 255.0;
        let inv = 1.0 - a;
        let bg_c = bg.map(f64::from);
        let fg_c = [f64::from(px.0[0]), f64::from(px.0[1]), f64::from(px.0[2])];
        for ch in 0..3 {
            p.0[ch] = (bg_c[ch] * inv + fg_c[ch] * a).round().clamp(0.0, 255.0) as u8;
        }
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(out)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode");
    std::fs::write(&args[2], png).expect("save output");
    println!("flattened {} -> {} on #{:02X}{:02X}{:02X}", args[1], args[2], bg[0], bg[1], bg[2]);
}
