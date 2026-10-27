//! 视觉验收：src vs 渲染预览 → [预览 | ×8 放大差异热图] 竖排 PNG。
//! `visdiff <src.png> <preview.png> <out.png>`


fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src = image::load_from_memory(&std::fs::read(&a[1]).expect("read")).unwrap().to_rgba8();
    let prev = image::load_from_memory(&std::fs::read(&a[2]).expect("read")).unwrap().to_rgb8();
    let (w, h) = src.dimensions();
    let mut out = image::RgbImage::new(w, h * 2);
    let mut max_d = 0u32;
    for y in 0..h {
        for x in 0..w {
            let s = src.get_pixel(x, y);
            let c = prev.get_pixel(x, y);
            out.put_pixel(x, y, *c);
            // 差异热图（×8 裁剪，蓝-绿-红谱）
            let d: u32 = (0..3)
                .map(|ch| (s.0[ch] as i32 - c.0[ch] as i32).unsigned_abs() as u32)
                .max()
                .unwrap_or(0);
            max_d = max_d.max(d);
            let v = (d * 8).min(255) as u8;
            let heat = image::Rgb([v, (v / 2).min(180), 255 - v]);
            out.put_pixel(x, y + h, heat);
        }
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(out)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    std::fs::write(&a[3], png).unwrap();
    println!("visdiff: {}x{} max|d|={max_d} -> {}", w, h, a[3]);
}
