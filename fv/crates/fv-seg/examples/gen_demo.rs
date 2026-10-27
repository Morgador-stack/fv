//! 合成演示图：左半渐变背景 + 右半"主体"（彩色圆环+噪声纹理），配套 matte。

use image::RgbaImage;

fn main() {
    let (w, h) = (256u32, 256u32);
    let mut img = RgbaImage::new(w, h);
    let mut matte = image::GrayImage::new(w, h);
    let mut noise = 0x5EEDu32;
    let mut nz = || {
        noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
        ((noise >> 28) as i32) - 7
    };
    for y in 0..h {
        for x in 0..w {
            // 背景：左半水平渐变
            let t = x as f32 / w as f32;
            let (mut r, mut g, mut b) = ((t * 200.0) as i32, 60, ((1.0 - t) * 180.0) as i32);
            let dx = x as f32 - 176.0;
            let dy = y as f32 - 128.0;
            let d = (dx * dx + dy * dy).sqrt();
            // 主体：双环 + 中心盘，带纹理噪声
            if d < 56.0 {
                let ring = (d - 44.0).abs() < 6.0;
                let ring2 = (d - 28.0).abs() < 4.0;
                let core = d < 16.0;
                if ring || ring2 || core {
                    let hue = (d * 6.0 + (y as f32) * 0.5) % 360.0;
                    let (hr, hg, hb) = hsv_to_rgb(hue, 0.75, 0.95);
                    r = hr as i32;
                    g = hg as i32;
                    b = hb as i32;
                    // 主体加纹理
                    let n = nz() * 3;
                    r = (r + n).clamp(0, 255);
                    g = (g + n).clamp(0, 255);
                    b = (b + n).clamp(0, 255);
                }
            }
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    (r + nz()).clamp(0, 255) as u8,
                    (g + nz()).clamp(0, 255) as u8,
                    (b + nz()).clamp(0, 255) as u8,
                    255,
                ]),
            );
            // matte：主体区域（双环+盘）= 255，其余 0；边缘羽化 2px
            let inside = d < 52.0;
            let edge = ((d - 50.0).abs() < 2.5) as u8 * 128;
            let v = if inside { 255 } else { edge };
            matte.put_pixel(x, y, image::Luma([v]));
        }
    }
    let out = std::env::temp_dir();
    img.save(out.join("fv_demo.png")).unwrap();
    matte.save(out.join("fv_demo_matte.png")).unwrap();
    println!("saved {} / fv_demo.png + fv_demo_matte.png", out.display());
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 / 60 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0)
}
