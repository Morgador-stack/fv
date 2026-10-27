//! 客观评估任意 geometrize design JSON（含 228 软椭圆 S 曲线语义）：
//! `cargo run --release --example eval_design -- <source.png> <design.json> [--bg auto|RRGGBB] [--save out.png]`
//!
//! 228 语义（源自 vinylizer alpha_228.h）：像素归一化半径 rn ∈ [0,1)，
//! coverage = a 满足 1 − C1·a − C2·a² − C3·a³ = rn（牛顿解，LUT 4096），
//! 有效 alpha = coverage × opacity；canvas = canvas·(1−alpha) + color·alpha。

use fv_common::json::read_design;

// alpha_228 统一走 fv_fit::soft 的唯一实现（f64 牛顿解）——旧 LUT+本地 f32
// 牛顿与引擎各一份实现，存在细微口径漂移风险（CODE_REVIEW §E 常量三拷贝）。
fn alpha_228_newton(rn: f32) -> f32 {
    fv_fit::soft::alpha_228(f64::from(rn)) as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (src_path, json_path) = (&args[1], &args[2]);
    let bg_spec = args
        .iter()
        .position(|a| a == "--bg")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "auto".into());
    let save = args.iter().position(|a| a == "--save").map(|i| args[i + 1].clone());

    let src = image::open(src_path).unwrap().to_rgba8();
    let (w, h) = src.dimensions();
    let n = (w * h) as usize;

    // 背景填充（与引擎/psnr 同规则）
    let (mut sum, mut cnt) = ([0u64; 3], 0u64);
    for px in src.pixels() {
        if px.0[3] >= 128 {
            for (s, v) in sum.iter_mut().zip(px.0.iter().take(3)) {
                *s += *v as u64;
            }
            cnt += 1;
        }
    }
    let bg: [f32; 3] = if bg_spec.eq_ignore_ascii_case("auto") {
        if cnt > 0 {
            core::array::from_fn(|ch| ((sum[ch] + cnt / 2) / cnt).clamp(0, 255) as f32)
        } else {
            [0.0; 3]
        }
    } else {
        let v = u32::from_str_radix(bg_spec.trim_start_matches('#'), 16).unwrap();
        [(v >> 16) as f32, ((v >> 8) & 0xFF) as f32, (v & 0xFF) as f32]
    };

    let target: Vec<f32> = {
        let mut t = vec![0f32; n * 3];
        for p in 0..n {
            let px = src.get_pixel(p as u32 % w, p as u32 / w);
            let c = if px.0[3] >= 128 {
                [px.0[0] as f32, px.0[1] as f32, px.0[2] as f32]
            } else {
                bg
            };
            for ch in 0..3 {
                t[ch * n + p] = c[ch];
            }
        }
        t
    };

    // 渐进合成（文件顺序 = z 序自底向上）
    let design = read_design(std::path::Path::new(json_path)).unwrap();
    let mut canvas = vec![0f32; n * 3];
    for ch in 0..3 {
        for p in 0..n {
            canvas[ch * n + p] = bg[ch];
        }
    }
    let mut type_count = [0usize; 3];
    for s in &design.shapes {
        let opacity = s.color[3] as f32 / 255.0;
        let color = [s.color[0] as f32, s.color[1] as f32, s.color[2] as f32];
        let (cx, cy) = (s.data[0] as f64, s.data[1] as f64);
        let (rx, ry) = (s.data[2] as f64, s.data[3] as f64);
        let ang = if s.data.len() >= 4 { s.data[4] as f64 } else { 0.0 };
        let kind = match s.kind {
            1 => {
                type_count[0] += 1;
                1
            }
            16 => {
                type_count[1] += 1;
                16
            }
            _ => {
                type_count[2] += 1;
                228
            }
        };
        // bbox
        let (c, sn) = (ang.to_radians().cos(), ang.to_radians().sin());
        // 旋转椭圆精确外接半宽：sqrt(rx²c²+ry²s²)（v1 用 max(|rxc|,|rys|) 会砍掉
        // 旋转层的四角，系统性低估旋转椭圆密集栈的质量）
        let (xext, yext) = (
            (rx * rx * c * c + ry * ry * sn * sn).sqrt(),
            (rx * rx * sn * sn + ry * ry * c * c).sqrt(),
        );
        let x0 = (cx - xext - 1.0).floor().max(0.0) as i64;
        let x1 = (cx + xext + 1.0).ceil().min(w as f64 - 1.0) as i64;
        let y0 = (cy - yext - 1.0).floor().max(0.0) as i64;
        let y1 = (cy + yext + 1.0).ceil().min(h as f64 - 1.0) as i64;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f64 + 0.5 - cx;
                let dy = y as f64 + 0.5 - cy;
                let u = c * dx + sn * dy;
                let v = -sn * dx + c * dy;
                let alpha = match kind {
                    1 => {
                        if u.abs() <= rx && v.abs() <= ry {
                            opacity.max(0.0)
                        } else {
                            0.0
                        }
                    }
                    16 => {
                        let r2 = u * u / (rx * rx) + v * v / (ry * ry);
                        if r2 <= 1.0 {
                            opacity.max(0.0)
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        let r2 = u * u / (rx * rx) + v * v / (ry * ry);
                        if r2 < 1.0 {
                            let rn = r2.sqrt() as f32;
                            alpha_228_newton(rn) * opacity
                        } else {
                            0.0
                        }
                    }
                };
                if alpha <= 0.0 {
                    continue;
                }
                let p = (y as usize) * w as usize + x as usize;
                for ch in 0..3 {
                    let old = canvas[ch * n + p];
                    canvas[ch * n + p] = old * (1.0 - alpha) + color[ch] * alpha;
                }
            }
        }
    }

    // PSNR / MAE
    let (mut se, mut ae) = (0f64, 0f64);
    for i in 0..n * 3 {
        let d = canvas[i] as f64 - target[i] as f64;
        se += d * d;
        ae += d.abs();
    }
    let mse = se / (n * 3) as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    println!(
        "shapes={} (rect {}/ellipse {}/soft {})  dims={w}x{h}  MSE={mse:.2}  PSNR={psnr:.2} dB  MAE={:.2}",
        design.shapes.len(), type_count[0], type_count[1], type_count[2],
        ae / (n * 3) as f64
    );

    if let Some(out) = save {
        let mut img = image::RgbaImage::new(w, h);
        for p in 0..n {
            let x = (p as u32) % w;
            let y = (p as u32) / w;
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    canvas[p].round().clamp(0.0, 255.0) as u8,
                    canvas[n + p].round().clamp(0.0, 255.0) as u8,
                    canvas[2 * n + p].round().clamp(0.0, 255.0) as u8,
                    255,
                ]),
            );
        }
        img.save(&out).unwrap();
        println!("render saved: {out}");
    }
}
