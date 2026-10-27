//! C0 特征提取：图像 → 墨迹统计（与 orient.rs 同式的 chamfer/Sobel，自包含）。
//! `imgstats <img.png> [RRGGBB]` → 一行 CSV：
//! name,fg_frac,w25,w50,w95,edge_density,orient_var,long_edge
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: imgstats <img.png> [RRGGBB]");
        std::process::exit(2);
    }
    let img = image::open(&args[1]).expect("open").to_rgba8();
    let (w, h) = img.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let n = wu * hu;
    let name = std::path::Path::new(&args[1])
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    // bg：不透明均值（或指定）——仅用于参考；实际判背景用灰度中位数
    // 灰度中位数当背景（均值会被 >50% 墨量图污染——stripes 全图误判 fg 的教训）
    let mut gray = vec![0f32; n];
    for (p, px) in gray.iter_mut().zip(img.pixels()) {
        *p = (px.0[0] as f32 + px.0[1] as f32 + px.0[2] as f32) / 3.0;
    }
    let mut sorted_gray = gray.clone();
    sorted_gray.sort_by(|a, b| a.total_cmp(b));
    let bg_gray = sorted_gray[n / 2];
    let mut fg = vec![false; n];
    for p in 0..n {
        fg[p] = (gray[p] - bg_gray).abs() > 60.0;
    }
    let fg_count = fg.iter().filter(|&&f| f).count();
    let fg_frac = fg_count as f32 / n as f32;

    // chamfer 距离（1/√2 两遍）→ 宽度分位
    let inf = f32::INFINITY;
    let mut dist = vec![inf; n];
    for p in 0..n {
        if fg[p] {
            dist[p] = 0.0;
        }
    }
    let s2 = 2f32.sqrt();
    let nb = |x: i64, y: i64, d: &[f32]| -> f32 {
        if x < 0 || y < 0 || x >= wu as i64 || y >= hu as i64 {
            f32::INFINITY
        } else {
            d[y as usize * wu + x as usize]
        }
    };
    for y in 0..hu as i64 {
        for x in 0..wu as i64 {
            let p = (y as usize) * wu + x as usize;
            let d = dist[p];
            dist[p] = d
                .min(nb(x - 1, y, &dist) + 1.0)
                .min(nb(x - 1, y - 1, &dist) + s2)
                .min(nb(x, y - 1, &dist) + 1.0)
                .min(nb(x + 1, y - 1, &dist) + s2);
        }
    }
    for y in (0..hu as i64).rev() {
        for x in (0..wu as i64).rev() {
            let p = (y as usize) * wu + x as usize;
            let d = dist[p];
            dist[p] = d
                .min(nb(x + 1, y, &dist) + 1.0)
                .min(nb(x + 1, y + 1, &dist) + s2)
                .min(nb(x, y + 1, &dist) + 1.0)
                .min(nb(x - 1, y + 1, &dist) + s2);
        }
    }
    let mut widths: Vec<f32> = (0..n)
        .filter(|&p| fg[p] && dist[p].is_finite())
        .map(|p| dist[p] * 2.0 + 1.0) // 像素含身：1px 笔画宽 1 而非 0
        .collect();
    widths.sort_by(|a, b| a.total_cmp(b));
    let q = |f: f64| -> f32 {
        if widths.is_empty() {
            0.0
        } else {
            widths[((widths.len() as f64 * f) as usize).min(widths.len() - 1)]
        }
    };
    let (w25, w50, w95) = (q(0.25), q(0.50), q(0.95));

    // 边界密度：fg 与 bg 4 邻接的对数 / fg 数
    let mut bound = 0usize;
    for y in 0..hu {
        for x in 0..wu {
            let p = y * wu + x;
            if !fg[p] {
                continue;
            }
            for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let xx = (x as i64 + dx).clamp(0, wu as i64 - 1) as usize;
                let yy = (y as i64 + dy).clamp(0, hu as i64 - 1) as usize;
                if !fg[yy * wu + xx] {
                    bound += 1;
                    break;
                }
            }
        }
    }
    let edge_density = if fg_count > 0 {
        bound as f32 / fg_count as f32
    } else {
        0.0
    };

    // 走向方差：fg 边界像素的 Sobel 角度（ doubled to [0,π) 的单位向量均值模长 → 1−mod = 各向异性）
    let g = |x: i64, y: i64| gray[(y.clamp(0, hu as i64 - 1) as usize) * wu + (x.clamp(0, wu as i64 - 1) as usize)];
    let (mut sx, mut sy, mut cnt) = (0f32, 0f32, 0usize);
    for y in 0..hu as i64 {
        for x in 0..wu as i64 {
            let p = (y as usize) * wu + x as usize;
            if !fg[p] {
                continue;
            }
            let gx = -g(x - 1, y - 1) - 2.0 * g(x - 1, y) - g(x - 1, y + 1)
                + g(x + 1, y - 1) + 2.0 * g(x + 1, y) + g(x + 1, y + 1);
            let gy = -g(x - 1, y - 1) - 2.0 * g(x, y - 1) - g(x + 1, y - 1)
                + g(x - 1, y + 1) + 2.0 * g(x, y + 1) + g(x + 1, y + 1);
            if gx * gx + gy * gy < 1e-6 {
                continue;
            }
            let th = gy.atan2(gx) + std::f32::consts::FRAC_PI_2; // 墨迹走向
            let (t2x, t2y) = ((2.0 * th).cos(), (2.0 * th).sin()); // 轴对称消歧
            sx += t2x;
            sy += t2y;
            cnt += 1;
        }
    }
    let orient_var = if cnt > 0 {
        let m = (sx * sx + sy * sy).sqrt() / cnt as f32;
        1.0 - m // 0=单一定向（条纹），1=完全乱向
    } else {
        1.0
    };

    let long_edge = w.max(h) as f32;
    println!(
        "{name},{fg_frac:.4},{w25:.2},{w50:.2},{w95:.2},{edge_density:.4},{orient_var:.4},{long_edge:.0}"
    );
}
