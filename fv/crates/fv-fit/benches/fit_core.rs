//! M1 核心热点基准：光栅化 / 前缀和刷新 / 单候选评估 / 小规模全流程。

use criterion::{Criterion, criterion_group, criterion_main};
use fv_fit::engine::{self, FitConfig};
use fv_fit::raster::{self, EllipseParams};
use fv_fit::rowsum::{MapId, RowSums};
use std::hint::black_box;

const W: u32 = 448;
const H: u32 = 448;

fn synthetic_planes(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w * h) as usize;
    let mut target = vec![0u8; n * 3];
    let canvas = vec![40u8; n * 3];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = y * w as usize + x;
            target[p] = (x * 255 / w as usize) as u8;
            target[n + p] = (y * 255 / h as usize) as u8;
            target[2 * n + p] = ((x ^ y) & 0xff) as u8;
        }
    }
    (target, canvas)
}

fn synthetic_image(w: u32, h: u32) -> image::RgbaImage {
    let (t, _) = synthetic_planes(w, h);
    let n = (w * h) as usize;
    let mut img = image::RgbaImage::new(w, h);
    for (p, px) in img.pixels_mut().enumerate() {
        *px = image::Rgba([t[p], t[n + p], t[2 * n + p], 255]);
    }
    img
}

fn bench_m1(c: &mut Criterion) {
    let mut group = c.benchmark_group("m1");

    // 1. 光栅化：中等椭圆 @448
    let e = EllipseParams {
        cx: 224.0,
        cy: 224.0,
        rx: 60.0,
        ry: 30.0,
        angle_rad: 0.7,
    };
    group.bench_function("ellipse_rows_448", |b| {
        b.iter(|| raster::ellipse_rows(black_box(&e), W, H))
    });

    // 2. 前缀和刷新（每层接纳后一次）
    let (target, canvas) = synthetic_planes(W, H);
    let mut sums = RowSums::new(W, H);
    sums.set_target(&target);
    group.bench_function("rowsum_refresh_448", |b| {
        b.iter(|| sums.refresh_canvas(black_box(&canvas)))
    });

    // 3. 单候选评估（O(高度) 核心）
    group.bench_function("evaluate_448", |b| {
        b.iter(|| engine::evaluate(black_box(&sums), black_box(&e), W, H))
    });

    // 4. 段查询微基准
    group.bench_function("seg_query", |b| {
        b.iter(|| black_box(sums.seg(MapId::T2, 2, 200, 100, 300)))
    });

    // 5. 小规模全流程（256×192，10 层 × 4000 采样）
    let img = synthetic_image(256, 192);
    group.bench_function("fit_10x4000_256", |b| {
        b.iter(|| {
            let cfg = FitConfig {
                layers: 10,
                samples_per_shape: 4000,
                opt_max_dim: 256,
                seed: 1,
                refine_iters: 12,
                checkpoints: vec![],
                out_prefix: None,
                bg: [0, 0, 0],
                polish_sweeps: 0,
                soft_layers: 0,
                soft_refit: false,
                auto_soft: false,
                structured: true,
            };
            let mut p = |_, _| {};
            engine::fit(black_box(&img), &cfg, &mut p)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_m1);
criterion_main!(benches);
