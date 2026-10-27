//! 复刻引擎调用序列的软通道探针：default() → on_ready → 贪心式 on_accept → eval_soft_batch。

use fv_fit::candidate::{self, CandidateSpec, Rng};
use fv_fit::engine::BatchEvaluator;
use fv_fit::rowsum::RowSums;
use fv_gpu::GpuEvaluator;

const W: u32 = 448;
const H: u32 = 448;

#[test]
fn engine_sequence_soft_probe() {
    let n = (W * H) as usize;
    let mut seed = 0xBEEFu64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u8
    };
    let target: Vec<u8> = (0..n * 3).map(|_| next()).collect();
    let mut canvas: Vec<u8> = (0..n * 3).map(|_| next()).collect();
    let mut sums = RowSums::new(W, H);
    sums.set_target(&target);
    sums.refresh_canvas(&canvas);

    let Ok(mut gpu) = GpuEvaluator::new(W, H, &sums) else {
        eprintln!("跳过：无 GPU");
        return;
    };
    // 引擎序列：default/new 之后 on_ready（真实平面）
    gpu.on_ready(&sums, &target, &canvas, W, H);

    // 模拟 3 次"贪心接纳"：改画布 → refresh → on_accept
    for i in 0..3 {
        for v in canvas.iter_mut() {
            *v = v.wrapping_add(i * 7 + 3);
        }
        sums.refresh_canvas(&canvas);
        gpu.on_accept(&sums, &target, &canvas, W, H);
    }

    let mut rng = Rng::new(0x50F7);
    let spec = CandidateSpec {
        min_radius: 2.0,
        max_radius: 120.0,
    };
    let cands: Vec<_> = (0..20000)
        .map(|_| candidate::sample(&mut rng, W as f32, H as f32, &spec))
        .collect();
    let results = gpu.eval_soft_batch(&cands, W, H);
    let mut ok = 0usize;
    let mut k_match = 0usize;
    for (e, g) in cands.iter().zip(&results) {
        if let Some((gk, gd)) = g {
            ok += 1;
            if let Some((ck, cd)) = fv_fit::soft::eval_soft(&target, &canvas, W, H, e) {
                let diff = ck.iter().zip(gk.iter()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                if diff <= 1 {
                    k_match += 1;
                }
                let denom = cd.abs().max(1);
                let rel = (gd - cd).abs() as f64 / denom as f64;
                assert!(rel < 0.05, "delta 偏差 gpu={gd} cpu={cd}");
            } else {
                panic!("GPU 有值但 CPU 无：gpu={g:?} e={e:?}");
            }
        }
    }
    eprintln!("序列探针：gpu 有值 {ok}/200，k±1 一致 {k_match}/{ok}");
    assert!(ok > 50);
}
