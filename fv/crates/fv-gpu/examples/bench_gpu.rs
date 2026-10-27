//! GPU vs CPU 候选评估吞吐基准：448² 随机图，100k 候选。

use fv_fit::candidate::{self, CandidateSpec, Rng};
use fv_fit::engine;
use fv_fit::rowsum::RowSums;
use fv_gpu::GpuEvaluator;
use std::time::Instant;

const W: u32 = 448;
const H: u32 = 448;
const N: usize = 100_000;

fn main() {
    let n = (W * H) as usize;
    let mut seed = 0xBEEFu64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u8
    };
    let target: Vec<u8> = (0..n * 3).map(|_| next()).collect();
    let canvas: Vec<u8> = (0..n * 3).map(|_| next()).collect();
    let mut sums = RowSums::new(W, H);
    sums.set_target(&target);
    sums.refresh_canvas(&canvas);

    let mut rng = Rng::new(42);
    let spec = CandidateSpec {
        min_radius: 2.0,
        max_radius: 160.0,
    };
    let cands: Vec<_> = (0..N)
        .map(|_| candidate::sample(&mut rng, W as f32, H as f32, &spec))
        .collect();

    // CPU 串行基准
    let t0 = Instant::now();
    let mut cpu_best = i64::MIN;
    for e in &cands {
        if let Some((_, d)) = engine::evaluate(&sums, e, W, H) {
            cpu_best = cpu_best.max(d);
        }
    }
    let cpu_dt = t0.elapsed();

    // GPU 基准（两轮：首轮含初始化，次轮纯评估）
    let mut gpu = GpuEvaluator::new(W, H, &sums).expect("GPU 初始化失败");
    let t1 = Instant::now();
    let _ = gpu.evaluate(&cands).expect("GPU 评估失败");
    let first = t1.elapsed();
    let t2 = Instant::now();
    let r2 = gpu.evaluate(&cands).expect("GPU 评估失败");
    let steady = t2.elapsed();

    let gpu_best = r2.iter().map(|r| r.delta).max().unwrap();
    let cpu_best = if cpu_best == i64::MIN { 0 } else { cpu_best };
    println!("候选数 {N} @ {W}x{H}");
    println!("CPU 单线程 : {cpu_dt:?}  ({:.0} 候选/秒)", N as f64 / cpu_dt.as_secs_f64());
    println!("GPU 首轮   : {first:?}  ({:.0} 候选/秒)", N as f64 / first.as_secs_f64());
    println!("GPU 稳态   : {steady:?}  ({:.0} 候选/秒)", N as f64 / steady.as_secs_f64());
    println!("加速比(稳态 vs CPU 单线程): {:.1}x", cpu_dt.as_secs_f64() / steady.as_secs_f64());
    println!("最优 delta 一致性: gpu={gpu_best} cpu={cpu_best}");
}
