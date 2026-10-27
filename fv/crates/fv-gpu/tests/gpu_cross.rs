//! GPU 评估器 vs CPU evaluate 交叉验证：随机图 + 随机候选，数值必须一致（边界 ±1px 容差）。

use fv_fit::candidate::{self, CandidateSpec, Rng};
use fv_fit::engine::{self, BatchEvaluator};
use fv_fit::rowsum::RowSums;
use fv_gpu::GpuEvaluator;

const W: u32 = 448;
const H: u32 = 448;

fn fixture() -> (Vec<u8>, Vec<u8>, RowSums) {
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
    (target, canvas, sums)
}

#[test]
fn gpu_matches_cpu() {
    let (_t, _c, sums) = fixture();
    let mut gpu = match GpuEvaluator::new(W, H, &sums) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("跳过（无可用 GPU/特性）：{e}");
            return;
        }
    };

    let mut rng = Rng::new(0x5EED_1234);
    let spec = CandidateSpec {
        min_radius: 2.0,
        max_radius: 160.0,
    };
    let cands: Vec<_> = (0..5000)
        .map(|_| candidate::sample(&mut rng, W as f32, H as f32, &spec))
        .collect();

    let gpu_results = gpu.evaluate(&cands).expect("GPU 评估失败");

    let mut exact = 0usize;
    let mut tol_ok = 0usize;
    for (e, g) in cands.iter().zip(&gpu_results) {
        let c = engine::evaluate(&sums, e, W, H);
        match (c, g.ok) {
            (None, false) => {
                exact += 1;
            }
            (Some((ccolor, cdelta)), true) => {
                // f32 vs f64 几何：行端点可能 ±1px。delta 差上界按边界行数放宽。
                let tol = 255i64 * 255 * 3 * 8; // ≤8 行边界 × 3ch × 255²（很宽）
                if g.color == ccolor && g.delta == cdelta {
                    exact += 1;
                }
                if (g.delta - cdelta).abs() <= tol {
                    tol_ok += 1;
                } else {
                    panic!(
                        "候选 ({:.1},{:.1},{:.1},{:.1},{:.2}) gpu={} cpu={} area={}",
                        e.cx, e.cy, e.rx, e.ry, e.angle_rad, g.delta, cdelta, g.area
                    );
                }
            }
            _ => panic!("ok 标志不一致：cpu_ok={} gpu_ok={}", c.is_some(), g.ok),
        }
    }
    eprintln!("精确一致 {exact}/5000，容差内 {tol_ok}/5000");
    assert!(tol_ok == 5000, "全部必须落在容差内");
    assert!(exact * 100 >= 80 * 5000, "精确一致率 {exact}/5000 过低（<80%）");
}

#[test]
fn gpu_soft_matches_cpu() {
    let (target, canvas, sums) = fixture();
    let mut gpu = match GpuEvaluator::new(W, H, &sums) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("跳过（无可用 GPU/特性）：{e}");
            return;
        }
    };
    // 软通道需要真实 target/canvas 上下文
    gpu.on_ready(&sums, &target, &canvas, W, H);

    let mut rng = Rng::new(0x50F7_BEEF);
    let spec = CandidateSpec {
        min_radius: 2.0,
        max_radius: 120.0,
    };
    let cands: Vec<_> = (0..800)
        .map(|_| candidate::sample(&mut rng, W as f32, H as f32, &spec))
        .collect();

    let gpu_results = gpu.eval_soft_batch(&cands, W, H);
    let mut k_match = 0usize;
    let mut k_near = 0usize;
    let mut checked = 0usize;
    for (e, g) in cands.iter().zip(&gpu_results) {
        let c = fv_fit::soft::eval_soft(&target, &canvas, W, H, e);
        match (c, g) {
            (None, None) => {}
            (Some((ck, cd)), Some((gk, gd))) => {
                checked += 1;
                let max_diff = ck.iter().zip(gk.iter()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                if max_diff == 0 {
                    k_match += 1;
                }
                if max_diff <= 1 {
                    k_near += 1;
                }
                // f32 vs f64 + 取整路径：delta 相对误差放宽到 5%（排序用途）
                let denom = cd.abs().max(1);
                let rel = (gd - cd).abs() as f64 / denom as f64;
                assert!(rel < 0.05, "soft delta 偏差 {gd} vs {cd}（{e:?}）");
            }
            _ => panic!("soft ok 标志不一致：cpu={c:?} gpu={g:?}"),
        }
    }
    eprintln!("soft 交叉：有值 {checked}，k 全等 {k_match}，k±1 {k_near}/{checked}");
    assert!(checked > 400, "有效 soft 候选过少");
    assert!(
        k_near * 100 >= 95 * checked,
        "soft k±1 一致率过低：{k_near}/{checked}"
    );
}
