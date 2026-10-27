//! 重放引擎现场：加载 dump 的 (target, canvas, 候选)，GPU 与 CPU 各评一次。
//! 前置：先用 fvfit-gpu 跑出 C:/Users/Lenovo/AppData/Local/Temp/fv_dump_*.bin
//!（engine.rs 软通道调试 dump；无 dump 时本测试跳过）。

use fv_fit::candidate::Rng;
use fv_fit::engine::BatchEvaluator;
use fv_fit::raster::EllipseParams;
use fv_fit::rowsum::RowSums;
use fv_gpu::GpuEvaluator;

#[test]
fn replay_engine_divergence() {
    let dir = "C:/Users/Lenovo/AppData/Local/Temp";
    let Ok(target) = std::fs::read(format!("{dir}/fv_dump_target.bin")) else {
        eprintln!("跳过：无 dump 文件");
        return;
    };
    let canvas = std::fs::read(format!("{dir}/fv_dump_canvas.bin")).unwrap();
    let cand_txt = std::fs::read_to_string(format!("{dir}/fv_dump_cand.txt")).unwrap();
    let dims = std::fs::read_to_string(format!("{dir}/fv_dump_dims.txt")).unwrap();
    let (w, h) = dims.split_once(',').unwrap();
    let (w, h) = (w.parse::<u32>().unwrap(), h.parse::<u32>().unwrap());
    let v: Vec<f32> = cand_txt.split(',').map(|s| s.parse().unwrap()).collect();
    let e = EllipseParams {
        cx: v[0],
        cy: v[1],
        rx: v[2],
        ry: v[3],
        angle_rad: v[4],
    };
    let k_dump = [v[5] as u8, v[6] as u8, v[7] as u8];
    eprintln!("重放 {w}x{h} 候选 {e:?} k={k_dump:?}");

    let n = (w * h) as usize;
    assert_eq!(target.len(), n * 3);
    let mut sums = RowSums::new(w, h);
    sums.set_target(&target);
    sums.refresh_canvas(&canvas);

    let cpu = fv_fit::soft::eval_soft(&target, &canvas, w, h, &e);
    eprintln!("cpu = {cpu:?}");

    // 引擎当时实测：混合 k_dump 后 bbox 内误差变化
    let mut sim = canvas.clone();
    let before: i64 = (0..n * 3)
        .map(|i| (i64::from(target[i]) - i64::from(sim[i])).pow(2))
        .sum();
    fv_fit::soft::blend_composite(&mut sim, w, h, &e, k_dump, 1.0);
    let after: i64 = (0..n * 3)
        .map(|i| (i64::from(target[i]) - i64::from(sim[i])).pow(2))
        .sum();
    eprintln!("引擎实测 delta = {}（全画布精确重算）", before - after);

    let Ok(mut gpu) = GpuEvaluator::new(w, h, &sums) else {
        eprintln!("跳过：无 GPU");
        return;
    };
    gpu.on_ready(&sums, &target, &canvas, w, h);
    let gpu_r = gpu.eval_soft_batch(&[e], w, h);
    eprintln!("gpu = {gpu_r:?}");

    // 单候选重放若一致，则分歧来自批内多候选交互/竞态
    let mut rng = Rng::new(0x50F7);
    let mut batch = vec![e];
    for _ in 0..19999 {
        batch.push(EllipseParams {
            cx: rng.next_f32() * w as f32,
            cy: rng.next_f32() * h as f32,
            rx: 2.0 + rng.next_f32() * 140.0,
            ry: 2.0 + rng.next_f32() * 140.0,
            angle_rad: rng.next_f32() * core::f32::consts::PI,
        });
    }
    let gpu_batch = gpu.eval_soft_batch(&batch, w, h);
    eprintln!("batch[0] = {:?}", gpu_batch[0]);
    let cpu_batch0 = fv_fit::soft::eval_soft(&target, &canvas, w, h, &batch[0]);
    eprintln!("cpu_batch[0] = {cpu_batch0:?}");

    let mut mismatch = 0usize;
    for (bb, gr) in batch.iter().zip(&gpu_batch) {
        let c = fv_fit::soft::eval_soft(&target, &canvas, w, h, bb);
        match (c, gr) {
            (None, None) => {}
            // 边界带：|delta| < 1 的 f32/f64 取整口径差（噪声级收益，无实际影响）
            (None, Some((_, gd))) if *gd <= 1 => {}
            (Some((_, cd)), None) if cd.abs() <= 1 => {}
            (Some((_, cd)), Some((_gk, gd))) if cd.abs() <= 1 => {}
            (Some((ck, cd)), Some((gk, gd))) => {
                let denom = cd.abs().max(1);
                if (gd - cd).abs() as f64 / denom as f64 > 0.05 {
                    mismatch += 1;
                    if mismatch <= 3 {
                        eprintln!("mismatch {bb:?}: cpu=({ck:?},{cd}) gpu=({gk:?},{gd})");
                    }
                }
            }
            _ => {
                mismatch += 1;
                if mismatch <= 3 {
                    eprintln!("ok-flag mismatch {bb:?}: cpu={c:?} gpu={gr:?}");
                }
            }
        }
    }
    eprintln!("批内不一致数：{mismatch}/20000");
    assert!(mismatch == 0, "GPU 与 CPU 在重放数据上仍有分歧");
}
