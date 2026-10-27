//! 软通道 GPU 缓冲生命周期/同步探针（引擎规模 20000 候选、448×448）。
//!
//! 针对五个假设逐条验证：
//! 1. ensure_cap 换新 cand/out 后两个 bind group 是否正确（软路径/实心路径各自触发扩容）；
//! 2. write_buffer → dispatch 顺序（on_accept 重传平面 + 每批重写 cand/params）；
//! 3. dispatch_and_read 的 map_async/poll(Wait) 读侧（同输入重复调用应逐位一致；
//!    提前读只会表现为hang/错位，不会表现为确定性结果漂移）；
//! 4. out 缓冲两次 dispatch 之间的残留值是否泄漏（实心 ok=1 满槽残留 → soft）；
//! 5. solid/soft 内核共用 params/cand/out 交替 dispatch 是否相互污染。
//!
//! 另附引擎软循环复刻测试：证明「接纳 → 种子随 soft_out.len() 变化」以及
//! 「软接纳分支缺 on_accept → GPU 画布 stale → pred 虚高」两个引擎侧行为。

use fv_fit::candidate::{self, CandidateSpec, Rng};
use fv_fit::engine::{BatchEvaluator, EvalResults};
use fv_fit::raster::EllipseParams;
use fv_fit::rowsum::RowSums;
use fv_fit::soft;
use fv_gpu::GpuEvaluator;

const W: u32 = 448;
const H: u32 = 448;
const N: usize = 20000; // 引擎实际批量（--samples 20000，触发 4096→32768 扩容）

fn fixture() -> (Vec<u8>, Vec<u8>, RowSums) {
    let n = (W * H) as usize;
    let mut seed = 0xBEEFu64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u8
    };
    let target: Vec<u8> = (0..n * 3).map(|_| next()).collect();
    let canvas: Vec<u8> = (0..n * 3).map(|_| next()).collect();
    let mut sums = RowSums::new(W, H);
    sums.set_target(&target);
    sums.refresh_canvas(&canvas);
    (target, canvas, sums)
}

fn spec() -> CandidateSpec {
    CandidateSpec {
        min_radius: 2.0,
        max_radius: 120.0,
    }
}

fn gen_cands(seed: u64) -> Vec<EllipseParams> {
    let mut rng = Rng::new(seed);
    (0..N)
        .map(|_| candidate::sample(&mut rng, W as f32, H as f32, &spec()))
        .collect()
}

fn cands_eq(a: &[EllipseParams], b: &[EllipseParams]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.cx == y.cx && x.cy == y.cy && x.rx == y.rx && x.ry == y.ry && x.angle_rad == y.angle_rad
        })
}

/// 复刻 engine.rs 软循环 argmax 语义：delta 大者优先（先到先得平手）。
fn argmax(rs: &EvalResults) -> Option<(usize, [u8; 3], i64)> {
    let mut best: Option<(usize, [u8; 3], i64)> = None;
    for (i, r) in rs.iter().enumerate() {
        if let Some((color, delta)) = r {
            let (color, delta) = (*color, *delta);
            if best.as_ref().is_none_or(|(_, _, bd)| delta > *bd) {
                best = Some((i, color, delta));
            }
        }
    }
    best
}

fn ok_count(rs: &EvalResults) -> usize {
    rs.iter().filter(|r| r.is_some()).count()
}

fn paint_block(canvas: &mut [u8], bx: u32, by: u32, bw: u32, bh: u32, color: [u8; 3]) {
    let n = (W * H) as usize;
    for y in by..(by + bh).min(H) {
        for x in bx..(bx + bw).min(W) {
            let p = y as usize * W as usize + x as usize;
            for ch in 0..3 {
                canvas[ch * n + p] = color[ch];
            }
        }
    }
}

/// 一轮实心层：GPU 批评 → argmax → 画实块 → refresh → on_accept（复刻 engine.rs:504-598）。
fn solid_round(
    gpu: &mut GpuEvaluator,
    sums: &mut RowSums,
    target: &[u8],
    canvas: &mut [u8],
    seed: u64,
) {
    let cands = gen_cands(seed);
    let rs = gpu.eval_batch(&cands, W, H);
    if let Some((i, color, _)) = argmax(&rs) {
        let e = cands[i];
        let bx = e.cx.clamp(0.0, W as f32 - 1.0) as u32;
        let by = e.cy.clamp(0.0, H as f32 - 1.0) as u32;
        paint_block(canvas, bx, by, 48, 48, color);
        sums.refresh_canvas(canvas);
        gpu.on_accept(sums, target, canvas, W, H);
    }
}

/// argmax 与 CPU 软参考交叉验证（k±1、delta 5%，与 gpu_cross 同口径）。
fn check_argmax_vs_cpu(
    target: &[u8],
    canvas: &[u8],
    cands: &[EllipseParams],
    (i, gk, gd): (usize, [u8; 3], i64),
    ctx: &str,
) {
    let Some((ck, cd)) = soft::eval_soft(target, canvas, W, H, &cands[i]) else {
        panic!("{ctx}: GPU argmax 有值但 CPU 无：e={:?}", cands[i]);
    };
    let diff = ck
        .iter()
        .zip(gk.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    let denom = cd.abs().max(1);
    let rel = (gd - cd).abs() as f64 / denom as f64;
    assert!(
        diff <= 1 && rel < 0.05,
        "{ctx}: argmax 偏差 k_diff={diff} gpu={gd} cpu={cd} e={:?}",
        cands[i]
    );
}

/// 前 probe_n 个候选的 ok 标志 + delta 与 CPU 一致性（GPU 全量确定性之外的正确性锚点）。
fn check_prefix_vs_cpu(target: &[u8], canvas: &[u8], cands: &[EllipseParams], rs: &EvalResults, probe_n: usize, ctx: &str) {
    let mut checked = 0usize;
    for (e, g) in cands.iter().zip(rs).take(probe_n) {
        let c = soft::eval_soft(target, canvas, W, H, e);
        match (c, g) {
            (None, None) => {}
            (Some((ck, cd)), Some((gk, gd))) => {
                checked += 1;
                let diff = ck
                    .iter()
                    .zip(gk.iter())
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                let denom = cd.abs().max(1);
                let rel = (gd - cd).abs() as f64 / denom as f64;
                assert!(diff <= 1 && rel < 0.05, "{ctx}: 偏差 gpu={gd} cpu={cd} e={e:?}");
            }
            _ => panic!("{ctx}: ok 标志不一致 cpu={c:?} gpu={g:?}"),
        }
    }
    eprintln!("{ctx}: 前 {probe_n} 候选 ok 一致，其中 {checked} 个有值且 delta 5% 内");
}

#[test]
fn soft_engine_scale_triple_call_deterministic() {
    let (target, mut canvas, mut sums) = fixture();
    let Ok(mut gpu) = GpuEvaluator::new(W, H, &sums) else {
        eprintln!("跳过：无 GPU");
        return;
    };
    gpu.on_ready(&sums, &target, &canvas, W, H);

    // 引擎实心层序列：两次 20000 批（首次在实心路径触发 ensure_cap 4096→32768，
    // 并经 on_accept 重传 maps/target/canvas —— 与真实 20 层轨迹等价的状态）
    solid_round(&mut gpu, &mut sums, &target, &mut canvas, 0xA11CE);
    solid_round(&mut gpu, &mut sums, &target, &mut canvas, 0xB0B);

    // 同输入连续 3 次 soft 批评（引擎轨迹中第 1 次与第 2/3 次分歧的点）
    let cands = gen_cands(0x50F7);
    let r1 = gpu.eval_soft_batch(&cands, W, H);
    let r2 = gpu.eval_soft_batch(&cands, W, H);
    let r3 = gpu.eval_soft_batch(&cands, W, H);

    assert_eq!(ok_count(&r1), ok_count(&r2), "ok 标志数 call1 != call2");
    assert_eq!(r1, r2, "soft 全量结果 call1 != call2（同输入）");
    assert_eq!(r2, r3, "soft 全量结果 call2 != call3（同输入）");
    let a1 = argmax(&r1).expect("iter1 应有正收益候选");
    let a2 = argmax(&r2).expect("iter2 应有正收益候选");
    assert_eq!(a1, a2, "argmax 不稳定");

    check_argmax_vs_cpu(&target, &canvas, &cands, a1, "triple-call");
    eprintln!(
        "triple-call 确定性：ok {ok}/{}，argmax idx={}",
        N,
        a1.0,
        ok = ok_count(&r1)
    );
}

#[test]
fn soft_first_call_after_cap_growth_on_soft_path() {
    // 用户假设的触发方式：首次 eval_soft_batch 即扩容（换 cand/out + 重建 soft_bind/bind）
    let (target, canvas, sums) = fixture();
    let Ok(mut gpu) = GpuEvaluator::new(W, H, &sums) else {
        eprintln!("跳过：无 GPU");
        return;
    };
    gpu.on_ready(&sums, &target, &canvas, W, H);

    let cands = gen_cands(0x50F7);
    let r1 = gpu.eval_soft_batch(&cands, W, H); // 内部 ensure_cap(20000)：4096→32768
    let r2 = gpu.eval_soft_batch(&cands, W, H);
    let r3 = gpu.eval_soft_batch(&cands, W, H);

    assert_eq!(r1, r2, "扩容后首次调用与第二次不一致");
    assert_eq!(r2, r3, "第二次与第三次不一致");
    let a1 = argmax(&r1).expect("应有正收益候选");
    check_argmax_vs_cpu(&target, &canvas, &cands, a1, "cap-growth-soft");
    // 扩容后 bind group 数值正确性锚点：前 150 个候选与 CPU 对齐
    check_prefix_vs_cpu(&target, &canvas, &cands, &r1, 150, "cap-growth-soft");
    eprintln!(
        "soft 路径扩容：ok {ok}/{}，argmax idx={}",
        N,
        a1.0,
        ok = ok_count(&r1)
    );
}

#[test]
fn soft_solid_interleave_no_pollution() {
    // solid→soft→solid→soft：out 缓冲里实心 ok=1 满槽残留不得泄漏进 soft 结果；
    // 两内核共用 params/cand/out 交替 dispatch 不得相互污染。
    let (target, canvas, sums) = fixture();
    let Ok(mut gpu) = GpuEvaluator::new(W, H, &sums) else {
        eprintln!("跳过：无 GPU");
        return;
    };
    gpu.on_ready(&sums, &target, &canvas, W, H);

    let cs = gen_cands(0x5EED_1234);
    let ca = gen_cands(0x50F7);

    let solid1 = gpu.eval_batch(&cs, W, H); // out ← 实心结果（大量 ok=1 槽）
    let s1 = gpu.eval_soft_batch(&ca, W, H); // 紧随其后读 out 的 soft
    let solid2 = gpu.eval_batch(&cs, W, H);
    let s2 = gpu.eval_soft_batch(&ca, W, H);

    assert_eq!(s1, s2, "solid/soft 交替后 soft 结果不一致（残留泄漏/污染）");
    // 实心自身也应确定（同一批两次结果一致）
    assert_eq!(solid1, solid2, "solid 结果不稳定");
    let a1 = argmax(&s1).expect("soft 应有正收益候选");
    check_argmax_vs_cpu(&target, &canvas, &ca, a1, "interleave");
    check_prefix_vs_cpu(&target, &canvas, &ca, &s1, 150, "interleave");
    eprintln!(
        "solid/soft 交替：soft ok {ok}/{}，argmax idx={}",
        N,
        a1.0,
        ok = ok_count(&s1)
    );
}

#[test]
fn engine_soft_loop_replica_accept_seed_and_stale_canvas() {
    // 复刻 engine.rs 软循环关键机制：
    //   seed = cfg.seed ^ 0x50F7 ^ ((soft_out.len() as u64) << 32)   （engine.rs:662）
    //   接纳分支 blend_composite + grid.rebuild，但【不调 on_accept】（engine.rs:757-762）
    let (target, mut canvas, mut sums) = fixture();
    let Ok(mut gpu) = GpuEvaluator::new(W, H, &sums) else {
        eprintln!("跳过：无 GPU");
        return;
    };
    gpu.on_ready(&sums, &target, &canvas, W, H);
    let base = 0x1234_5678u64 ^ 0x50F7;

    // 迭代 1：soft_out.len()==0
    let c0 = gen_cands(base);
    let r1 = gpu.eval_soft_batch(&c0, W, H);
    let (i1, k1, d1) = argmax(&r1).expect("iter1 应有 argmax");
    let e1 = c0[i1];

    // 引擎接纳路径：改画布 + refresh，但 GPU 不通知（缺 on_accept）
    soft::blend_composite(&mut canvas, W, H, &e1, k1, 1.0);
    sums.refresh_canvas(&canvas);

    // 迭代 2 种子：soft_out.len() 0→1 → 候选列表确实变了（「候选列表完全相同」前提不成立）
    let c1 = gen_cands(base ^ (1u64 << 32));
    assert!(
        !cands_eq(&c0, &c1),
        "接纳一次后候选列表应不同（种子依赖 soft_out.len()）"
    );

    let r2a = gpu.eval_soft_batch(&c1, W, H);
    let r2b = gpu.eval_soft_batch(&c1, W, H);
    assert_eq!(r2a, r2b, "同输入 soft 批评不确定（iter2 vs iter3 场景）");

    // GPU 侧画布未同步的直接证据：重评 iter1 的列表必须逐位复现 r1
    let r1_again = gpu.eval_soft_batch(&c0, W, H);
    assert_eq!(
        r1_again, r1,
        "GPU 结果漂移：同输入同 GPU 状态不应变化（说明 GPU 画布被意外更新）"
    );

    // 补上引擎漏掉的 on_accept：同列表结果必须改变，且与 CPU（真画布）对齐
    gpu.on_accept(&sums, &target, &canvas, W, H);
    let r1_sync = gpu.eval_soft_batch(&c0, W, H);
    assert_ne!(
        r1_sync, r1,
        "on_accept 后 GPU 结果不变——平面同步失效（不应发生）"
    );

    // 刚被软混的同一候选：stale 画布上 gpu=d1（大额虚高）；真画布上应≈0/None
    let cpu_after = soft::eval_soft(&target, &canvas, W, H, &e1);
    let gpu_after = r1_sync[i1];
    match cpu_after {
        None => assert!(
            gpu_after.is_none() || gpu_after.unwrap().1 < d1 / 10,
            "sync 后该候选 pred 仍居高：gpu={gpu_after:?} stale_pred={d1}（stale 画布虚高证据不足）"
        ),
        Some((ck, cd)) => {
            let Some((gk, gd)) = gpu_after else {
                panic!("sync 后 CPU 有值 GPU 无：e={e1:?}");
            };
            let diff = ck
                .iter()
                .zip(gk.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            let denom = cd.abs().max(1);
            let rel = (gd - cd).abs() as f64 / denom as f64;
            assert!(diff <= 1 && rel < 0.05, "sync 后偏差 gpu={gd} cpu={cd}");
        }
    }
    eprintln!(
        "复刻确认：iter1 接纳（gpu_pred={d1}）→ 种子变化 → iter2 候选不同；GPU 画布 stale（重评逐位一致），on_accept 后同候选 pred={:?} vs stale={d1}",
        r1_sync[i1].map(|(_, d)| d)
    );
}

#[test]
fn soft_borderline_delta_zero_rows_are_f32_truncation() {
    // 平滑画布（模拟软层混入后的低残差区域，同 replay.rs 的 dump 场景）：
    // 同一画布喂 GPU 与 CPU，ok 标志分歧只允许出现在 |真 delta| < 1 的边界带
    //（shader ok==1 ⟹ 0 < f32 delta；i64 delta==0 ⟹ f32 delta < 1，纯截断）。
    // 若出现 GPU delta 远离 0 而 CPU None（或反之且幅度大），才算真分歧。
    let (target, mut canvas, mut sums) = fixture();
    let Ok(mut gpu) = GpuEvaluator::new(W, H, &sums) else {
        eprintln!("跳过：无 GPU");
        return;
    };
    gpu.on_ready(&sums, &target, &canvas, W, H);

    // 混入几个大软椭圆制造低残差平滑区（复刻软通道接纳后的画布形态）
    let mut rng = Rng::new(0xC0FFEE);
    let big = CandidateSpec {
        min_radius: 60.0,
        max_radius: 140.0,
    };
    let mut blended: Vec<EllipseParams> = Vec::new();
    for _ in 0..6 {
        let e = candidate::sample(&mut rng, W as f32, H as f32, &big);
        if let Some((k, _)) = soft::eval_soft(&target, &canvas, W, H, &e) {
            soft::blend_composite(&mut canvas, W, H, &e, k, 1.0);
            blended.push(e);
        }
    }
    sums.refresh_canvas(&canvas);
    gpu.on_accept(&sums, &target, &canvas, W, H);

    // 刚混过的椭圆及其微抖动变体的真 delta ≈ 0（已最优化）——正是 f32/f64 边界带
    let mut jrng = Rng::new(0x17A7_17A7);
    let mut jittered: Vec<EllipseParams> = Vec::new();
    for e in &blended {
        for _ in 0..40 {
            jittered.push(EllipseParams {
                cx: e.cx + (jrng.next_f32() - 0.5),
                cy: e.cy + (jrng.next_f32() - 0.5),
                rx: e.rx * (1.0 + (jrng.next_f32() - 0.5) * 0.02),
                ry: e.ry * (1.0 + (jrng.next_f32() - 0.5) * 0.02),
                angle_rad: e.angle_rad + (jrng.next_f32() - 0.5) * 0.02,
            });
        }
    }
    let rs_j = gpu.eval_soft_batch(&jittered, W, H);
    let (mut zj, mut bigj) = (0usize, 0usize);
    for (i, r) in rs_j.iter().enumerate() {
        let cpu = soft::eval_soft(&target, &canvas, W, H, &jittered[i]);
        match (r, cpu) {
            (Some((gk, gd)), Some((ck, cd))) => {
                if *gd == 0 {
                    zj += 1;
                    assert!(cd.abs() <= 1, "delta==0 行 CPU delta={cd} 超边界带");
                    let diff = gk
                        .iter()
                        .zip(ck.iter())
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap();
                    assert!(diff <= 2, "delta==0 行 k 偏差 {diff}");
                }
            }
            (Some(_), None) => bigj += 1,
            (None, Some((_, cd))) if cd.abs() > 1 => bigj += 1,
            _ => {}
        }
    }
    eprintln!("抖动组：delta==0 行 {zj}/{}，超边界带翻转 {bigj}", jittered.len());
    let (mut zero_rows, mut big_flips, mut gpu_none_cpu_some) = (bigj, 0usize, 0usize);
    for batch_seed in 1..=3u64 {
        let cands = gen_cands(batch_seed * 0x9E37);
        let rs = gpu.eval_soft_batch(&cands, W, H);
        for (i, r) in rs.iter().enumerate() {
            let cpu = soft::eval_soft(&target, &canvas, W, H, &cands[i]);
            match (r, cpu) {
                (Some((gk, gd)), Some((ck, cd))) => {
                    if *gd == 0 {
                        zero_rows += 1;
                        assert!(
                            cd.abs() <= 1,
                            "delta==0 行的 CPU delta 过大：{cd}（超出截断边界带）e={:?}",
                            cands[i]
                        );
                        let diff = gk
                            .iter()
                            .zip(ck.iter())
                            .map(|(a, b)| a.abs_diff(*b))
                            .max()
                            .unwrap();
                        assert!(diff <= 2, "delta==0 行 k 偏差过大 {diff}");
                    }
                }
                (Some(_), None) => {
                    big_flips += 1;
                    eprintln!("GPU Some / CPU None：{:?} gpu={r:?}", cands[i]);
                }
                (None, Some((_, cd))) => {
                    gpu_none_cpu_some += 1;
                    if cd.abs() > 1 {
                        big_flips += 1;
                        eprintln!("GPU None / CPU 有大值 {cd}：{:?}", cands[i]);
                    }
                }
                (None, None) => {}
            }
        }
    }
    eprintln!(
        "边界带统计：GPU i64-delta==0 且 ok=1 共 {zero_rows} 行（须全部落在 CPU |delta|≤1 内）；大幅度翻转 {big_flips}（应为 0 或极小）；GPU None/CPU 小正值 {gpu_none_cpu_some}"
    );
    assert_eq!(big_flips, 0, "存在超出 f32/f64 边界带的真实分歧");
}
