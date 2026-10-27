//! 分层拟合编排库（自 bin 抽出，解锁集成测试）。
//!
//! 编排：bg → subject（实心）→ edge（实心或 `--strokes` 画家式描边）→ soft 终段
//!（目标=原图全域，实心三振后的残差归软层认领——真图上只画边界带会输盲拟合 2dB
//! 的教训，见 STUDY L1.2b）。`--tune` 坐标搜索与 `--plan-params` 直通亦在此。

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fv_common::shape::Shape;
use fv_fit::engine::{fit, fit_gpu, resume, resume_gpu, BatchEvaluator, FitConfig, FitResult};
use fv_fit::render::render_shapes;
use fv_gpu::GpuEvaluator;

use crate::plan_layers_with;
use crate::{LayerPlan, Matte, PlanParams};

/// 单层统计（stderr 汇总用）。
pub struct LayerStat {
    pub name: &'static str,
    pub budget: usize,
    pub solid: usize,
    pub soft: usize,
    pub secs: f64,
    pub mse: f64,
}

pub struct PipelineResult {
    /// z 序自底向上：bg → subject → edge（实心 + 软），原图像素坐标。
    pub shapes: Vec<Shape>,
    pub stats: Vec<LayerStat>,
    pub psnr: f64,
}

/// 三层拟合编排。预算为 0 的层直接跳过；`samples` 为每形状采样数，
/// 边界层自动加码 ×1.5 并把精修轮数 24→32；`soft_edge` 只作用在终段软相。
/// `gpu` 为 Some 时全部走 GPU 批量评估引擎路径（fit_gpu/resume_gpu），
/// 否则走 CPU rayon 路径——参数与目标图构造完全一致。
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline(
    orig: &image::RgbaImage,
    plan: &LayerPlan,
    samples: usize,
    seed: u64,
    bg: [u8; 3],
    soft_edge: usize,
    strokes: bool,
    gpu: &mut Option<GpuEvaluator>,
) -> Result<PipelineResult, String> {
    let (w, h) = (plan.w, plan.h);
    let mut shapes: Vec<Shape> = Vec::new();
    let mut stats: Vec<LayerStat> = Vec::new();

    // ---- 1. 背景层：主体掩码内像素换成 bg 色 → 主体区零残差 ----
    if plan.budget.background > 0 {
        let keep: Vec<bool> = plan.subject_mask.iter().map(|&v| v == 0).collect();
        let target = patched_target(orig, &keep, move |_| [bg[0], bg[1], bg[2], 255]);
        let cfg = FitConfig {
            layers: plan.budget.background,
            samples_per_shape: samples,
            seed,
            bg,
            ..FitConfig::default()
        };
        let (r, secs) = run_stage(
            "bg",
            &target,
            &[],
            &cfg,
            gpu.as_mut().map(|g| g as &mut dyn BatchEvaluator),
        );
        shapes = r.shapes;
        stats.push(LayerStat {
            name: "bg",
            budget: plan.budget.background,
            solid: shapes.len(),
            soft: 0,
            secs,
            mse: r.final_mse,
        });
    }

    // ---- 2. 主体层：背景掩码内像素换成 canvas1（下层渲染），resume 续拟合 ----
    if plan.budget.subject > 0 {
        let canvas = render_shapes(w, h, &shapes, bg);
        let craw = canvas.into_raw();
        let keep: Vec<bool> = plan.background_mask.iter().map(|&v| v == 0).collect();
        let target =
            patched_target(orig, &keep, |i| [craw[i * 4], craw[i * 4 + 1], craw[i * 4 + 2], 255]);
        let cfg = FitConfig {
            layers: shapes.len() + plan.budget.subject,
            samples_per_shape: samples,
            seed: seed ^ 0x5EED_0001,
            bg,
            ..FitConfig::default()
        };
        let before = shapes.len();
        let (r, secs) = run_stage(
            "subject",
            &target,
            &shapes,
            &cfg,
            gpu.as_mut().map(|g| g as &mut dyn BatchEvaluator),
        );
        shapes = r.shapes;
        stats.push(LayerStat {
            name: "subject",
            budget: plan.budget.subject,
            solid: shapes.len() - before,
            soft: 0,
            secs,
            mse: r.final_mse,
        });
    }

    // ---- 3. 边界层：带外像素换成 canvas2；默认贪心拟合，strokes 走画家式描边 ----
    if plan.budget.edge > 0 {
        let before = shapes.len();
        if strokes {
            // M1.7 描边：field = 边界带强度场，颜色取原图局部均值（planar 目标）
            let n = (w * h) as usize;
            let mut tgt = vec![0u8; n * 3];
            for p in 0..n {
                let px = orig.get_pixel((p % w as usize) as u32, (p / w as usize) as u32);
                tgt[p] = px.0[0];
                tgt[n + p] = px.0[1];
                tgt[2 * n + p] = px.0[2];
            }
            let strokes = fv_fit::stroke::contour_strokes(
                &plan.boundary_band,
                &tgt,
                w,
                h,
                96,
                plan.budget.edge,
                1.5,
                14.0,
                2.0,
            );
            for (e, c) in strokes.shapes {
                shapes.push(Shape::ellipse(
                    e.cx,
                    e.cy,
                    e.rx,
                    e.ry,
                    e.angle_rad.to_degrees(),
                    c,
                ));
            }
            stats.push(LayerStat {
                name: "edge(strokes)",
                budget: plan.budget.edge,
                solid: shapes.len() - before,
                soft: 0,
                secs: 0.0,
                mse: f64::NAN,
            });
        } else {
            let canvas = render_shapes(w, h, &shapes, bg);
            let craw = canvas.into_raw();
            let keep: Vec<bool> = plan.boundary_band.iter().map(|&v| v != 0).collect();
            let target = patched_target(orig, &keep, |i| {
                [craw[i * 4], craw[i * 4 + 1], craw[i * 4 + 2], 255]
            });
            let cfg = FitConfig {
                layers: shapes.len() + plan.budget.edge,
                samples_per_shape: samples + samples / 2,
                seed: seed ^ 0x5EED_0002,
                refine_iters: 32,
                bg,
                ..FitConfig::default()
            };
            let (r, secs) = run_stage(
                "edge",
                &target,
                &shapes,
                &cfg,
                gpu.as_mut().map(|g| g as &mut dyn BatchEvaluator),
            );
            shapes = r.shapes;
            stats.push(LayerStat {
                name: "edge",
                budget: plan.budget.edge,
                solid: shapes.len() - before,
                soft: 0,
                secs,
                mse: r.final_mse,
            });
        }
    }

    // ---- 4. 软椭圆终段：目标 = 原图全域，按残差自由放置（语义路由的"收尾笔"）----
    // 不再局限边界带——实心三振留下的主体内部残差也归它认领（真图实测：
    // 只画带会让盲拟合 auto-soft 反超 2dB）。resume 层数=现有实心数 → 实心循环
    // 立即跳过，直接进引擎软相。
    if soft_edge > 0 {
        let target = orig.clone();
        let cfg = FitConfig {
            layers: shapes.len(),
            samples_per_shape: samples,
            seed: seed ^ 0x5EED_0003,
            bg,
            soft_layers: soft_edge,
            ..FitConfig::default()
        };
        let before = shapes.len();
        let (r, secs) = run_stage(
            "soft",
            &target,
            &shapes,
            &cfg,
            gpu.as_mut().map(|g| g as &mut dyn BatchEvaluator),
        );
        shapes = r.shapes;
        let soft = shapes.len() - before;
        stats.push(LayerStat {
            name: "soft",
            budget: soft_edge,
            solid: 0,
            soft,
            secs,
            mse: r.final_mse,
        });
    }

    let preview = render_shapes(w, h, &shapes, bg);
    let psnr = psnr_db(&preview, orig, bg);
    Ok(PipelineResult { shapes, stats, psnr })
}

/// `--plan-params` 直通单跑（C0 评估通道）。
#[allow(clippy::too_many_arguments)]
pub fn run_with_params(
    img: &image::RgbaImage,
    matte: &Matte,
    p: &PlanParams,
    total: usize,
    samples: usize,
    seed: u64,
    bg: [u8; 3],
    soft_edge: usize,
    strokes: bool,
    gpu: &mut Option<GpuEvaluator>,
) -> Result<PipelineResult, String> {
    let plan = plan_layers_with(matte, total, p)?;
    run_pipeline(img, &plan, samples, seed, bg, soft_edge, strokes, gpu)
}

/// `--tune` 坐标搜索（扩网格 + 两轮，STUDY tune v2）。
/// 返回 (最优结果, 最优参数, 耗时秒)。
#[allow(clippy::too_many_arguments)]
pub fn tune_search(
    img: &image::RgbaImage,
    matte: &Matte,
    total: usize,
    samples: usize,
    seed: u64,
    bg: [u8; 3],
    soft_edge: usize,
    strokes: bool,
    gpu: &mut Option<GpuEvaluator>,
) -> Result<(PipelineResult, PlanParams, f64), String> {
    let t0 = Instant::now();
    let eval = |p: &PlanParams,
                g: &mut Option<GpuEvaluator>|
     -> Result<PipelineResult, String> {
        let plan = plan_layers_with(matte, total, p)?;
        run_pipeline(img, &plan, samples, seed, bg, soft_edge, strokes, g)
    };
    let mut best_p = PlanParams::default();
    let mut best = eval(&best_p, gpu)?;
    eprintln!(
        "[tune] baseline          psnr={:.3} shapes={}",
        best.psnr,
        best.shapes.len()
    );

    // 通用旋钮试探：apply 把候选值写进参数副本，更优则采纳
    #[allow(clippy::too_many_arguments)]
    let try_knob = |name: &str,
                    vals: &[f64],
                    apply: &dyn Fn(&mut PlanParams, f64),
                    best_p: &mut PlanParams,
                    best: &mut PipelineResult,
                    eval: &mut dyn FnMut(&PlanParams) -> Result<PipelineResult, String>|
     -> Result<(), String> {
        for v in vals {
            let mut p = *best_p;
            apply(&mut p, *v);
            let r = eval(&p)?;
            if r.psnr > best.psnr {
                eprintln!(
                    "[tune] {name:<14}={v:<5.2} psnr={:.3} shapes={}  ← best",
                    r.psnr,
                    r.shapes.len()
                );
                *best = r;
                *best_p = p;
            } else {
                eprintln!(
                    "[tune] {name:<14}={v:<5.2} psnr={:.3} shapes={}",
                    r.psnr,
                    r.shapes.len()
                );
            }
        }
        Ok(())
    };

    type Knob = (&'static str, Vec<f64>, Box<dyn Fn(&mut PlanParams, f64)>);
    // v2 候选网格（post-B1 头顶还有 3dB+，旧 2 候选网格吃不完）：
    // 阈值 ±16/±32、带宽 1/5/7、边缘占比 0.10/0.35/0.45 等
    let knobs: Vec<Knob> = vec![
        ("thresh_bias", vec![-32.0, -16.0, 16.0, 32.0], Box::new(|p, v| p.thresh_bias = v as i32)),
        ("smooth_rounds", vec![0.0, 2.0, 3.0], Box::new(|p, v| p.smooth_rounds = v as u32)),
        ("band_radius", vec![1.0, 5.0, 7.0], Box::new(|p, v| p.band_radius = v as usize)),
        ("edge_frac", vec![0.10, 0.35, 0.45], Box::new(|p, v| p.edge_frac = v)),
        ("subject_gain", vec![0.35, 0.55, 0.65], Box::new(|p, v| p.subject_gain = v)),
        ("subject_floor", vec![0.20, 0.40], Box::new(|p, v| p.subject_floor = v)),
    ];
    // 两轮坐标下降（旋钮间有交互——第一轮定大局、第二轮精修）
    for round in 0..2 {
        eprintln!("[tune] round {round}");
        for (name, vals, apply) in &knobs {
            let mut eval_fn = |p: &PlanParams| eval(p, gpu);
            try_knob(name, vals, apply, &mut best_p, &mut best, &mut eval_fn)?;
        }
    }

    eprintln!(
        "[tune] done in {:.1}s: best psnr={:.3} shapes={} params={:?}",
        t0.elapsed().as_secs_f64(),
        best.psnr,
        best.shapes.len(),
        best_p
    );
    Ok((best, best_p, t0.elapsed().as_secs_f64()))
}

/// 跑一层拟合（initial 为空走 fit，否则 resume 续拟合），带 stderr 进度条。
pub fn run_stage(
    tag: &str,
    target: &image::RgbaImage,
    initial: &[Shape],
    cfg: &FitConfig,
    gpu: Option<&mut dyn BatchEvaluator>,
) -> (FitResult, f64) {
    let total = cfg.layers;
    let add = cfg.layers.saturating_sub(initial.len());
    let t0 = Instant::now();
    let mut progress = |done: usize, mse: f64| {
        if done.is_multiple_of(25) || done == total {
            eprint!("\r  [{tag} {done}/{total}] mse={mse:.1}  ");
            let _ = std::io::stderr().flush();
        }
    };
    let r = match (initial.is_empty(), gpu) {
        (true, Some(g)) => fit_gpu(target, cfg, g, &mut progress),
        (true, None) => fit(target, cfg, &mut progress),
        (false, Some(g)) => resume_gpu(target, cfg, initial, g, &mut progress),
        (false, None) => resume(target, cfg, initial, &mut progress),
    };
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "\r  [{tag}] {}/{} accepted in {secs:.1}s",
        r.shapes.len() - initial.len(),
        add
    );
    (r, secs)
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

/// 预初始化 GPU 评估器：按引擎 `fit_impl` 的同一公式把输入尺寸换算成
/// 拟合工作分辨率（opt_max_dim=448），用空 RowSums 探测 Vulkan 设备；
/// 失败返回 Err（调用方回退 CPU，不 panic）。
pub fn init_gpu_evaluator(w: u32, h: u32) -> Result<GpuEvaluator, String> {
    let opt_max_dim = FitConfig::default().opt_max_dim;
    let long_edge = w.max(h).max(1);
    let scale = (opt_max_dim as f32 / long_edge as f32).min(1.0);
    let (fw, fh) = (
        ((w as f32 * scale) as u32).max(4),
        ((h as f32 * scale) as u32).max(4),
    );
    let sums = fv_fit::rowsum::RowSums::new(fw, fh);
    GpuEvaluator::new(fw, fh, &sums)
}

/// 目标图：`keep[i]` 为真的像素保留原图，其余替换为 `fill(i)`。
pub fn patched_target(
    orig: &image::RgbaImage,
    keep: &[bool],
    fill: impl Fn(usize) -> [u8; 4],
) -> image::RgbaImage {
    let mut out = orig.clone();
    let raw: &mut [u8] = out.as_mut();
    for (i, &k) in keep.iter().enumerate() {
        if !k {
            raw[i * 4..i * 4 + 4].copy_from_slice(&fill(i));
        }
    }
    out
}

/// 全分辨率 PSNR：渲染 vs 原图（透明像素按 bg 规则），逐像素 3 通道 MSE。
pub fn psnr_db(render: &image::RgbaImage, orig: &image::RgbaImage, bg: [u8; 3]) -> f64 {
    let mut se = 0i64;
    let mut n = 0usize;
    for (o, r) in orig.pixels().zip(render.pixels()) {
        let c = if o.0[3] < 128 { bg } else { [o.0[0], o.0[1], o.0[2]] };
        for (cv, rv) in c.iter().zip(r.0.iter().take(3)) {
            let d = i64::from(*cv) - i64::from(*rv);
            se += d * d;
        }
        n += 1;
    }
    let mse = se as f64 / (3.0 * n.max(1) as f64);
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// 背景色：`auto` → 干净背景（主体掩码外且边界带外）不透明像素均值，
/// 逐级退化到背景掩码像素、全图像素（仍无 → 黑）；否则解析 RRGGBB。
pub fn resolve_bg(orig: &image::RgbaImage, plan: &LayerPlan, spec: &str) -> Result<[u8; 3], String> {
    if !spec.eq_ignore_ascii_case("auto") {
        let s = spec.trim_start_matches('#');
        if s.len() != 6 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("非法背景色 \"{spec}\"：应为 auto 或 RRGGBB"));
        }
        let v = u32::from_str_radix(s, 16).map_err(|e| e.to_string())?;
        return Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8]);
    }
    let raw = orig.as_raw();
    let n = raw.len() / 4;
    let mean = |sel: &dyn Fn(usize) -> bool| -> Option<[u8; 3]> {
        let (mut sum, mut cnt) = ([0u64; 3], 0u64);
        for i in 0..n {
            if sel(i) && raw[i * 4 + 3] >= 128 {
                for ch in 0..3 {
                    sum[ch] += u64::from(raw[i * 4 + ch]);
                }
                cnt += 1;
            }
        }
        (cnt > 0).then(|| [0, 1, 2].map(|ch| ((sum[ch] + cnt / 2) / cnt).clamp(0, 255) as u8))
    };
    mean(&|i| plan.subject_mask[i] == 0 && plan.boundary_band[i] == 0)
        .or_else(|| mean(&|i| plan.subject_mask[i] == 0))
        .or_else(|| mean(&|_| true))
        .ok_or_else(|| "auto 背景色：输入图无有效不透明像素".to_string())
}

/// 从灰度 PNG 读 matte，尺寸不符自动 Triangle 缩放。
pub fn matte_from_png(path: &Path, w: u32, h: u32) -> Result<Matte, String> {
    let img = image::open(path).map_err(|e| format!("打开 matte {}: {e}", path.display()))?;
    let mut gray = img.to_luma8();
    if gray.width() != w || gray.height() != h {
        eprintln!(
            "matte {}x{} 与输入图 {w}x{h} 不一致，已自动缩放",
            gray.width(),
            gray.height()
        );
        gray = image::imageops::resize(&gray, w, h, image::imageops::FilterType::Triangle);
    }
    Ok(Matte { width: w, height: h, data: gray.into_raw() })
}

/// `{prefix}{suffix}`（追加而非替换扩展名）。
pub fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_os_string();
    s.push(suffix);
    s.into()
}

/// 默认输出前缀 = 输入文件去扩展名（保留父目录）。
pub fn default_prefix(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".into());
    match input.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(stem),
        _ => PathBuf::from(stem),
    }
}
