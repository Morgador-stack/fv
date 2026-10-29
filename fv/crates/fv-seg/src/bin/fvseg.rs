//! fvseg CLI：语义分层拟合（matte → 三层编排 → geometrize JSON）。
//! 编排/plan-params 逻辑在 `fv_seg::pipeline`（库，可测试）；本文件做参数解析、
//! matte 获取与产物落盘，并承载 tune v3 搜索引擎（TUNE_V3_DESIGN：连续化坐标
//! 下降 + 三点抛物线顶点 + E1/E3/E5 早停）。v2 固定网格引擎原样保留在
//! `pipeline::tune_search`，经 `--tune-algo v2` 供 A/B 回退对照。

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use fv_common::json::write_design;
use fv_fit::render::render_shapes;
use fv_gpu::GpuEvaluator;
use fv_seg::pipeline::{
    append_suffix, default_prefix, init_gpu_evaluator, matte_from_png, resolve_bg, run_pipeline,
    run_with_params, tune_search, PipelineResult,
};
use fv_seg::{plan_layers, segment_by, Matte, PlanParams, SegModel};

#[derive(Debug, Parser)]
#[command(name = "fvseg", about = "语义分层拟合：图像+matte → 三层 vinyl 形状（geometrize JSON）")]
struct Args {
    /// 输入图片（PNG/JPEG/BMP）
    input: PathBuf,

    /// 现成 matte（8 位灰度 PNG）；缺省则按 --model 推理（尺寸不符自动缩放）
    #[arg(long)]
    matte: Option<PathBuf>,

    /// 分割模型（--matte 缺省时生效）：modnet=人像 matting，u2net=通用显著性
    #[arg(long, default_value = "modnet")]
    model: String,

    /// 三层总形状预算（背景+主体+边界）
    #[arg(long, default_value_t = 1500)]
    total: usize,

    /// 每形状候选采样数（边界层自动 ×1.5）
    #[arg(long, default_value_t = 20000)]
    samples: usize,

    /// 随机种子
    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// 背景色：auto（干净背景像素均值）或 RRGGBB 十六进制
    #[arg(long, default_value = "auto")]
    bg: String,

    /// 软椭圆（228 混合）终段层数；0 = 关闭
    #[arg(long, default_value_t = 300)]
    soft: usize,

    /// 用 GPU（wgpu/Vulkan compute）批量评估热路径；初始化失败自动回退 CPU
    #[arg(long)]
    gpu: bool,

    /// 边界层改用画家式描边（M1.7：沿边界带等值线放实心小椭圆，风格化）
    #[arg(long)]
    strokes: bool,

    /// L1.3-lite：对分层规划参数（PlanParams）做搜索，目标 = 管线终评 PSNR。
    /// v3（默认）：连续化坐标下降 + 顶点求解 + 早停，≤40 次评估（标准档典型 ~33）；
    /// v2：固定网格两轮坐标下降（--tune-algo v2）。建议小预算跑搜索、最优参数再放大
    #[arg(long)]
    tune: bool,

    /// tune 算法：v3 = 连续化+早停（默认）；v2 = 旧固定网格坐标下降（A/B 回退）
    #[arg(long, default_value = "v3")]
    tune_algo: String,

    /// tune v3 坐标下降轮数（默认 2；round≥1 步长减半，E1 门控 subject_*）
    #[arg(long, default_value_t = 2)]
    tune_rounds: usize,

    /// tune v3 快速档：单轮、只探 band_radius+thresh_bias+edge_frac（≤12 次评估）
    #[arg(long)]
    tune_fast: bool,

    /// 直接给定 PlanParams 六参数（thresh_bias,smooth_rounds,band_radius,
    /// edge_frac,subject_gain,subject_floor）——单跑一次（C0 评估通道）
    #[arg(long, value_delimiter = ',')]
    plan_params: Vec<f64>,

    /// 输出前缀（默认 = 输入文件去扩展名）
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("fvseg: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.total == 0 {
        return Err("--total 须 ≥ 1".into());
    }
    if args.samples == 0 {
        return Err("--samples 须 ≥ 1".into());
    }
    let img = image::open(&args.input)
        .map_err(|e| format!("打开输入图 {}: {e}", args.input.display()))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err("输入图尺寸为 0".into());
    }

    // ---- matte：--matte 直读灰度 PNG，否则按 --model 推理（缺模型 → 指引 + 退出码 2）----
    let matte = match &args.matte {
        Some(p) => matte_from_png(p, w, h)?,
        None => {
            let model = match args.model.to_ascii_lowercase().as_str() {
                "modnet" => SegModel::Modnet,
                "u2net" => SegModel::U2Net,
                other => {
                    return Err(format!("未知模型 \"{other}\"（可选 modnet / u2net）").into());
                }
            };
            match segment_by(&args.input, model) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("fvseg: {} 推理不可用：{e}", args.model);
                    eprintln!("指引：把对应 onnx 放到 <repo>/deps/models/（modnet.onnx / u2net.onnx，或设 FV_SEG_MODEL / FV_SEG_U2NET），");
                    eprintln!("      把 onnxruntime.dll 放到 <repo>/deps/ort/（或设 FV_SEG_ORT_DYLIB）；");
                    eprintln!("      或改用 --matte <matte.png> 直接给定掩码绕过模型。");
                    std::process::exit(2);
                }
            }
        }
    };

    let plan = plan_layers(&matte, args.total)?;
    if plan.w != w || plan.h != h {
        return Err(format!(
            "matte 尺寸 {}x{} 与输入图 {w}x{h} 不一致",
            plan.w, plan.h
        )
        .into());
    }
    let bg = resolve_bg(&img, &plan, &args.bg)?;
    eprintln!(
        "plan: {w}x{h} subject={}px bg={}px band={}px | budget: bg={} subject={} edge={} (total {})",
        plan.subject_area(),
        plan.background_area(),
        plan.band_area(),
        plan.budget.background,
        plan.budget.subject,
        plan.budget.edge,
        args.total
    );
    eprintln!("bg color: {:02X}{:02X}{:02X}", bg[0], bg[1], bg[2]);

    // ---- 评估后端：--gpu 预初始化（探测 Vulkan 设备；失败 → 警告 + 回退 CPU）----
    let mut gpu_store: Option<GpuEvaluator> = if args.gpu {
        match init_gpu_evaluator(w, h) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("fvseg: GPU 评估器初始化失败（{e}），回退 CPU 路径");
                None
            }
        }
    } else {
        None
    };
    eprintln!(
        "eval backend: {}",
        if gpu_store.is_some() { "GPU (wgpu/Vulkan)" } else { "CPU (rayon)" }
    );

    let save = |result: &fv_seg::pipeline::PipelineResult, suffix: &str| -> Result<(), String> {
        let prefix = args.out.clone().unwrap_or_else(|| default_prefix(&args.input));
        let json_path = append_suffix(&prefix, suffix);
        let preview_path = append_suffix(&prefix, &suffix.replace(".json", ".preview.png"));
        write_design(&json_path, &result.shapes)
            .map_err(|e| format!("写 {}: {e}", json_path.display()))?;
        render_shapes(w, h, &result.shapes, bg)
            .save(&preview_path)
            .map_err(|e| format!("写 {}: {e}", preview_path.display()))?;
        println!("json:    {}", json_path.display());
        println!("preview: {}", preview_path.display());
        Ok(())
    };

    // ---- C0 评估通道：--plan-params 直通单跑 ----
    if !args.plan_params.is_empty() {
        if args.plan_params.len() != 6 {
            return Err("--plan-params 需要恰好 6 个值".into());
        }
        let v = &args.plan_params;
        let p = PlanParams {
            thresh_bias: v[0] as i32,
            smooth_rounds: v[1] as u32,
            band_radius: v[2] as usize,
            edge_frac: v[3],
            subject_gain: v[4],
            subject_floor: v[5],
        };
        let result = run_with_params(
            &img, &matte, &p, args.total, args.samples, args.seed, bg, args.soft, args.strokes,
            &mut gpu_store,
        )?;
        eprintln!(
            "plan-params: {:?} psnr={:.3} shapes={}",
            p,
            result.psnr,
            result.shapes.len()
        );
        save(&result, ".plan.json")?;
        return Ok(());
    }

    // ---- L1.3-lite：--tune 规划参数搜索（v3 连续化+早停；v2 网格回退对照）----
    if args.tune {
        if args.tune_rounds == 0 {
            return Err("--tune-rounds 须 ≥ 1".into());
        }
        let algo = args.tune_algo.to_ascii_lowercase();
        let tuned = match algo.as_str() {
            "v3" => tune_v3(
                &img, &matte, args.total, args.samples, args.seed, bg, args.soft, args.strokes,
                args.tune_rounds, args.tune_fast, &mut gpu_store,
            ),
            "v2" => tune_search(
                &img, &matte, args.total, args.samples, args.seed, bg, args.soft, args.strokes,
                &mut gpu_store,
            ),
            other => return Err(format!("未知 --tune-algo \"{other}\"（可选 v2 / v3）").into()),
        };
        let (best, _best_p, _secs) = tuned?;
        save(&best, ".tuned.json")?;
        return Ok(());
    }

    // ---- 默认：单跑默认参数 ----
    let t_all = Instant::now();
    let result = run_pipeline(
        &img,
        &plan,
        args.samples,
        args.seed,
        bg,
        args.soft,
        args.strokes,
        &mut gpu_store,
    )?;
    for st in &result.stats {
        eprintln!(
            "  layer {:<13} budget={:<5} solid={:<5} soft={:<4} {:>5.1}s  mse={:.1}",
            st.name, st.budget, st.solid, st.soft, st.secs, st.mse
        );
    }
    let psnr = if result.psnr.is_infinite() {
        "inf".to_string()
    } else {
        format!("{:.2} dB", result.psnr)
    };
    eprintln!(
        "total: {} shapes, wall {:.1}s, PSNR {psnr}",
        result.shapes.len(),
        t_all.elapsed().as_secs_f64()
    );
    save(&result, ".json")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// tune v3（TUNE_V3_DESIGN §D）：连续化坐标下降 + 三点抛物线顶点 + 早停
// ---------------------------------------------------------------------------

/// E2 预算硬顶：每图总评估数（含 baseline）。
const EVAL_CAP: usize = 40;
/// DIMS 中 subject_gain / subject_floor 的下标（E1 记账位，§D.2）。
const IDX_GAIN: usize = 4;
const IDX_FLOOR: usize = 5;
/// 快速档（--tune-fast）只探的维（§D.3 快速档：≤12 次评估）。
const FAST_DIMS: [&str; 3] = ["band_radius", "thresh_bias", "edge_frac"];

/// 维的处理方式（§A.1/§A.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// 纯网格：两轮都用固定 probes0（thresh_bias；极致档细分不在本实现范围）。
    Grid,
    /// 网格 + 邻值：round-0 用 probes0，round≥1 用 incumbent ±1（smooth_rounds）。
    Neighbors,
    /// 网格 + 顶点：round-0 用 probes0（v2 网格保覆盖）+ 拉格朗日顶点；
    /// round≥1 用 incumbent ±d1·0.5^(r-1) + 顶点（edge_frac）。
    GridVertex,
    /// 连续：±δ 探针（round-0 δ=d0，之后减半）+ 顶点（subject_gain/floor）。
    Cont,
    /// 连续取整：同 Cont，但顶点 round 成整数再评估（band_radius，§E-R1）。
    SnapCont,
}

/// 单维规格：get/set 把 f64 潜变量映射到 PlanParams（整数维在 set 内取整）。
struct DimSpec {
    name: &'static str,
    kind: Kind,
    get: fn(&PlanParams) -> f64,
    set: fn(&mut PlanParams, f64),
    /// round-0 步长（Cont/SnapCont；round≥1 自动减半，§D.2）。
    d0: f64,
    /// round-1 步长（GridVertex：§A.3 明示 edge_frac round-1 = incumbent ±0.10）。
    d1: f64,
    /// Grid/Neighbors/GridVertex 的 round-0 固定探针。
    probes0: &'static [f64],
    lo: f64,
    hi: f64,
}

/// 维顺序 = §A.4：band 提前（胜率×增益最高、与阈值/掩码独立），掩码侧先于预算侧。
const DIMS: [DimSpec; 6] = [
    DimSpec {
        name: "band_radius",
        kind: Kind::SnapCont,
        get: |p: &PlanParams| p.band_radius as f64,
        set: |p: &mut PlanParams, v| p.band_radius = v.round().clamp(0.0, 12.0) as usize,
        d0: 2.0,
        d1: 1.0,
        probes0: &[],
        lo: 0.0,
        hi: 12.0,
    },
    DimSpec {
        name: "thresh_bias",
        kind: Kind::Grid,
        get: |p: &PlanParams| p.thresh_bias as f64,
        set: |p: &mut PlanParams, v| p.thresh_bias = v.round().clamp(-64.0, 64.0) as i32,
        d0: 0.0,
        d1: 0.0,
        probes0: &[-32.0, -16.0, 16.0, 32.0],
        lo: -64.0,
        hi: 64.0,
    },
    DimSpec {
        name: "smooth_rounds",
        kind: Kind::Neighbors,
        get: |p: &PlanParams| p.smooth_rounds as f64,
        set: |p: &mut PlanParams, v| p.smooth_rounds = v.round().clamp(0.0, 3.0) as u32,
        d0: 0.0,
        d1: 1.0,
        probes0: &[0.0, 2.0, 3.0],
        lo: 0.0,
        hi: 3.0,
    },
    DimSpec {
        name: "edge_frac",
        kind: Kind::GridVertex,
        get: |p: &PlanParams| p.edge_frac,
        set: |p: &mut PlanParams, v| p.edge_frac = v.clamp(0.02, 0.60),
        d0: 0.0,
        d1: 0.10,
        probes0: &[0.10, 0.35, 0.45],
        lo: 0.02,
        hi: 0.60,
    },
    DimSpec {
        name: "subject_gain",
        kind: Kind::Cont,
        get: |p: &PlanParams| p.subject_gain,
        set: |p: &mut PlanParams, v| p.subject_gain = v.clamp(0.20, 0.90),
        d0: 0.15,
        d1: 0.075,
        probes0: &[],
        lo: 0.20,
        hi: 0.90,
    },
    DimSpec {
        name: "subject_floor",
        kind: Kind::Cont,
        get: |p: &PlanParams| p.subject_floor,
        set: |p: &mut PlanParams, v| p.subject_floor = v.clamp(0.10, 0.50),
        d0: 0.10,
        d1: 0.05,
        probes0: &[],
        lo: 0.10,
        hi: 0.50,
    },
];

/// E5 评估缓存 + E1 记账 + 平台标记（§D.2 TuneCtx）。
#[derive(Default)]
struct TuneCtx {
    /// 六参位模式键（f64 用位模式，禁止容差合并——§E-R7）。
    cache: HashSet<[i64; 6]>,
    /// round-0 各维是否胜出（E1：gain/floor 第一轮均无胜出 → 后续轮全跳）。
    won_r0: [bool; 6],
    /// §A.2-4 平台/惰性维（三点严格相等），后续轮次跳过。
    inert: [bool; 6],
    evals: usize,
    capped: bool,
}

/// 评估通道的固定实参包（等价于 --plan-params 直通单跑的 C0 通道）。
struct TuneEnv<'a> {
    img: &'a image::RgbaImage,
    matte: &'a Matte,
    total: usize,
    samples: usize,
    seed: u64,
    bg: [u8; 3],
    soft_edge: usize,
    strokes: bool,
}

impl TuneEnv<'_> {
    fn eval(&self, p: &PlanParams, gpu: &mut Option<GpuEvaluator>) -> Result<PipelineResult, String> {
        run_with_params(
            self.img, self.matte, p, self.total, self.samples, self.seed, self.bg,
            self.soft_edge, self.strokes, gpu,
        )
    }
}

/// E5 缓存键：六参位模式（§D.2 key()）。
fn params_key(p: &PlanParams) -> [i64; 6] {
    [
        i64::from(p.thresh_bias),
        i64::from(p.smooth_rounds),
        p.band_radius as i64,
        p.edge_frac.to_bits() as i64,
        p.subject_gain.to_bits() as i64,
        p.subject_floor.to_bits() as i64,
    ]
}

/// E5+E2 门控的评估：缓存命中或额度耗尽 → None（调用方跳过该候选）。
fn eval_cached(
    ctx: &mut TuneCtx,
    env: &TuneEnv<'_>,
    p: &PlanParams,
    gpu: &mut Option<GpuEvaluator>,
) -> Result<Option<PipelineResult>, String> {
    if !ctx.cache.insert(params_key(p)) {
        return Ok(None); // 确定性评估：同参必不优于已记录 best（§B-E5）
    }
    if ctx.evals >= EVAL_CAP {
        ctx.capped = true; // §B-E2 预算硬顶
        return Ok(None);
    }
    ctx.evals += 1;
    Ok(Some(env.eval(p, gpu)?))
}

/// round r 的探针步长（Cont/SnapCont 减半；GridVertex round-1 起 §A.3 口径）。
fn round_delta(dim: &DimSpec, round: usize) -> f64 {
    match dim.kind {
        Kind::Grid => 0.0,
        Kind::Neighbors => 1.0,
        Kind::GridVertex if round == 0 => 0.0,
        Kind::GridVertex => dim.d1 * 0.5f64.powi(round as i32 - 1),
        Kind::Cont | Kind::SnapCont => dim.d0 * 0.5f64.powi(round as i32),
    }
}

/// 本轮探针集（§A.3）：Grid 维固定网格；Neighbors 维 incumbent 邻值（≤2 个）；
/// 其余 ±δ，clamp 后与 incumbent 重合的塌缩侧剔除。
fn probe_values(dim: &DimSpec, x0: f64, round: usize) -> Vec<f64> {
    let raw: Vec<f64> = match (dim.kind, round) {
        (Kind::Grid, _) | (Kind::Neighbors, 0) | (Kind::GridVertex, 0) => dim.probes0.to_vec(),
        (Kind::Neighbors, _) => vec![x0 - 1.0, x0 + 1.0],
        (Kind::GridVertex, _) | (Kind::Cont, _) | (Kind::SnapCont, _) => {
            let d = round_delta(dim, round);
            vec![x0 - d, x0 + d]
        }
    };
    raw.into_iter()
        .map(|x| x.clamp(dim.lo, dim.hi))
        .filter(|&x| (x - x0).abs() > 1e-12)
        .collect()
}

/// 三点抛物线顶点（§A.2-2）。等距走二阶差分特例（denom<0 才有内凹峰）；
/// 不等距（clamp 探针）走拉格朗日一般式并以二次项系数 a<0 复核凹性。
/// 任一点非有限（inf/NaN，§E-R4）、单调/凸/平台 → None。
fn vertex3(xm: f64, ym: f64, x0: f64, y0: f64, xp: f64, yp: f64) -> Option<f64> {
    if !(ym.is_finite() && y0.is_finite() && yp.is_finite()) {
        return None;
    }
    let (hm, hp) = (x0 - xm, xp - x0);
    if hm <= 0.0 || hp <= 0.0 {
        return None; // 需严格两侧
    }
    if (hm - hp).abs() <= 1e-9 * hm.max(1.0) {
        let denom = ym - 2.0 * y0 + yp;
        if denom >= -1e-9 {
            return None; // 单调/凸/平台 → 无内部峰（方向信息供补差轮，本档不延伸）
        }
        Some(x0 + hp * (ym - yp) / (2.0 * denom))
    } else {
        let a = ym / ((xm - x0) * (xm - xp))
            + y0 / ((x0 - xm) * (x0 - xp))
            + yp / ((xp - xm) * (xp - x0));
        if a >= -1e-12 {
            return None;
        }
        let num =
            ym * (x0 * x0 - xp * xp) + y0 * (xp * xp - xm * xm) + yp * (xm * xm - x0 * x0);
        let den = 2.0 * (ym * (x0 - xp) + y0 * (xp - xm) + yp * (xm - x0));
        let xs = num / den;
        xs.is_finite().then_some(xs)
    }
}

/// 取 incumbent 两侧最近的已评估点（顶点三点中的 x−/x+；缺任一侧 → None）。
fn bracket3(pts: &[(f64, f64)], x0: f64) -> Option<((f64, f64), (f64, f64))> {
    let lo = pts
        .iter()
        .filter(|&&(x, _)| x < x0 - 1e-12)
        .max_by(|a, b| a.0.total_cmp(&b.0))?;
    let hi = pts
        .iter()
        .filter(|&&(x, _)| x > x0 + 1e-12)
        .min_by(|a, b| a.0.total_cmp(&b.0))?;
    Some((*lo, *hi))
}

/// 评估行日志：v2 正则可解析格式（tools/tune_stats.py L12-17），顶点行尾追加
/// ` (vertex)` 标记（§D.2 日志兼容约束；正则对尾随文本不敏感）。
fn log_try(name: &str, v: f64, r: &PipelineResult, won: bool, vertex: bool) {
    let mark = if vertex { " (vertex)" } else { "" };
    if won {
        eprintln!(
            "[tune] {name:<14}={v:<5.2} psnr={:.3} shapes={}  ← best{mark}",
            r.psnr,
            r.shapes.len()
        );
    } else {
        eprintln!(
            "[tune] {name:<14}={v:<5.2} psnr={:.3} shapes={}{mark}",
            r.psnr,
            r.shapes.len()
        );
    }
}

/// 单维连续化探查（§D.2 probe_dim）：双侧探针 → 顶点求解与实测验证采纳 → 平台判定。
/// 顶点预测值只决定“要不要花这 1 次评估”，采纳判据 = 实测 psnr 超当前 best（§A.2-3）。
#[allow(clippy::too_many_arguments)]
fn probe_dim(
    ctx: &mut TuneCtx,
    env: &TuneEnv<'_>,
    dim: &DimSpec,
    di: usize,
    round: usize,
    best_p: &mut PlanParams,
    best: &mut PipelineResult,
    gpu: &mut Option<GpuEvaluator>,
) -> Result<(), String> {
    if ctx.inert[di] {
        return Ok(()); // §A.2-4：平台维后续轮次跳过
    }
    let x0 = (dim.get)(best_p);
    let y0 = best.psnr;
    let mut tried = vec![x0];
    let mut pts = vec![(x0, y0)];

    // 1. 双侧探针（Grid → probes0；E5 命中即跳过）
    for x in probe_values(dim, x0, round) {
        let mut p = *best_p;
        (dim.set)(&mut p, x);
        let Some(r) = eval_cached(ctx, env, &p, gpu)? else {
            continue;
        };
        let won = r.psnr > best.psnr;
        log_try(dim.name, x, &r, won, false);
        let y = r.psnr;
        if won {
            *best = r;
            *best_p = p;
            if round == 0 {
                ctx.won_r0[di] = true; // E1 记账
            }
        }
        tried.push(x);
        pts.push((x, y));
    }

    // 2. 顶点（Grid/Neighbors 不做；SnapCont 顶点取整评估，§E-R1）
    if matches!(dim.kind, Kind::GridVertex | Kind::Cont | Kind::SnapCont)
        && let Some(((xm, ym), (xp, yp))) = bracket3(&pts, x0)
        && let Some(mut xs) = vertex3(xm, ym, x0, y0, xp, yp)
    {
        xs = xs.clamp(dim.lo, dim.hi);
        if dim.kind == Kind::SnapCont {
            xs = xs.round();
        }
        // SnapCont 顶点须移动 ≥1 整步；其余维 |x*−x0| < max(δ/10, 0.005) 无区分度
        let eps = if dim.kind == Kind::SnapCont {
            0.5
        } else {
            (round_delta(dim, round) * 0.1).max(0.005)
        };
        if !tried.iter().any(|&t| (t - xs).abs() < 1e-9) && (xs - x0).abs() >= eps {
            let mut p = *best_p;
            (dim.set)(&mut p, xs);
            if let Some(r) = eval_cached(ctx, env, &p, gpu)? {
                let won = r.psnr > best.psnr;
                log_try(dim.name, xs, &r, won, true);
                let y = r.psnr;
                if won {
                    *best = r;
                    *best_p = p;
                    if round == 0 {
                        ctx.won_r0[di] = true;
                    }
                }
                pts.push((xs, y));
            }
        }
    }

    // 3. 平台/惰性判定：两侧都探过且全部点与 incumbent 严格相等（确定性评估）→ 惰性
    let has_lo = pts.iter().any(|&(x, _)| x < x0 - 1e-12);
    let has_hi = pts.iter().any(|&(x, _)| x > x0 + 1e-12);
    if has_lo && has_hi && pts.iter().all(|&(_, y)| y == y0) {
        ctx.inert[di] = true;
    }
    Ok(())
}

/// `--tune` v3 主流程（§D.2）：baseline → round-0 全六维 → round≥1 减半+E1 门控，
/// E2 硬顶 40 评估；E3 inf 基线即出。日志行格式与 v2 逐字节兼容（tune_stats.py）。
#[allow(clippy::too_many_arguments)]
fn tune_v3(
    img: &image::RgbaImage,
    matte: &Matte,
    total: usize,
    samples: usize,
    seed: u64,
    bg: [u8; 3],
    soft_edge: usize,
    strokes: bool,
    rounds: usize,
    fast: bool,
    gpu: &mut Option<GpuEvaluator>,
) -> Result<(PipelineResult, PlanParams, f64), String> {
    let t0 = Instant::now();
    let env = TuneEnv { img, matte, total, samples, seed, bg, soft_edge, strokes };
    let mut ctx = TuneCtx::default();
    let mut best_p = PlanParams::default();
    let mut best = eval_cached(&mut ctx, &env, &best_p, gpu)?
        .expect("baseline 是首个评估，必为缓存未命中");
    eprintln!(
        "[tune] baseline          psnr={:.3} shapes={}",
        best.psnr,
        best.shapes.len()
    );

    let done_line = |ctx: &TuneCtx, best: &PipelineResult, best_p: &PlanParams, secs: f64| {
        eprintln!(
            "[tune] done in {secs:.1}s: best psnr={:.3} shapes={} evals={} params={best_p:?}",
            best.psnr,
            best.shapes.len(),
            ctx.evals
        );
    };

    // E3：inf 基线 = 完美重构，不可被有限值超过 → 立即返回默认参数（§B）
    if best.psnr.is_infinite() {
        eprintln!("[tune] baseline inf → 完美重构，跳过搜索");
        let secs = t0.elapsed().as_secs_f64();
        done_line(&ctx, &best, &best_p, secs);
        return Ok((best, best_p, secs));
    }

    // 快速档强制单轮（§D.3）；round 循环内 E2 触顶即断
    let rounds = if fast { 1 } else { rounds };
    for round in 0..rounds {
        eprintln!("[tune] round {round}");
        // E1：gain/floor 第一轮均无胜出 → 后续轮两者全跳（§B，实测零损失）
        let e1_skip = round > 0 && !ctx.won_r0[IDX_GAIN] && !ctx.won_r0[IDX_FLOOR];
        for (di, dim) in DIMS.iter().enumerate() {
            if fast && !FAST_DIMS.contains(&dim.name) {
                continue;
            }
            if e1_skip && (di == IDX_GAIN || di == IDX_FLOOR) {
                continue;
            }
            probe_dim(&mut ctx, &env, dim, di, round, &mut best_p, &mut best, gpu)?;
            if ctx.evals >= EVAL_CAP {
                break;
            }
        }
        if ctx.evals >= EVAL_CAP {
            break;
        }
    }
    if ctx.capped {
        eprintln!("[tune] eval budget cap {EVAL_CAP} reached, truncated");
    }

    let secs = t0.elapsed().as_secs_f64();
    done_line(&ctx, &best, &best_p, secs);
    Ok((best, best_p, secs))
}

#[cfg(test)]
mod tune_v3_tests {
    use super::*;

    #[test]
    fn dims_layout_matches_a4_and_e1_indices() {
        assert_eq!(DIMS[0].name, "band_radius");
        assert_eq!(DIMS[1].name, "thresh_bias");
        assert_eq!(DIMS[2].name, "smooth_rounds");
        assert_eq!(DIMS[3].name, "edge_frac");
        assert_eq!(DIMS[IDX_GAIN].name, "subject_gain");
        assert_eq!(DIMS[IDX_FLOOR].name, "subject_floor");
    }

    #[test]
    fn vertex3_equidistant_and_lagrange() {
        // 等距内凹：(1,-1),(3,0),(5,-4) → x* = 2.4
        let v = vertex3(1.0, -1.0, 3.0, 0.0, 5.0, -4.0).expect("concave has vertex");
        assert!((v - 2.4).abs() < 1e-9);
        // 不等距（clamp 探针）：(0.10,0),(0.25,0),(0.35,-1) → 抛物线 k(x-0.1)(x-0.25),
        // k=-40 → x* = 0.175（拉格朗日一般式路径）
        let v = vertex3(0.10, 0.0, 0.25, 0.0, 0.35, -1.0).expect("asymmetric concave");
        assert!((v - 0.175).abs() < 1e-9);
    }

    #[test]
    fn vertex3_rejects_convex_monotone_flat_and_non_finite() {
        assert_eq!(vertex3(1.0, 1.0, 3.0, 0.0, 5.0, 1.0), None); // 凸
        assert_eq!(vertex3(1.0, 1.0, 3.0, 2.0, 5.0, 4.0), None); // 单调
        assert_eq!(vertex3(1.0, 2.0, 3.0, 2.0, 5.0, 2.0), None); // 平台
        assert_eq!(vertex3(1.0, f64::INFINITY, 3.0, 2.0, 5.0, 1.0), None);
        assert_eq!(vertex3(1.0, f64::NAN, 3.0, 2.0, 5.0, 1.0), None);
    }

    #[test]
    fn bracket3_picks_nearest_sides() {
        let pts = vec![(0.25, 1.0), (0.10, 2.0), (0.35, 3.0), (0.45, 4.0)];
        let ((xm, ym), (xp, yp)) = bracket3(&pts, 0.25).expect("both sides present");
        assert_eq!((xm, ym), (0.10, 2.0));
        assert_eq!((xp, yp), (0.35, 3.0));
        let one_side = vec![(3.0, 1.0), (5.0, 2.0)];
        assert_eq!(bracket3(&one_side, 3.0), None);
    }

    #[test]
    fn probe_values_round0_match_a3_table() {
        // round-0 探针与 §A.3 逐点对照（incumbent = PlanParams::default()）
        let d = PlanParams::default();
        for dim in &DIMS {
            let pv = probe_values(dim, (dim.get)(&d), 0);
            let expect: &[f64] = match dim.name {
                "band_radius" => &[1.0, 5.0],
                "thresh_bias" => &[-32.0, -16.0, 16.0, 32.0],
                "smooth_rounds" => &[0.0, 2.0, 3.0],
                "edge_frac" => &[0.10, 0.35, 0.45],
                "subject_gain" => &[0.30, 0.60],
                "subject_floor" => &[0.20, 0.40],
                other => panic!("unknown dim {other}"),
            };
            assert_eq!(pv.len(), expect.len(), "dim {}", dim.name);
            for (a, b) in pv.iter().zip(expect) {
                assert!((a - b).abs() < 1e-9, "dim {}: {a} vs {b}", dim.name);
            }
        }
    }

    #[test]
    fn probe_values_round1_and_boundary_clamp() {
        let band = &DIMS[0];
        assert_eq!(probe_values(band, 5.0, 1), vec![4.0, 6.0]); // r±1
        assert_eq!(probe_values(band, 0.0, 0), vec![2.0]); // 下界塌缩侧剔除
        assert_eq!(probe_values(band, 12.0, 0), vec![10.0]); // 上界同
        let smooth = &DIMS[2];
        assert_eq!(probe_values(smooth, 0.0, 1), vec![1.0]); // 邻值下界截断
        assert_eq!(probe_values(smooth, 3.0, 1), vec![2.0]);
        let near = |pv: Vec<f64>, expect: &[f64]| {
            assert_eq!(pv.len(), expect.len());
            for (a, b) in pv.into_iter().zip(expect) {
                assert!((a - b).abs() < 1e-9, "{a} vs {b}");
            }
        };
        let edge = &DIMS[3];
        near(probe_values(edge, 0.35, 1), &[0.25, 0.45]); // ±0.10
        let gain = &DIMS[IDX_GAIN];
        near(probe_values(gain, 0.45, 1), &[0.375, 0.525]); // ±0.075
    }

    #[test]
    fn params_key_distinguishes_all_six_fields() {
        let base = PlanParams::default();
        let k0 = params_key(&base);
        let variants = [
            PlanParams { thresh_bias: 1, ..base },
            PlanParams { smooth_rounds: 2, ..base },
            PlanParams { band_radius: 4, ..base },
            PlanParams { edge_frac: base.edge_frac + 1e-9, ..base },
            PlanParams { subject_gain: base.subject_gain + 1e-9, ..base },
            PlanParams { subject_floor: base.subject_floor + 1e-9, ..base },
        ];
        for (i, v) in variants.iter().enumerate() {
            assert_ne!(params_key(v), k0, "field {i} 应改变键（位模式，无容差合并）");
        }
        assert_eq!(params_key(&base), params_key(&base));
    }

    /// 96px 合成图冒烟 + v3 确定性（同 tests/tune_determinism.rs 的合成口径）：
    /// CPU 路径跑通、PSNR 有限、双跑同参同 PSNR（E5 前提 = 评估确定性）。
    #[test]
    fn tune_v3_smoke_deterministic() {
        let (w, h) = (96u32, 96u32);
        let mut img = image::RgbaImage::new(w, h);
        let mut matte = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let t = f64::from(x) / f64::from(w);
                let (dx, dy) = (f64::from(x) - 48.0, f64::from(y) - 48.0);
                let d2 = dx * dx + dy * dy;
                let mut px = [40.0 + 160.0 * t, 60.0 + 120.0 * t, 100.0];
                if d2 <= 30.0 * 30.0 {
                    px = [200.0 + 40.0 * (d2.sqrt() / 30.0).sin(), 90.0, 60.0];
                    matte[(y * w + x) as usize] = 255;
                }
                img.put_pixel(x, y, image::Rgba([px[0] as u8, px[1] as u8, px[2] as u8, 255]));
            }
        }
        let matte = Matte { width: w, height: h, data: matte };
        let bg = [30u8, 60, 90];

        let mut g1 = None;
        let (r1, p1, _) = tune_v3(&img, &matte, 12, 120, 7, bg, 2, false, 1, false, &mut g1)
            .expect("tune v3 run 1");
        let mut g2 = None;
        let (r2, p2, _) = tune_v3(&img, &matte, 12, 120, 7, bg, 2, false, 1, false, &mut g2)
            .expect("tune v3 run 2");

        assert!(r1.psnr.is_finite(), "PSNR 应有限");
        assert_eq!(p1, p2, "两次搜索应收敛到同一参数");
        assert_eq!(r1.psnr.to_bits(), r2.psnr.to_bits(), "PSNR 逐位一致");
    }
}
