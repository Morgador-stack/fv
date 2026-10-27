//! fvseg CLI：语义分层拟合（matte → 三层编排 → geometrize JSON）。
//! 编排/tune/plan-params 逻辑在 `fv_seg::pipeline`（库，可测试）；本文件只做
//! 参数解析、matte 获取与产物落盘。

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use fv_common::json::write_design;
use fv_fit::render::render_shapes;
use fv_gpu::GpuEvaluator;
use fv_seg::pipeline::{
    append_suffix, default_prefix, init_gpu_evaluator, matte_from_png, resolve_bg, run_pipeline,
    run_with_params, tune_search,
};
use fv_seg::{plan_layers, segment_by, PlanParams, SegModel};

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

    /// L1.3-lite：对分层规划参数（PlanParams）做坐标搜索，目标 = 管线终评
    /// PSNR（扩网格 + 两轮，~32 次评估；建议小预算跑搜索、最优参数再放大）
    #[arg(long)]
    tune: bool,

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

    // ---- L1.3-lite：--tune 规划参数坐标搜索 ----
    if args.tune {
        let (best, best_p, _secs) = tune_search(
            &img, &matte, args.total, args.samples, args.seed, bg, args.soft, args.strokes,
            &mut gpu_store,
        )?;
        let _ = best_p;
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
