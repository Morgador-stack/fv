//! fvfit-gpu：GPU 批量评估版拟合 CLI（参数与 fv-fit 完全一致，走 Vulkan compute）。
//! 需要支持 SHADER_INT64 的适配器（NVIDIA/AMD/Intel 近年驱动均支持；N1X 实测通过）。
//!
//! 另含 M1.8 可微精修后处理模式：`--refine-json X.json --input img.png`，
//! 读入纯实心（type 16）结果 JSON，做联合梯度精修（Adam + τ 退火 + 硬光栅收口），
//! 写回 `{out}.refined.json` / `{out}.refined.preview.png`。
//! 注意 --opt-res 必须与原拟合一致（形状坐标按其缩放）。

use clap::Parser;
use fv_fit::cli::{self, Args as FitArgs};
use fv_gpu::diff::refine::{diff_refine, RefineConfig};
use fv_gpu::diff::ShapeParams;

#[derive(Debug, Parser)]
#[command(name = "fvfit-gpu", about = "GPU 拟合 + M1.8 可微精修 CLI")]
struct Cli {
    #[command(flatten)]
    fit: FitArgs,

    /// 精修模式：既有 geometrize JSON 路径（全部 type 16 实心层）
    #[arg(long)]
    refine_json: Option<std::path::PathBuf>,

    /// 精修迭代数（0 = 只评测不精修）
    #[arg(long, default_value_t = 50)]
    diff_iters: usize,

    /// 几何学习率（px/步）
    #[arg(long, default_value_t = 0.35)]
    diff_lr_geo: f32,

    /// 颜色学习率（255 标度/步）
    #[arg(long, default_value_t = 1.2)]
    diff_lr_col: f32,

    /// τ 起始值（覆盖带宽）
    #[arg(long, default_value_t = 4.0)]
    diff_tau0: f32,

    /// τ 每步增长率
    #[arg(long, default_value_t = 1.3)]
    diff_tau_growth: f32,

    /// τ 上限
    #[arg(long, default_value_t = 192.0)]
    diff_tau_max: f32,
}

fn main() {
    let cli = Cli::parse();
    let code = if cli.refine_json.is_some() {
        run_refine(&cli)
    } else {
        let mut gpu = fv_gpu::GpuEvaluator::default();
        match cli::run_with(cli.fit, Some(&mut gpu)) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("error: {e}");
                1
            }
        }
    };
    std::process::exit(code);
}

/// 精修模式：JSON（原图坐标）→ 工作分辨率联合梯度精修 → 写回原图坐标。
/// 角度约定：引擎光栅 R(−θ) 与 diff 内核 R(+θ)，边界处 θ 取负号互换。
fn run_refine(cli: &Cli) -> i32 {
    use fv_common::json;

    let json_path = cli.refine_json.as_ref().unwrap();
    let result = (|| -> Result<(), String> {
        let img = image::open(&cli.fit.input).map_err(|e| format!("open: {e}"))?.to_rgba8();
        let bg = cli::resolve_bg(&img, &cli.fit.bg)?;
        let design =
            json::read_design(json_path).map_err(|e| format!("read {}: {e}", json_path.display()))?;
        let shapes = cli::shapes_from_design(&design, usize::MAX, json_path)?;

        let (ow, oh) = (img.width(), img.height());
        let long_edge = ow.max(oh);
        let scale = (cli.fit.opt_res as f32 / long_edge as f32).min(1.0);
        let w = (((ow as f32) * scale) as u32).max(4);
        let h = (((oh as f32) * scale) as u32).max(4);
        let small =
            image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);

        // 目标平面：alpha<128 → bg（与引擎/裁判同规则），f32 0-255
        let mut target = vec![0f32; (w * h * 3) as usize];
        for (i, px) in small.pixels().enumerate() {
            let c = if px.0[3] < 128 {
                [f32::from(bg[0]), f32::from(bg[1]), f32::from(bg[2])]
            } else {
                [f32::from(px.0[0]), f32::from(px.0[1]), f32::from(px.0[2])]
            };
            target[3 * i..3 * i + 3].copy_from_slice(&c);
        }

        // 原图坐标 → 工作坐标；引擎角度（deg，R(−θ)）→ diff 角度（rad，R(+θ)）
        let to_params = |s: &fv_common::shape::Shape| ShapeParams {
            cx: s.cx * scale,
            cy: s.cy * scale,
            a: s.rx * scale,
            b: s.ry * scale,
            theta: -s.angle_deg.to_radians(),
            r: f32::from(s.color[0]),
            g: f32::from(s.color[1]),
            bl: f32::from(s.color[2]),
        };
        let n_solid = shapes.len();
        eprintln!(
            "refine: {n_solid} 层（全部 type 16 实心），工作分辨率 {w}x{h}（scale={scale:.4}）"
        );
        let params: Vec<ShapeParams> = shapes.iter().map(to_params).collect();

        let cfg = RefineConfig {
            iters: cli.diff_iters,
            lr_geo: cli.diff_lr_geo,
            lr_col: cli.diff_lr_col,
            tau0: cli.diff_tau0,
            tau_growth: cli.diff_tau_growth,
            tau_max: cli.diff_tau_max,
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let (refined, report) =
            diff_refine(params, w, h, [f32::from(bg[0]), f32::from(bg[1]), f32::from(bg[2])], &target, &cfg)?;
        eprintln!(
            "refine: {:.1}s psnr {:.3} → {:.3} dB（best@iter {}，accepted={}），trail {:?}",
            t0.elapsed().as_secs_f64(),
            report.psnr_start,
            report.psnr_best,
            report.best_iter,
            report.accepted,
            report.trail.iter().map(|(k, p)| (k, (p * 1000.0).round() / 1000.0)).collect::<Vec<_>>()
        );

        // 工作坐标 → 原图坐标；角度换回 deg 规范化 [0,360)
        let inv = 1.0 / scale;
        let out: Vec<fv_common::shape::Shape> = refined
            .iter()
            .map(|p| {
                fv_common::shape::Shape::ellipse(
                    p.cx * inv,
                    p.cy * inv,
                    p.a * inv,
                    p.b * inv,
                    (-p.theta).to_degrees().rem_euclid(360.0),
                    [p.r.round().clamp(0.0, 255.0) as u8,
                     p.g.round().clamp(0.0, 255.0) as u8,
                     p.bl.round().clamp(0.0, 255.0) as u8],
                )
            })
            .collect();

        let out_prefix = cli.fit.out_prefix.clone().unwrap_or_else(|| {
            json_path.parent().unwrap_or(std::path::Path::new(".")).join(
                json_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "refined".into()),
            )
        });
        let json_out = out_prefix.with_extension("refined.json");
        let png_out = out_prefix.with_extension("refined.preview.png");
        json::write_design(&json_out, &out).map_err(|e| format!("write json: {e}"))?;
        let preview = fv_fit::render::render_shapes(ow, oh, &out, bg);
        preview.save(&png_out).map_err(|e| format!("write png: {e}"))?;
        println!("json:    {}", json_out.display());
        println!("preview: {}", png_out.display());
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
