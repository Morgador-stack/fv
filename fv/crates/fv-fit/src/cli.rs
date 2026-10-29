//! 命令行入口：图片 → {prefix}.json + {prefix}.preview.png（含 checkpoint）。

use crate::engine::{self, FitConfig};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "fv-fit", about = "图片 → Forza vinyl 形状（geometrize JSON）")]
pub struct Args {
    /// 输入图片（PNG/JPEG）
    pub input: std::path::PathBuf,

    /// 目标形状（图层）数
    #[arg(long, default_value_t = 1000)]
    pub layers: usize,

    /// 每个形状的随机候选采样数
    #[arg(long, default_value_t = 20000)]
    pub samples: usize,

    /// 拟合工作分辨率（长边上限）
    #[arg(long, default_value_t = 448)]
    pub opt_res: u32,

    /// 随机种子（可复现）
    #[arg(long, default_value_t = 0xC0FFEE)]
    pub seed: u64,

    /// checkpoint 层数，逗号分隔（如 500,1000）
    #[arg(long, value_delimiter = ',')]
    pub checkpoints: Vec<usize>,

    /// 输出前缀（默认 = 输入文件去扩展名）
    #[arg(long)]
    pub out_prefix: Option<std::path::PathBuf>,

    /// 断点续拟合：已有 geometrize JSON 路径。读入其形状（须全部 type==16 旋转椭圆、
    /// 数量 ≤ --layers）作为底部图层，在其上继续拟合到 --layers 层；
    /// 输出 JSON 的前缀与该文件逐位一致，新层追加其上
    #[arg(long)]
    pub resume: Option<std::path::PathBuf>,

    /// 背景色：auto（不透明像素均值）或 RRGGBB 十六进制
    #[arg(long, default_value = "auto")]
    pub bg: String,

    /// 遮挡感知精确抛光轮数（0 = 关闭）
    #[arg(long, default_value_t = 0)]
    pub polish: usize,

    /// 软椭圆（228 混合语义）透明层数（0 = 关闭）
    #[arg(long, default_value_t = 0)]
    pub soft: usize,

    /// 关闭软栈颜色重拟合（默认开启）
    #[arg(long, default_value_t = false)]
    pub no_soft_refit: bool,

    /// 自适应预算：实心饱和后剩余层预算自动给软层（--layers 变为总预算）
    #[arg(long, default_value_t = false)]
    pub auto_soft: bool,

    /// 关闭 B1 结构导引采样（回到纯 sqrt-面积均匀采样，旧行为）
    #[arg(long, default_value_t = false)]
    pub no_structured: bool,
}

/// 解析背景色：`auto` → 不透明像素（a≥128）逐通道均值四舍五入（无不透明像素则黑）；
/// 否则解析 6 位十六进制 RRGGBB（大小写均可）。
/// 语义真源 = `fv_common::bg::resolve`（薄壳，签名/行为/错误文案逐位不变）。
pub fn resolve_bg(img: &image::RgbaImage, spec: &str) -> Result<[u8; 3], String> {
    fv_common::bg::resolve(img, spec)
}

/// checkpoint 归一化：去重、升序、丢弃 0 与 >layers。
pub(crate) fn normalize_checkpoints(raw: &[usize], layers: usize) -> Vec<usize> {
    let mut v: Vec<usize> = raw
        .iter()
        .copied()
        .filter(|&n| n > 0 && n <= layers)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// 续拟合 JSON → 引擎初始形状。校验：全部 type==16、data 恰 5 项
/// （[cx, cy, rx, ry, angle]，原图像素）、数量 ≤ layers。
pub fn shapes_from_design(
    design: &fv_common::json::DesignJson,
    layers: usize,
    path: &std::path::Path,
) -> Result<Vec<fv_common::shape::Shape>, String> {
    if design.shapes.len() > layers {
        return Err(format!(
            "resume: {} 含 {} 个形状，超过 --layers={layers}",
            path.display(),
            design.shapes.len()
        ));
    }
    design
        .shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.kind != fv_common::shape::ShapeKind::RotatedEllipse as u32 {
                return Err(format!(
                    "resume: {} shapes[{i}] type={} 不支持（只接受 16=旋转椭圆）",
                    path.display(),
                    s.kind
                ));
            }
            if s.data.len() != 5 {
                return Err(format!(
                    "resume: {} shapes[{i}] data 长度 {} ≠ 5（type 16 应为 [cx,cy,rx,ry,angle]）",
                    path.display(),
                    s.data.len()
                ));
            }
            // is_finite 闸（CODE_REVIEW §E.5）：JSON "1e40" 经 f32 反序列化即变 inf，
            // 非有限坐标会原样穿透引擎（渲染为空、逐位保留）最后进入注入器。
            for (j, v) in s.data.iter().enumerate() {
                if !v.is_finite() {
                    return Err(format!(
                        "resume: {} shapes[{i}].data[{j}] = {v} 非有限（inf/NaN）",
                        path.display()
                    ));
                }
            }
            let [r, g, b, _a] = s.color;
            Ok(fv_common::shape::Shape::ellipse(
                s.data[0], s.data[1], s.data[2], s.data[3], s.data[4], [r, g, b],
            ))
        })
        .collect()
}

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    run_with(args, None)
}

/// `run` 的可注入评估后端版本：`gpu = Some(..)` 时走 GPU 批量评估路径。
pub fn run_with(
    args: Args,
    gpu: Option<&mut dyn engine::BatchEvaluator>,
) -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open(&args.input)?.to_rgba8();
    let bg = resolve_bg(&img, &args.bg).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let checkpoints = normalize_checkpoints(&args.checkpoints, args.layers);

    // 断点续拟合：读入既有 JSON 作为底部图层（z 序自底向上，原图坐标）
    let initial = match &args.resume {
        Some(path) => {
            let design = fv_common::json::read_design(path)
                .map_err(|e| format!("resume: 读取 {} 失败: {e}", path.display()))?;
            let shapes = shapes_from_design(&design, args.layers, path)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            eprintln!(
                "resume: 从 {} 读入 {} 个初始形状，续拟合到 {} 层",
                path.display(),
                shapes.len(),
                args.layers
            );
            shapes
        }
        None => Vec::new(),
    };

    let out_prefix = args.out_prefix.clone().unwrap_or_else(|| {
        let stem = args
            .input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out".into());
        match args.input.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.join(stem),
            _ => std::path::PathBuf::from(stem),
        }
    });

    let cfg = FitConfig {
        layers: args.layers,
        samples_per_shape: args.samples,
        opt_max_dim: args.opt_res,
        seed: args.seed,
        refine_iters: 24,
        checkpoints,
        out_prefix: Some(out_prefix.clone()),
        bg,
        polish_sweeps: args.polish,
        soft_layers: args.soft,
        soft_refit: !args.no_soft_refit,
        auto_soft: args.auto_soft,
        structured: !args.no_structured,
    };

    let layers = cfg.layers;
    let mut last_done = 0usize;
    let t0 = std::time::Instant::now();
    {
        let mut progress = |done: usize, mse: f64| {
            if done.is_multiple_of(25) || done == layers {
                // t= 累计耗时供规模基准从日志分段取数
                eprint!(
                    "\r[{done}/{layers}] mse={mse:.1} t={:.1}s   ",
                    t0.elapsed().as_secs_f64()
                );
                use std::io::Write;
                let _ = std::io::stderr().flush();
            }
            last_done = done;
        };
        let result = match (&args.resume, gpu) {
            (Some(_), Some(g)) => engine::resume_gpu(&img, &cfg, &initial, g, &mut progress),
            (Some(_), None) => engine::resume(&img, &cfg, &initial, &mut progress),
            (None, Some(g)) => engine::fit_gpu(&img, &cfg, g, &mut progress),
            (None, None) => engine::fit(&img, &cfg, &mut progress),
        };
        eprintln!();
        eprintln!("完成：{} 个形状，final mse = {:.2}", result.shapes.len(), result.final_mse);
        let _ = last_done;
    }

    println!("json:    {}.json", out_prefix.display());
    println!("preview: {}.preview.png", out_prefix.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_with(alpha: u8, rgb: [u8; 3]) -> image::RgbaImage {
        let mut img = image::RgbaImage::new(2, 2);
        for px in img.pixels_mut() {
            *px = image::Rgba([rgb[0], rgb[1], rgb[2], alpha]);
        }
        img
    }

    #[test]
    fn bg_auto_mean_rounding() {
        // 4 像素：两个不透明 (10,20,30)/(11,21,31)，两个透明 (255,255,255)
        let mut img = image::RgbaImage::new(2, 2);
        let vals = [[10, 20, 30, 255], [11, 21, 31, 255], [255, 255, 255, 0], [1, 2, 3, 10]];
        for (i, px) in img.pixels_mut().enumerate() {
            *px = image::Rgba(vals[i]);
        }
        assert_eq!(resolve_bg(&img, "auto").unwrap(), [11, 21, 31]);
    }

    #[test]
    fn bg_auto_all_transparent_is_black() {
        assert_eq!(resolve_bg(&img_with(0, [9, 9, 9]), "auto").unwrap(), [0, 0, 0]);
    }

    #[test]
    fn bg_hex_variants() {
        assert_eq!(resolve_bg(&img_with(255, [0; 3]), "FF8001").unwrap(), [255, 128, 1]);
        assert_eq!(resolve_bg(&img_with(255, [0; 3]), "#ff8001").unwrap(), [255, 128, 1]);
    }

    #[test]
    fn bg_hex_rejects_garbage() {
        for bad in ["12345", "zzz888", "", "1234567", "12 456"] {
            assert!(resolve_bg(&img_with(255, [0; 3]), bad).is_err(), "应拒绝 {bad}");
        }
    }

    #[test]
    fn checkpoints_normalized() {
        assert_eq!(normalize_checkpoints(&[1000, 500, 500, 0, 2000, 300], 1000), vec![300, 500, 1000]);
        assert_eq!(normalize_checkpoints(&[], 100), Vec::<usize>::new());
    }

    // ---- resume JSON → 初始形状 ----

    use fv_common::json::{DesignJson, ShapeJson};

    fn design(shapes: Vec<ShapeJson>) -> DesignJson {
        DesignJson { shapes }
    }

    fn sj(kind: u32, data: Vec<f32>) -> ShapeJson {
        ShapeJson { kind, data, color: [9, 8, 7, 255] }
    }

    #[test]
    fn resume_json_converts_ellipse() {
        let d = design(vec![sj(16, vec![1.5, 2.5, 3.5, 4.5, 30.0])]);
        let s = shapes_from_design(&d, 10, std::path::Path::new("x.json")).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].cx, s[0].cy, s[0].rx, s[0].ry, s[0].angle_deg),
                   (1.5, 2.5, 3.5, 4.5, 30.0));
        assert_eq!(s[0].color, [9, 8, 7, 255]);
    }

    #[test]
    fn resume_json_rejects_non_ellipse_type() {
        let d = design(vec![sj(16, vec![1.0, 2.0, 3.0, 4.0, 0.0]), sj(1, vec![1.0, 2.0, 3.0, 4.0])]);
        let err = shapes_from_design(&d, 10, std::path::Path::new("x.json")).unwrap_err();
        assert!(err.contains("type=1"), "{err}");
    }

    #[test]
    fn resume_json_rejects_bad_data_len() {
        let d = design(vec![sj(16, vec![1.0, 2.0, 3.0, 4.0])]);
        let err = shapes_from_design(&d, 10, std::path::Path::new("x.json")).unwrap_err();
        assert!(err.contains("长度 4"), "{err}");
    }

    #[test]
    fn resume_json_rejects_too_many_shapes() {
        let d = design(vec![sj(16, vec![1.0; 5]), sj(16, vec![1.0; 5])]);
        let err = shapes_from_design(&d, 1, std::path::Path::new("x.json")).unwrap_err();
        assert!(err.contains("超过 --layers=1"), "{err}");
    }

    #[test]
    fn resume_json_rejects_non_finite_data() {
        // intake 闸（CODE_REVIEW §E.5）：JSON "1e40" 经 f32 反序列化 = inf，
        // 必须在此拦下并指明 shapes[i].data[j] 位置。
        let d = design(vec![
            sj(16, vec![10.0, 20.0, 3.0, 4.0, 0.0]),
            sj(16, vec![1.0, 2.0, 1e40_f64 as f32, 4.0, 0.0]),
        ]);
        let err = shapes_from_design(&d, 10, std::path::Path::new("x.json")).unwrap_err();
        assert!(err.contains("shapes[1].data[2]"), "{err}");
        assert!(err.contains("非有限"), "{err}");
    }
}
