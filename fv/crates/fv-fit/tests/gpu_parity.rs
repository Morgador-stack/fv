//! GPU≡CPU 回归锁（STUDY §排查链：GPU 候选流同源 / 复核顺序 / accept 顺序逐位敏感）。
//!
//! [`CpuMirrorEvaluator`] 用引擎自身的 CPU 精确评分（`engine::evaluate` /
//! `soft::eval_soft`）逐位复刻 GPU 后端的对外契约（on_ready/on_accept 缓存同步 +
//! 整批评估）。因此 `fit_gpu` 与 `fit` 的候选流、评分、接纳决策必须完全一致：
//! 任一处分叉（候选 RNG 流不同源、GPU 复核 top-k/兜底顺序被改、accept 记账顺序
//! 被改）都会让 mse/progress 序列/形状参数在噪声图上立即失配——本文件即失败。

use fv_fit::engine::{self, BatchEvaluator, EvalResults, FitConfig};
use fv_fit::raster::EllipseParams;
use fv_fit::rowsum::RowSums;
use fv_fit::soft;

/// 用 CPU 精确评分镜像 GPU 批评估后端（语义等价后端，无真实 GPU 依赖）。
struct CpuMirrorEvaluator {
    sums: RowSums,
    target: Vec<u8>,
    canvas: Vec<u8>,
    w: u32,
    h: u32,
}

impl CpuMirrorEvaluator {
    fn new() -> Self {
        Self {
            sums: RowSums::new(4, 4),
            target: Vec::new(),
            canvas: Vec::new(),
            w: 4,
            h: 4,
        }
    }

    /// 与引擎约定一致：on_ready/on_accept 时画布已是最新，按其重放缓存。
    fn sync(&mut self, target: &[u8], canvas: &[u8], w: u32, h: u32) {
        let mut sums = RowSums::new(w, h);
        sums.set_target(target);
        sums.refresh_canvas(canvas);
        self.sums = sums;
        self.target = target.to_vec();
        self.canvas = canvas.to_vec();
        self.w = w;
        self.h = h;
    }
}

impl BatchEvaluator for CpuMirrorEvaluator {
    fn on_ready(&mut self, _sums: &RowSums, target: &[u8], canvas: &[u8], w: u32, h: u32) {
        self.sync(target, canvas, w, h);
    }
    fn on_accept(&mut self, _sums: &RowSums, target: &[u8], canvas: &[u8], w: u32, h: u32) {
        self.sync(target, canvas, w, h);
    }
    fn eval_batch(&mut self, cands: &[EllipseParams], w: u32, h: u32) -> EvalResults {
        assert_eq!((w, h), (self.w, self.h), "引擎分辨率与后端缓存不一致");
        cands.iter()
            .map(|e| engine::evaluate(&self.sums, e, w, h))
            .collect()
    }
    fn eval_soft_batch(&mut self, cands: &[EllipseParams], w: u32, h: u32) -> EvalResults {
        assert_eq!((w, h), (self.w, self.h), "引擎分辨率与后端缓存不一致");
        cands.iter()
            .map(|e| soft::eval_soft(&self.target, &self.canvas, w, h, e))
            .collect()
    }
}

/// 确定性渐变+噪声图（双向渐变 + LCG 噪声，避免平滑图过早收敛掩盖分叉）。
fn gradient_noise_image(w: u32, h: u32) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(w, h);
    let mut noise = 0x5EEDu32;
    let mut nz = || {
        noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
        ((noise >> 28) as i32) - 8
    };
    for y in 0..h {
        for x in 0..w {
            let (t, s) = (x as f32 / w as f32, y as f32 / h as f32);
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    ((t * 255.0) as i32 + nz()).clamp(0, 255) as u8,
                    ((s * 200.0) as i32 + nz()).clamp(0, 255) as u8,
                    (((t + s) * 0.5 * 255.0) as i32 + nz()).clamp(0, 255) as u8,
                    255,
                ]),
            );
        }
    }
    img
}

/// 形状序列逐位比较（f32 按 to_bits；含 kind 与 RGBA 色）。
fn assert_same_shapes(a: &[fv_common::shape::Shape], b: &[fv_common::shape::Shape]) {
    assert_eq!(a.len(), b.len(), "层数分叉: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.kind, y.kind, "shape[{i}] kind");
        assert_eq!(x.cx.to_bits(), y.cx.to_bits(), "shape[{i}] cx");
        assert_eq!(x.cy.to_bits(), y.cy.to_bits(), "shape[{i}] cy");
        assert_eq!(x.rx.to_bits(), y.rx.to_bits(), "shape[{i}] rx");
        assert_eq!(x.ry.to_bits(), y.ry.to_bits(), "shape[{i}] ry");
        assert_eq!(x.angle_deg.to_bits(), y.angle_deg.to_bits(), "shape[{i}] angle");
        assert_eq!(x.color, y.color, "shape[{i}] color");
    }
}

/// mse 序列逐位比较（progress 回调序列是接纳顺序一致性的可观测证据）。
fn assert_same_progress(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "progress 回调次数分叉");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "mse[{i}] {} vs {}", x, y);
    }
}

#[test]
fn gpu_mock_parity_solid() {
    let img = gradient_noise_image(96, 72);
    let cfg = FitConfig {
        layers: 60,
        samples_per_shape: 2000,
        opt_max_dim: 192,
        seed: 123,
        refine_iters: 16,
        checkpoints: vec![],
        out_prefix: None,
        bg: [0, 0, 0],
        polish_sweeps: 0,
        soft_layers: 0,
        soft_refit: false,
        auto_soft: false,
        structured: true,
    };

    let mut cpu_mses = Vec::new();
    let cpu = engine::fit(&img, &cfg, &mut |_, m| cpu_mses.push(m));

    let mut gpu_backend = CpuMirrorEvaluator::new();
    let mut gpu_mses = Vec::new();
    let gpu = engine::fit_gpu(&img, &cfg, &mut gpu_backend, &mut |_, m| gpu_mses.push(m));

    assert_same_progress(&cpu_mses, &gpu_mses);
    assert_same_shapes(&cpu.shapes, &gpu.shapes);
    assert_eq!(
        cpu.final_mse.to_bits(),
        gpu.final_mse.to_bits(),
        "final_mse 分叉: {} vs {}",
        cpu.final_mse,
        gpu.final_mse
    );
}

#[test]
fn gpu_mock_parity_polish_and_soft() {
    // polish + soft + refit 全通道：GPU 的 on_ready/on_accept 与网格刷新时序
    // 在三个阶段间被逐位复刻，任一同步遗漏都会让软通道对着 stale 画布预测。
    let img = gradient_noise_image(80, 60);
    let cfg = FitConfig {
        layers: 40,
        samples_per_shape: 1200,
        opt_max_dim: 160,
        seed: 0x50F7,
        refine_iters: 12,
        checkpoints: vec![],
        out_prefix: None,
        bg: [0, 0, 0],
        polish_sweeps: 2,
        soft_layers: 10,
        soft_refit: true,
        auto_soft: false,
        structured: true,
    };

    let mut cpu_mses = Vec::new();
    let cpu = engine::fit(&img, &cfg, &mut |_, m| cpu_mses.push(m));

    let mut gpu_backend = CpuMirrorEvaluator::new();
    let mut gpu_mses = Vec::new();
    let gpu = engine::fit_gpu(&img, &cfg, &mut gpu_backend, &mut |_, m| gpu_mses.push(m));

    assert_same_progress(&cpu_mses, &gpu_mses);
    assert_same_shapes(&cpu.shapes, &gpu.shapes);
    assert_eq!(cpu.final_mse.to_bits(), gpu.final_mse.to_bits());
    // 该配置必须实际产生软层（通道确实被覆盖）
    let soft_count = cpu
        .shapes
        .iter()
        .filter(|s| s.kind == fv_common::shape::ShapeKind::SoftEllipse)
        .count();
    assert!(soft_count > 0, "软通道未生效，parity 覆盖失效");
}

#[test]
fn gpu_mock_parity_auto_soft() {
    // auto_soft：实心三振后剩余预算给软层——早停路径 + 软预算换算的 GPU/CPU 一致性
    let img = gradient_noise_image(80, 60);
    let cfg = FitConfig {
        layers: 50,
        samples_per_shape: 1200,
        opt_max_dim: 160,
        seed: 42,
        refine_iters: 12,
        checkpoints: vec![],
        out_prefix: None,
        bg: [0, 0, 0],
        polish_sweeps: 0,
        soft_layers: 0,
        soft_refit: true,
        auto_soft: true,
        structured: true,
    };

    let mut cpu_mses = Vec::new();
    let cpu = engine::fit(&img, &cfg, &mut |_, m| cpu_mses.push(m));

    let mut gpu_backend = CpuMirrorEvaluator::new();
    let mut gpu_mses = Vec::new();
    let gpu = engine::fit_gpu(&img, &cfg, &mut gpu_backend, &mut |_, m| gpu_mses.push(m));

    assert_same_progress(&cpu_mses, &gpu_mses);
    assert_same_shapes(&cpu.shapes, &gpu.shapes);
    assert_eq!(cpu.final_mse.to_bits(), gpu.final_mse.to_bits());
}
