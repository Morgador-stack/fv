//! 贪心拟合主循环（引擎）。M1：多线程候选评分 + 爬山精修 + 增量画布 + checkpoint 落盘。
//!
//! 评分核心（O(高度)，见 rowsum 模块文档）：
//! ```text
//! delta = cur_err(R) − new_err(R, 最优均值色)，整数 i64 全程精确、可复现
//! ```

use crate::candidate::{self, CandidateSpec, Rng};
use crate::raster::{self, EllipseParams};
use crate::rowsum::{MapId, RowSums};
use fv_common::shape::{Shape, ShapeKind};
use rayon::prelude::*;

/// 批量评估结果：与候选等长，None = 无效/无正收益。
pub type EvalResults = Vec<Option<([u8; 3], i64)>>;

/// 批量候选评估后端（由 GPU 实现，如 fv-gpu 的 GpuEvaluator）。
/// 引擎在 CPU 采样、GPU 整批评估、CPU 精修之间分工。
pub trait BatchEvaluator {
    /// 后端是否健康可用（on_ready 失败/运行中出错后为 false——引擎据此回退 CPU，
    /// 而不是把全 None 喂给三振早停静默产出空设计）。
    fn available(&self) -> bool {
        true
    }
    /// 画布/前缀和就绪后调用一次（w/h 与 fit 工作分辨率一致）。
    /// target/canvas 为 3 平面 SoA u8（软椭圆评估需要逐像素值）。
    fn on_ready(&mut self, sums: &RowSums, target: &[u8], canvas: &[u8], w: u32, h: u32);
    /// 每次接纳后调用（画布已更新，实现需重传其缓存）。
    fn on_accept(&mut self, sums: &RowSums, target: &[u8], canvas: &[u8], w: u32, h: u32);
    /// 批量评估实心候选；返回与 cands 等长，None = 候选无效（空椭圆等）。
    fn eval_batch(&mut self, cands: &[EllipseParams], w: u32, h: u32) -> EvalResults;
    /// 批量评估软椭圆候选（228 语义，opacity=1）；None = 无正收益。
    fn eval_soft_batch(&mut self, cands: &[EllipseParams], w: u32, h: u32) -> EvalResults;
}

/// 拟合配置（算法参数 + 输出控制）。
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// 目标形状数
    pub layers: usize,
    /// 每个形状的随机候选采样数
    pub samples_per_shape: usize,
    /// 拟合工作分辨率（长边上限）
    pub opt_max_dim: u32,
    /// 随机种子
    pub seed: u64,
    /// 最优候选的爬山精修轮数
    pub refine_iters: usize,
    /// checkpoint 层数（升序；在完成该层后落盘 JSON+预览）
    pub checkpoints: Vec<usize>,
    /// 输出前缀：Some(p) → 写 `{p}.{n}.json` / `{p}.{n}.preview.png`，终稿不带层数后缀
    pub out_prefix: Option<std::path::PathBuf>,
    /// 背景色（画布初始色 = 透明区目标色）
    pub bg: [u8; 3],
    /// 遮挡感知精确抛光的扫描轮数（0 = 关闭）
    pub polish_sweeps: usize,
    /// 软椭圆（228 混合语义）透明层数：实心+抛光之后追加，攻击纹理残差
    pub soft_layers: usize,
    /// 软栈颜色重拟合（默认开）
    pub soft_refit: bool,
    /// 自适应预算：soft_layers==0 时生效——实心三振出局后，剩余层预算全部给软层
    ///（此时 layers 语义 = 总预算）
    pub auto_soft: bool,
    /// B1 结构导引采样（对数半轴 + Pareto 长宽比 + Sobel 走向；照片类靠
    /// 50/50 混采原 spec 保护。关闭 = 纯 sqrt-面积均匀，旧行为）
    pub structured: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            layers: 1000,
            samples_per_shape: 20000,
            opt_max_dim: 448,
            seed: 0xC0FFEE,
            refine_iters: 24,
            checkpoints: Vec::new(),
            out_prefix: None,
            bg: [0, 0, 0],
            polish_sweeps: 0,
            soft_layers: 0,
            soft_refit: true,
            auto_soft: false,
            structured: true,
        }
    }
}

/// 拟合结果：形状按 z 序自底向上，坐标为**原图像素**。
pub struct FitResult {
    pub shapes: Vec<Shape>,
    pub final_mse: f64,
}

#[derive(Clone)]
struct Candidate {
    e: EllipseParams,
    color: [u8; 3],
    delta: i64,
}

/// 粗粒度误差网格：块误差前缀和（CDF 反演采样用）。
/// 把候选中心按残差分布加权——贪心版"重定位"，解决均匀采样在纹理残差上的平台期。
struct ErrorGrid {
    bw: usize,
    bh: usize,
    block: usize,
    cum: Vec<u64>,
    total: u64,
}

impl ErrorGrid {
    fn new(w: u32, h: u32, block: usize) -> Self {
        let bw = (w as usize).div_ceil(block);
        let bh = (h as usize).div_ceil(block);
        Self {
            bw,
            bh,
            block,
            cum: vec![0; bw * bh],
            total: 0,
        }
    }

    /// 由 target/canvas 全量重算块误差前缀和（O(像素数)，每层接纳后一次，开销可忽略）。
    fn rebuild(&mut self, target: &[u8], canvas: &[u8], w: u32, h: u32) {
        let n = (w * h) as usize;
        let mut sums = vec![0u64; self.bw * self.bh];
        let (w, h) = (w as usize, h as usize);
        for y in 0..h {
            let by = y / self.block;
            for x in 0..w {
                let bx = x / self.block;
                let p = y * w + x;
                let mut e = 0u64;
                for ch in 0..3 {
                    let d = target[ch * n + p] as i64 - canvas[ch * n + p] as i64;
                    e += (d * d) as u64;
                }
                sums[by * self.bw + bx] += e;
            }
        }
        let mut acc = 0u64;
        for v in sums.iter_mut() {
            acc += *v;
            *v = acc;
        }
        self.cum = sums;
        self.total = acc;
    }

    /// 按块误差加权采一个像素中心（total==0 时退化为均匀）。
    fn sample_center(&self, rng: &mut Rng, w: u32, h: u32) -> (f32, f32) {
        if self.total == 0 {
            return (rng.next_f32() * w as f32, rng.next_f32() * h as f32);
        }
        let r = rng.next_f32() as f64 * self.total as f64;
        let idx = self
            .cum
            .partition_point(|&c| (c as f64) < r)
            .min(self.cum.len() - 1);
        let bx = idx % self.bw;
        let by = idx / self.bw;
        let b = self.block as f32;
        (
            bx as f32 * b + rng.next_f32() * b,
            by as f32 * b + rng.next_f32() * b,
        )
    }
}

/// 全序比较（delta 大者优先；平手按参数字典序）——保证 rayon reduce 结果与调度顺序无关。
fn better(a: &Option<Candidate>, b: &Option<Candidate>) -> Option<Candidate> {
    match (a, b) {
        (None, x) | (x, None) => x.clone(),
        (Some(a), Some(b)) => {
            let a_wins = a.delta > b.delta
                || (a.delta == b.delta
                    && (a.e.cx, a.e.cy, a.e.rx, a.e.ry, a.e.angle_rad)
                        < (b.e.cx, b.e.cy, b.e.rx, b.e.ry, b.e.angle_rad));
            Some(if a_wins { a.clone() } else { b.clone() })
        }
    }
}

/// 评估候选：返回最优均值色与误差下降量 delta（>0 才值得接纳）。
/// 公开供基准测试与调试工具使用。
pub fn evaluate(sums: &RowSums, e: &EllipseParams, w: u32, h: u32) -> Option<([u8; 3], i64)> {
    let spans = raster::ellipse_rows(e, w, h);
    if spans.is_empty() {
        return None;
    }
    let area = raster::spans_pixel_count(&spans);
    if area == 0 {
        return None;
    }
    let area = area as i64;

    let mut t = [0i64; 3];
    let mut t2 = [0i64; 3];
    let mut tc = [0i64; 3];
    let mut c2 = [0i64; 3];
    for s in &spans {
        let (y, x0, x1) = (s.y, s.x0, s.x1);
        for ch in 0..3 {
            t[ch] += sums.seg(MapId::T, ch, y, x0, x1);
            t2[ch] += sums.seg(MapId::T2, ch, y, x0, x1);
            tc[ch] += sums.seg(MapId::TC, ch, y, x0, x1);
            c2[ch] += sums.seg(MapId::C2, ch, y, x0, x1);
        }
    }

    let mut color = [0u8; 3];
    let mut delta = 0i64;
    for ch in 0..3 {
        // 最优整数色 = 四舍五入均值
        let mean = ((t[ch] + area / 2) / area).clamp(0, 255);
        color[ch] = mean as u8;
        let new_err = t2[ch] - 2 * mean * t[ch] + area * mean * mean;
        let cur_err = t2[ch] - 2 * tc[ch] + c2[ch];
        delta += cur_err - new_err;
    }
    Some((color, delta))
}

/// 对当前最优候选做确定性爬山精修。
fn refine(
    sums: &RowSums,
    w: u32,
    h: u32,
    mut cand: Candidate,
    rng: &mut Rng,
    iters: usize,
) -> Candidate {
    let long_edge = w.max(h) as f32;
    let mut sigma_pos = long_edge * 0.02;
    let mut sigma_rad = 0.18f32;
    let mut sigma_ang = 10.0f32; // 度

    for _ in 0..iters {
        let mut improved = false;
        for _ in 0..6 {
            let mut e2 = cand.e;
            let pick = rng.next_u32() % 6;
            let sign = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            match pick {
                0 => e2.cx += sign * sigma_pos,
                1 => e2.cy += sign * sigma_pos,
                2 => e2.rx = (e2.rx * (1.0 + sign * sigma_rad)).max(0.5),
                3 => e2.ry = (e2.ry * (1.0 + sign * sigma_rad)).max(0.5),
                4 => e2.angle_rad += sign * sigma_ang.to_radians(),
                _ => {
                    e2.rx = (e2.rx * (1.0 + sign * sigma_rad)).max(0.5);
                    e2.ry = (e2.ry * (1.0 - sign * sigma_rad)).max(0.5);
                }
            }
            if let Some((color, delta)) = evaluate(sums, &e2, w, h)
                && (delta > cand.delta
                    || (delta == cand.delta
                        && (e2.cx, e2.cy, e2.rx, e2.ry, e2.angle_rad)
                            < (cand.e.cx, cand.e.cy, cand.e.rx, cand.e.ry, cand.e.angle_rad)))
                {
                    cand = Candidate { e: e2, color, delta };
                    improved = true;
                    break;
                }
        }
        if !improved {
            sigma_pos *= 0.6;
            sigma_rad *= 0.6;
            sigma_ang *= 0.6;
        }
    }
    cand
}

/// 主入口。`progress(done_layers, mse)` 初始回调一次 (0, mse0)，此后每完成一层回调。
pub fn fit(
    target_rgba: &image::RgbaImage,
    cfg: &FitConfig,
    progress: &mut dyn FnMut(usize, f64),
) -> FitResult {
    fit_impl(target_rgba, cfg, &[], None, progress)
}

/// GPU 批量评估版 `fit`。
pub fn fit_gpu(
    target_rgba: &image::RgbaImage,
    cfg: &FitConfig,
    gpu: &mut dyn BatchEvaluator,
    progress: &mut dyn FnMut(usize, f64),
) -> FitResult {
    fit_impl(target_rgba, cfg, &[], Some(gpu), progress)
}

/// 从已有形状（原图坐标，z 序自底向上）续拟合：先把它们缩放到 opt 分辨率、
/// 合成到画布、刷新前缀和，再继续贪心到 cfg.layers 层。
///
/// 契约（违反即 panic，带清晰消息）：
/// - `initial` 只接受旋转椭圆 `ShapeKind::RotatedEllipse`（JSON type==16，即本引擎
///   自己的输出）；矩形/软椭圆等其他类型一律拒绝。
/// - `initial.len() ≤ cfg.layers`。
/// - 初始总误差由画布**逐像素重算**，不信任旧 JSON 携带的任何统计值。
/// - 输出前 `initial.len()` 个形状**逐位保留**输入值（不经过缩放往返浮点误差）；
///   progress 的 done 与 checkpoints 均按绝对层数（含初始形状）计。
pub fn resume(
    target_rgba: &image::RgbaImage,
    cfg: &FitConfig,
    initial: &[Shape],
    progress: &mut dyn FnMut(usize, f64),
) -> FitResult {
    for (i, s) in initial.iter().enumerate() {
        assert!(
            s.kind == ShapeKind::RotatedEllipse,
            "resume: initial[{i}] kind={:?} 不支持（只接受 RotatedEllipse / JSON type==16）",
            s.kind
        );
    }
    assert!(
        initial.len() <= cfg.layers,
        "resume: initial.len()={} 超过 cfg.layers={}",
        initial.len(),
        cfg.layers
    );
    fit_impl(target_rgba, cfg, initial, None, progress)
}

/// GPU 批量评估版 `resume`。
pub fn resume_gpu(
    target_rgba: &image::RgbaImage,
    cfg: &FitConfig,
    initial: &[Shape],
    gpu: &mut dyn BatchEvaluator,
    progress: &mut dyn FnMut(usize, f64),
) -> FitResult {
    fit_impl(target_rgba, cfg, initial, Some(gpu), progress)
}

/// fit/resume 共用主体：`initial` 为原图坐标的初始形状（fit 传空切片）。
fn fit_impl(
    target_rgba: &image::RgbaImage,
    cfg: &FitConfig,
    initial: &[Shape],
    mut gpu: Option<&mut dyn BatchEvaluator>,
    progress: &mut dyn FnMut(usize, f64),
) -> FitResult {
    assert!(cfg.layers > 0 && cfg.samples_per_shape > 0);

    // ---- 1. 缩放到工作分辨率（保纵横比） ----
    let (ow, oh) = target_rgba.dimensions();
    let long_edge = ow.max(oh).max(1);
    let scale = (cfg.opt_max_dim as f32 / long_edge as f32).min(1.0);
    let (w, h) = (
        ((ow as f32 * scale) as u32).max(4),
        ((oh as f32 * scale) as u32).max(4),
    );
    let small = image::imageops::resize(target_rgba, w, h, image::imageops::FilterType::Triangle);

    // ---- 2. 目标/画布 3 平面 SoA（半透明区按背景色处理） ----
    let n = (w as usize) * (h as usize);
    let mut target = vec![0u8; n * 3];
    let mut canvas = vec![0u8; n * 3];
    for ch in 0..3 {
        canvas[ch * n..(ch + 1) * n].fill(cfg.bg[ch]);
    }
    let raw = small.as_raw();
    for p in 0..n {
        let px = &raw[p * 4..p * 4 + 4];
        let c = if px[3] < 128 { cfg.bg } else { [px[0], px[1], px[2]] };
        target[p] = c[0];
        target[n + p] = c[1];
        target[2 * n + p] = c[2];
    }

    // ---- 2b. resume 初始形状：原图坐标 → opt 坐标（×scale，即 to_orig 的 ×inv_scale 逆变换），
    //      依 z 序自底向上合成到画布 ----
    let base = initial.len();
    let initial_opt: Vec<(EllipseParams, [u8; 3])> = initial
        .iter()
        .map(|s| {
            (
                EllipseParams {
                    cx: s.cx * scale,
                    cy: s.cy * scale,
                    rx: s.rx * scale,
                    ry: s.ry * scale,
                    angle_rad: s.angle_deg.to_radians(),
                },
                [s.color[0], s.color[1], s.color[2]],
            )
        })
        .collect();
    for (e, c) in &initial_opt {
        raster::composite_ellipse(&mut canvas, w, h, e, *c);
    }

    // ---- 3. 行前缀和 ----
    let mut sums = RowSums::new(w, h);
    sums.set_target(&target);
    sums.refresh_canvas(&canvas);

    // 初始总误差：逐像素重算（resume 时画布已含初始形状，不能信旧值；
    // fit 时画布即背景，与原 Σ(t−bg)² 写法等价）
    let mut total_err: i64 = (0..3 * n)
        .map(|i| {
            let d = target[i] as i64 - canvas[i] as i64;
            d * d
        })
        .sum();
    let denom = (3 * n) as f64;
    progress(0, total_err as f64 / denom);

    // ---- 4. 贪心主循环 ----
    let spec = CandidateSpec {
        min_radius: 1.2,
        max_radius: (w.max(h) as f32) * 0.32,
    };
    // B1 结构导引场（一次预处理；采样时 thin/orig 各 50%）
    let guide = crate::orient::build_guide(&target, cfg.bg, w, h, &spec);
    let threads = rayon::current_num_threads().clamp(1, 32);
    let per_thread = cfg.samples_per_shape.div_ceil(threads);

    // 全量形状账本：shapes_opt 含 resume 初始层（缩放到 opt）+ 新接纳层；
    // srcs[i] = 初始层在 initial 中的下标（输出保真用），u32::MAX = 本轮新增。
    let mut shapes_opt: Vec<(EllipseParams, [u8; 3])> = initial_opt.clone();
    let mut srcs: Vec<u32> = (0..base as u32).collect();
    let mut changed: Vec<bool> = vec![false; base];
    let inv_scale = 1.0 / scale;
    let mut grid = ErrorGrid::new(w, h, 16);
    let mut empty_streak = 0u32;

    // 输出转换：未改动的初始层逐位原样返回（不经缩放往返）；其余 opt → 原图坐标。
    fn to_full(
        initial: &[Shape],
        initial_opt: &[(EllipseParams, [u8; 3])],
        shapes_opt: &[(EllipseParams, [u8; 3])],
        srcs: &[u32],
        changed: &[bool],
        inv_scale: f32,
    ) -> Vec<Shape> {
        shapes_opt
            .iter()
            .enumerate()
            .map(|(i, (e, c))| {
                let src = srcs[i];
                if src != u32::MAX
                    && !changed[i]
                    && initial_opt
                        .get(src as usize)
                        .is_some_and(|(e0, c0)| {
                            e0.cx.to_bits() == e.cx.to_bits()
                                && e0.cy.to_bits() == e.cy.to_bits()
                                && e0.rx.to_bits() == e.rx.to_bits()
                                && e0.ry.to_bits() == e.ry.to_bits()
                                && e0.angle_rad.to_bits() == e.angle_rad.to_bits()
                                && c0 == c
                        })
                {
                    return initial[src as usize];
                }
                Shape::ellipse(
                    e.cx * inv_scale,
                    e.cy * inv_scale,
                    e.rx * inv_scale,
                    e.ry * inv_scale,
                    e.angle_rad.to_degrees(),
                    *c,
                )
            })
            .collect()
    }

    let write_outputs = |shapes_opt: &[(EllipseParams, [u8; 3])],
                         srcs: &[u32],
                         changed: &[bool],
                         soft: &[(EllipseParams, [u8; 3])],
                         suffix: &str| {
        if let Some(prefix) = &cfg.out_prefix {
            let mut shapes =
                to_full(initial, &initial_opt, shapes_opt, srcs, changed, inv_scale);
            for (e, c) in soft {
                shapes.push(Shape {
                    kind: ShapeKind::SoftEllipse,
                    cx: e.cx * inv_scale,
                    cy: e.cy * inv_scale,
                    rx: e.rx * inv_scale,
                    ry: e.ry * inv_scale,
                    angle_deg: e.angle_rad.to_degrees(),
                    color: [c[0], c[1], c[2], 255],
                });
            }
            let mut jp = prefix.clone().into_os_string();
            jp.push(suffix);
            jp.push(".json");
            let mut pp = prefix.clone().into_os_string();
            pp.push(suffix);
            pp.push(".preview.png");
            let jp: std::path::PathBuf = jp.into();
            let pp: std::path::PathBuf = pp.into();
            if let Err(e) = fv_common::json::write_design(&jp, &shapes) {
                eprintln!("checkpoint json 写入失败: {e}");
            }
            if let Err(e) = crate::render::render_shapes(ow, oh, &shapes, cfg.bg).save(&pp) {
                eprintln!("checkpoint 预览写入失败: {e}");
            }
        }
    };

    for layer in 0..cfg.layers {
        // 绝对层数（含 resume 初始层）达到 cfg.layers 即收工
        if shapes_opt.len() >= cfg.layers {
            break;
        }
        // 误差网格：块误差 CDF（每层接纳后重建；首层用初始 bg 画布）
        if layer == 0 {
            grid.rebuild(&target, &canvas, w, h);
            if let Some(g) = gpu.as_deref_mut() {
                g.on_ready(&sums, &target, &canvas, w, h);
            }
        }
        // 采样评分：GPU 批量路径 或 CPU 多线程路径（reduce 用全序保证确定性）
        // GPU 不健康（无 Vulkan/运行中断）→ 走 CPU 路径（真回退，不再静默空结果）
        let gpu_ok = gpu.as_deref_mut().is_some_and(|g| g.available());
        let best: Option<Candidate> = if let (true, Some(g)) = (gpu_ok, gpu.as_deref_mut()) {
            // 候选集与 CPU 路径逐位同源：复刻 CPU 的分块 RNG 方案（chunk 顺序拼接、
            // 同 seed 混合、同 grid 加权中心）。此前 GPU 用单流独立采样，候选分布
            // 本身就与 CPU 不同——fvseg demo 上表现为 1dB 级质量差的主因。
            let mut cands = Vec::with_capacity(cfg.samples_per_shape);
            for chunk in 0..threads {
                let mut rng = Rng::new(
                    cfg.seed
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        ^ ((layer as u64) << 32)
                        ^ (chunk as u64).wrapping_mul(0x100000001B3),
                );
                for _ in 0..per_thread {
                    let (cx, cy) = if rng.next_u32() & 3 == 0 {
                        (rng.next_f32() * w as f32, rng.next_f32() * h as f32)
                    } else {
                        grid.sample_center(&mut rng, w, h)
                    };
                    cands.push(if cfg.structured {
                        candidate::sample_structured(&mut rng, &guide, cx, cy)
                    } else {
                        candidate::sample_with_center(&mut rng, &spec, cx, cy)
                    });
                }
            }
            let results = g.eval_batch(&cands, w, h);
            // GPU i64 评分与 CPU 在边带有 0.7% ±1 分歧，深饱和/近tie 处放大贪心分叉。
            // 取 GPU 报告 top-k 用 CPU 精确评分复核（~4µs/层，相对 GPU 批评估可忽略）；
            // 全负批复核最接近的 1 个，防 ±1 误差触发假三振早停。
            const VERIFY_K: usize = 4;
            let mut ranked: Vec<(usize, i64)> = results
                .iter()
                .enumerate()
                .filter_map(|(i, r)| r.map(|(_, d)| (i, d)))
                .collect();
            ranked.sort_by_key(|&(_, d)| std::cmp::Reverse(d));
            let mut best: Option<Candidate> = None;
            for (i, _) in ranked.iter().take_while(|&&(_, d)| d > 0).take(VERIFY_K) {
                let e = cands[*i];
                if let Some((color, delta)) = evaluate(&sums, &e, w, h)
                    && delta > 0
                {
                    best = better(&Some(Candidate { e, color, delta }), &best);
                }
            }
            // 全负批：复核 GPU 认为最接近的 1 个，防 ±1 误差触发假三振早停
            if best.is_none()
                && let Some(&(i, _)) = ranked.first()
            {
                let e = cands[i];
                if let Some((color, delta)) = evaluate(&sums, &e, w, h)
                    && delta > 0
                {
                    best = Some(Candidate { e, color, delta });
                }
            }
            best
        } else {
            (0..threads)
                .into_par_iter()
                .map(|chunk| {
                    let mut rng = Rng::new(
                        cfg.seed
                            .wrapping_mul(0x9E3779B97F4A7C15)
                            ^ ((layer as u64) << 32)
                            ^ (chunk as u64).wrapping_mul(0x100000001B3),
                    );
                    let mut local: Option<Candidate> = None;
                    for _ in 0..per_thread {
                        // 75% 误差加权中心 + 25% 均匀探索（网格有滞后性）
                        let (cx, cy) = if rng.next_u32() & 3 == 0 {
                            (rng.next_f32() * w as f32, rng.next_f32() * h as f32)
                        } else {
                            grid.sample_center(&mut rng, w, h)
                        };
                        let e = if cfg.structured {
                            candidate::sample_structured(&mut rng, &guide, cx, cy)
                        } else {
                            candidate::sample_with_center(&mut rng, &spec, cx, cy)
                        };
                        if let Some((color, delta)) = evaluate(&sums, &e, w, h)
                            && delta > 0
                            {
                                let c = Candidate { e, color, delta };
                                local = match local {
                                    None => Some(c),
                                    Some(prev) => {
                                        
                                        better(&Some(c.clone()), &Some(prev))
                                    }
                                };
                            }
                    }
                    local
                })
                .reduce(|| None, |a, b| {
                    
                    better(&a, &b)
                })
        };

        let best = match best {
            Some(b) if b.delta > 0 => b,
            _ => {
                // 三振出局：连续 3 层采样池无改进才终止（误差网格随接纳刷新）
                empty_streak += 1;
                if empty_streak >= 3 {
                    break;
                }
                continue;
            }
        };

        // 爬山精修
        let mut rrng = Rng::new(cfg.seed ^ ((layer as u64) + 1).wrapping_mul(0x5EED5) ^ 0xABCD);
        let best = refine(&sums, w, h, best, &mut rrng, cfg.refine_iters);

        // 接纳：合成 + 刷新画布侧前缀和 + 重建误差网格 + 通知 GPU 后端
        raster::composite_ellipse(&mut canvas, w, h, &best.e, best.color);
        sums.refresh_canvas(&canvas);
        grid.rebuild(&target, &canvas, w, h);
        if let Some(g) = gpu.as_deref_mut() {
            g.on_accept(&sums, &target, &canvas, w, h);
        }
        total_err -= best.delta;
        empty_streak = 0;
        shapes_opt.push((best.e, best.color));
        srcs.push(u32::MAX);
        changed.push(false);

        let done = shapes_opt.len();
        progress(done, total_err as f64 / denom);

        if cfg.checkpoints.contains(&done) {
            write_outputs(&shapes_opt, &srcs, &changed, &[], &format!(".{done}"));
        }
    }

    // ---- 5. 遮挡感知精确抛光（可选）：逐形状在可见集上重拟合 + 删无效层 ----
    if cfg.polish_sweeps > 0 {
        let (rep, survivors) = crate::polish::polish_opt(
            &target,
            w,
            h,
            &mut shapes_opt,
            cfg.bg,
            cfg.polish_sweeps,
            cfg.refine_iters,
            cfg.seed,
        );
        // 同步血统/改动账本，并重合成画布精确重算总误差
        srcs.truncate(shapes_opt.len());
        changed.truncate(shapes_opt.len());
        for (i, s) in survivors.iter().enumerate() {
            srcs[i] = s.src;
            changed[i] = s.changed;
        }
        for ch in 0..3 {
            canvas[ch * n..(ch + 1) * n].fill(cfg.bg[ch]);
        }
        for (e, c) in &shapes_opt {
            raster::composite_ellipse(&mut canvas, w, h, e, *c);
        }
        total_err = 0i64;
        for p in 0..n * 3 {
            let d = target[p] as i64 - canvas[p] as i64;
            total_err += d * d;
        }
        eprintln!(
            "polish: {} sweeps, {} accepted, {} deleted, mse {:.2} -> {:.2}",
            rep.sweeps_run, rep.accepted, rep.deleted, rep.mse_before, rep.mse_after
        );
        // 画布已重算：同步前缀和与 GPU 缓存（否则后续软通道对着 stale 画布预测）
        sums.refresh_canvas(&canvas);
        if let Some(g) = gpu.as_deref_mut() {
            g.on_accept(&sums, &target, &canvas, w, h);
        }
        progress(shapes_opt.len(), total_err as f64 / denom);
    }

    // ---- 6. 软椭圆（228）透明层通道：混合语义攻击纹理残差 ----
    // 实心覆盖在纹理区是罚分（平色盖高频必涨误差）；软层把像素往均值拉（降方差）。
    // 吸纳前先快照 bbox 试混，实际 delta ≤ 0 则回滚（u8 舍入防虚账）。
    // M1.5 自适应预算：soft_layers==0 且 auto_soft 时，实心三振出局后的剩余层预算全给软层
    //（此时 cfg.layers 语义 = 总预算）。
    let soft_budget = if cfg.soft_layers > 0 {
        cfg.soft_layers
    } else if cfg.auto_soft {
        cfg.layers.saturating_sub(shapes_opt.len())
    } else {
        0
    };
    let mut soft_out: Vec<(EllipseParams, [u8; 3])> = Vec::new();
    let solid_canvas = if soft_budget > 0 {
        canvas.clone()
    } else {
        Vec::new()
    };
    if soft_budget > 0 {
        // resume 层数已满（layers==base）等路径会跳过主循环的 grid 重建/on_ready，
        // polish 后 grid 也可能陈旧——软相入口无条件刷新一次（幂等，开销可忽略）
        grid.rebuild(&target, &canvas, w, h);
        if let Some(g) = gpu.as_deref_mut() {
            g.on_ready(&sums, &target, &canvas, w, h);
        }
        let mut soft_streak = 0u32;
        for _iter in 0..soft_budget {
            if soft_streak >= 3 {
                break;
            }
            let gpu_ok = gpu.as_deref_mut().is_some_and(|g| g.available());
            let best: Option<(EllipseParams, [u8; 3])> = if let (true, Some(g)) = (gpu_ok, gpu.as_deref_mut()) {
                let mut rng = Rng::new(cfg.seed ^ 0x50F7 ^ ((soft_out.len() as u64) << 32));
                let mut cands = Vec::with_capacity(cfg.samples_per_shape);
                for _ in 0..cfg.samples_per_shape {
                    let (cx, cy) = if rng.next_u32() & 3 == 0 {
                        (rng.next_f32() * w as f32, rng.next_f32() * h as f32)
                    } else {
                        grid.sample_center(&mut rng, w, h)
                    };
                    cands.push(if cfg.structured {
                        candidate::sample_structured(&mut rng, &guide, cx, cy)
                    } else {
                        candidate::sample_with_center(&mut rng, &spec, cx, cy)
                    });
                }
                let results = g.eval_soft_batch(&cands, w, h);
                let mut best: Option<(EllipseParams, [u8; 3], i64)> = None;
                for (e, r) in cands.iter().zip(results) {
                    if let Some((color, delta)) = r
                        && best.as_ref().is_none_or(|(_, _, bd)| delta > *bd)
                    {
                        best = Some((*e, color, delta));
                    }
                }
                best.map(|(e, k, _)| (e, k))
            } else {
                let mut rng = Rng::new(cfg.seed ^ 0x50F7 ^ ((soft_out.len() as u64) << 32));
                let mut best: Option<(EllipseParams, [u8; 3], i64)> = None;
                for _ in 0..cfg.samples_per_shape {
                    let (cx, cy) = if rng.next_u32() & 3 == 0 {
                        (rng.next_f32() * w as f32, rng.next_f32() * h as f32)
                    } else {
                        grid.sample_center(&mut rng, w, h)
                    };
                    let e = if cfg.structured {
                        candidate::sample_structured(&mut rng, &guide, cx, cy)
                    } else {
                        candidate::sample_with_center(&mut rng, &spec, cx, cy)
                    };
                    if let Some((k, delta)) = crate::soft::eval_soft(&target, &canvas, w, h, &e)
                        && best.as_ref().is_none_or(|(_, _, bd)| delta > *bd)
                    {
                        best = Some((e, k, delta));
                    }
                }
                best.map(|(e, k, _)| (e, k))
            };
            let Some((e, k)) = best else {
                soft_streak += 1;
                continue;
            };
            let spans = raster::ellipse_rows(&e, w, h);
            let Some(bbox) = raster::spans_bbox(&spans) else {
                soft_streak += 1;
                continue;
            };
            // bbox 内 before 误差 + 快照
            let (bx0, by0, bx1, by1) = (bbox.0 as usize, bbox.1 as usize, bbox.2 as usize, bbox.3 as usize);
            let mut before = 0i64;
            let mut snap = Vec::with_capacity((bx1 - bx0) * (by1 - by0) * 3);
            for y in by0..by1 {
                for x in bx0..bx1 {
                    let p = y * w as usize + x;
                    for ch in 0..3 {
                        let t = target[ch * n + p] as i64;
                        let c = canvas[ch * n + p] as i64;
                        before += (t - c) * (t - c);
                        snap.push(canvas[ch * n + p]);
                    }
                }
            }
            crate::soft::blend_composite(&mut canvas, w, h, &e, k, 1.0);
            let mut after = 0i64;
            for y in by0..by1 {
                for x in bx0..bx1 {
                    let p = y * w as usize + x;
                    for ch in 0..3 {
                        let t = target[ch * n + p] as i64;
                        let c = canvas[ch * n + p] as i64;
                        after += (t - c) * (t - c);
                    }
                }
            }
            if after < before {
                total_err -= before - after;
                soft_out.push((e, k));
                soft_streak = 0;
                sums.refresh_canvas(&canvas);
                grid.rebuild(&target, &canvas, w, h);
                if let Some(g) = gpu.as_deref_mut() {
                    g.on_accept(&sums, &target, &canvas, w, h);
                }
                progress(shapes_opt.len() + soft_out.len(), total_err as f64 / denom);
            } else {
                // 回滚
                let mut si = 0usize;
                for y in by0..by1 {
                    for x in bx0..bx1 {
                        let p = y * w as usize + x;
                        for ch in 0..3 {
                            canvas[ch * n + p] = snap[si];
                            si += 1;
                        }
                    }
                }
                soft_streak += 1;
            }
        }
    }

    // ---- 7. 软栈颜色重拟合（M1.4）：坐标下降修正被后续软层改变的残差 ----
    if cfg.soft_refit && !soft_out.is_empty() {
        let (rep, final_canvas) =
            crate::soft::refit_soft_stack(&target, w, h, &solid_canvas, &mut soft_out);
        canvas.copy_from_slice(&final_canvas);
        total_err = 0i64;
        for p in 0..n * 3 {
            let d = target[p] as i64 - canvas[p] as i64;
            total_err += d * d;
        }
        sums.refresh_canvas(&canvas);
        grid.rebuild(&target, &canvas, w, h);
        if let Some(g) = gpu {
            g.on_accept(&sums, &target, &canvas, w, h);
        }
        eprintln!(
            "soft-refit: {}/{} layers refit, mse {:.2} -> {:.2}",
            rep.refits, rep.layers, rep.mse_before, rep.mse_after
        );
        progress(shapes_opt.len() + soft_out.len(), total_err as f64 / denom);
    }

    write_outputs(&shapes_opt, &srcs, &changed, &soft_out, "");
    let mut full = to_full(initial, &initial_opt, &shapes_opt, &srcs, &changed, inv_scale);
    for (e, c) in &soft_out {
        full.push(Shape {
            kind: ShapeKind::SoftEllipse,
            cx: e.cx * inv_scale,
            cy: e.cy * inv_scale,
            rx: e.rx * inv_scale,
            ry: e.ry * inv_scale,
            angle_deg: e.angle_rad.to_degrees(),
            color: [c[0], c[1], c[2], 255],
        });
    }
    FitResult {
        shapes: full,
        final_mse: total_err as f64 / denom,
    }
}
