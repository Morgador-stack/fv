//! 贪心拟合主循环（引擎）。M1：多线程候选评分 + 爬山精修 + 增量画布 + checkpoint 落盘。
//!
//! 评分核心（O(高度)，见 rowsum 模块文档）：
//! ```text
//! delta = cur_err(R) − new_err(R, 最优均值色)，整数 i64 全程精确、可复现
//! ```
//!
//! fit_impl 按阶段拆分（原 542 行巨函数 → 编排器 + 5 个阶段函数，共享状态走
//! [`Workspace`]）：
//!
//! ```text
//! fit_impl
//! ├── prepare_workspace      缩放 + target/canvas + resume 初始合成 + 前缀和 + 初始误差
//! ├── solid_greedy_loop      实心贪心主循环（候选生成/评分/精修/接纳/checkpoint）
//! │   ├── generate_solid_candidates   GPU 路径候选集（与 CPU 分块流逐位同源）
//! │   ├── sample_and_score_chunk      CPU 路径单 chunk 采样评分（rayon map 闭包体）
//! │   └── verify_gpu_results          GPU top-k 用 CPU 精确评分复核 + 全负批兜底
//! ├── polish_phase           遮挡感知精确抛光（可选）
//! ├── soft_phase             软椭圆（228）通道 + 软栈颜色重拟合
//! │   ├── soft_candidates / select_soft_best   软候选采样与 GPU/CPU 评分
//! │   ├── soft_try_accept           bbox 快照试混，负收益回滚
//! │   └── soft_refit                软栈坐标下降重拟合
//! └── finalize_outputs       write_outputs + to_full 原图坐标组装
//! ```

use crate::candidate::{self, CandidateSpec, Rng};
use crate::orient::StructureGuide;
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
pub(crate) struct Candidate {
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

/// fit_impl 各阶段共享的工作区：opt 分辨率下的画布/前缀和/误差记账/形状账本。
///
/// 中间状态集中于此，阶段函数经 `&mut Workspace` 传递——避免巨函数与十参数函数。
/// 字段语义与拆分前 fit_impl 的局部变量一一对应，勿单独改动（多项逐位敏感：
/// GPU≡CPU 候选同源、resume 血统保真、accept 顺序、checkpoint 触发时机）。
pub(crate) struct Workspace<'a> {
    /// resume 初始形状（原图坐标；fit 传空切片）
    initial: &'a [Shape],
    /// 原图尺寸（checkpoint 预览渲染用）
    ow: u32,
    oh: u32,
    /// opt 工作分辨率
    w: u32,
    h: u32,
    /// opt → 原图的输出缩放系数（= 1/scale）
    inv_scale: f32,
    /// 像素数（w·h）与 MSE 分母（3n）
    n: usize,
    denom: f64,
    /// 目标/画布 3 平面 SoA u8（半透明区按背景色处理）
    target: Vec<u8>,
    canvas: Vec<u8>,
    /// 行前缀和（画布侧随接纳刷新）
    sums: RowSums,
    /// 块误差 CDF 网格（候选重定位采样用）
    grid: ErrorGrid,
    /// resume 初始层（opt 坐标，z 序自底向上）
    initial_opt: Vec<(EllipseParams, [u8; 3])>,
    /// 全量形状账本：初始层 + 新接纳层（z 序自底向上）
    shapes_opt: Vec<(EllipseParams, [u8; 3])>,
    /// srcs[i] = 初始层在 initial 中的下标（输出保真用），u32::MAX = 本轮新增
    srcs: Vec<u32>,
    /// 初始层是否被 polish 改动（改动即不再逐位保真输出）
    changed: Vec<bool>,
    /// 当前总误差（整数 i64 精确记账，MSE = total_err/denom）
    total_err: i64,
    /// 候选采样参数与 B1 结构导引场
    spec: CandidateSpec,
    guide: StructureGuide,
    /// 分块采样线程数与每 chunk 采样数
    threads: usize,
    per_thread: usize,
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

/// fit/resume 共用主体（编排器）：`initial` 为原图坐标的初始形状（fit 传空切片）。
fn fit_impl(
    target_rgba: &image::RgbaImage,
    cfg: &FitConfig,
    initial: &[Shape],
    mut gpu: Option<&mut dyn BatchEvaluator>,
    progress: &mut dyn FnMut(usize, f64),
) -> FitResult {
    assert!(cfg.layers > 0 && cfg.samples_per_shape > 0);

    let mut ws = prepare_workspace(target_rgba, cfg, initial);
    // progress 契约：初始回调一次 (0, mse0)——resume 时画布已含初始形状，
    // 逐像素重算的 mse0 不可信旧值；此后每完成一层回调。
    progress(0, ws.total_err as f64 / ws.denom);

    // gpu 经 &mut Option 传给各阶段（dyn 引用不变性 → 逐次 as_deref_mut 重借用）
    solid_greedy_loop(&mut ws, cfg, &mut gpu, progress);
    polish_phase(&mut ws, cfg, &mut gpu, progress);
    let soft_out = soft_phase(&mut ws, cfg, &mut gpu, progress);
    finalize_outputs(ws, cfg, &soft_out)
}

// ---- 阶段 1：工作区准备（缩放 + target/canvas 构造 + resume 初始形状合成） ----

/// 缩放到工作分辨率、构造 target/canvas 3 平面 SoA、依 z 序合成 resume 初始形状、
/// 建行前缀和并逐像素重算初始总误差。
pub(crate) fn prepare_workspace<'a>(
    target_rgba: &image::RgbaImage,
    cfg: &FitConfig,
    initial: &'a [Shape],
) -> Workspace<'a> {
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
    let total_err: i64 = (0..3 * n)
        .map(|i| {
            let d = target[i] as i64 - canvas[i] as i64;
            d * d
        })
        .sum();

    // ---- 4. 采样基础设施（分块数/每块采样数/B1 导引场） ----
    let spec = CandidateSpec {
        min_radius: 1.2,
        max_radius: (w.max(h) as f32) * 0.32,
    };
    let guide = crate::orient::build_guide(&target, cfg.bg, w, h, &spec);
    let threads = rayon::current_num_threads().clamp(1, 32);
    let per_thread = cfg.samples_per_shape.div_ceil(threads);

    Workspace {
        initial,
        ow,
        oh,
        w,
        h,
        inv_scale: 1.0 / scale,
        n,
        denom: (3 * n) as f64,
        target,
        canvas,
        sums,
        grid: ErrorGrid::new(w, h, 16),
        initial_opt: initial_opt.clone(),
        shapes_opt: initial_opt,
        srcs: (0..initial.len() as u32).collect(),
        changed: vec![false; initial.len()],
        total_err,
        spec,
        guide,
        threads,
        per_thread,
    }
}

// ---- 阶段 2：实心贪心主循环 ----

/// 实心通道第 `chunk` 流的种子派生——CPU 多线程评分与 GPU 候选生成共用，
/// 是 GPU≡CPU 候选流逐位同源的锚点，勿单独改动任一侧。
#[inline]
pub(crate) fn solid_chunk_rng(cfg: &FitConfig, layer: usize, chunk: usize) -> Rng {
    Rng::new(
        cfg.seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            ^ ((layer as u64) << 32)
            ^ (chunk as u64).wrapping_mul(0x100000001B3),
    )
}

/// 候选中心：75% 误差加权 + 25% 均匀探索（网格有滞后性）。
/// RNG 消耗顺序固定——GPU/CPU 候选流逐位同源的关键，勿改分支结构。
#[inline]
fn sample_center_mixed(rng: &mut Rng, grid: &ErrorGrid, w: u32, h: u32) -> (f32, f32) {
    if rng.next_u32() & 3 == 0 {
        (rng.next_f32() * w as f32, rng.next_f32() * h as f32)
    } else {
        grid.sample_center(rng, w, h)
    }
}

/// 按配置采一个候选椭圆（B1 结构导引 或 纯 sqrt-面积均匀）。
#[inline]
fn sample_candidate(
    rng: &mut Rng,
    structured: bool,
    guide: &StructureGuide,
    spec: &CandidateSpec,
    cx: f32,
    cy: f32,
) -> EllipseParams {
    if structured {
        candidate::sample_structured(rng, guide, cx, cy)
    } else {
        candidate::sample_with_center(rng, spec, cx, cy)
    }
}

/// GPU 实心路径的候选集：复刻 CPU 的分块 RNG 方案（chunk 顺序拼接、同 seed 混合、
/// 同 grid 加权中心），保证与 CPU 路径候选逐位同源。此前 GPU 用单流独立采样，
/// 候选分布本身就与 CPU 不同——fvseg demo 上 1dB 级质量差的主因。
pub(crate) fn generate_solid_candidates(
    ws: &Workspace,
    cfg: &FitConfig,
    layer: usize,
) -> Vec<EllipseParams> {
    let mut cands = Vec::with_capacity(cfg.samples_per_shape);
    for chunk in 0..ws.threads {
        let mut rng = solid_chunk_rng(cfg, layer, chunk);
        for _ in 0..ws.per_thread {
            let (cx, cy) = sample_center_mixed(&mut rng, &ws.grid, ws.w, ws.h);
            cands.push(sample_candidate(
                &mut rng,
                cfg.structured,
                &ws.guide,
                &ws.spec,
                cx,
                cy,
            ));
        }
    }
    cands
}

/// CPU 实心路径单 chunk 的采样评分（rayon map 闭包体）。采样序列与
/// [`generate_solid_candidates`] 逐位同源（同一 RNG 流，逐候选同决策点）。
pub(crate) fn sample_and_score_chunk(
    ws: &Workspace,
    cfg: &FitConfig,
    layer: usize,
    chunk: usize,
) -> Option<Candidate> {
    let mut rng = solid_chunk_rng(cfg, layer, chunk);
    let mut local: Option<Candidate> = None;
    for _ in 0..ws.per_thread {
        let (cx, cy) = sample_center_mixed(&mut rng, &ws.grid, ws.w, ws.h);
        let e = sample_candidate(
            &mut rng,
            cfg.structured,
            &ws.guide,
            &ws.spec,
            cx,
            cy,
        );
        if let Some((color, delta)) = evaluate(&ws.sums, &e, ws.w, ws.h)
            && delta > 0
        {
            let c = Candidate { e, color, delta };
            local = match local {
                None => Some(c),
                Some(prev) => better(&Some(c.clone()), &Some(prev)),
            };
        }
    }
    local
}

/// GPU 批结果复核：GPU i64 评分与 CPU 在边带有 0.7% ±1 分歧，深饱和/近 tie 处
/// 放大贪心分叉。取 GPU 报告 top-k 用 CPU 精确评分复核（~4µs/层，相对 GPU 批评估
/// 可忽略）；全负批复核最接近的 1 个，防 ±1 误差触发假三振早停。
pub(crate) fn verify_gpu_results(
    sums: &RowSums,
    cands: &[EllipseParams],
    results: EvalResults,
    w: u32,
    h: u32,
) -> Option<Candidate> {
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
        if let Some((color, delta)) = evaluate(sums, &e, w, h)
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
        if let Some((color, delta)) = evaluate(sums, &e, w, h)
            && delta > 0
        {
            best = Some(Candidate { e, color, delta });
        }
    }
    best
}

/// 实心贪心主循环：候选生成/评分（GPU 批量 或 CPU 多线程，reduce 用全序保证
/// 确定性）→ 爬山精修 → 接纳（合成/前缀和/误差网格/GPU 通知/记账）→ checkpoint。
/// 三振早停：连续 3 层采样池无改进才终止（误差网格随接纳刷新）。
fn solid_greedy_loop(
    ws: &mut Workspace,
    cfg: &FitConfig,
    gpu: &mut Option<&mut dyn BatchEvaluator>,
    progress: &mut dyn FnMut(usize, f64),
) {
    let mut empty_streak = 0u32;
    for layer in 0..cfg.layers {
        // 绝对层数（含 resume 初始层）达到 cfg.layers 即收工
        if ws.shapes_opt.len() >= cfg.layers {
            break;
        }
        // 误差网格：块误差 CDF（每层接纳后重建；首层用初始 bg 画布）
        if layer == 0 {
            ws.grid.rebuild(&ws.target, &ws.canvas, ws.w, ws.h);
            if let Some(g) = gpu.as_deref_mut() {
                g.on_ready(&ws.sums, &ws.target, &ws.canvas, ws.w, ws.h);
            }
        }
        // 采样评分：GPU 批量路径 或 CPU 多线程路径（reduce 用全序保证确定性）
        // GPU 不健康（无 Vulkan/运行中断）→ 走 CPU 路径（真回退，不再静默空结果）
        let gpu_ok = gpu.as_deref_mut().is_some_and(|g| g.available());
        let best: Option<Candidate> = if let (true, Some(g)) = (gpu_ok, gpu.as_deref_mut()) {
            // 候选集与 CPU 路径逐位同源（见 generate_solid_candidates）
            let cands = generate_solid_candidates(ws, cfg, layer);
            let results = g.eval_batch(&cands, ws.w, ws.h);
            verify_gpu_results(&ws.sums, &cands, results, ws.w, ws.h)
        } else {
            let ws_ref: &Workspace = ws;
            (0..ws_ref.threads)
                .into_par_iter()
                .map(|chunk| sample_and_score_chunk(ws_ref, cfg, layer, chunk))
                .reduce(|| None, |a, b| better(&a, &b))
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
        let best = refine(&ws.sums, ws.w, ws.h, best, &mut rrng, cfg.refine_iters);

        // 接纳：合成 + 刷新画布侧前缀和 + 重建误差网格 + 通知 GPU 后端
        raster::composite_ellipse(&mut ws.canvas, ws.w, ws.h, &best.e, best.color);
        ws.sums.refresh_canvas(&ws.canvas);
        ws.grid.rebuild(&ws.target, &ws.canvas, ws.w, ws.h);
        if let Some(g) = gpu.as_deref_mut() {
            g.on_accept(&ws.sums, &ws.target, &ws.canvas, ws.w, ws.h);
        }
        ws.total_err -= best.delta;
        empty_streak = 0;
        ws.shapes_opt.push((best.e, best.color));
        ws.srcs.push(u32::MAX);
        ws.changed.push(false);

        let done = ws.shapes_opt.len();
        progress(done, ws.total_err as f64 / ws.denom);

        if cfg.checkpoints.contains(&done) {
            write_outputs(cfg, ws, &[], &format!(".{done}"));
        }
    }
}

// ---- 阶段 3：遮挡感知精确抛光（可选） ----

/// 逐像素重算总误差（polish/soft-refit 后画布全量重合成时的精确记账）。
fn recompute_total_err(ws: &mut Workspace) {
    ws.total_err = 0i64;
    for p in 0..ws.n * 3 {
        let d = ws.target[p] as i64 - ws.canvas[p] as i64;
        ws.total_err += d * d;
    }
}

/// 遮挡感知精确抛光：逐形状在可见集上重拟合 + 删无效层；同步血统/改动账本，
/// 重合成画布精确重算总误差。
fn polish_phase(
    ws: &mut Workspace,
    cfg: &FitConfig,
    gpu: &mut Option<&mut dyn BatchEvaluator>,
    progress: &mut dyn FnMut(usize, f64),
) {
    if cfg.polish_sweeps == 0 {
        return;
    }
    let n = ws.n;
    let (rep, survivors) = crate::polish::polish_opt(
        &ws.target,
        ws.w,
        ws.h,
        &mut ws.shapes_opt,
        cfg.bg,
        cfg.polish_sweeps,
        cfg.refine_iters,
        cfg.seed,
    );
    // 同步血统/改动账本，并重合成画布精确重算总误差
    ws.srcs.truncate(ws.shapes_opt.len());
    ws.changed.truncate(ws.shapes_opt.len());
    for (i, s) in survivors.iter().enumerate() {
        ws.srcs[i] = s.src;
        ws.changed[i] = s.changed;
    }
    for ch in 0..3 {
        ws.canvas[ch * n..(ch + 1) * n].fill(cfg.bg[ch]);
    }
    for (e, c) in &ws.shapes_opt {
        raster::composite_ellipse(&mut ws.canvas, ws.w, ws.h, e, *c);
    }
    recompute_total_err(ws);
    eprintln!(
        "polish: {} sweeps, {} accepted, {} deleted, mse {:.2} -> {:.2}",
        rep.sweeps_run, rep.accepted, rep.deleted, rep.mse_before, rep.mse_after
    );
    // 画布已重算：同步前缀和与 GPU 缓存（否则后续软通道对着 stale 画布预测）
    ws.sums.refresh_canvas(&ws.canvas);
    if let Some(g) = gpu.as_deref_mut() {
        g.on_accept(&ws.sums, &ws.target, &ws.canvas, ws.w, ws.h);
    }
    progress(ws.shapes_opt.len(), ws.total_err as f64 / ws.denom);
}

// ---- 阶段 4：软椭圆（228）透明层通道 + 软栈颜色重拟合 ----

/// 软候选单流生成（GPU/CPU 共用）：单 RNG 流（`stream_idx` 入种子）、同 grid
/// 加权中心。软通道 GPU 与 CPU 候选流天然逐位同源（与实心通道的分块方案不同）。
fn soft_candidates(ws: &Workspace, cfg: &FitConfig, stream_idx: usize) -> Vec<EllipseParams> {
    let mut rng = Rng::new(cfg.seed ^ 0x50F7 ^ ((stream_idx as u64) << 32));
    let mut cands = Vec::with_capacity(cfg.samples_per_shape);
    for _ in 0..cfg.samples_per_shape {
        let (cx, cy) = sample_center_mixed(&mut rng, &ws.grid, ws.w, ws.h);
        cands.push(sample_candidate(
            &mut rng,
            cfg.structured,
            &ws.guide,
            &ws.spec,
            cx,
            cy,
        ));
    }
    cands
}

/// 软通道单迭代的最优候选：GPU 批评估 或 CPU 逐个精评（后端不健康时回退）。
/// 返回 (椭圆, 颜色)；None = 全负/无效。评分取严格更大 delta（平手先到先得）。
fn select_soft_best(
    ws: &Workspace,
    cfg: &FitConfig,
    gpu: &mut Option<&mut dyn BatchEvaluator>,
    stream_idx: usize,
) -> Option<(EllipseParams, [u8; 3])> {
    let gpu_ok = gpu.as_deref_mut().is_some_and(|g| g.available());
    if let (true, Some(g)) = (gpu_ok, gpu.as_deref_mut()) {
        let cands = soft_candidates(ws, cfg, stream_idx);
        let results = g.eval_soft_batch(&cands, ws.w, ws.h);
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
        let cands = soft_candidates(ws, cfg, stream_idx);
        let mut best: Option<(EllipseParams, [u8; 3], i64)> = None;
        for e in &cands {
            if let Some((k, delta)) = crate::soft::eval_soft(&ws.target, &ws.canvas, ws.w, ws.h, e)
                && best.as_ref().is_none_or(|(_, _, bd)| delta > *bd)
            {
                best = Some((*e, k, delta));
            }
        }
        best.map(|(e, k, _)| (e, k))
    }
}

/// 软通道接纳判定：吸纳前先快照 bbox 试混，实测 after < before 才保留
///（u8 舍入防虚账），否则回滚快照。返回是否接纳。
fn soft_try_accept(
    ws: &mut Workspace,
    gpu: &mut Option<&mut dyn BatchEvaluator>,
    e: &EllipseParams,
    k: [u8; 3],
    soft_out: &mut Vec<(EllipseParams, [u8; 3])>,
    progress: &mut dyn FnMut(usize, f64),
) -> bool {
    let spans = raster::ellipse_rows(e, ws.w, ws.h);
    let Some(bbox) = raster::spans_bbox(&spans) else {
        return false;
    };
    // bbox 内 before 误差 + 快照
    let (bx0, by0, bx1, by1) = (
        bbox.0 as usize,
        bbox.1 as usize,
        bbox.2 as usize,
        bbox.3 as usize,
    );
    let mut before = 0i64;
    let mut snap = Vec::with_capacity((bx1 - bx0) * (by1 - by0) * 3);
    for y in by0..by1 {
        for x in bx0..bx1 {
            let p = y * ws.w as usize + x;
            for ch in 0..3 {
                let t = ws.target[ch * ws.n + p] as i64;
                let c = ws.canvas[ch * ws.n + p] as i64;
                before += (t - c) * (t - c);
                snap.push(ws.canvas[ch * ws.n + p]);
            }
        }
    }
    crate::soft::blend_composite(&mut ws.canvas, ws.w, ws.h, e, k, 1.0);
    let mut after = 0i64;
    for y in by0..by1 {
        for x in bx0..bx1 {
            let p = y * ws.w as usize + x;
            for ch in 0..3 {
                let t = ws.target[ch * ws.n + p] as i64;
                let c = ws.canvas[ch * ws.n + p] as i64;
                after += (t - c) * (t - c);
            }
        }
    }
    if after < before {
        ws.total_err -= before - after;
        soft_out.push((*e, k));
        ws.sums.refresh_canvas(&ws.canvas);
        ws.grid.rebuild(&ws.target, &ws.canvas, ws.w, ws.h);
        if let Some(g) = gpu.as_deref_mut() {
            g.on_accept(&ws.sums, &ws.target, &ws.canvas, ws.w, ws.h);
        }
        progress(
            ws.shapes_opt.len() + soft_out.len(),
            ws.total_err as f64 / ws.denom,
        );
        true
    } else {
        // 回滚
        let mut si = 0usize;
        for y in by0..by1 {
            for x in bx0..bx1 {
                let p = y * ws.w as usize + x;
                for ch in 0..3 {
                    ws.canvas[ch * ws.n + p] = snap[si];
                    si += 1;
                }
            }
        }
        false
    }
}

/// 软栈颜色重拟合（M1.4）：坐标下降修正被后续软层改变的残差。
fn soft_refit(
    ws: &mut Workspace,
    gpu: &mut Option<&mut dyn BatchEvaluator>,
    solid_canvas: &[u8],
    soft_out: &mut [(EllipseParams, [u8; 3])],
    progress: &mut dyn FnMut(usize, f64),
) {
    let (rep, final_canvas) =
        crate::soft::refit_soft_stack(&ws.target, ws.w, ws.h, solid_canvas, soft_out);
    ws.canvas.copy_from_slice(&final_canvas);
    recompute_total_err(ws);
    ws.sums.refresh_canvas(&ws.canvas);
    ws.grid.rebuild(&ws.target, &ws.canvas, ws.w, ws.h);
    if let Some(g) = gpu.as_deref_mut() {
        g.on_accept(&ws.sums, &ws.target, &ws.canvas, ws.w, ws.h);
    }
    eprintln!(
        "soft-refit: {}/{} layers refit, mse {:.2} -> {:.2}",
        rep.refits, rep.layers, rep.mse_before, rep.mse_after
    );
    progress(
        ws.shapes_opt.len() + soft_out.len(),
        ws.total_err as f64 / ws.denom,
    );
}

/// 软椭圆（228）透明层通道：混合语义攻击纹理残差——实心覆盖在纹理区是罚分
///（平色盖高频必涨误差），软层把像素往均值拉（降方差）。
/// M1.5 自适应预算：soft_layers==0 且 auto_soft 时，实心三振出局后的剩余层预算
/// 全给软层（此时 cfg.layers 语义 = 总预算）。返回软层（opt 坐标，z 序在实心之上）。
fn soft_phase(
    ws: &mut Workspace,
    cfg: &FitConfig,
    gpu: &mut Option<&mut dyn BatchEvaluator>,
    progress: &mut dyn FnMut(usize, f64),
) -> Vec<(EllipseParams, [u8; 3])> {
    let soft_budget = if cfg.soft_layers > 0 {
        cfg.soft_layers
    } else if cfg.auto_soft {
        cfg.layers.saturating_sub(ws.shapes_opt.len())
    } else {
        0
    };
    let mut soft_out: Vec<(EllipseParams, [u8; 3])> = Vec::new();
    // refit 的基准画布 = 软相入口时刻的实心画布快照
    let solid_canvas = if soft_budget > 0 {
        ws.canvas.clone()
    } else {
        Vec::new()
    };
    if soft_budget > 0 {
        // resume 层数已满（layers==base）等路径会跳过主循环的 grid 重建/on_ready，
        // polish 后 grid 也可能陈旧——软相入口无条件刷新一次（幂等，开销可忽略）
        ws.grid.rebuild(&ws.target, &ws.canvas, ws.w, ws.h);
        if let Some(g) = gpu.as_deref_mut() {
            g.on_ready(&ws.sums, &ws.target, &ws.canvas, ws.w, ws.h);
        }
        let mut soft_streak = 0u32;
        for _ in 0..soft_budget {
            if soft_streak >= 3 {
                break;
            }
            let Some((e, k)) = select_soft_best(ws, cfg, gpu, soft_out.len()) else {
                soft_streak += 1;
                continue;
            };
            if soft_try_accept(ws, gpu, &e, k, &mut soft_out, progress) {
                soft_streak = 0;
            } else {
                soft_streak += 1;
            }
        }
    }

    // ---- 软栈颜色重拟合（M1.4）：坐标下降修正被后续软层改变的残差 ----
    if cfg.soft_refit && !soft_out.is_empty() {
        soft_refit(ws, gpu, &solid_canvas, &mut soft_out, progress);
    }
    soft_out
}

// ---- 阶段 5：终稿输出 ----

/// 输出转换：未改动的初始层逐位原样返回（不经缩放往返）；其余 opt → 原图坐标。
pub(crate) fn to_full(ws: &Workspace) -> Vec<Shape> {
    ws.shapes_opt
        .iter()
        .enumerate()
        .map(|(i, (e, c))| {
            let src = ws.srcs[i];
            if src != u32::MAX
                && !ws.changed[i]
                && ws.initial_opt.get(src as usize).is_some_and(|(e0, c0)| {
                    e0.cx.to_bits() == e.cx.to_bits()
                        && e0.cy.to_bits() == e.cy.to_bits()
                        && e0.rx.to_bits() == e.rx.to_bits()
                        && e0.ry.to_bits() == e.ry.to_bits()
                        && e0.angle_rad.to_bits() == e.angle_rad.to_bits()
                        && c0 == c
                })
            {
                return ws.initial[src as usize];
            }
            Shape::ellipse(
                e.cx * ws.inv_scale,
                e.cy * ws.inv_scale,
                e.rx * ws.inv_scale,
                e.ry * ws.inv_scale,
                e.angle_rad.to_degrees(),
                *c,
            )
        })
        .collect()
}

/// checkpoint/终稿落盘：`{prefix}{suffix}.json` + 预览 PNG（未配置 out_prefix 时跳过）。
pub(crate) fn write_outputs(
    cfg: &FitConfig,
    ws: &Workspace,
    soft: &[(EllipseParams, [u8; 3])],
    suffix: &str,
) {
    let Some(prefix) = &cfg.out_prefix else {
        return;
    };
    let mut shapes = to_full(ws);
    for (e, c) in soft {
        shapes.push(Shape {
            kind: ShapeKind::SoftEllipse,
            cx: e.cx * ws.inv_scale,
            cy: e.cy * ws.inv_scale,
            rx: e.rx * ws.inv_scale,
            ry: e.ry * ws.inv_scale,
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
    if let Err(e) = crate::render::render_shapes(ws.ow, ws.oh, &shapes, cfg.bg).save(&pp) {
        eprintln!("checkpoint 预览写入失败: {e}");
    }
}

/// 终稿落盘 + 原图坐标输出组装（实心层血统保真 + 软层缩放追加）。
fn finalize_outputs(
    ws: Workspace,
    cfg: &FitConfig,
    soft_out: &[(EllipseParams, [u8; 3])],
) -> FitResult {
    write_outputs(cfg, &ws, soft_out, "");
    let mut full = to_full(&ws);
    let inv_scale = ws.inv_scale;
    for (e, c) in soft_out {
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
        final_mse: ws.total_err as f64 / ws.denom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fv_common::shape::ShapeKind;

    /// 确定性渐变+噪声图（贴近真实图片特性，避免平坦图上的退化路径）。
    fn noise_image(w: u32, h: u32, seed: u32) -> image::RgbaImage {
        let mut img = image::RgbaImage::new(w, h);
        let mut noise = seed;
        let mut nz = || {
            noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
            ((noise >> 28) as i32) - 8
        };
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / w as f32;
                img.put_pixel(
                    x,
                    y,
                    image::Rgba([
                        ((t * 255.0) as i32 + nz()).clamp(0, 255) as u8,
                        (60 + nz()).clamp(0, 255) as u8,
                        ((255.0 - t * 255.0) as i32 + nz()).clamp(0, 255) as u8,
                        255,
                    ]),
                );
            }
        }
        img
    }

    fn test_cfg(opt_max_dim: u32) -> FitConfig {
        FitConfig {
            layers: 1,
            samples_per_shape: 1,
            opt_max_dim,
            seed: 7,
            refine_iters: 1,
            checkpoints: vec![],
            out_prefix: None,
            bg: [0, 0, 0],
            polish_sweeps: 0,
            soft_layers: 0,
            soft_refit: false,
            auto_soft: false,
            structured: true,
        }
    }

    /// 逐位比较形状（f32 按 to_bits）。
    fn assert_bits_eq(a: &Shape, b: &Shape, idx: usize) {
        assert_eq!(a.kind, b.kind, "shape[{idx}] kind");
        assert_eq!(a.cx.to_bits(), b.cx.to_bits(), "shape[{idx}] cx");
        assert_eq!(a.cy.to_bits(), b.cy.to_bits(), "shape[{idx}] cy");
        assert_eq!(a.rx.to_bits(), b.rx.to_bits(), "shape[{idx}] rx");
        assert_eq!(a.ry.to_bits(), b.ry.to_bits(), "shape[{idx}] ry");
        assert_eq!(a.angle_deg.to_bits(), b.angle_deg.to_bits(), "shape[{idx}] angle");
        assert_eq!(a.color, b.color, "shape[{idx}] color");
    }

    #[test]
    fn solid_chunk_rng_contract() {
        // 同参同流（GPU/CPU 候选同源的锚点）；跨 chunk/跨层异流。
        let cfg = test_cfg(64);
        let stream = |layer: usize, chunk: usize| {
            let mut r = solid_chunk_rng(&cfg, layer, chunk);
            (r.next_u32(), r.next_u32(), r.next_f32())
        };
        assert_eq!(stream(3, 5), stream(3, 5), "同参必须同流");
        assert_ne!(stream(3, 5), stream(3, 6), "跨 chunk 必须异流");
        assert_ne!(stream(3, 5), stream(4, 5), "跨层必须异流");
    }

    /// GPU 候选集与 CPU 分块采样方案逐位同源：独立按文档方案重放（chunk 顺序
    /// 拼接、同 seed 混合、75/25 加权中心、structured 采样），对比参数逐位一致。
    #[test]
    fn gpu_candidate_stream_matches_cpu_chunk_scheme() {
        let img = noise_image(96, 72, 0xAB);
        let cfg = FitConfig {
            samples_per_shape: 4000,
            ..test_cfg(96)
        };
        let ws = prepare_workspace(&img, &cfg, &[]);
        let layer = 3;
        let cands = generate_solid_candidates(&ws, &cfg, layer);

        let mut replay: Vec<EllipseParams> = Vec::with_capacity(cfg.samples_per_shape);
        for chunk in 0..ws.threads {
            let mut rng = solid_chunk_rng(&cfg, layer, chunk);
            for _ in 0..ws.per_thread {
                // 独立重放 CPU 方案（不经 sample_center_mixed/sample_candidate）：
                // 锁住"75% 网格加权 + 25% 均匀"与 structured 分支的 RNG 消耗顺序
                let (cx, cy) = if rng.next_u32() & 3 == 0 {
                    (rng.next_f32() * ws.w as f32, rng.next_f32() * ws.h as f32)
                } else {
                    ws.grid.sample_center(&mut rng, ws.w, ws.h)
                };
                replay.push(candidate::sample_structured(&mut rng, &ws.guide, cx, cy));
            }
        }
        assert_eq!(cands.len(), replay.len());
        for (i, (a, b)) in cands.iter().zip(&replay).enumerate() {
            assert_eq!(a.cx.to_bits(), b.cx.to_bits(), "cand[{i}] cx");
            assert_eq!(a.cy.to_bits(), b.cy.to_bits(), "cand[{i}] cy");
            assert_eq!(a.rx.to_bits(), b.rx.to_bits(), "cand[{i}] rx");
            assert_eq!(a.ry.to_bits(), b.ry.to_bits(), "cand[{i}] ry");
            assert_eq!(a.angle_rad.to_bits(), b.angle_rad.to_bits(), "cand[{i}] angle");
        }
    }

    /// GPU top-k 复核 ≡ CPU 全量扫描（评分精确一致时必须选出同一最优）。
    #[test]
    fn verify_gpu_topk_matches_full_scan() {
        let (w, h) = (48u32, 48u32);
        let n = (w * h) as usize;
        let mut target = vec![90u8; n * 3];
        let mut rng = Rng::new(7);
        for t in target.iter_mut() {
            *t = t.wrapping_add((rng.next_u32() & 31) as u8);
        }
        let canvas = vec![0u8; n * 3];
        let mut sums = RowSums::new(w, h);
        sums.set_target(&target);
        sums.refresh_canvas(&canvas);

        let spec = CandidateSpec {
            min_radius: 1.2,
            max_radius: 15.0,
        };
        let mut cands = Vec::new();
        for _ in 0..200 {
            cands.push(candidate::sample(&mut rng, w as f32, h as f32, &spec));
        }
        // CPU 全量扫描（实心 CPU 路径同一全序 reduce 语义）
        let mut best: Option<Candidate> = None;
        let mut results = Vec::with_capacity(cands.len());
        for e in &cands {
            let r = evaluate(&sums, e, w, h);
            results.push(r);
            if let Some((color, delta)) = r
                && delta > 0
            {
                best = better(&Some(Candidate { e: *e, color, delta }), &best);
            }
        }
        let b = best.expect("噪声画布上应有正收益候选");
        let v = verify_gpu_results(&sums, &cands, results, w, h)
            .expect("复核应复现 CPU 最优");
        assert_eq!(b.delta, v.delta);
        assert_eq!(b.e.cx.to_bits(), v.e.cx.to_bits());
        assert_eq!(b.e.cy.to_bits(), v.e.cy.to_bits());
        assert_eq!(b.e.rx.to_bits(), v.e.rx.to_bits());
        assert_eq!(b.e.ry.to_bits(), v.e.ry.to_bits());
        assert_eq!(b.e.angle_rad.to_bits(), v.e.angle_rad.to_bits());
        assert_eq!(b.color, v.color);
    }

    /// 全负批兜底：GPU 评分报负但 CPU 精评为正 → 复核翻案（防假三振）；
    /// GPU 报 None（无效）→ 直接 None；真负 → None。
    #[test]
    fn verify_gpu_all_negative_fallback() {
        let (w, h) = (48u32, 48u32);
        let n = (w * h) as usize;
        let mut target = vec![180u8; n * 3];
        let mut rng = Rng::new(9);
        for t in target.iter_mut() {
            *t = t.wrapping_add((rng.next_u32() & 15) as u8);
        }
        let e = EllipseParams {
            cx: 24.0,
            cy: 24.0,
            rx: 20.0,
            ry: 14.0,
            angle_rad: 0.3,
        };
        // sums_pos：画布黑、目标亮 → 实心候选正收益
        let mut sums_pos = RowSums::new(w, h);
        sums_pos.set_target(&target);
        sums_pos.refresh_canvas(&vec![0u8; n * 3]);
        // sums_neg：画布 == 目标 → 一切实心候选 delta ≤ 0
        let mut sums_neg = RowSums::new(w, h);
        sums_neg.set_target(&target);
        sums_neg.refresh_canvas(&target);

        assert!(evaluate(&sums_pos, &e, w, h).unwrap().1 > 0);

        // GPU 全负（±1 边带误差场景）且把正收益候选排第一 → CPU 复核翻案
        let results = vec![Some(([9, 9, 9], -1i64)), Some(([9, 9, 9], -7i64))];
        let v = verify_gpu_results(&sums_pos, &[e, e], results, w, h)
            .expect("兜底应复核出正收益");
        assert_eq!(v.e.cx.to_bits(), e.cx.to_bits());
        assert_eq!(v.e.cy.to_bits(), e.cy.to_bits());
        // GPU 报 None（无效批）→ 无候选可复核
        assert!(verify_gpu_results(&sums_pos, &[e], vec![None], w, h).is_none());
        // 真负批 → 兜底复核后仍 None
        assert!(evaluate(&sums_neg, &e, w, h).is_none_or(|(_, d)| d <= 0));
        let results = vec![Some(([9, 9, 9], -7i64))];
        assert!(verify_gpu_results(&sums_neg, std::slice::from_ref(&e), results, w, h).is_none());
    }

    /// to_full 血统契约：未改动的初始层逐位原样返回；新增层经 inv_scale 缩放；
    /// polish 改动过的初始层改走缩放往返（不再逐位保真）。
    #[test]
    fn to_full_lineage_contract() {
        let img = noise_image(64, 64, 11);
        let s0 = Shape::ellipse(10.3, 12.7, 4.0, 5.0, 30.0, [1, 2, 3]);
        let s1 = Shape::ellipse(40.1, 44.3, 6.0, 3.0, 100.0, [200, 100, 50]);
        // 长边 64 → opt_max_dim 48 → scale = 0.75（非 2 的幂，缩放往返有浮点扰动）
        let cfg = test_cfg(48);
        let initial = [s0, s1];
        let mut ws = prepare_workspace(&img, &cfg, &initial);

        // 未改动初始层：逐位原样（不经缩放往返）
        let full = to_full(&ws);
        assert_eq!(full.len(), 2);
        assert_bits_eq(&full[0], &initial[0], 0);
        assert_bits_eq(&full[1], &initial[1], 1);

        // 新增层：opt → 原图坐标
        let ne = EllipseParams {
            cx: 8.0,
            cy: 9.0,
            rx: 2.0,
            ry: 2.5,
            angle_rad: 0.5,
        };
        ws.shapes_opt.push((ne, [9, 9, 9]));
        ws.srcs.push(u32::MAX);
        ws.changed.push(false);
        let full = to_full(&ws);
        assert_eq!(full.len(), 3);
        assert_bits_eq(&full[0], &s0, 0);
        assert_bits_eq(&full[1], &s1, 1);
        let last = &full[2];
        let scale = (48.0f32 / 64.0).min(1.0);
        let inv_scale = 1.0 / scale;
        assert_eq!(last.kind, ShapeKind::RotatedEllipse);
        assert_eq!(last.cx.to_bits(), (ne.cx * inv_scale).to_bits());
        assert_eq!(last.cy.to_bits(), (ne.cy * inv_scale).to_bits());
        assert_eq!(last.angle_deg.to_bits(), ne.angle_rad.to_degrees().to_bits());
        assert_eq!(last.color, [9, 9, 9, 255]);

        // polish 改动过的初始层：改走缩放往返输出（有浮点扰动，不再逐位一致）
        ws.changed[0] = true;
        let full = to_full(&ws);
        assert_ne!(
            full[0].angle_deg.to_bits(),
            initial[0].angle_deg.to_bits(),
            "改动层应经缩放往返"
        );
    }

    /// 软通道接纳/回滚：正收益候选接纳（画布变更、总误差下降、progress 回调）；
    /// 零/负收益候选回滚（画布逐位还原、账目不变、不进 soft_out）。
    #[test]
    fn soft_try_accept_and_rollback() {
        // 左半亮红、右半纯 bg 黑：左半软层正收益，右半零收益
        let (w, h) = (64u32, 64u32);
        let mut img = image::RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = if x < w / 2 { [200u8, 30, 30] } else { [0, 0, 0] };
                img.put_pixel(x, y, image::Rgba([c[0], c[1], c[2], 255]));
            }
        }
        let cfg = test_cfg(64);
        let mut ws = prepare_workspace(&img, &cfg, &[]);
        let mut soft_out = Vec::new();
        let mut no_gpu: Option<&mut dyn BatchEvaluator> = None;
        let mut progress_calls = 0usize;

        let e_left = EllipseParams {
            cx: 16.0,
            cy: 32.0,
            rx: 10.0,
            ry: 10.0,
            angle_rad: 0.0,
        };
        let err0 = ws.total_err;
        assert!(soft_try_accept(
            &mut ws,
            &mut no_gpu,
            &e_left,
            [200, 30, 30],
            &mut soft_out,
            &mut |_: usize, _: f64| progress_calls += 1,
        ));
        assert_eq!(soft_out.len(), 1);
        assert!(ws.total_err < err0, "接纳后总误差应下降");
        assert_eq!(progress_calls, 1, "接纳应回调一次 progress");

        // 右半：目标==画布（纯 bg）→ 无正收益 → 回滚
        let err1 = ws.total_err;
        let canvas1 = ws.canvas.clone();
        let e_right = EllipseParams {
            cx: 48.0,
            cy: 32.0,
            rx: 10.0,
            ry: 10.0,
            angle_rad: 0.0,
        };
        assert!(!soft_try_accept(
            &mut ws,
            &mut no_gpu,
            &e_right,
            [200, 30, 30],
            &mut soft_out,
            &mut |_, _| {},
        ));
        assert_eq!(soft_out.len(), 1, "回滚候选不得进 soft_out");
        assert_eq!(ws.total_err, err1, "回滚不得改账");
        assert_eq!(ws.canvas, canvas1, "回滚应逐位还原画布");
        assert_eq!(progress_calls, 1, "回滚不得回调 progress");
    }
}
