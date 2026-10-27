//! 分层拟合端到端（L1.2 → M1 最小链路，v2：直调 `fv_seg::pipeline` 库编排）。
//!
//! 旧版在测试里手工复刻 bin 的三段编排（bin 不可测的年代产物），已随生产
//! 演进漂移（bin 早已是四段 + 全局软相）。库化后本测试直接调真编排，
//! 断言升级为对 `stats`/`shapes` 分段的语义校验。

use fv_common::shape::{Shape, ShapeKind};
use fv_fit::render::render_shapes;
use fv_seg::pipeline::{psnr_db, run_pipeline};
use fv_seg::{plan_layers, LayerPlan, Matte};

const W: u32 = 192;
const H: u32 = 192;
const TOTAL: usize = 60;
const SAMPLES: usize = 3000;
const SEED: u64 = 7;
const SOFT: usize = 8;
const BG: [u8; 3] = [12, 24, 48];

/// 合成图：横向渐变底 + 中央多彩圆环纹理；matte = 圆内 255。
fn synth() -> (image::RgbaImage, Matte) {
    let (cx, cy, r) = (W as f32 * 0.5, H as f32 * 0.5, W as f32 * 0.30);
    let mut img = image::RgbaImage::new(W, H);
    let mut matte = vec![0u8; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let t = x as f32 / W as f32;
            let grad = [40.0 + 160.0 * t, 70.0 + 130.0 * t, 120.0 + 80.0 * t];
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let d2 = dx * dx + dy * dy;
            let mut px = grad;
            if d2 <= r * r {
                // 径向条纹环（圆环纹理）：主体区有真实残差可拟合
                let ring = ((d2.sqrt() / r) * 6.0 * std::f32::consts::PI).cos();
                let stripe = ((x + y) as f32 * 0.25).sin();
                px = [200.0 + 55.0 * ring, 90.0 + 60.0 * stripe, 60.0 + 40.0 * ring];
                matte[(y * W + x) as usize] = 255;
            }
            img.put_pixel(x, y, image::Rgba([px[0] as u8, px[1] as u8, px[2] as u8, 255]));
        }
    }
    (img, Matte { width: W, height: H, data: matte })
}

/// (cx, cy)（原图像素）是否落在掩码（0/255）内。
fn in_mask(mask: &[u8], x: f32, y: f32) -> bool {
    let (xi, yi) = (x.floor() as i64, y.floor() as i64);
    if xi < 0 || yi < 0 || xi >= W as i64 || yi >= H as i64 {
        return false;
    }
    mask[yi as usize * W as usize + xi as usize] != 0
}

#[test]
fn layered_e2e_library_pipeline() {
    let (img, matte) = synth();
    let plan: LayerPlan = plan_layers(&matte, TOTAL).expect("plan_layers");
    assert_eq!((plan.w, plan.h), (W, H), "plan 尺寸应与输入一致");

    let mut gpu = None; // CPU 路径（无 Vulkan 依赖的确定性测试）
    let result = run_pipeline(&img, &plan, SAMPLES, SEED, BG, SOFT, false, &mut gpu)
        .expect("run_pipeline");

    // ---- 四段 stats：bg / subject / edge / soft，分段不超预算 ----
    let names: Vec<&str> = result.stats.iter().map(|s| s.name).collect();
    assert_eq!(names, vec!["bg", "subject", "edge", "soft"], "四段编排");
    let (bg_n, sub_n, edge_n, soft_n) = (
        result.stats[0].solid,
        result.stats[1].solid,
        result.stats[2].solid,
        result.stats[3].soft,
    );
    assert!(bg_n > 0 && bg_n <= plan.budget.background, "背景层 {bg_n}");
    assert!(sub_n > 0 && sub_n <= plan.budget.subject, "主体层 {sub_n}");
    assert!(edge_n <= plan.budget.edge, "边界层 {edge_n}");
    assert!(soft_n <= SOFT, "软段 {soft_n} ≤ 预算 {SOFT}");
    assert_eq!(
        result.shapes.len(),
        bg_n + sub_n + edge_n + soft_n,
        "stats 分段与 shapes 总数一致"
    );

    // ---- 语义放置：bg 段形状中心在主体掩码外，subject 段在掩码内 ----
    //（贪心对"不关心区域零残差"的响应——L1.2 的核心机制，见 pipeline.rs 文档）
    let bg_off_mask = result.shapes[..bg_n]
        .iter()
        .filter(|s| !in_mask(&plan.subject_mask, s.cx, s.cy))
        .count();
    assert!(
        bg_off_mask >= bg_n * 7 / 10,
        "bg 段应有 ≥70% 中心在主体掩码外（允许骑边形状；实得 {bg_off_mask}/{bg_n}）"
    );
    let sub_in_mask = result.shapes[bg_n..bg_n + sub_n]
        .iter()
        .filter(|s| in_mask(&plan.subject_mask, s.cx, s.cy))
        .count();
    assert!(
        sub_in_mask >= sub_n * 8 / 10,
        "subject 段应有 ≥80% 中心在主体掩码内（实得 {sub_in_mask}/{sub_n}）"
    );

    // ---- 软段形状全部为 228 ----
    let tail = &result.shapes[bg_n + sub_n + edge_n..];
    assert!(
        tail.iter().all(|s| s.kind == ShapeKind::SoftEllipse),
        "soft 段全部 kind=228"
    );

    // ---- PSNR 自洽：库内 psnr 与重渲染一致，且分层显著优于纯背景 ----
    let render = render_shapes(W, H, &result.shapes, BG);
    let got = psnr_db(&render, &img, BG);
    assert!((got - result.psnr).abs() < 1e-9, "psnr 口径一致");
    let bg_only: Vec<Shape> = Vec::new();
    let base = psnr_db(&render_shapes(W, H, &bg_only, BG), &img, BG);
    assert!(got > base + 6.0, "分层拟合应比纯背景好 6dB+（{got:.1} vs {base:.1}）");
}
