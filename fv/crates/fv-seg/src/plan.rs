//! Semantic layer planning on top of an alpha matte.
//!
//! Consumes a [`Matte`] and produces a [`LayerPlan`]: two complementary
//! binary masks (subject / background), a boundary intensity field, and a
//! per-layer stroke budget. Downstream (M1) fits ellipses per layer
//! independently and paints background → subject → boundary; this module
//! does *no* fitting itself — masks + budget + boundary field only.
//!
//! Pipeline:
//! 1. **Binarize** the matte with a self-implemented Otsu threshold
//!    ([`otsu_threshold`]).
//! 2. **Smooth** the subject mask with one round of 3×3 majority voting
//!    (removes isolated speckles); background is the exact complement, so
//!    the two masks stay strictly complementary after smoothing.
//! 3. **Boundary band**: Sobel gradient magnitude of the *original* matte
//!    (f32), normalized to 0..=255 by its maximum, then dilated with a 3 px
//!    square structuring element (separable max-filter, Chebyshev radius 3).
//! 4. **Budget**: split `total_budget` across layers (formula on
//!    [`LayerBudget`]).

use std::io;
use std::path::Path;

use crate::Matte;

/// Per-layer stroke budget (see [`plan_layers`] / `allocate_budget_with` for the
/// split formula).
///
/// Invariant: `subject + background + edge == total_budget` for any input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerBudget {
    /// Strokes allotted to the subject layer.
    pub subject: usize,
    /// Strokes allotted to the background layer.
    pub background: usize,
    /// Strokes allotted to the boundary band layer.
    pub edge: usize,
}

/// The full semantic layer plan for one image.
///
/// All three fields are row-major, `w*h` bytes:
/// - `subject_mask` / `background_mask` are strictly 0 or 255 and exact
///   complements of each other.
/// - `boundary_band` is a 0..=255 intensity field marking the matte's edge
///   (Sobel magnitude, dilated 3 px).
#[derive(Debug, Clone)]
pub struct LayerPlan {
    /// Plan width (== matte width).
    pub w: u32,
    /// Plan height (== matte height).
    pub h: u32,
    /// 0/255 subject mask (matte ≥ Otsu threshold, majority-smoothed).
    pub subject_mask: Vec<u8>,
    /// 0/255 background mask (exact complement of `subject_mask`).
    pub background_mask: Vec<u8>,
    /// 0..255 boundary band intensity (Sobel magnitude, dilated 3 px).
    pub boundary_band: Vec<u8>,
    /// Stroke budget split across the three layers.
    pub budget: LayerBudget,
}

impl LayerPlan {
    /// Number of subject pixels (== 255 in `subject_mask`).
    pub fn subject_area(&self) -> usize {
        self.subject_mask.iter().filter(|&&v| v != 0).count()
    }

    /// Number of background pixels (== 255 in `background_mask`).
    pub fn background_area(&self) -> usize {
        self.background_mask.iter().filter(|&&v| v != 0).count()
    }

    /// Number of pixels with nonzero boundary-band intensity.
    pub fn band_area(&self) -> usize {
        self.boundary_band.iter().filter(|&&v| v != 0).count()
    }
}

/// Otsu's method, self-implemented on a 256-bin histogram of `matte`.
///
/// Returns the threshold `t` with the convention **subject = pixel ≥ t**:
/// the search runs over `t` in `(min_value, max_value]` so the darkest
/// occupied bin always lands in the background class and ties are broken
/// toward the smallest such `t`.
///
/// Degenerate input (constant matte): Otsu is undefined on a unimodal
/// histogram, so we fall back to the constant's own value when it is bright
/// (≥ 128 → all subject) and to `value + 1` when it is dark (< 128 → all
/// background). Empty input yields 0.
pub fn otsu_threshold(matte: &[u8]) -> u8 {
    if matte.is_empty() {
        return 0;
    }
    let mut hist = [0u64; 256];
    for &v in matte {
        hist[v as usize] += 1;
    }
    let Some(vmin) = (0..256).find(|&i| hist[i] > 0) else {
        return 0;
    };
    let Some(vmax) = (vmin..256).rev().find(|&i| hist[i] > 0) else {
        return 0;
    };
    if vmin == vmax {
        // Constant matte: bright → foreground, dark → background.
        return if vmin >= 128 { vmin as u8 } else { (vmin + 1) as u8 };
    }

    let total = matte.len() as f64;
    let sum_all: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();

    // Between-class variance σ²(t) = w0·w1·(μ0−μ1)², maximized over t.
    let (mut w0, mut sum0, mut best_var, mut best_t) = (0.0f64, 0.0f64, -1.0f64, vmin as u8);
    for t in vmin + 1..=vmax {
        // class0 = bins < t, class1 = bins ≥ t
        w0 += hist[t - 1] as f64;
        sum0 += (t - 1) as f64 * hist[t - 1] as f64;
        if w0 <= 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 <= 0.0 {
            break;
        }
        let mu0 = sum0 / w0;
        let mu1 = (sum_all - sum0) / w1;
        let var = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// L1.3-lite 可调规划参数：把分层规划的硬编码常数暴露为可搜索旋钮
/// （默认值 = 原行为，`plan_layers` 即默认参数的别名）。
/// 搜索目标由调用方定义（典型 = 管线终评 PSNR），见 fvseg `--tune`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanParams {
    /// Otsu 阈值偏置：正 = 收紧主体（少像素进主体层），负 = 放宽。
    pub thresh_bias: i32,
    /// 3×3 多数投票平滑轮数（0 = 不平滑）。
    pub smooth_rounds: u32,
    /// 边界带膨胀半径（Chebyshev px，0 = 不膨胀）。
    pub band_radius: usize,
    /// 预算：边缘层占总预算比例。
    pub edge_frac: f64,
    /// 预算：主体层增益（乘 (0.5+alpha)）。
    pub subject_gain: f64,
    /// 预算：主体层下限比例。
    pub subject_floor: f64,
}

impl Default for PlanParams {
    fn default() -> Self {
        Self {
            thresh_bias: 0,
            smooth_rounds: 1,
            band_radius: 3,
            edge_frac: 0.25,
            subject_gain: 0.45,
            subject_floor: 0.30,
        }
    }
}

/// Build the semantic layer plan for `matte` with a total stroke budget of
/// `total_budget`.
///
/// Steps (see module docs): Otsu binarize → 3×3 majority smooth →
/// complementary background → Sobel boundary band (normalize + dilate 3 px)
/// → budget split. `total_budget` may be 0 (all layers get 0 strokes).
pub fn plan_layers(matte: &Matte, total_budget: usize) -> Result<LayerPlan, String> {
    plan_layers_with(matte, total_budget, &PlanParams::default())
}

/// `plan_layers` 的参数化版本（L1.3-lite 旋钮）。
pub fn plan_layers_with(
    matte: &Matte,
    total_budget: usize,
    p: &PlanParams,
) -> Result<LayerPlan, String> {
    let (w, h) = (matte.width as usize, matte.height as usize);
    let n = w.checked_mul(h).ok_or("matte dimensions overflow usize")?;
    if n == 0 {
        return Err("matte is empty (zero width or height)".to_string());
    }
    if matte.data.len() != n {
        return Err(format!(
            "matte buffer size mismatch: len={}, expected {n} ({}x{})",
            matte.data.len(),
            matte.width,
            matte.height
        ));
    }

    // 1. Otsu binarization: subject = matte ≥ threshold（+ 偏置，clamp 进合法域）。
    let t0 = otsu_threshold(&matte.data);
    let t = (t0 as i32 + p.thresh_bias).clamp(1, 255) as u8;
    let bin: Vec<u8> = matte
        .data
        .iter()
        .map(|&v| if v >= t { 255 } else { 0 })
        .collect();

    // 2. 3×3 majority voting（p.smooth_rounds 轮）; background is
    //    its exact complement so complementarity survives smoothing.
    let mut subject_mask = bin;
    for _ in 0..p.smooth_rounds {
        subject_mask = majority_smooth3x3(&subject_mask, w, h);
    }
    let background_mask: Vec<u8> = subject_mask
        .iter()
        .map(|&v| if v == 0 { 255 } else { 0 })
        .collect();

    // 3. Boundary band from the *original* (soft) matte.
    let mag = sobel_magnitude(&matte.data, w, h);
    let band = normalize_to_u8(&mag);
    let boundary_band = dilate_radius(&band, w, h, p.band_radius);

    // 4. Budget split by subject area fraction.
    let subject_area = subject_mask.iter().filter(|&&v| v != 0).count();
    let alpha = subject_area as f64 / n as f64;
    let budget = allocate_budget_with(total_budget, alpha, p);

    Ok(LayerPlan {
        w: matte.width,
        h: matte.height,
        subject_mask,
        background_mask,
        boundary_band,
        budget,
    })
}

/// Save the plan's masks as PNGs into `dir` (created if missing):
/// `subject.png` / `background.png` / `boundary_band.png` (grayscale) plus
/// `overlay.png`, an RGB visualization with subject = red, background = blue
/// and boundary intensity = green.
pub fn save_masks(plan: &LayerPlan, dir: impl AsRef<Path>) -> io::Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let save_gray = |name: &str, data: &[u8]| -> io::Result<()> {
        let img: image::GrayImage = image::ImageBuffer::from_raw(plan.w, plan.h, data.to_vec())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, format!("{name}: mask size mismatch"))
            })?;
        img.save(dir.join(name))
            .map_err(|e| io::Error::other(format!("{name}: {e}")))?;
        Ok(())
    };
    save_gray("subject.png", &plan.subject_mask)?;
    save_gray("background.png", &plan.background_mask)?;
    save_gray("boundary_band.png", &plan.boundary_band)?;

    let n = plan.w as usize * plan.h as usize;
    let mut rgb = vec![0u8; 3 * n];
    for i in 0..n {
        rgb[3 * i] = plan.subject_mask[i];
        rgb[3 * i + 1] = plan.boundary_band[i];
        rgb[3 * i + 2] = plan.background_mask[i];
    }
    let overlay: image::RgbImage = image::ImageBuffer::from_raw(plan.w, plan.h, rgb)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "overlay size mismatch"))?;
    overlay
        .save(dir.join("overlay.png"))
        .map_err(|e| io::Error::other(format!("overlay.png: {e}")))?;
    Ok(())
}

/// Split `total` strokes across layers.
///
/// With `alpha` = subject area fraction (clamped to [0, 1]) and `T` = total:
/// - `edge`       = round(0.25·T)
/// - `subject`    = clamp(round(0.45·T·(0.5 + alpha)), floor(0.30·T), T − edge)
///   — the 0.30·T floor guarantees the subject layer keeps a minimum share
///   even when the subject area is tiny; the T − edge cap keeps the sum valid.
/// - `background` = T − subject − edge (the remainder).
///
/// Guarantees: all three values are usize ≥ 0 and
/// `subject + background + edge == T` for every `T ≥ 0` and `alpha`.
/// 参数化预算分配（语义见下；系数来自 `p`）。
fn allocate_budget_with(total: usize, alpha: f64, p: &PlanParams) -> LayerBudget {
    let alpha = alpha.clamp(0.0, 1.0);
    let edge = (p.edge_frac * total as f64).round() as usize;
    let floor = (p.subject_floor * total as f64).round() as usize;
    let cap = total - edge.min(total); // ≥ floor for every T ≥ 0 (checked in tests)
    let subject = ((p.subject_gain * total as f64) * (0.5 + alpha)).round() as usize;
    // `.min(cap)` applied last so `background = total − subject − edge ≥ 0`.
    let subject = subject.max(floor).min(cap);
    LayerBudget {
        subject,
        background: total - subject - edge,
        edge,
    }
}

/// One round of 3×3 majority voting on a 0/255 mask. A pixel becomes 255 iff
/// at least 5 of the 9 window samples (out-of-bounds replicated) are 255.
fn majority_smooth3x3(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut ones = 0u32;
            for dy in -1i32..=1 {
                let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    ones += u32::from(mask[yy * w + xx] != 0);
                }
            }
            out[y * w + x] = if ones >= 5 { 255 } else { 0 };
        }
    }
    out
}

/// Sobel gradient magnitude of an 8-bit image, as f32 (border replicated).
/// Kernels: Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = Gxᵀ.
fn sobel_magnitude(img: &[u8], w: usize, h: usize) -> Vec<f32> {
    let at = |x: i64, y: i64| -> f32 {
        let xx = x.clamp(0, w as i64 - 1) as usize;
        let yy = y.clamp(0, h as i64 - 1) as usize;
        img[yy * w + xx] as f32
    };
    let mut mag = vec![0.0f32; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let tl = at(x - 1, y - 1);
            let t = at(x, y - 1);
            let tr = at(x + 1, y - 1);
            let l = at(x - 1, y);
            let r = at(x + 1, y);
            let bl = at(x - 1, y + 1);
            let b = at(x, y + 1);
            let br = at(x + 1, y + 1);
            let gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
            let gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
            mag[y as usize * w + x as usize] = (gx * gx + gy * gy).sqrt();
        }
    }
    mag
}

/// Scale an f32 field to 0..=255 by its maximum (all-zero input → all-zero).
fn normalize_to_u8(field: &[f32]) -> Vec<u8> {
    let max = field.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return vec![0; field.len()];
    }
    let scale = 255.0 / max;
    field
        .iter()
        .map(|&v| (v * scale).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// N px dilation via a separable max-filter (square structuring element,
/// Chebyshev radius `r`; `r == 0` returns a copy): horizontal pass then
/// vertical pass.
fn dilate_radius(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0u8; w * h];
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            tmp[row + x] = src[row + lo..=row + hi].iter().copied().max().unwrap_or(0);
        }
    }
    for y in 0..h {
        for x in 0..w {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            out[y * w + x] = (lo..=hi).map(|yy| tmp[yy * w + x]).max().unwrap_or(0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hard-edged disk of 255s on a 0 background.
    fn circle_matte(w: u32, h: u32, cx: f32, cy: f32, r: f32) -> Matte {
        let mut data = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r * r {
                    data[(y * w + x) as usize] = 255;
                }
            }
        }
        Matte { width: w, height: h, data }
    }

    #[test]
    fn otsu_finds_bimodal_valley() {
        // 40% of pixels at 30, 60% at 200: the threshold must land strictly
        // between the two peaks and classify both peaks correctly.
        let n = 64 * 64;
        let mut data = vec![200u8; n];
        for v in data.iter_mut().take(n * 40 / 100) {
            *v = 30;
        }
        let t = otsu_threshold(&data);
        assert!(t > 30 && t <= 200, "threshold {t} not in valley (30, 200]");
        assert_eq!(data.iter().filter(|&&v| v == 30).count(), n * 40 / 100);
        assert_eq!(
            data.iter().filter(|&&v| v >= t).count(),
            n - n * 40 / 100,
            "binarization must classify every peak-200 pixel as subject"
        );
    }

    #[test]
    fn otsu_constant_matte_fallback() {
        assert_eq!(otsu_threshold(&[200u8; 16]), 200); // bright constant → all subject
        assert_eq!(otsu_threshold(&[10u8; 16]), 11); // dark constant → all background
        assert_eq!(otsu_threshold(&[]), 0);
    }

    #[test]
    fn subject_background_are_complementary() {
        let matte = circle_matte(64, 64, 32.0, 32.0, 20.0);
        let plan = plan_layers(&matte, 1500).expect("plan_layers");
        let n = 64 * 64;
        assert_eq!(plan.subject_mask.len(), n);
        assert_eq!(plan.w, 64);
        assert_eq!(plan.h, 64);
        for i in 0..n {
            let (s, b) = (plan.subject_mask[i], plan.background_mask[i]);
            assert!(s == 0 || s == 255, "subject value {s} not binary at {i}");
            assert!(b == 0 || b == 255, "background value {b} not binary at {i}");
            assert_eq!(
                u8::from(s != 0) + u8::from(b != 0),
                1,
                "pixel {i} not exactly one of subject/background"
            );
        }
        // Interior stays subject, far corner stays background.
        assert_eq!(plan.subject_mask[32 * 64 + 32], 255);
        assert_eq!(plan.background_mask[2 * 64 + 2], 255);
        assert_eq!(plan.subject_area() + plan.background_area(), n);
    }

    #[test]
    fn boundary_band_lies_on_circle_rim() {
        let matte = circle_matte(64, 64, 32.0, 32.0, 20.0);
        let plan = plan_layers(&matte, 1500).expect("plan_layers");
        let idx = |x: usize, y: usize| y * 64 + x;
        assert_eq!(plan.boundary_band[idx(32, 32)], 0, "center must be flat");
        assert!(plan.boundary_band[idx(52, 32)] > 0, "east rim must be in band");
        assert!(plan.boundary_band[idx(32, 12)] > 0, "north rim must be in band");
        assert!(plan.boundary_band[idx(15, 47)] > 0, "southwest rim must be in band");
        assert_eq!(plan.boundary_band[idx(2, 2)], 0, "far corner must be outside band");
        // Band is thin: a pixel 6+ px inside the disk is untouched by the
        // 3 px dilation of a ~1.5 px wide Sobel ring.
        assert_eq!(plan.boundary_band[idx(32, 26)], 0, "6 px inside rim must be flat");
    }

    #[test]
    fn budget_sums_to_total() {
        for total in [0usize, 1, 2, 3, 7, 10, 1500, 99_999] {
            for &alpha in &[0.0f64, 0.05, 0.3, 0.5, 0.9, 1.0] {
                let b = allocate_budget_with(total, alpha, &PlanParams::default());
                assert_eq!(b.subject + b.background + b.edge, total, "total={total} alpha={alpha}");
                assert!(b.edge <= total);
            }
        }
        // Exact shares at T=1500: edge = 0.25·T, floor = 0.30·T.
        let b = allocate_budget_with(1500, 0.5, &PlanParams::default());
        assert_eq!(b.edge, 375);
        assert_eq!(b.subject, 675); // 0.45·1500·(0.5+0.5)
        assert_eq!(b.background, 450);
        // Small-subject guarantee: floor keeps subject ≥ 0.30·T.
        let b = allocate_budget_with(1500, 0.02, &PlanParams::default());
        assert!(b.subject >= 450, "subject floor violated: {}", b.subject);
    }

    #[test]
    fn plan_budget_sums_to_total() {
        let matte = circle_matte(64, 64, 32.0, 32.0, 20.0);
        let plan = plan_layers(&matte, 1500).expect("plan_layers");
        let b = plan.budget;
        assert_eq!(b.subject + b.background + b.edge, 1500);
        assert!(b.subject >= 450, "alpha≈0.31 should still hit the 0.30·T floor");
    }

    #[test]
    fn constant_mattes_degrade_gracefully() {
        let bright = Matte { width: 8, height: 8, data: vec![255u8; 64] };
        let plan = plan_layers(&bright, 100).expect("plan_layers");
        assert_eq!(plan.subject_area(), 64);
        assert_eq!(plan.band_area(), 0); // no gradient anywhere

        let dark = Matte { width: 8, height: 8, data: vec![0u8; 64] };
        let plan = plan_layers(&dark, 100).expect("plan_layers");
        assert_eq!(plan.subject_area(), 0);
        assert_eq!(plan.background_area(), 64);
        assert_eq!(plan.budget.subject + plan.budget.background + plan.budget.edge, 100);
    }

    #[test]
    fn rejects_bad_mattes() {
        assert!(plan_layers(&Matte { width: 0, height: 0, data: vec![] }, 10).is_err());
        assert!(plan_layers(&Matte { width: 4, height: 4, data: vec![0; 15] }, 10).is_err());
    }

    #[test]
    fn save_masks_writes_all_pngs() {
        let matte = circle_matte(32, 32, 16.0, 16.0, 10.0);
        let plan = plan_layers(&matte, 300).expect("plan_layers");
        let dir = std::env::temp_dir().join(format!("fv_plan_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        save_masks(&plan, &dir).expect("save_masks");
        for name in ["subject.png", "background.png", "boundary_band.png", "overlay.png"] {
            assert!(dir.join(name).is_file(), "missing {name}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
