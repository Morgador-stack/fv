//! fv-seg — matting foundation on ONNX Runtime.
//!
//! "image in → alpha matte out": loads a portrait-matting ONNX model
//! (MODNet by default) via the `ort` crate in load-dynamic mode, runs CPU
//! inference, and returns a grayscale alpha matte at the input resolution.
//!
//! Paths are configurable via env vars:
//! - `FV_SEG_MODEL`    — path to the `.onnx` model
//! - `FV_SEG_ORT_DYLIB`— path to `onnxruntime.dll` (forwarded to `ORT_DYLIB_PATH`)
//!
//! Default locations live in the repo-level `deps/` directory.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

pub mod pipeline;
pub mod plan;
pub use plan::{plan_layers, plan_layers_with, save_masks, LayerBudget, LayerPlan, PlanParams};

/// Env var overriding the model path.
pub const MODEL_ENV: &str = "FV_SEG_MODEL";
/// Env var overriding the onnxruntime dylib path (forwarded to `ORT_DYLIB_PATH`).
pub const DYLIB_ENV: &str = "FV_SEG_ORT_DYLIB";
/// Env var read directly by `ort` (load-dynamic). Must be set before the
/// first ORT call in the process.
pub const ORT_DYLIB_PATH_ENV: &str = "ORT_DYLIB_PATH";

/// Default repo-level deps dir: `<repo>/deps` — three levels above this crate.
fn default_deps_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("manifest dir has enough ancestors")
        .join("deps")
        .join(rel)
}

/// Default model: MODNet (Apache-2.0), dynamic `input` [1,3,H,W] f32.
pub fn default_model_path() -> PathBuf {
    std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_deps_path("models/modnet.onnx"))
}

/// Default onnxruntime dylib (1.22.0 win-arm64, matches ort 2.0.0-rc.10).
pub fn default_dylib_path() -> PathBuf {
    std::env::var_os(DYLIB_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_deps_path("ort/onnxruntime.dll"))
}

/// Model preprocessing parameters (MODNet): shortest edge, both edges kept
/// a multiple of this divisor.
const INPUT_SHORT_EDGE: u32 = 512;
const INPUT_SIZE_DIVISIBILITY: u32 = 32;

/// Grayscale alpha matte, row-major, 0 (background) .. 255 (foreground),
/// at the resolution of the segmented input image.
pub struct Matte {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Matte {
    /// Save as an 8-bit grayscale PNG.
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let img: image::GrayImage = image::ImageBuffer::from_raw(self.width, self.height, self.data.clone())
            .ok_or_else(|| "matte buffer size mismatch".to_string())?;
        img.save(path.as_ref()).map_err(|e| format!("save matte: {e}"))
    }

    /// (min, max, mean) over the matte, for sanity checks.
    pub fn stats(&self) -> (u8, u8, f32) {
        let mut min = u8::MAX;
        let mut max = u8::MIN;
        let mut sum: u64 = 0;
        for &v in &self.data {
            min = min.min(v);
            max = max.max(v);
            sum += v as u64;
        }
        let mean = if self.data.is_empty() { 0.0 } else { sum as f32 / self.data.len() as f32 };
        (min, max, mean)
    }
}

/// Segment an image file, producing an alpha matte at the same resolution.
/// 默认 MODNet（人像专用）；任意显著物体用 [`segment_by`] + [`SegModel::U2Net`]。
pub fn segment(path: impl AsRef<Path>) -> Result<Matte, String> {
    segment_by(path, SegModel::Modnet)
}

/// 模型注册表。U2-Net 为通用显著性（Apache-2.0，xuebinqin/U-2-Net 的 ONNX 转换，
/// 320×320，ImageNet mean/std 归一化，输出 logits 需 min-max 归一）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegModel {
    /// MODNet 人像 matting（Xenova/modnet，512 短边，[-1,1]，输出已含 sigmoid）
    Modnet,
    /// U2-Net 通用显著物体（frankminors123/U2Net_ONNX，320×320，ImageNet 归一化）
    U2Net,
}

/// env 覆盖：U2Net 模型路径（缺省 deps/models/u2net.onnx）
pub const U2NET_MODEL_ENV: &str = "FV_SEG_U2NET";

/// U2Net 模型缺省路径。
pub fn default_u2net_path() -> PathBuf {
    std::env::var_os(U2NET_MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_deps_path("models/u2net.onnx"))
}

/// 按模型分割。
pub fn segment_by(path: impl AsRef<Path>, model: SegModel) -> Result<Matte, String> {
    let (mp, dp): (PathBuf, PathBuf) = match model {
        SegModel::Modnet => (default_model_path(), default_dylib_path()),
        SegModel::U2Net => (default_u2net_path(), default_dylib_path()),
    };
    segment_with(path, &mp, &dp)
}

/// `segment` with explicit model/dylib paths (used by the probe and tests).
pub fn segment_with(
    image_path: impl AsRef<Path>,
    model_path: &Path,
    dylib_path: &Path,
) -> Result<Matte, String> {
    let img = image::open(image_path.as_ref()).map_err(|e| format!("open image: {e}"))?;
    let model = if model_path.ends_with("u2net.onnx") {
        SegModel::U2Net
    } else {
        SegModel::Modnet
    };
    segment_image_by(&img, model_path, dylib_path, model)
}

/// Full pipeline on an already-decoded image（保留旧签名：按文件名推断模型）。
pub fn segment_image(
    img: &image::DynamicImage,
    model_path: &Path,
    dylib_path: &Path,
) -> Result<Matte, String> {
    let model = if model_path.ends_with("u2net.onnx") {
        SegModel::U2Net
    } else {
        SegModel::Modnet
    };
    segment_image_by(img, model_path, dylib_path, model)
}

/// 按模型规格的完整管线：resize → 归一化 → ORT 推理 → 后处理 → 原分辨率 matte。
pub fn segment_image_by(
    img: &image::DynamicImage,
    model_path: &Path,
    dylib_path: &Path,
    model: SegModel,
) -> Result<Matte, String> {
    if !model_path.is_file() {
        return Err(format!("model file not found: {}", model_path.display()));
    }
    if !dylib_path.is_file() {
        return Err(format!("onnxruntime dylib not found: {}", dylib_path.display()));
    }

    // ort (load-dynamic) reads ORT_DYLIB_PATH when the library is first loaded,
    // which happens on the first Session creation. Edition 2024 requires unsafe.
    if std::env::var_os(ORT_DYLIB_PATH_ENV).is_none() {
        // SAFETY: happens before any ORT session exists; other threads must not
        // have touched the ORT runtime yet (single-threaded init path).
        unsafe { std::env::set_var(ORT_DYLIB_PATH_ENV, dylib_path) };
    }

    let rgb = img.to_rgb8();
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    // ---- preprocess: 按模型规格 ----
    let (tw, th) = match model {
        SegModel::Modnet => {
            let scale = INPUT_SHORT_EDGE as f32 / orig_w.min(orig_h) as f32;
            let th = ((orig_h as f32 * scale).round() as u32).max(INPUT_SHORT_EDGE);
            let tw = ((orig_w as f32 * scale).round() as u32).max(INPUT_SHORT_EDGE);
            (
                tw.div_ceil(INPUT_SIZE_DIVISIBILITY) * INPUT_SIZE_DIVISIBILITY,
                th.div_ceil(INPUT_SIZE_DIVISIBILITY) * INPUT_SIZE_DIVISIBILITY,
            )
        }
        // U2-Net 训练分辨率 320×320（模型卡）
        SegModel::U2Net => (320, 320),
    };

    let resized = image::imageops::resize(&rgb, tw, th, image::imageops::FilterType::Triangle);

    let n = (tw * th) as usize;
    let mut chw = vec![0.0f32; 3 * n];
    match model {
        SegModel::Modnet => {
            // CHW, (v/255 - 0.5) / 0.5  →  [-1, 1]
            for (i, px) in resized.pixels().enumerate() {
                let [r, g, b] = px.0;
                chw[i] = (r as f32 / 255.0 - 0.5) / 0.5;
                chw[n + i] = (g as f32 / 255.0 - 0.5) / 0.5;
                chw[2 * n + i] = (b as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        // U2-Net: (v/255 − mean)/std, ImageNet mean/std（模型卡）
        SegModel::U2Net => {
            const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
            const STD: [f32; 3] = [0.229, 0.224, 0.225];
            for (i, px) in resized.pixels().enumerate() {
                let [r, g, b] = px.0;
                chw[i] = (r as f32 / 255.0 - MEAN[0]) / STD[0];
                chw[n + i] = (g as f32 / 255.0 - MEAN[1]) / STD[1];
                chw[2 * n + i] = (b as f32 / 255.0 - MEAN[2]) / STD[2];
            }
        }
    }

    // ---- inference (CPU EP by default; dylib path set above) ----
    let mut session = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("load model {}: {e}", model_path.display()))?;

    let in_name = session
        .inputs
        .first()
        .map(|i| i.name.clone())
        .unwrap_or_else(|| "input".to_string());
    let out_name = session
        .outputs
        .first()
        .map(|o| o.name.clone())
        .unwrap_or_else(|| "output".to_string());

    let input = Tensor::from_array(([1usize, 3, th as usize, tw as usize], chw))
        .map_err(|e| format!("build input tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs![in_name.as_str() => input])
        .map_err(|e| format!("inference: {e}"))?;

    let out_value = outputs
        .get(out_name.as_str())
        .ok_or_else(|| format!("output '{out_name}' missing"))?;
    let (_, raw) = out_value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("extract output: {e}"))?;
    if raw.len() != n {
        return Err(format!(
            "unexpected matte element count: got {}, expected {n} ({tw}x{th})",
            raw.len()
        ));
    }

    // ---- postprocess: [0,1] → u8 at model resolution, then resize back ----
    let matte_small: image::GrayImage = {
        let buf: Vec<u8> = match model {
            SegModel::Modnet => raw
                .iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
            // U2-Net 输出是未归一的显著性图：min-max 归一到 [0,1]
            SegModel::U2Net => {
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for &v in raw.iter() {
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let span = (mx - mn).max(1e-6);
                raw.iter()
                    .map(|&v| ((v - mn) / span * 255.0).round() as u8)
                    .collect()
            }
        };
        image::ImageBuffer::from_raw(tw, th, buf).ok_or_else(|| "matte buffer mismatch".to_string())?
    };
    let matte_full =
        image::imageops::resize(&matte_small, orig_w, orig_h, image::imageops::FilterType::Triangle);

    Ok(Matte {
        width: orig_w,
        height: orig_h,
        data: matte_full.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Missing model/dylib must surface as a clean `Err`, never a panic.
    #[test]
    fn missing_files_are_err_not_panic() {
        let img = image::DynamicImage::new_rgb8(8, 8);
        let r = segment_image(
            &img,
            Path::new("Z:/definitely/not/here.onnx"),
            Path::new("Z:/definitely/not/here/onnxruntime.dll"),
        );
        assert!(r.is_err(), "expected Err when model/dylib are missing");
    }

    /// End-to-end smoke test; skipped (not failed) when the model or dylib
    /// is not present on this machine.
    #[test]
    fn segment_smoke_if_deps_present() {
        let model = default_model_path();
        let dylib = default_dylib_path();
        if !model.is_file() || !dylib.is_file() {
            eprintln!(
                "skip: model ({}) or dylib ({}) not present",
                model.display(),
                dylib.display()
            );
            return;
        }

        // 64x64 test image: left half white, right half black.
        let mut img = image::GrayImage::new(64, 64);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            *p = if x < 32 { image::Luma([255]) } else { image::Luma([0]) };
        }
        let img = image::DynamicImage::ImageLuma8(img);

        // Pipeline assertions: runs cleanly and returns the input resolution.
        // Note: MODNet is a *portrait* matting model, so a synthetic
        // (person-free) image may legitimately produce a near-zero matte —
        // that is model behavior, not a pipeline failure. Real-content
        // validation lives in examples/seg_probe.rs.
        let matte = segment_image(&img, &model, &dylib).expect("segment should succeed");
        assert_eq!(matte.width, 64);
        assert_eq!(matte.height, 64);
        let (min, max, mean) = matte.stats();
        eprintln!("synthetic-image matte stats: min={min} max={max} mean={mean:.2}");
    }
}
