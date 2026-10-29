//! Layer-planning probe: image in → matte (MODNet, or synthetic fallback) →
//! [`plan_layers`] → masks + overlay written to an output directory, with the
//! budget split and mask area fractions printed.
//!
//! Degradation: if the image, the ONNX model, or the onnxruntime dylib is
//! missing (or inference fails), a synthetic soft-ellipse matte is used so
//! the planning pipeline can always be exercised.
//!
//! Usage:
//! cargo run --release -p fv-seg --example plan_probe [image] [out_dir]
//! Both arguments are optional: with no image a synthetic matte is used;
//! out_dir defaults to the system temp directory.

use std::path::Path;

use fv_seg::Matte;

const TOTAL_BUDGET: usize = 1500;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args.get(1).map(String::as_str);
    let out_dir = args
        .get(2)
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::temp_dir().join("fv_plan_out").display().to_string());
    println!("image   : {}", image_path.unwrap_or("<none: synthetic matte>"));
    println!("out dir : {out_dir}");
    println!("budget  : total={TOTAL_BUDGET}");

    // ---- matte: real MODNet when possible, synthetic fallback otherwise ----
    let (matte, source) = match image_path {
        Some(path) => load_or_synthesize_matte(path),
        None => (
            synthetic_matte(320, 240),
            "synthetic (no image argument given)".to_string(),
        ),
    };
    println!("matte   : {source} ({}x{})", matte.width, matte.height);
    let (min, max, mean) = matte.stats();
    println!("matte stats: min={min} max={max} mean={mean:.2}");

    // ---- plan ----
    let t_plan = std::time::Instant::now();
    let plan = fv_seg::plan_layers(&matte, TOTAL_BUDGET)
        .unwrap_or_else(|e| panic!("plan_layers: {e}"));
    println!("planned in {:?}", t_plan.elapsed());
    println!(
        "otsu threshold: {}",
        fv_seg::plan::otsu_threshold(&matte.data)
    );

    // ---- save ----
    let out = Path::new(&out_dir);
    fv_seg::save_masks(&plan, out).unwrap_or_else(|e| panic!("save_masks: {e}"));
    matte
        .save_png(out.join("matte.png"))
        .unwrap_or_else(|e| eprintln!("warn: save matte.png: {e}"));
    println!(
        "wrote subject.png / background.png / boundary_band.png / overlay.png / matte.png into {}",
        out.display()
    );

    // ---- report ----
    let total_px = (plan.w as usize) * (plan.h as usize);
    let pct = |count: usize| 100.0 * count as f64 / total_px as f64;
    let b = plan.budget;
    println!("budget split : subject={} background={} edge={} (sum={})", b.subject, b.background, b.edge, b.subject + b.background + b.edge);
    println!(
        "mask areas   : subject {:.1}% | background {:.1}% | band(nonzero) {:.1}% | mean band intensity {:.1}",
        pct(plan.subject_area()),
        pct(plan.background_area()),
        pct(plan.band_area()),
        plan.boundary_band.iter().map(|&v| v as u64).sum::<u64>() as f64 / total_px as f64
    );
}

/// Try the real segmentation path; fall back to a synthetic matte with a
/// printed reason. Returns (matte, human-readable source description).
fn load_or_synthesize_matte(image_path: &str) -> (Matte, String) {
    match image::open(image_path) {
        Err(e) => {
            let m = synthetic_matte(320, 240);
            (m, format!("synthetic (image failed to open: {e})"))
        }
        Ok(img) => {
            let model = fv_seg::default_model_path();
            let dylib = fv_seg::default_dylib_path();
            if !model.is_file() || !dylib.is_file() {
                let m = synthetic_matte(img.width(), img.height());
                return (
                    m,
                    format!(
                        "synthetic (model {} or dylib {} missing)",
                        model.display(),
                        dylib.display()
                    ),
                );
            }
            match fv_seg::segment_image(&img, &model, &dylib) {
                Ok(m) => (m, "MODNet (fv_seg::segment_image)".to_string()),
                Err(e) => {
                    let m = synthetic_matte(img.width(), img.height());
                    (m, format!("synthetic (inference failed: {e})"))
                }
            }
        }
    }
}

/// Synthetic test matte at `w`×`h`: a soft-edged ellipse (torso stand-in)
/// plus a smaller circle blob, with a ~1/10-radius soft transition so the
/// Sobel band has something realistic to bite on.
fn synthetic_matte(w: u32, h: u32) -> Matte {
    let (w, h) = (w.max(64), h.max(64));
    let (wf, hf) = (w as f32, h as f32);
    let (cx, cy) = (0.42 * wf, 0.48 * hf);
    let (ax, ay) = (0.30 * wf, 0.38 * hf);
    let (bcx, bcy, br) = (0.72 * wf, 0.74 * hf, 0.10 * wf);

    let mut data = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            let q_ellipse =
                (((fx - cx) / ax).powi(2) + ((fy - cy) / ay).powi(2)).sqrt();
            let q_blob = ((fx - bcx).powi(2) + (fy - bcy).powi(2)).sqrt() / br;
            let q = q_ellipse.min(q_blob);
            let alpha = ((1.0 - q) * 255.0 * 10.0).clamp(0.0, 255.0);
            data[(y * w + x) as usize] = alpha as u8;
        }
    }
    Matte { width: w, height: h, data }
}
