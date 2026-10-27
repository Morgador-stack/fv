//! End-to-end probe: image in → matte PNG out, with timings and stats.
//!
//! Usage: cargo run --release -p fv-seg --example seg_probe [image] [out.png]

use std::time::Instant;

use fv_seg::Matte;

const DEFAULT_IMAGE: &str = r"D:\WorkSpace\ZCWorkspace\forza-vinyl\refs\forza-painter-fh6\assets\imgs\ayylmao.png";
const DEFAULT_OUT: &str = "matte.png";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args.get(1).map(String::as_str).unwrap_or(DEFAULT_IMAGE);
    let out_path = args.get(2).map(String::as_str).unwrap_or(DEFAULT_OUT);

    let model = fv_seg::default_model_path();
    let dylib = fv_seg::default_dylib_path();
    println!("image : {image_path}");
    println!("model : {}", model.display());
    println!("dylib : {}", dylib.display());

    let t_load = Instant::now();
    let img = image::open(image_path).unwrap_or_else(|e| panic!("open image: {e}"));
    println!(
        "loaded image {}x{} in {:?}",
        img.width(),
        img.height(),
        t_load.elapsed()
    );

    let t_seg = Instant::now();
    let _cold_matte: Matte = fv_seg::segment_image(&img, &model, &dylib)
        .unwrap_or_else(|e| panic!("segmentation failed: {e}"));
    let cold = t_seg.elapsed();
    println!("inference+pre/post (cold, incl. session+model load): {cold:?}");

    // Warm pass in the same process: session already loaded.
    let t_warm = Instant::now();
    let matte: Matte = fv_seg::segment_image(&img, &model, &dylib)
        .unwrap_or_else(|e| panic!("warm segmentation failed: {e}"));
    println!("inference+pre/post (warm): {:?}", t_warm.elapsed());

    let t_save = Instant::now();
    matte
        .save_png(out_path)
        .unwrap_or_else(|e| panic!("save matte: {e}"));
    println!("saved {out_path} in {:?}", t_save.elapsed());

    let (min, max, mean) = matte.stats();
    println!("matte stats: min={min} max={max} mean={mean:.2} ({}x{})", matte.width, matte.height);

    // Read the saved PNG back to prove the artifact is a valid, non-degenerate file.
    let reloaded = image::open(out_path).unwrap_or_else(|e| panic!("re-open matte: {e}"));
    let gray = reloaded.to_luma8();
    let rmin = gray.pixels().map(|p| p.0[0]).min().unwrap();
    let rmax = gray.pixels().map(|p| p.0[0]).max().unwrap();
    println!("re-read {}: {}x{}, min={rmin} max={rmax}", out_path, gray.width(), gray.height());
    if min == max {
        eprintln!("WARNING: matte is constant ({min}) — likely degenerate output");
    } else if rmax == 0 {
        eprintln!("WARNING: saved matte is all-zero");
    } else {
        println!("OK: matte non-empty, non-constant");
    }
}
