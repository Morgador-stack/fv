//! U2-Net 通用显著性 vs MODNet 人像对比探针：ayylmao（外星人）上 MODNet 应近零、U2Net 应圈出头。
//! `cargo run --release -p fv-seg --example u2net_probe -- <img>`

fn main() {
    let img = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "D:/WorkSpace/ZCWorkspace/forza-vinyl/refs/forza-painter-fh6/assets/imgs/ayylmao.png".into());
    for (name, model) in [("modnet", fv_seg::SegModel::Modnet), ("u2net", fv_seg::SegModel::U2Net)] {
        let t0 = std::time::Instant::now();
        match fv_seg::segment_by(&img, model) {
            Ok(matte) => {
                let (mn, mx, mean) = matte.stats();
                let out = format!(
                    "C:/Users/Lenovo/AppData/Local/Temp/fv_matte_{name}.png"
                );
                let _ = matte.save_png(&out);
                println!(
                    "{name}: min={mn} max={mx} mean={mean:.1} {}ms -> {out}",
                    t0.elapsed().as_millis()
                );
            }
            Err(e) => println!("{name}: ERR {e}"),
        }
    }
}
