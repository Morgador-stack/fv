//! DUT-OMRON parquet → (image, mask) PNG 对提取。
//! `pqdump <in.parquet> <outdir> [n]`（默认 10 对；自动识别 byte 列）。
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pqdump <in.parquet> <outdir> [n]");
        std::process::exit(2);
    }
    let n: usize = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(10);
    let outdir = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&outdir).unwrap();

    let file = File::open(&args[1]).expect("open parquet");
    let reader = SerializedFileReader::new(file).expect("read parquet footer");
    let schema = reader.metadata().file_metadata().schema();
    // 递归收集叶子列（HF image 列是 group{bytes,path}）
    fn leaves(prefix: &str, f: &parquet::schema::types::Type, out: &mut Vec<(String, String)>) {
        let name = if prefix.is_empty() {
            f.name().to_string()
        } else {
            format!("{prefix}.{}", f.name())
        };
        if f.is_primitive() {
            out.push((name, format!("{:?}", f.get_physical_type())));
        } else {
            for c in f.get_fields() {
                leaves(&name, c, out);
            }
        }
    }
    let mut fields = Vec::new();
    for f in schema.get_fields() {
        leaves("", f, &mut fields);
    }
    eprintln!("schema: {fields:?}");

    // 找 byte 列（image/mask 通常是 BYTE_ARRAY）
    let byte_cols: Vec<String> = fields
        .iter()
        .filter(|(_, t)| t.contains("BYTE_ARRAY"))
        .map(|(n, _)| n.clone())
        .collect();
    eprintln!("byte columns: {byte_cols:?}");

    for (ri, row) in reader.get_row_iter(None).expect("row iter").take(n).enumerate() {
        let row = row.expect("row");
        let names: Vec<String> = row
            .get_column_iter()
            .map(|(n, _)| n.to_string())
            .collect();
        let mut saved = Vec::new();
        for (ci, name) in names.iter().enumerate() {
            // HF 嵌套：顶层列是 group{bytes,path} → 进组取 bytes（首叶子）
            let bytes = if let Ok(g) = row.get_group(ci) {
                g.get_bytes(0).ok().map(|b| b.data().to_vec())
            } else {
                row.get_bytes(ci).ok().map(|b| b.data().to_vec())
            };
            if let Some(b) = bytes {
                // 试 PNG/JPEG 魔数，原样落盘（PNG 直接用；JPEG 由后续转换）
                let ext = if b.starts_with(&[0x89, b'P', b'N', b'G']) {
                    "png"
                } else if b.starts_with(&[0xFF, 0xD8]) {
                    "jpg"
                } else {
                    "bin"
                };
                let path = outdir.join(format!("{ri:03}_{name}.{ext}"));
                std::fs::write(&path, &b).unwrap();
                saved.push(format!("{}={}B", path.display(), b.len()));
            }
        }
        println!("row {ri}: {}", saved.join(" "));
    }
}
