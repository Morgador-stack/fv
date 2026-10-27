//! geometrize 兼容 JSON（与 vinylizer / forza-painter 注入器互通的交换格式）。
//!
//! 格式（vinylizer `json_writer.cpp` 实测）：
//! ```json
//! {"shapes": [
//!   {"type": 16, "data": [cx, cy, rx, ry, angle], "color": [r, g, b, a]},
//!   {"type": 1,  "data": [cx, cy, w, h],           "color": [r, g, b, a]}
//! ]}
//! ```
//! data 坐标为原图像素；color[3] = 不透明度×255（SoftEllipse 专用，基础形状 255）。

use crate::shape::{Shape, ShapeKind};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeJson {
    #[serde(rename = "type")]
    pub kind: u32,
    pub data: Vec<f32>,
    pub color: [u8; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignJson {
    pub shapes: Vec<ShapeJson>,
}

pub fn shape_to_json(s: &Shape) -> ShapeJson {
    let mut data = vec![s.cx, s.cy, s.rx, s.ry];
    if s.kind != ShapeKind::Rect {
        data.push(s.angle_deg.rem_euclid(360.0));
    }
    ShapeJson {
        kind: s.kind as u32,
        data,
        color: s.color,
    }
}

/// 以 2 空格缩进写出设计 JSON（与 vinylizer `dump(2)` 输出一致）。
pub fn write_design(path: &Path, shapes: &[Shape]) -> std::io::Result<()> {
    let design = DesignJson {
        shapes: shapes.iter().map(shape_to_json).collect(),
    };
    let mut buf = Vec::new();
    serde_json::to_writer_pretty(&mut buf, &design).map_err(io_err)?;
    std::fs::write(path, buf)
}

/// 读回设计 JSON（测试与后续注入器校验用）。
pub fn read_design(path: &Path) -> std::io::Result<DesignJson> {
    let buf = std::fs::read(path)?;
    read_design_bytes(&buf)
}

/// `read_design` 的字节版（CLI 工具从已读字节解析）。
/// 读入即校验有限性（1e999 经 f64→f32 变 inf 的入口闸）。
pub fn read_design_bytes(buf: &[u8]) -> std::io::Result<DesignJson> {
    let d: DesignJson = serde_json::from_slice(buf).map_err(io_err)?;
    for (i, s) in d.shapes.iter().enumerate() {
        if !s.data.iter().all(|v| v.is_finite()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("shapes[{i}] 含非有限值（inf/NaN）"),
            ));
        }
    }
    Ok(d)
}

fn io_err(e: serde_json::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipse_shape_serializes_with_angle() {
        let s = Shape::ellipse(10.0, 20.0, 3.5, 4.5, 370.0, [1, 2, 3]);
        let j = shape_to_json(&s);
        assert_eq!(j.kind, 16);
        assert_eq!(j.data.len(), 5);
        assert!((j.data[4] - 10.0).abs() < 1e-4, "angle fmod 360");
        assert_eq!(j.color, [1, 2, 3, 255]);
        let json = serde_json::to_string(&DesignJson { shapes: vec![j] }).unwrap();
        assert!(json.contains(r#""type":16"#));
        assert!(json.contains(r#""color":"#));
    }

    #[test]
    fn write_read_roundtrip() {
        let dir = std::env::temp_dir().join("fv-common-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.json");
        let shapes = vec![
            Shape::ellipse(5.0, 6.0, 1.0, 2.0, 30.0, [9, 8, 7]),
            Shape {
                kind: ShapeKind::Rect,
                cx: 1.0,
                cy: 2.0,
                rx: 3.0,
                ry: 4.0,
                angle_deg: 0.0,
                color: [0, 0, 0, 255],
            },
        ];
        write_design(&path, &shapes).unwrap();
        let back = read_design(&path).unwrap();
        assert_eq!(back.shapes.len(), 2);
        assert_eq!(back.shapes[0].data.len(), 5);
        assert_eq!(back.shapes[1].data.len(), 4);
        assert_eq!(back.shapes[1].kind, 1);
    }
}
