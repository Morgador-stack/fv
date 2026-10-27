//! 设计 JSON → 内存层字段的坐标换算（规则来自 vinylizer write_layer_h，三方交叉验证）。
//!
//! - 位置：`(x, −y)`（JSON 图像系 y 向下，游戏系 y 向上）
//! - 缩放：`w / divisor`（矩形 127，椭圆/软椭圆 63；divisor 是"游戏缩放 1.0 对应的宽"）
//! - 旋转：`360° − angle`
//! - 颜色：RGBA 原样（基础形状 alpha 必须 255，228 软椭圆 alpha=不透明度）
//! - mask 恒 0

use crate::layout::Layout;
use fv_common::json::ShapeJson;

/// 一层待写入的内存字段值（序列化前）。
#[derive(Debug, Clone, PartialEq)]
pub struct LayerFields {
    pub pos: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
    pub color: [u8; 4],
    pub shape_id: u8,
}

/// 画布平移（--center 用）：设计坐标原点从左上移到画布中心。
#[derive(Debug, Clone, Copy)]
pub struct CanvasOffset {
    pub dx: f32,
    pub dy: f32,
}

impl CanvasOffset {
    pub fn center_for(canvas_w: f32, canvas_h: f32) -> Self {
        Self {
            dx: canvas_w / 2.0,
            dy: canvas_h / 2.0,
        }
    }
    pub const NONE: CanvasOffset = CanvasOffset { dx: 0.0, dy: 0.0 };
}

pub fn prepare(s: &ShapeJson, lay: &Layout, offset: CanvasOffset) -> Result<LayerFields, String> {
    let shape_id = lay
        .shape_id_of(s.kind)
        .ok_or_else(|| format!("不支持的形状 type {}", s.kind))?;
    let divisor = lay
        .divisor_of(s.kind)
        .ok_or_else(|| format!("缺少 type {} 的缩放除数", s.kind))?;
    if s.data.len() < 4 {
        return Err(format!("type {} 的 data 长度 {} < 4", s.kind, s.data.len()));
    }
    let angle = if s.data.len() >= 5 { s.data[4] } else { 0.0 };
    // is_finite 闸（写内存前最后一道）：JSON 的 1e999 会经 f64→f32 变 inf，
    // inf/NaN 坐标写进游戏内存 = 崩游戏。CODE_REVIEW §E。
    let fields = LayerFields {
        pos: [s.data[0] - offset.dx, -(s.data[1] - offset.dy)],
        scale: [s.data[2] / divisor, s.data[3] / divisor],
        rotation: 360.0 - angle.rem_euclid(360.0),
        color: s.color,
        shape_id,
    };
    let all_finite = fields.pos.iter().chain(fields.scale.iter())
        .all(|v| v.is_finite())
        && fields.rotation.is_finite();
    if !all_finite {
        return Err(format!(
            "type {} 含非有限坐标/缩放（inf/NaN）：{:?}",
            s.kind, fields.pos
        ));
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ellipse_json(x: f32, y: f32, rx: f32, ry: f32, angle: f32) -> ShapeJson {
        ShapeJson {
            kind: 16,
            data: vec![x, y, rx, ry, angle],
            color: [10, 20, 30, 255],
        }
    }

    #[test]
    fn ellipse_conversion_rules() {
        let lay = Layout::default();
        let f = prepare(&ellipse_json(100.0, 200.0, 63.0, 126.0, 30.0), &lay, CanvasOffset::NONE).unwrap();
        assert_eq!(f.pos, [100.0, -200.0], "y 取负");
        assert_eq!(f.scale, [1.0, 2.0], "w/63");
        assert!((f.rotation - 330.0).abs() < 1e-4, "360-angle");
        assert_eq!(f.color, [10, 20, 30, 255]);
        assert_eq!(f.shape_id, 102);
    }

    #[test]
    fn rect_divisor_and_no_angle() {
        let lay = Layout::default();
        let s = ShapeJson {
            kind: 1,
            data: vec![0.0, 0.0, 127.0, 254.0],
            color: [0, 0, 0, 255],
        };
        let f = prepare(&s, &lay, CanvasOffset::NONE).unwrap();
        assert_eq!(f.scale, [1.0, 2.0]);
        assert_eq!(f.shape_id, 101);
        assert!((f.rotation - 360.0).abs() < 1e-4);
    }

    #[test]
    fn center_offset_applies() {
        let lay = Layout::default();
        let off = CanvasOffset::center_for(328.0, 328.0);
        let f = prepare(&ellipse_json(0.0, 0.0, 63.0, 63.0, 0.0), &lay, off).unwrap();
        assert_eq!(f.pos, [-164.0, 164.0], "左上角 → 画布中心系的 (-164, +164)");
    }

    #[test]
    fn rejects_unknown_type_and_short_data() {
        let lay = Layout::default();
        let bad_type = ShapeJson {
            kind: 99,
            data: vec![0.0; 5],
            color: [0; 4],
        };
        assert!(prepare(&bad_type, &lay, CanvasOffset::NONE).is_err());
        let short = ShapeJson {
            kind: 16,
            data: vec![1.0, 2.0],
            color: [0; 4],
        };
        assert!(prepare(&short, &lay, CanvasOffset::NONE).is_err());
    }
}
