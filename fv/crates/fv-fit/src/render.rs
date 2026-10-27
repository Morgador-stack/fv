//! 最终形状列表 → 预览 PNG。

use crate::raster::{self, EllipseParams};
use fv_common::shape::Shape;

/// 形状列表（原图坐标，z 序自底向上）→ 3 平面 SoA u8 画布。
/// 实心形状覆盖；软椭圆（228）按 α=β·opacity 混合。
fn shapes_to_canvas(w: u32, h: u32, shapes: &[Shape], bg: [u8; 3]) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut canvas = vec![0u8; n * 3];
    for ch in 0..3 {
        let plane = &mut canvas[ch * n..(ch + 1) * n];
        plane.fill(bg[ch]);
    }
    for s in shapes {
        let e = EllipseParams {
            cx: s.cx,
            cy: s.cy,
            rx: s.rx,
            ry: s.ry,
            angle_rad: s.angle_deg.to_radians(),
        };
        if s.kind == fv_common::shape::ShapeKind::SoftEllipse {
            let opacity = f64::from(s.color[3]) / 255.0;
            crate::soft::blend_composite(
                &mut canvas,
                w,
                h,
                &e,
                [s.color[0], s.color[1], s.color[2]],
                opacity,
            );
        } else {
            raster::composite_ellipse(&mut canvas, w, h, &e, [s.color[0], s.color[1], s.color[2]]);
        }
    }
    canvas
}

/// 3 平面 SoA 画布 → RGBA 图（alpha 恒 255）。
fn canvas_to_image(w: u32, h: u32, canvas: &[u8]) -> image::RgbaImage {
    let n = (w * h) as usize;
    let mut img = image::RgbaImage::new(w, h);
    for (p, px) in img.pixels_mut().enumerate() {
        *px = image::Rgba([canvas[p], canvas[n + p], canvas[2 * n + p], 255]);
    }
    img
}

/// 把形状列表（**原图坐标**，按 z 序自底向上）光栅化到 (w,h) 画布并导出 RGBA。
/// 画布初始填充 `bg`，逐形状实心覆盖（SoftEllipse 的 alpha 渐变 M1 暂按实心处理）。
pub fn render_shapes(w: u32, h: u32, shapes: &[Shape], bg: [u8; 3]) -> image::RgbaImage {
    canvas_to_image(w, h, &shapes_to_canvas(w, h, shapes, bg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_shapes_is_uniform_bg() {
        let img = render_shapes(16, 8, &[], [10, 20, 30]);
        assert_eq!(img.dimensions(), (16, 8));
        for px in img.pixels() {
            assert_eq!(px.0, [10, 20, 30, 255]);
        }
    }

    #[test]
    fn canvas_to_image_layout() {
        let w = 4;
        let h = 2;
        let canvas: Vec<u8> = vec![1; 8]
            .into_iter()
            .chain(vec![2; 8])
            .chain(vec![3; 8])
            .collect();
        let img = canvas_to_image(w, h, &canvas);
        assert_eq!(img.get_pixel(0, 0).0, [1, 2, 3, 255]);
        assert_eq!(img.get_pixel(3, 1).0, [1, 2, 3, 255]);
    }
}
