//! 形状类型定义。JSON type 值与 FH6 内存 shape_id 的映射发生在注入器侧，
//! 这里只保留 geometrize JSON 层面的 type 编号（与 vinylizer/forsa-painter 生态兼容）。

/// geometrize JSON 层面的形状类型编号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ShapeKind {
    /// 矩形（data 无角度项）。FH6 shape_id 101，缩放除数 127。
    Rect = 1,
    /// 旋转椭圆（data 含角度项）。FH6 shape_id 102，缩放除数 63。
    RotatedEllipse = 16,
    /// 软(渐变)椭圆，alpha 通道承载不透明度。FH6 shape_id 228，缩放除数 63。
    SoftEllipse = 228,
}

/// 一个 vinyl 形状。坐标/半径为**原图像素**单位，y 向下；
/// rx/ry 为半轴长；angle_deg 顺时针存储、写入 JSON 时 fmod 360。
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub kind: ShapeKind,
    pub cx: f32,
    pub cy: f32,
    pub rx: f32,
    pub ry: f32,
    pub angle_deg: f32,
    /// RGBA。基础形状恒为 255；SoftEllipse 的 a = opacity*255。
    pub color: [u8; 4],
}

impl Shape {
    pub fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32, angle_deg: f32, color: [u8; 3]) -> Self {
        Self {
            kind: ShapeKind::RotatedEllipse,
            cx,
            cy,
            rx,
            ry,
            angle_deg,
            color: [color[0], color[1], color[2], 255],
        }
    }
}
