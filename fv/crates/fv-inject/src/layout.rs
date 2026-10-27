//! 层布局：FH6 LiveryGroup/Layer 结构偏移，数据文件化（游戏更新后只改 JSON）。
//! 数值来源：vinylizer inject.cpp ≡ FD6 fh6_patterns.json ≡ bvzrays DEVELOPMENT.md 三方一致（2026-05 确认）。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupLayout {
    /// u16 模板层数偏移（定位锚点）
    pub layer_count_offset: u64,
    /// u64 层表指针偏移（指向 u64 指针数组，8 字节步长）
    pub layer_table_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLayout {
    /// 单层 blob 尺寸（校验区域不越界用）
    pub blob_size: u64,
    pub position_offset: u64, // f32×2
    pub scale_offset: u64,    // f32×2
    pub rotation_offset: u64, // f32
    pub color_offset: u64,    // u8×4
    pub mask_offset: u64,     // u8，必须 0
    pub shape_id_offset: u64, // u8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeIds {
    pub rect: u8,
    pub ellipse: u8,
    pub soft_ellipse: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanConfig {
    /// 单区域最大读取块
    pub region_chunk: usize,
    /// 严格校验的采样层数（全部必须通过）
    pub sample_layers: usize,
}

/// 一套游戏的完整注入描述。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layout {
    pub game: String,
    pub executables: Vec<String>,
    pub livery_group: GroupLayout,
    pub layer: LayerLayout,
    pub shape_ids: ShapeIds,
    /// JSON type → 缩放除数（1=矩形 127，16=椭圆 63，228=软椭圆 63）
    pub scale_divisors: BTreeMap<u32, f32>,
    pub scan: ScanConfig,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            game: "Forza Horizon 6".into(),
            executables: vec![
                "forzahorizon6.exe".into(),
                "forzahorizon6-win64-shipping.exe".into(),
            ],
            livery_group: GroupLayout {
                layer_count_offset: 0x5A,
                layer_table_offset: 0x78,
            },
            layer: LayerLayout {
                blob_size: 0x80,
                position_offset: 0x18,
                scale_offset: 0x28,
                rotation_offset: 0x50,
                color_offset: 0x74,
                mask_offset: 0x78,
                shape_id_offset: 0x7A,
            },
            shape_ids: ShapeIds {
                rect: 101,
                ellipse: 102,
                soft_ellipse: 228,
            },
            scale_divisors: BTreeMap::from([(1, 127.0), (16, 63.0), (228, 63.0)]),
            scan: ScanConfig {
                region_chunk: 64 * 1024 * 1024,
                sample_layers: 64,
            },
        }
    }
}

impl Layout {
    pub fn load(path: &Path) -> Result<Self, String> {
        let buf = std::fs::read(path).map_err(|e| format!("读取布局文件失败: {e}"))?;
        serde_json::from_slice(&buf).map_err(|e| format!("解析布局文件失败: {e}"))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let buf = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, buf).map_err(|e| format!("写入布局文件失败: {e}"))
    }

    /// JSON type → 内存 shape_id。
    pub fn shape_id_of(&self, json_type: u32) -> Option<u8> {
        match json_type {
            1 => Some(self.shape_ids.rect),
            16 => Some(self.shape_ids.ellipse),
            228 => Some(self.shape_ids.soft_ellipse),
            _ => None,
        }
    }

    pub fn divisor_of(&self, json_type: u32) -> Option<f32> {
        self.scale_divisors.get(&json_type).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roundtrip_json() {
        let lay = Layout::default();
        let s = serde_json::to_string_pretty(&lay).unwrap();
        let back: Layout = serde_json::from_str(&s).unwrap();
        assert_eq!(back.layer.shape_id_offset, 0x7A);
        assert_eq!(back.livery_group.layer_count_offset, 0x5A);
        assert_eq!(back.shape_id_of(16), Some(102));
        assert_eq!(back.divisor_of(1), Some(127.0));
        assert_eq!(back.shape_id_of(99), None);
    }
}
