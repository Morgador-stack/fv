//! 层写入：读旧值 → 逐字段写 → 失败逐字段回滚。写入即时生效（游戏内渲染同步，无 commit）。

use crate::convert::LayerFields;
use crate::layout::Layout;
use crate::mem::MemIo;

struct FieldWrite {
    offset: u64,
    bytes: Vec<u8>,
}

/// 把 LayerFields 序列化成（偏移, 字节）列表。
fn field_writes(lay: &Layout, f: &LayerFields) -> Vec<FieldWrite> {
    let l = &lay.layer;
    vec![
        FieldWrite {
            offset: l.position_offset,
            bytes: f.pos.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        FieldWrite {
            offset: l.scale_offset,
            bytes: f.scale.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        FieldWrite {
            offset: l.rotation_offset,
            bytes: f.rotation.to_le_bytes().to_vec(),
        },
        FieldWrite {
            offset: l.color_offset,
            bytes: f.color.to_vec(),
        },
        FieldWrite {
            offset: l.shape_id_offset,
            bytes: vec![f.shape_id],
        },
        FieldWrite {
            offset: l.mask_offset,
            bytes: vec![0],
        },
    ]
}

/// 写一层（带回滚）。返回 Err(已完成字段数, 失败描述) 供诊断。
pub fn write_layer(
    m: &mut dyn MemIo,
    lay: &Layout,
    table_addr: u64,
    index: usize,
    fields: &LayerFields,
) -> Result<(), (usize, String)> {
    let ptr = match crate::mem::read_u64(m, table_addr + (index as u64) * 8) {
        Some(p) => p,
        None => return Err((0, format!("layer[{index}] 指针不可读"))),
    };
    let writes = field_writes(lay, fields);

    // 先读旧值
    let mut olds: Vec<(u64, Vec<u8>)> = Vec::with_capacity(writes.len());
    for w in &writes {
        let mut buf = vec![0u8; w.bytes.len()];
        if !m.read_exact(ptr + w.offset, &mut buf) {
            return Err((0, format!("layer[{index}] 旧值不可读 @0x{:X}", ptr + w.offset)));
        }
        olds.push((ptr + w.offset, buf));
    }

    for (done, w) in writes.iter().enumerate() {
        if !m.write_all(ptr + w.offset, &w.bytes) {
            // 回滚
            for (addr, old) in olds.iter().take(done) {
                let _ = m.write_all(*addr, old);
            }
            return Err((done, format!("layer[{index}] 写失败 @0x{:X}", ptr + w.offset)));
        }
    }
    Ok(())
}

/// 清空一层（缩放 0 的黑色矩形，游戏内不可见）。
pub fn clear_layer(m: &mut dyn MemIo, lay: &Layout, table_addr: u64, index: usize) -> Result<(), (usize, String)> {
    let fields = LayerFields {
        pos: [0.0, 0.0],
        scale: [0.0, 0.0],
        rotation: 0.0,
        color: [0, 0, 0, 255],
        shape_id: lay.shape_ids.rect,
    };
    write_layer(m, lay, table_addr, index, &fields)
}

/// 批量注入编排：写入前 n 层，多余模板层清空。返回 (成功, 失败) 数。
pub fn inject_all(
    m: &mut dyn MemIo,
    lay: &Layout,
    table_addr: u64,
    layer_count: usize,
    fields_list: &[LayerFields],
) -> (usize, usize) {
    let mut ok = 0;
    let mut fail = 0;
    for i in 0..layer_count {
        let r = if i < fields_list.len() {
            write_layer(m, lay, table_addr, i, &fields_list[i])
        } else {
            clear_layer(m, lay, table_addr, i)
        };
        match r {
            Ok(()) => ok += 1,
            Err(e) => {
                fail += 1;
                eprintln!("layer[{i}] 写入失败（已写字段 {} 个）: {}", e.0, e.1);
                if fail >= 10 && ok == 0 {
                    eprintln!("连续失败过多，中止");
                    break;
                }
            }
        }
    }
    (ok, fail)
}
