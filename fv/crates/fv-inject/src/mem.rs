//! 进程内存抽象：扫描/写入逻辑全部面向此 trait，可 Mock 离线全链路测试。

/// 一块可扫描的内存区域（committed、private、可读写）。
#[derive(Debug, Clone, Copy)]
pub struct Region {
    pub base: u64,
    pub size: usize,
}

pub trait MemIo {
    /// 枚举可扫描区域。
    fn regions(&self) -> Vec<Region>;
    /// 精确读取（全部成功才 true）。
    fn read_exact(&self, addr: u64, buf: &mut [u8]) -> bool;
    /// 尽量读取，返回实际字节数（用于分块扫描）。
    fn read_partial(&self, addr: u64, buf: &mut [u8]) -> usize;
    /// 全量写入。
    fn write_all(&mut self, addr: u64, buf: &[u8]) -> bool;
}

/// 便捷读取原语。
pub fn read_u16(m: &dyn MemIo, addr: u64) -> Option<u16> {
    let mut b = [0u8; 2];
    if m.read_exact(addr, &mut b) {
        Some(u16::from_le_bytes(b))
    } else {
        None
    }
}

pub fn read_u8(m: &dyn MemIo, addr: u64) -> Option<u8> {
    let mut b = [0u8; 1];
    if m.read_exact(addr, &mut b) {
        Some(b[0])
    } else {
        None
    }
}

pub fn read_u64(m: &dyn MemIo, addr: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    if m.read_exact(addr, &mut b) {
        Some(u64::from_le_bytes(b))
    } else {
        None
    }
}

/// Mock 地址空间：若干独立区块（各自成 Region），供离线测试。
#[derive(Default)]
pub struct MockMem {
    blocks: std::collections::BTreeMap<u64, Vec<u8>>,
    /// 写保护区间（读正常、写失败），用于测试回滚路径。
    protected: Vec<(u64, u64)>,
}

impl MockMem {
    /// 放置一块内存（base 对齐随意），返回 base。
    pub fn add_block(&mut self, base: u64, data: Vec<u8>) -> u64 {
        assert!(!data.is_empty(), "空块无意义");
        self.blocks.insert(base, data);
        base
    }

    /// 在块内写原语（测试构造用）。
    pub fn poke(&mut self, addr: u64, bytes: &[u8]) {
        let base = self
            .blocks
            .range(..=addr)
            .next_back()
            .map(|(b, _)| *b)
            .expect("poke 到未知块");
        let off = (addr - base) as usize;
        let blk = self.blocks.get_mut(&base).unwrap();
        blk[off..off + bytes.len()].copy_from_slice(bytes);
    }

    /// 把 [addr, addr+len) 设为写保护（写返回失败，读不受影响）。
    pub fn protect(&mut self, addr: u64, len: u64) {
        self.protected.push((addr, addr + len));
    }

    fn is_protected(&self, addr: u64, len: usize) -> bool {
        let end = addr + len as u64;
        self.protected.iter().any(|&(a, b)| addr < b && end > a)
    }
}

impl MemIo for MockMem {
    fn regions(&self) -> Vec<Region> {
        self.blocks
            .iter()
            .map(|(base, data)| Region {
                base: *base,
                size: data.len(),
            })
            .collect()
    }

    fn read_exact(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.read_partial(addr, buf) == buf.len()
    }

    fn read_partial(&self, addr: u64, buf: &mut [u8]) -> usize {
        for (base, data) in self.blocks.range(..=addr).rev() {
            if addr >= *base {
                let off = (addr - base) as usize;
                if off < data.len() {
                    let n = (data.len() - off).min(buf.len());
                    buf[..n].copy_from_slice(&data[off..off + n]);
                    return n;
                }
                return 0;
            }
        }
        0
    }

    fn write_all(&mut self, addr: u64, buf: &[u8]) -> bool {
        if self.is_protected(addr, buf.len()) {
            return false;
        }
        for base in self.blocks.range(..=addr).rev().map(|(b, _)| *b) {
            if addr >= base {
                let off = (addr - base) as usize;
                let blk_len = self.blocks.get(&base).unwrap().len();
                if off + buf.len() <= blk_len {
                    let blk = self.blocks.get_mut(&base).unwrap();
                    blk[off..off + buf.len()].copy_from_slice(buf);
                    return true;
                }
                return false; // 跨块尾 = 写失败（模拟不可写边界）
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_read_write_roundtrip() {
        let mut m = MockMem::default();
        m.add_block(0x1000, vec![0u8; 256]);
        assert!(m.write_all(0x1010, &[1, 2, 3, 4]));
        let mut out = [0u8; 4];
        assert!(m.read_exact(0x1010, &mut out));
        assert_eq!(out, [1, 2, 3, 4]);
    }

    #[test]
    fn mock_partial_and_boundary() {
        let mut m = MockMem::default();
        m.add_block(0x100, vec![7u8; 16]);
        let mut buf = [0u8; 8];
        // 从块尾前 4 字节起读 8 → 只能读 4
        assert_eq!(m.read_partial(0x10C, &mut buf), 4);
        assert_eq!(&buf[..4], &[7, 7, 7, 7]);
        // 写越过块尾 → 失败
        assert!(!m.write_all(0x10C, &[0; 8]));
    }

    #[test]
    fn mock_primitives() {
        let mut m = MockMem::default();
        m.add_block(0x2000, vec![0u8; 64]);
        m.poke(0x2010, &0x1234u16.to_le_bytes());
        m.poke(0x2020, &0xDEADBEEFu64.to_le_bytes());
        assert_eq!(read_u16(&m, 0x2010), Some(0x1234));
        assert_eq!(read_u64(&m, 0x2020), Some(0xDEADBEEF));
        assert_eq!(read_u16(&m, 0xFFFF_0000), None);
    }
}
