//! Windows 实机实现：ToolHelp 找进程 + VirtualQueryEx 枚举区域 + RPM/WPM。

use crate::mem::{MemIo, Region};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Diagnostics::Debug::{
    ReadProcessMemory, WriteProcessMemory,
};
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
    TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Memory::{
    MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_PRIVATE, VirtualQueryEx, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY, PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY,
};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_ACCESS_RIGHTS, PROCESS_QUERY_INFORMATION, PROCESS_VM_OPERATION,
    PROCESS_VM_READ, PROCESS_VM_WRITE,
};

const ACCESS: PROCESS_ACCESS_RIGHTS = PROCESS_ACCESS_RIGHTS(
    PROCESS_VM_READ.0 | PROCESS_VM_WRITE.0 | PROCESS_VM_OPERATION.0 | PROCESS_QUERY_INFORMATION.0,
);

/// 按可执行名列表找进程（大小写不敏感），返回 (pid, 名字)。
pub fn find_process(names: &[String]) -> Option<(u32, String)> {
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0).ok()?;
        let mut pe = PROCESSENTRY32W {
            dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
            ..Default::default()
        };
        let mut found = None;
        if Process32FirstW(snap, &mut pe).is_ok() {
            loop {
                let name = String::from_utf16_lossy(
                    &pe.szExeFile[..pe.szExeFile.iter().position(|&c| c == 0).unwrap_or(0)],
                )
                .to_ascii_lowercase();
                if names.iter().any(|n| name == n.to_ascii_lowercase()) {
                    found = Some((pe.th32ProcessID, name));
                    break;
                }
                if Process32NextW(snap, &mut pe).is_err() {
                    break;
                }
            }
        }
        let _ = CloseHandle(snap);
        found
    }
}

pub struct WinMem {
    handle: HANDLE,
}

impl WinMem {
    pub fn open(pid: u32) -> Result<Self, String> {
        unsafe {
            let handle = OpenProcess(ACCESS, false, pid)
                .map_err(|e| format!("OpenProcess({pid}) 失败: {e}（试试管理员权限）"))?;
            Ok(Self { handle })
        }
    }
}

impl Drop for WinMem {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.handle);
        }
    }
}

impl MemIo for WinMem {
    fn regions(&self) -> Vec<Region> {
        let mut out = Vec::new();
        let mut addr: u64 = 0x1_0000;
        unsafe {
            loop {
                let mut mbi = MEMORY_BASIC_INFORMATION::default();
                if VirtualQueryEx(
                    self.handle,
                    Some(addr as *const _),
                    &mut mbi,
                    std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                ) == 0
                {
                    break;
                }
                let base = mbi.BaseAddress as u64;
                let size = mbi.RegionSize as u64;
                let prot = mbi.Protect;
                let committed = MEM_COMMIT;
                if mbi.State == committed
                    && mbi.Type == MEM_PRIVATE
                    && !prot.contains(PAGE_GUARD)
                    && !prot.contains(PAGE_NOACCESS)
                    && (prot.contains(PAGE_READONLY)
                        || prot.contains(PAGE_READWRITE)
                        || prot.contains(PAGE_WRITECOPY)
                        || prot.contains(PAGE_EXECUTE_READ)
                        || prot.contains(PAGE_EXECUTE_READWRITE)
                        || prot.contains(PAGE_EXECUTE_WRITECOPY))
                {
                    out.push(Region {
                        base,
                        size: size as usize,
                    });
                }
                let next = base.saturating_add(size);
                if next <= addr || next > 0x7FFF_FFFF_FFFF {
                    break;
                }
                addr = next;
            }
        }
        out
    }

    fn read_exact(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.read_partial(addr, buf) == buf.len()
    }

    fn read_partial(&self, addr: u64, buf: &mut [u8]) -> usize {
        unsafe {
            let mut nread = 0usize;
            let ok = ReadProcessMemory(
                self.handle,
                addr as *const _,
                buf.as_mut_ptr().cast(),
                buf.len(),
                Some(&mut nread),
            );
            if ok.is_ok() {
                nread
            } else {
                0
            }
        }
    }

    fn write_all(&mut self, addr: u64, buf: &[u8]) -> bool {
        unsafe {
            let mut nwritten = 0usize;
            WriteProcessMemory(
                self.handle,
                addr as *mut _,
                buf.as_ptr().cast(),
                buf.len(),
                Some(&mut nwritten),
            )
            .is_ok()
                && nwritten == buf.len()
        }
    }
}
