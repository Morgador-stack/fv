//! fv-inject CLI：设计 JSON → 运行中游戏的 vinyl 图层。
//!
//! 用法（实机，需管理员）：
//! ```text
//! fv-inject --json out.json --layer-count 1004 [--pid 0] [--center 328,328] [--yes]
//! ```
//! 流程：定位（u16 扫描 + 严格校验）→ 模板统计 → 抽样 6 层人工核对 → `--yes` 才写入。

use clap::Parser;
use fv_inject::convert::{self, CanvasOffset};
use fv_inject::layout::Layout;

#[derive(Debug, Parser)]
#[command(name = "fv-inject", about = "geometrize JSON → Forza vinyl 图层（内存写入）")]
struct Args {
    /// 设计 JSON（fv-fit 产出）
    #[arg(long)]
    json: std::path::PathBuf,

    /// 游戏内模板层数（必须与实际一致；含 4 个边界层，如 1000 层设计用 1004）
    #[arg(long)]
    layer_count: usize,

    /// 目标进程 PID（0 = 按 layout 里的可执行名自动查找）
    #[arg(long, default_value_t = 0)]
    pid: u32,

    /// 布局 JSON（默认内置 FH6）
    #[arg(long)]
    layout: Option<std::path::PathBuf>,

    /// 画布居中：设计坐标系原点移到画布中心，格式 W,H（默认不平移，兼容 vinylizer 语义）
    #[arg(long, value_parser = parse_dims)]
    center: Option<(f32, f32)>,

    /// 抽样确认的层数
    #[arg(long, default_value_t = 6)]
    samples: usize,

    /// 跳过确认提示直接写入（危险）
    #[arg(long)]
    yes: bool,

    /// 只定位+校验+抽样，不写
    #[arg(long)]
    dry_run: bool,

    /// 导出当前布局到指定 JSON 后退出（游戏更新后改偏移用）
    #[arg(long)]
    dump_layout: Option<std::path::PathBuf>,
}

fn parse_dims(s: &str) -> Result<(f32, f32), String> {
    let (a, b) = s.split_once(',').ok_or("格式 W,H")?;
    let w: f32 = a.parse().map_err(|e| format!("W 解析失败: {e}"))?;
    let h: f32 = b.parse().map_err(|e| format!("H 解析失败: {e}"))?;
    Ok((w, h))
}

fn main() {
    let args = Args::parse();
    if let Some(out) = &args.dump_layout {
        if let Err(e) = Layout::default().save(out) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        eprintln!("布局已导出: {}", out.display());
        return;
    }
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(windows)]
fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let lay = match &args.layout {
        Some(p) => Layout::load(p)?,
        None => Layout::default(),
    };
    let design = fv_common::json::read_design(&args.json)?;
    if design.shapes.is_empty() {
        return Err("设计为空".into());
    }
    eprintln!("设计载入：{} 个形状", design.shapes.len());

    let pid = if args.pid != 0 {
        args.pid
    } else {
        let (pid, name) = fv_inject::win::find_process(&lay.executables)
            .ok_or("未找到游戏进程（确认游戏在涂装编辑器界面并已解组模板）")?;
        eprintln!("找到进程 {name} (PID={pid})");
        pid
    };
    let mut mem = fv_inject::win::WinMem::open(pid)?;

    eprintln!("扫描层表（可能需要数分钟，请保持游戏停在编辑器界面）…");
    let located = fv_inject::scan::locate_layer_table(&mem, &lay, args.layer_count)
        .ok_or("层表定位失败：确认模板层数输入与游戏内一致，且模板全部为基础形状（球/矩形）")?;
    eprintln!(
        "命中：group=0x{:X} table=0x{:X}",
        located.group_addr, located.table_addr
    );

    let (ok, rects, ellipses) =
        fv_inject::scan::count_layers(&mem, &lay, located.table_addr, args.layer_count);
    eprintln!("模板统计：有效 {ok}/{}（矩形 {rects}，椭圆 {ellipses}）", args.layer_count);

    eprintln!("抽样核对（这些值应像游戏里的模板球）：");
    for s in fv_inject::scan::sample_layers(&mem, &lay, located.table_addr, args.layer_count, args.samples) {
        eprintln!(
            "  layer[{:4}] ptr=0x{:X} pos=({:.1},{:.1}) scale=({:.2},{:.2}) rot={:.0}° color={:02X}{:02X}{:02X}{:02X} id={} mask={}",
            s.index, s.ptr, s.pos[0], s.pos[1], s.scale[0], s.scale[1], s.rotation,
            s.color[0], s.color[1], s.color[2], s.color[3], s.shape_id, s.mask
        );
    }

    if args.dry_run {
        eprintln!("dry-run 结束，未写入任何数据。");
        return Ok(());
    }
    if !args.yes {
        eprintln!("核对无误后加 --yes 重新执行以真正写入。");
        return Ok(());
    }

    let offset = match args.center {
        Some((w, h)) => CanvasOffset::center_for(w, h),
        None => CanvasOffset::NONE,
    };
    let mut fields = Vec::with_capacity(design.shapes.len());
    for s in &design.shapes {
        fields.push(convert::prepare(s, &lay, offset).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?);
    }
    eprintln!("写入 {} 层（多余模板层将清空）…", fields.len());
    let (ok, fail) = fv_inject::write_process::inject_all(&mut mem, &lay, located.table_addr, args.layer_count, &fields);
    eprintln!("完成：成功 {ok}，失败 {fail}。回游戏查看；保存涂装由你在游戏内操作。");
    if fail > 0 {
        std::process::exit(2);
    }
    Ok(())
}

#[cfg(not(windows))]
fn run(_args: Args) -> Result<(), Box<dyn std::error::Error>> {
    Err("实机注入仅支持 Windows（本构建用于开发/测试）".into())
}
