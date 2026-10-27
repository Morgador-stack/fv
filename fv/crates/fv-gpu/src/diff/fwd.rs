//! 前向软光栅（M18_DESIGN §3）：双扫描单内核。
//!
//! 每像素两次层循环：自底向上 over 合成（层前画布写 `cbelow[i]`）+ 自顶向下透射
//! （层前透射写 `tabove[i]`，随后乘 (1−β)）。β=σ((1−q)·τ)。
//! q 远超 1 时 β≈0 直接跳过混合（数值等价，省 sincos/mix）。
//! 快照平面长 = w*h；全部缓冲按容量分配（wgpu 拒零/变尺寸）。

use super::ShapeParams;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

/// 内核 uniform（8×f32）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uni {
    w: f32,
    h: f32,
    n: f32,
    tau: f32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    snap_start: f32,
    snap_end: f32,
}

pub struct DiffRenderer {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    pub(crate) params: wgpu::Buffer,
    pub(crate) shapes_buf: wgpu::Buffer,
    pub(crate) target_buf: wgpu::Buffer,
    pub(crate) canvas_buf: wgpu::Buffer,
    pub(crate) cbelow_buf: wgpu::Buffer,
    pub(crate) tabove_buf: wgpu::Buffer,
    pub(crate) grads_buf: wgpu::Buffer,
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) cap: usize,
    /// 上次 render 的层数（backward 用）。
    pub(crate) n: usize,
    /// 上次 render 的快照窗口（backward 派发用）。
    pub(crate) snap: (usize, usize),
    /// 上次 render 的画布（调试对账用）。
    pub(crate) last_canvas: Vec<f32>,
    /// 反向管线缓存（bwd.rs 懒构建）。
    pub(crate) bwd: Option<super::bwd::BwdPipeline>,
}

impl DiffRenderer {
    pub fn new(w: u32, h: u32, max_layers: usize) -> Result<Self, String> {
        if w == 0 || h == 0 || max_layers == 0 {
            return Err("DiffRenderer: 尺寸/容量须非零".to_string());
        }
        let plane = (w * h) as usize;
        // v2 量化布局：cbelow 4B/px + tabove 1B/px = 5B/px·层
        let want_layers_bytes = (max_layers * plane * 12) as u64;

        // 先请求设备（上限要得越大越好，失败递降到默认），再按实际 limit 截容量。
        let (device, queue, dev_limit) = pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|_| "无 Vulkan 适配器".to_string())?;
            let mut want = want_layers_bytes.clamp(1 << 28, 4 << 30);
            loop {
                let limits = wgpu::Limits {
                    max_buffer_size: want,
                    max_storage_buffer_binding_size: want.min(u32::MAX as u64) as u32,
                    ..wgpu::Limits::default()
                };
                match adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("fv-diff"),
                        required_features: wgpu::Features::empty(),
                        required_limits: limits,
                        ..Default::default()
                    })
                    .await
                {
                    Ok(pair) => {
                        let lim = pair.0.limits().max_buffer_size;
                        return Ok((pair.0, pair.1, lim));
                    }
                    Err(_) => {
                        if want <= (1 << 28) {
                            return Err("适配器连 256MiB buffer 都不给".to_string());
                        }
                        want = (want / 2).max(1 << 28);
                    }
                }
            }
        })?;
        // 容量 = 设备 limit 的 96% 能装多少层
        let cap = ((dev_limit * 96 / 100) / (plane as u64 * 12)) as usize;
        let cap = cap.clamp(1, max_layers);
        if cap < max_layers {
            eprintln!(
                "DiffRenderer: 驱动 buffer 上限 {dev_limit}B，容量截为 {cap} 层（请求 {max_layers}）"
            );
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diff-forward"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../diff_forward.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("diff-fwd-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(1),
                storage_rw(2),
                storage_rw(3),
                storage_rw(4),
            ],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("diff-fwd"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("diff-fwd-pl"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("diff-params"),
            contents: bytemuck::bytes_of(&Uni {
                w: w as f32,
                h: h as f32,
                n: 0.0,
                tau: 0.0,
                bg_r: 0.0,
                bg_g: 0.0,
                bg_b: 0.0,
                snap_start: 0.0,
                snap_end: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mk = |label: &str, bytes: usize, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let shapes_buf = mk(
            "diff-shapes",
            max_layers * super::SHAPE_FLOATS * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let target_buf = mk(
            "diff-tgt",
            plane * 3 * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let canvas_buf = mk(
            "diff-cnv",
            plane * 3 * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        // v2.2：cbelow 每像素 2 词（u16×257）；tabove f32（按容量 cap 分配）
        let cbelow_buf = mk(
            "diff-cbelow",
            cap * plane * 8,
            wgpu::BufferUsages::STORAGE,
        );
        let tabove_buf = mk(
            "diff-tabove",
            cap * plane * 4,
            wgpu::BufferUsages::STORAGE,
        );
        let grads_buf = mk(
            "diff-grads",
            max_layers * super::SHAPE_FLOATS * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diff-fwd-bg"),
            layout: &layout,
            entries: &[
                ent(0, &params),
                ent(1, &shapes_buf),
                ent(2, &canvas_buf),
                ent(3, &cbelow_buf),
                ent(4, &tabove_buf),
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind,
            params,
            shapes_buf,
            target_buf,
            canvas_buf,
            cbelow_buf,
            tabove_buf,
            grads_buf,
            w,
            h,
            cap,
            n: 0,
            snap: (0, 0),
            last_canvas: Vec::new(),
            bwd: None,
        })
    }

    /// 实际容量（受驱动 buffer 上限约束；调用方据此分块）。
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// 渲染 + 写双快照 + 存 target。返回画布（f32×3 平面交错 RGB，0-255 标度）。
    pub fn render(
        &mut self,
        shapes: &[ShapeParams],
        tau: f32,
        bg: [f32; 3],
        target: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.render_windowed(shapes, tau, bg, target, (0, shapes.len()))
    }

    /// 窗口版：全栈渲染，但只为 `[snap.0, snap.1)` 层写快照（分块 refine 用，
    /// 窗口层梯度与全量精确一致）。
    pub fn render_windowed(
        &mut self,
        shapes: &[ShapeParams],
        tau: f32,
        bg: [f32; 3],
        target: &[f32],
        snap: (usize, usize),
    ) -> Result<Vec<f32>, String> {
        let n = shapes.len();
        let snap_hi = snap.1.min(n); let snap_lo = snap.0.min(snap_hi);
        let snap_len = snap_hi - snap_lo;
        if n == 0 || snap_len > self.cap {
            return Err(format!("render: 层数 {n}/窗口 {snap_len} 超容量 {}", self.cap));
        }
        let plane = (self.w * self.h) as usize;
        if target.len() != plane * 3 {
            return Err(format!("target 长度 {} ≠ w*h*3 {}", target.len(), plane * 3));
        }

        let mut flat = Vec::with_capacity(n * super::SHAPE_FLOATS);
        for s in shapes {
            flat.extend_from_slice(&[s.cx, s.cy, s.a, s.b, s.theta, s.r, s.g, s.bl]);
        }
        self.queue.write_buffer(&self.shapes_buf, 0, bytemuck::cast_slice(&flat));
        self.queue.write_buffer(&self.target_buf, 0, bytemuck::cast_slice(target));
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Uni {
                w: self.w as f32,
                h: self.h as f32,
                n: n as f32,
                tau,
                bg_r: bg[0],
                bg_g: bg[1],
                bg_b: bg[2],
                snap_start: snap.0 as f32,
                snap_end: snap.1 as f32,
            }),
        );

        let bytes = (plane * 3 * 4) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diff-cnv-rb"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("diff-fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(self.w.div_ceil(8), self.h.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&self.canvas_buf, 0, &readback, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|_| "map_async 回调丢失".to_string())?
            .map_err(|e| e.to_string())?;
        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        self.n = n;
        self.snap = (snap_lo, snap_hi);
        self.last_canvas = out.clone();
        Ok(out)
    }

    /// 上次 render 的画布副本（测试对账）。
    pub fn last_canvas(&self) -> &[f32] {
        &self.last_canvas
    }

    /// 纯 Rust 前向参考（全程 f64）——GPU 交叉验证与 FD 的基准。
    pub fn cpu_forward_reference(
        shapes: &[ShapeParams],
        tau: f32,
        bg: [f32; 3],
        w: u32,
        h: u32,
    ) -> Vec<f64> {
        let mut out = vec![0f64; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let px = f64::from(x) + 0.5;
                let py = f64::from(y) + 0.5;
                let mut col = [
                    f64::from(bg[0]),
                    f64::from(bg[1]),
                    f64::from(bg[2]),
                ];
                for s in shapes {
                    let (sin, cos) = f64::from(s.theta).sin_cos();
                    let dx = px - f64::from(s.cx);
                    let dy = py - f64::from(s.cy);
                    let xp = dx * cos - dy * sin;
                    let yp = dx * sin + dy * cos;
                    let q = xp * xp / f64::from(s.a * s.a) + yp * yp / f64::from(s.b * s.b);
                    // 每层尺度归一化 τ（与内核同式）
                    let tl = (f64::from(tau) * f64::from(s.a.min(s.b)) / 8.0).clamp(4.0, 4096.0);
                    if q > 1.0 + 30.0 / tl {
                        continue; // β < σ(−30) ≈ 1e-13
                    }
                    let z = (1.0 - q) * tl;
                    let beta = 1.0 / (1.0 + (-z).exp());
                    let lc = [f64::from(s.r), f64::from(s.g), f64::from(s.bl)];
                    for ch in 0..3 {
                        col[ch] = col[ch] * (1.0 - beta) + lc[ch] * beta;
                    }
                }
                let i = ((y * w + x) * 3) as usize;
                out[i] = col[0];
                out[i + 1] = col[1];
                out[i + 2] = col[2];
            }
        }
        out
    }
}

fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn ent(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}
