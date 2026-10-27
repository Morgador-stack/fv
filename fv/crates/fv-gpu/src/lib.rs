//! fv-gpu：GPU（Vulkan/wgpu compute）候选椭圆评估器——贪心拟合的热循环上 GPU。
//! 双管线：实心（i64 精确积分图评分）+ 软椭圆 228（f32 混合闭式评分，仅排序；
//! 精确接纳由 CPU 侧快照实测把关）。数值口径与 fv-fit 对应实现一致。

use fv_fit::engine::{BatchEvaluator, EvalResults};
use fv_fit::raster::EllipseParams;
use fv_fit::rowsum::RowSums;
use wgpu::util::DeviceExt;

pub mod diff;

pub struct CandidateResult {
    pub delta: i64,
    pub color: [u8; 3],
    pub area: u64,
    pub ok: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    count: u32,
    stride: u32,
    /// maps 表（12 平面 SoA）每平面长 = h*stride——仅实心内核使用。
    maps_plane_len: u32,
    /// 像素平面（target/canvas）长 = w*h——仅软内核使用。
    /// （与 maps 长度曾共用一字段导致 G/B 错位，拆分防复发；两内核 WGSL
    ///  struct 必须同步六字段布局——软内核漏同步会把 maps 值当 plane_len 用。）
    plane_len: u32,
}

/// GPU 评估器：惰性初始化（`BatchEvaluator::on_ready` 时按 fit 实际分辨率建设备）。
/// `broken` 置位后 `available()` 为 false——引擎据此回退 CPU（静默空结果 bug 的修复）。
#[derive(Default)]
pub struct GpuEvaluator {
    inner: Option<Inner>,
    broken: bool,
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    solid_pipeline: wgpu::ComputePipeline,
    soft_pipeline: wgpu::ComputePipeline,
    maps: wgpu::Buffer,
    target: wgpu::Buffer,
    canvas: wgpu::Buffer,
    params: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    soft_layout: wgpu::BindGroupLayout,
    cand: wgpu::Buffer,
    out: wgpu::Buffer,
    bind: wgpu::BindGroup,
    soft_bind: wgpu::BindGroup,
    cap: usize,
    w: u32,
    h: u32,
}

fn planes_f32(planes: &[u8]) -> Vec<f32> {
    planes.iter().map(|&v| f32::from(v)).collect()
}

impl GpuEvaluator {
    /// 立即初始化（基准/测试用，软通道缓冲置零）；引擎路径走 `Default` + `on_ready`。
    pub fn new(w: u32, h: u32, sums: &RowSums) -> Result<Self, String> {
        let mut e = Self { inner: None, broken: false };
        e.ensure(w, h, sums)?;
        Ok(e)
    }

    fn ensure(&mut self, w: u32, h: u32, sums: &RowSums) -> Result<(), String> {
        self.ensure_with_planes(w, h, sums, &[], &[])
    }

    fn ensure_with_planes(
        &mut self,
        w: u32,
        h: u32,
        sums: &RowSums,
        target: &[u8],
        canvas: &[u8],
    ) -> Result<(), String> {
        if let Some(inner) = &self.inner
            && inner.w == w
            && inner.h == h
        {
            // 已存在也要补传平面（new() 路径建的是零缓冲）
            if !target.is_empty() {
                let inner = self.inner.as_mut().unwrap();
                inner.upload_maps_canvas(sums, target, canvas);
            }
            return Ok(());
        }
        self.inner = Some(Inner::new(w, h, sums, target, canvas)?);
        Ok(())
    }

    fn inner_or_err(&mut self) -> Result<&mut Inner, String> {
        self.inner
            .as_mut()
            .ok_or_else(|| "GpuEvaluator 未初始化（缺 on_ready）".to_string())
    }

    /// 每次接纳后重传 maps + canvas（on_accept 触发）。
    fn update_maps(&mut self, sums: &RowSums, target: &[u8], canvas: &[u8]) {
        let Some(inner) = self.inner.as_mut() else { return };
        inner.upload_maps_canvas(sums, target, canvas);
    }

    /// 批量评估实心候选（阻塞到读完结果）。容量不足时自动扩容。
    pub fn evaluate(&mut self, cands: &[EllipseParams]) -> Result<Vec<CandidateResult>, String> {
        self.inner_or_err()?.eval(cands)
    }
}

impl BatchEvaluator for GpuEvaluator {
    fn available(&self) -> bool {
        self.inner.is_some() && !self.broken
    }

    fn on_ready(&mut self, sums: &RowSums, target: &[u8], canvas: &[u8], w: u32, h: u32) {
        if let Err(e) = self.ensure_with_planes(w, h, sums, target, canvas) {
            eprintln!("GPU 初始化失败：{e}");
            self.inner = None;
        }
    }

    fn on_accept(&mut self, sums: &RowSums, target: &[u8], canvas: &[u8], _w: u32, _h: u32) {
        self.update_maps(sums, target, canvas);
    }

    fn eval_batch(
        &mut self,
        cands: &[EllipseParams],
        _w: u32,
        _h: u32,
    ) -> Vec<Option<([u8; 3], i64)>> {
        if self.inner.is_none() {
            // 不再静默全 None（旧 bug：三振早停 → 空 JSON exit 0）。
            // 未初始化 = on_ready 失败；available() 已让引擎走 CPU，这里到不了。
            return vec![None; cands.len()];
        }
        match self.evaluate(cands) {
            Ok(rs) => rs
                .into_iter()
                .map(|r| r.ok.then_some((r.color, r.delta)))
                .collect(),
            Err(e) => {
                eprintln!("GPU 批评估失败（后续回退 CPU）：{e}");
                self.broken = true;
                vec![None; cands.len()]
            }
        }
    }

    fn eval_soft_batch(
        &mut self,
        cands: &[EllipseParams],
        _w: u32,
        _h: u32,
    ) -> Vec<Option<([u8; 3], i64)>> {
        let Some(inner) = self.inner.as_mut() else {
            return vec![None; cands.len()];
        };
        match inner.eval_soft(cands) {
            Ok(rs) => rs,
            Err(e) => {
                eprintln!("GPU 软批评估失败：{e}");
                vec![None; cands.len()]
            }
        }
    }
}

fn buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Inner {
    fn new(w: u32, h: u32, sums: &RowSums, target: &[u8], canvas: &[u8]) -> Result<Self, String> {
        let maps_data = sums.planes_i64();
        let stride = w as usize + 1;
        let expected = 12 * h as usize * stride;
        if maps_data.len() != expected {
            return Err(format!(
                "RowSums 尺寸不符：{} != {expected}",
                maps_data.len()
            ));
        }

        let (device, queue) = pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok()?;
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("fv-gpu"),
                    required_features: wgpu::Features::SHADER_INT64,
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                })
                .await
                .ok()
        })
        .ok_or("无支持 SHADER_INT64 的适配器")?;

        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eval"),
            source: wgpu::ShaderSource::Wgsl(include_str!("eval.wgsl").into()),
        });
        let soft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eval_soft"),
            source: wgpu::ShaderSource::Wgsl(include_str!("eval_soft.wgsl").into()),
        });

        // 实心：params, maps, cands, out
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fv-bgl"),
            entries: &[
                uniform_entry(0),
                buffer_entry(1, true),
                buffer_entry(2, true),
                buffer_entry(3, false),
            ],
        });
        // 软：params, target, canvas, cands, out
        let soft_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fv-bgl-soft"),
            entries: &[
                uniform_entry(0),
                buffer_entry(1, true),
                buffer_entry(2, true),
                buffer_entry(3, true),
                buffer_entry(4, false),
            ],
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fv-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let soft_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fv-pl-soft"),
            bind_group_layouts: &[&soft_layout],
            push_constant_ranges: &[],
        });

        let solid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fv-eval"),
            layout: Some(&pl_layout),
            module: &solid_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
        let soft_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fv-eval-soft"),
            layout: Some(&soft_pl_layout),
            module: &soft_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&Params {
                w,
                h,
                count: 0,
                stride: stride as u32,
                maps_plane_len: h * stride as u32,
                plane_len: w * h,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let n3 = (w * h) as usize * 3;
        let maps = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("maps"),
            contents: bytemuck::cast_slice(maps_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // 软通道的 target/canvas 平面缓冲必须全尺寸（空输入零填），
        // 否则 bind group 因 0 大小绑定被拒。
        let tf = if target.is_empty() {
            vec![0f32; n3]
        } else {
            planes_f32(target)
        };
        let cf = if canvas.is_empty() {
            vec![0f32; n3]
        } else {
            planes_f32(canvas)
        };
        let target_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target"),
            contents: bytemuck::cast_slice(&tf),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let canvas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("canvas"),
            contents: bytemuck::cast_slice(&cf),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let cap = 4096usize;
        let (cand, out) = make_io_buffers(&device, cap);
        let bind = make_bind_group(&device, &layout, &maps, &params, &cand, &out);
        let soft_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fv-bg-soft"),
            layout: &soft_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: canvas_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cand.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out.as_entire_binding(),
                },
            ],
        });

        let inner = Self {
            device,
            queue,
            solid_pipeline,
            soft_pipeline,
            maps,
            target: target_buf,
            canvas: canvas_buf,
            params,
            layout,
            soft_layout,
            cand,
            out,
            bind,
            soft_bind,
            cap,
            w,
            h,
        };
        if !target.is_empty() {
            inner.upload_maps_canvas(sums, target, canvas);
        }
        Ok(inner)
    }

    fn upload_maps_canvas(&self, sums: &RowSums, target: &[u8], canvas: &[u8]) {
        self.queue
            .write_buffer(&self.maps, 0, bytemuck::cast_slice(sums.planes_i64()));
        if !target.is_empty() {
            self.queue
                .write_buffer(&self.target, 0, bytemuck::cast_slice(&planes_f32(target)));
        }
        if !canvas.is_empty() {
            self.queue
                .write_buffer(&self.canvas, 0, bytemuck::cast_slice(&planes_f32(canvas)));
        }
    }

    fn ensure_cap(&mut self, n: usize) {
        if n > self.cap {
            self.cap = n.next_power_of_two();
            let (cand, out) = make_io_buffers(&self.device, self.cap);
            self.cand = cand;
            self.out = out;
            self.bind = make_bind_group(&self.device, &self.layout, &self.maps, &self.params, &self.cand, &self.out);
            self.soft_bind = make_soft_bind_group(
                &self.device,
                &self.soft_layout,
                &self.params,
                &self.target,
                &self.canvas,
                &self.cand,
                &self.out,
            );
        }
    }

    fn write_cands_params(&self, cands: &[EllipseParams]) {
        let mut packed = Vec::with_capacity(cands.len() * 5);
        for e in cands {
            packed.extend_from_slice(&[e.cx, e.cy, e.rx, e.ry, e.angle_rad]);
        }
        self.queue.write_buffer(&self.cand, 0, bytemuck::cast_slice(&packed));
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                w: self.w,
                h: self.h,
                count: cands.len() as u32,
                stride: self.w + 1,
                maps_plane_len: self.h * (self.w + 1),
                plane_len: self.w * self.h,
            }),
        );
    }

    fn dispatch_and_read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        out: &wgpu::Buffer,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        n: usize,
    ) -> Result<Vec<[i64; 4]>, String> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (n * 4 * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eval"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            let wg = n.div_ceil(64) as u32;
            pass.dispatch_workgroups(wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(out, 0, &readback, 0, (n * 4 * 8) as u64);
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::Wait).map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|_| "map_async 回调丢失".to_string())?
            .map_err(|e| e.to_string())?;

        let data = slice.get_mapped_range();
        let raw: &[i64] = bytemuck::cast_slice(&data);
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
        }
        drop(data);
        readback.unmap();
        Ok(rows)
    }

    fn eval(&mut self, cands: &[EllipseParams]) -> Result<Vec<CandidateResult>, String> {
        if cands.is_empty() {
            return Ok(Vec::new());
        }
        self.ensure_cap(cands.len());
        self.write_cands_params(cands);
        let rows = Self::dispatch_and_read(
            &self.device,
            &self.queue,
            &self.out,
            &self.solid_pipeline,
            &self.bind,
            cands.len(),
        )?;
        let mut results = Vec::with_capacity(cands.len());
        for r in rows {
            let packed_rgb = r[1] as u64 as u32;
            results.push(CandidateResult {
                delta: r[0],
                color: [
                    (packed_rgb & 0xFF) as u8,
                    ((packed_rgb >> 8) & 0xFF) as u8,
                    ((packed_rgb >> 16) & 0xFF) as u8,
                ],
                area: r[2] as u64,
                ok: r[3] != 0,
            });
        }
        Ok(results)
    }

    fn eval_soft(
        &mut self,
        cands: &[EllipseParams],
    ) -> Result<EvalResults, String> {
        if cands.is_empty() {
            return Ok(Vec::new());
        }
        self.ensure_cap(cands.len());
        self.write_cands_params(cands);
        // 软内核平面寻址是 ch*plane_len + y*w + x，plane_len 必须是 w*h
        //（实心内核用的是行前缀布局 h*(w+1)，不能混用）
        self.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                w: self.w,
                h: self.h,
                count: cands.len() as u32,
                stride: self.w + 1,
                maps_plane_len: self.h * (self.w + 1),
                plane_len: self.w * self.h,
            }),
        );
        let rows = Self::dispatch_and_read(
            &self.device,
            &self.queue,
            &self.out,
            &self.soft_pipeline,
            &self.soft_bind,
            cands.len(),
        )?;
        Ok(rows
            .into_iter()
            .map(|r| {
                if r[3] == 0 {
                    return None;
                }
                let packed = r[1] as u64 as u32;
                Some((
                    [
                        (packed & 0xFF) as u8,
                        ((packed >> 8) & 0xFF) as u8,
                        ((packed >> 16) & 0xFF) as u8,
                    ],
                    r[0],
                ))
            })
            .collect())
    }
}

fn make_io_buffers(device: &wgpu::Device, cap: usize) -> (wgpu::Buffer, wgpu::Buffer) {
    let cand = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cand"),
        size: (cap * 5 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (cap * 4 * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (cand, out)
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    maps: &wgpu::Buffer,
    params: &wgpu::Buffer,
    cand: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fv-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: maps.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cand.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.as_entire_binding(),
            },
        ],
    })
}

fn make_soft_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    target: &wgpu::Buffer,
    canvas: &wgpu::Buffer,
    cand: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fv-bg-soft"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: canvas.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cand.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out.as_entire_binding(),
            },
        ],
    })
}
