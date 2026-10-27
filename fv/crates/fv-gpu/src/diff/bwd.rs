//! 反向梯度（M18_DESIGN §2/§3）：为 `DiffRenderer` 追加 `backward` impl 块。
//! 一 workgroup 一层；颜色梯度走 β·T 直通，几何梯度走 −τβ(1−β)·dq/dparam 链。

use super::fwd::DiffRenderer;
use std::sync::mpsc;

pub(crate) struct BwdPipeline {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
}

impl DiffRenderer {
    /// 梯度 of MSE wrt 每层 8 参数（[cx, cy, a, b, theta, r, g, bl] 行主序）。
    /// 必须在 `render` 之后调用（快照与 uniform 就绪）。
    pub fn backward(&mut self) -> Result<Vec<f32>, String> {
        let n = self.n;
        let snap_len = self.snap.1.saturating_sub(self.snap.0);
        if n == 0 || snap_len == 0 {
            return Err("backward: 尚未 render".to_string());
        }

        if self.bwd.is_none() {
            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("diff-backward"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../diff_backward.wgsl").into()),
            });
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("diff-bwd-bgl"),
                entries: &[
                    uni(0),
                    ro(1),
                    ro(2),
                    ro(3),
                    ro(4),
                    ro(5),
                    rw(6),
                ],
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("diff-bwd"),
                layout: Some(
                    &self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("diff-bwd-pl"),
                        bind_group_layouts: &[&layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
            // shapes/target/canvas/params 是 fwd 的私有字段 → 经 pub(crate) 直接引用。
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("diff-bwd-bg"),
                layout: &layout,
                entries: &[
                    ent(0, &self.params),
                    ent(1, &self.shapes_buf),
                    ent(2, &self.target_buf),
                    ent(3, &self.canvas_buf),
                    ent(4, &self.cbelow_buf),
                    ent(5, &self.tabove_buf),
                    ent(6, &self.grads_buf),
                ],
            });
            self.bwd = Some(BwdPipeline { pipeline, bind });
        }
        let bwd = self.bwd.as_ref().unwrap();

        let bytes = (n * super::SHAPE_FLOATS * 4) as u64;
        // 窗口外的 grads 不被内核写——清零防上一轮陈值
        self.queue.write_buffer(&self.grads_buf, 0, &vec![0u8; bytes as usize]);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diff-grads-rb"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("diff-bwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&bwd.pipeline);
            pass.set_bind_group(0, &bwd.bind, &[]);
            pass.dispatch_workgroups(snap_len as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.grads_buf, 0, &readback, 0, bytes);
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
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        Ok(out)
    }
}

fn ent(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn uni(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
