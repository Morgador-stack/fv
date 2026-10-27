//! GPU 环境探测：枚举适配器、后端、关键特性（含 SHADER_INT64——评估内核的数值范围前提）。

fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all());
        println!("适配器数量: {}", adapters.len());
        for (i, a) in adapters.iter().enumerate() {
            let info = a.get_info();
            let features = a.features();
            let int64 = format!("{:?}", features).contains("INT64");
            let limits = a.limits();
            println!(
                "[{i}] {:?} | {} ({}) | driver {} {:?}\n     int64={int64} | max_buffer={:.1} MB | max_compute_wg={}",
                info.backend,
                info.name,
                info.device,
                info.driver,
                info.driver_info,
                limits.max_buffer_size as f64 / 1048576.0,
                limits.max_compute_workgroups_per_dimension,
            );
        }
        if adapters.is_empty() {
            eprintln!("未找到任何 GPU 适配器（检查驱动/Vulkan）");
            std::process::exit(1);
        }
    });
}
