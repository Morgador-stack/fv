//! 反向梯度 vs 中心差分验证（M18_DESIGN §3 验收门槛 2）+ 平移梯度符号测试。
//! FD 基准 = cpu_forward_reference（f64 累积），比 GPU 前向更精确，排除
//! "前向误差冒充梯度误差"的混淆。

use fv_gpu::diff::fwd::DiffRenderer;
use fv_gpu::diff::ShapeParams;

const W: u32 = 40;
const H: u32 = 40;
const BG: [f32; 3] = [30.0, 60.0, 90.0];
const TAU: f32 = 4.0;

fn scene5() -> Vec<ShapeParams> {
    vec![
        ShapeParams { cx: 12.0, cy: 14.0, a: 6.0, b: 5.0, theta: 0.3, r: 200.0, g: 60.0, bl: 60.0 },
        ShapeParams { cx: 26.0, cy: 12.0, a: 8.0, b: 3.0, theta: -0.5, r: 30.0, g: 230.0, bl: 90.0 },
        ShapeParams { cx: 20.0, cy: 26.0, a: 10.0, b: 4.0, theta: 1.1, r: 250.0, g: 240.0, bl: 20.0 },
        ShapeParams { cx: 30.0, cy: 30.0, a: 5.0, b: 5.0, theta: 0.0, r: 90.0, g: 90.0, bl: 210.0 },
        ShapeParams { cx: 8.0, cy: 30.0, a: 7.0, b: 2.0, theta: 0.9, r: 10.0, g: 130.0, bl: 180.0 },
    ]
}

/// 与引擎裁判同规则的确定性伪随机目标（带结构：底色 + 块状目标区）。
fn target_img() -> Vec<f32> {
    let mut t = vec![0f32; (W * H * 3) as usize];
    let mut seed = 0xDA7A_u64;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / 4294967295.0
    };
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 3) as usize;
            let block = ((x / 8) + (y / 8)) % 3;
            for ch in 0..3usize {
                t[i + ch] = match (block as usize + ch) % 3 {
                    0 => 40.0 + 60.0 * next(),
                    1 => 120.0 + 60.0 * next(),
                    _ => 200.0 + 55.0 * next(),
                };
            }
        }
    }
    t
}

fn mse(shapes: &[ShapeParams], target: &[f32]) -> f64 {
    let c = DiffRenderer::cpu_forward_reference(shapes, TAU, BG, W, H);
    let mut se = 0f64;
    for (a, b) in c.iter().zip(target.iter()) {
        let d = *a - f64::from(*b);
        se += d * d;
    }
    se / (W * H * 3) as f64
}

fn perturb(shapes: &[ShapeParams], i: usize, p: usize, d: f32) -> Vec<ShapeParams> {
    let mut v = shapes.to_vec();
    let s = &mut v[i];
    match p {
        0 => s.cx += d,
        1 => s.cy += d,
        2 => s.a += d,
        3 => s.b += d,
        4 => s.theta += d,
        5 => s.r += d,
        6 => s.g += d,
        _ => s.bl += d,
    }
    v
}

#[test]
fn gradients_match_finite_difference() {
    let shapes = scene5();
    let target = target_img();
    let mut r = match DiffRenderer::new(W, H, shapes.len()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("跳过（无可用 GPU）：{e}");
            return;
        }
    };
    r.render(&shapes, TAU, BG, &target).expect("render");
    let grads = r.backward().expect("backward");

    // 前向对账：GPU 画布 vs CPU 参考（定位误差在前向还是反向）
    {
        let gpu_cnv = r.last_canvas();
        let cpu = DiffRenderer::cpu_forward_reference(&shapes, TAU, BG, W, H);
        let mut max_d = 0f64;
        let mut at = 0usize;
        for (k, (g, c)) in gpu_cnv.iter().zip(cpu.iter()).enumerate() {
            let d = f64::from(*g) - c;
            if d.abs() > max_d {
                max_d = d.abs();
                at = k;
            }
        }
        eprintln!("fwd 对账: max|diff|={max_d:.4} at idx={at} (px{} ch{})", at / 3, at % 3);
    }

    let h_geo = 1e-3f64;
    let h_col = 0.05f64;
    let mut worst_rel = 0f64;
    let mut worst = String::new();
    for i in 0..shapes.len() {
        for p in 0..8 {
            let h = if p < 5 { h_geo } else { h_col };
            let e_hi = mse(&perturb(&shapes, i, p, h as f32), &target);
            let e_lo = mse(&perturb(&shapes, i, p, -h as f32), &target);
            let fd = (e_hi - e_lo) / (2.0 * h);
            let g = f64::from(grads[i * 8 + p]);
            let rel = if fd.abs() < 1e-6 && g.abs() < 1e-6 {
                0.0
            } else if fd.abs() < 1e-6 {
                f64::INFINITY
            } else {
                ((g - fd) / fd).abs()
            };
            if rel < f64::INFINITY && rel > worst_rel {
                worst_rel = rel;
                worst = format!("shape{i} param{p}: fd={fd:.6e} gpu={g:.6e}");
            }
            // v2 量化快照（cbelow u8/tabove u8）引入 ~0.5/255 级 Cbelow 噪声，
            // 弱梯度参数（如 θ）相对误差可达 ~4%——Adam+accept-guard 实测无碍
            assert!(
                rel < 5e-2 || (g - fd).abs() < 3e-4,
                "shape{i} param{p}: FD={fd:.6e} vs GPU={g:.6e}（rel={rel}）"
            );
        }
    }
    eprintln!("FD 验证通过，最差 rel={worst_rel:.2e}（{worst}）");
}

#[test]
fn translation_gradient_sign_pushes_toward_target() {
    // 单层在画布左半，目标色块在右侧：C < target 在层覆盖区 → 残差为负。
    // 层右移（cx↑）会让更多目标像素被覆盖/画布均值上升 → E 下降 → dE/dcx < 0。
    let shapes = vec![ShapeParams {
        cx: 10.0, cy: 20.0, a: 7.0, b: 7.0, theta: 0.0,
        r: 60.0, g: 60.0, bl: 60.0,
    }];
    let mut target = vec![30.0f32; (W * H * 3) as usize];
    for y in 0..H {
        for x in (24..W).rev() {
            let i = ((y * W + x) * 3) as usize;
            target[i..i + 3].copy_from_slice(&[230.0, 230.0, 230.0]);
        }
    }
    let mut r = match DiffRenderer::new(W, H, 1) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("跳过（无可用 GPU）：{e}");
            return;
        }
    };
    r.render(&shapes, TAU, BG, &target).expect("render");
    let g = r.backward().expect("backward");
    assert!(
        g[0] < 0.0,
        "g_cx 应为负（推层右移），实得 {}",
        g[0]
    );
    eprintln!("g_cx={}（负号=右移方向正确）", g[0]);
}
