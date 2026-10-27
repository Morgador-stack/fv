// M1.8 反向梯度：一 workgroup 一层（64 线程），bbox 内累 8 梯度，固定树归约。
// 公式（M18_DESIGN §2，唯一事实来源）：
//   R = 2(C−tgt)/(3·H·W)
//   颜色:  g_c_ch   = Σ_p R_ch·β·T
//   几何:  s(p)     = Σ_ch R_ch·T·(c_ch − Cbelow_ch)·(−τ·β·(1−β))
//          g_param  = Σ_p s(p)·(dq/dparam)
//   dq/dcx = −(2x'·cosθ/a² + 2y'·sinθ/b²)
//   dq/dcy = −(2x'·sinθ/a² − 2y'·cosθ/b²)
//   dq/da  = −2x'²/a³        dq/db = −2y'²/b³
//   dq/dθ  = 2x'y'·(1/b² − 1/a²)
// 禁 atomics；确定性归约。

struct Uni {
    w: f32, h: f32, n: f32, tau: f32,
    bg_r: f32, bg_g: f32, bg_b: f32, snap_start: f32, snap_end: f32,
};

@group(0) @binding(0) var<uniform> P: Uni;
@group(0) @binding(1) var<storage, read> shapes: array<f32>;       // n×8
@group(0) @binding(2) var<storage, read> tgt: array<f32>;          // w*h*3
@group(0) @binding(3) var<storage, read> cnv: array<f32>;          // w*h*3
@group(0) @binding(4) var<storage, read> cbelow: array<u32>;       // n*(w*h)*2 每像素 2 词（u16×257）
@group(0) @binding(5) var<storage, read> tabove: array<f32>;       // n*(w*h)（f32 精确）
@group(0) @binding(6) var<storage, read_write> grads: array<f32>;  // n×8

var<workgroup> shared_red: array<vec4<f32>, 64>;   // 几何前 4 + 后 4 拆两 vec4
var<workgroup> shared_red2: array<vec4<f32>, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let w = u32(P.w);
    let h = u32(P.h);
    let tau = P.tau;
    let snap_lo = u32(P.snap_start);
    let i = snap_lo + wid.x;            // 实际层号（workgroup 0 = 窗口首层）
    let base = i * 8u;
    let cx = shapes[base];
    let cy = shapes[base + 1u];
    let a = shapes[base + 2u];
    let b = shapes[base + 3u];
    let theta = shapes[base + 4u];
    let col = vec3f(shapes[base + 5u], shapes[base + 6u], shapes[base + 7u]);

    let c = cos(theta);
    let s = sin(theta);
    let plane = w * h;
    let inv3 = 1.0 / (3.0 * f32(plane));

    // bbox：按每层 τ_l 覆盖显著带（等比放大 sqrt(1+30/τ_l) 包住 +1px 余量）
    let tl0 = clamp(tau * min(a, b) / 8.0, 4.0, 4096.0);
    let rad = max(a, b) * sqrt(1.0 + 30.0 / tl0) + 1.0;
    let x0 = max(cx - rad, 0.0);
    let x1 = min(cx + rad, f32(w));
    let y0 = max(cy - rad, 0.0);
    let y1 = min(cy + rad, f32(h));

    var part = vec4f(0.0, 0.0, 0.0, 0.0);   // g_cx, g_cy, g_a, g_b
    var part2 = vec4f(0.0, 0.0, 0.0, 0.0);  // g_theta, g_r, g_g, g_bl

    if (x1 > x0 && y1 > y0) {
        // 量化到像素中心网格：前向在 k+0.5 处采样，反向必须同网格
        // （从小数 bbox 起步会整体错开 frac，梯度系统性失真——s0/s2 对齐吻合、
        //  s1/s3/s4 错位的实测规律即由此来）。
        let x0i = u32(x0);
        let x1i = u32(min(x1 + 1.0, f32(w)));
        let y0i = u32(y0);
        let y1i = u32(min(y1 + 1.0, f32(h)));
        let rows = y1i - y0i;
        var ry = lid.x;
        loop {
            if (ry >= rows) { break; }
            let y = f32(y0i + ry);
            let py = y + 0.5;
            for (var xi = x0i; xi < x1i; xi = xi + 1u) {
                let px = f32(xi) + 0.5;
                let dx = px - cx;
                let dy = py - cy;
                let xp = dx * c - dy * s;
                let yp = dx * s + dy * c;
                let a2 = a * a;
                let b2 = b * b;
                let q = xp * xp / a2 + yp * yp / b2;
                // 每层尺度归一化 τ（同 fwd）：带宽恒为短轴固定比例
                let tl = clamp(tau * min(a, b) / 8.0, 4.0, 4096.0);
                if (abs(1.0 - q) * tl < 30.0) {
                    let z = (1.0 - q) * tl;
                    let beta = 1.0 / (1.0 + exp(-z));
                    let p = (y0i + ry) * w + xi;
                    let o = p * 3u;
                    // v2.2 解包（窗口相对索引）：tabove f32；cbelow u16×257 双词
                    let wi = i - snap_lo;
                    let t_above = tabove[wi * plane + p];
                    if (t_above > 1e-12) {
                        let base = (wi * plane + p) * 2u;
                        let w0 = cbelow[base];
                        let cb = vec3f(
                            f32(w0 & 65535u) / 257.0,
                            f32(w0 >> 16u) / 257.0,
                            f32(cbelow[base + 1u]) / 257.0,
                        );
                        let res = (cnv[o] - tgt[o]) * inv3 * 2.0;
                        let res1 = (cnv[o + 1u] - tgt[o + 1u]) * inv3 * 2.0;
                        let res2 = (cnv[o + 2u] - tgt[o + 2u]) * inv3 * 2.0;
                        // 颜色梯度（β·T 直通）
                        part2.y = part2.y + res * beta * t_above;
                        part2.z = part2.z + res1 * beta * t_above;
                        part2.w = part2.w + res2 * beta * t_above;
                        // 几何：s = Σ_ch R·T·(c−cbelow)·(−τβ(1−β))
                        let dbeta = -tl * beta * (1.0 - beta);
                        let diff = col - cb;
                        let sbeta = dot(vec3f(res, res1, res2), diff) * t_above * dbeta;
                        let dqdcx = -(2.0 * xp * c / a2 + 2.0 * yp * s / b2);
                        // ∂x'/∂cy=+s、∂y'/∂cy=−c 符号混合，无外提负号（文档 v1 此处有误）
                        let dqdcy = 2.0 * xp * s / a2 - 2.0 * yp * c / b2;
                        let dqda = -2.0 * xp * xp / (a2 * a);
                        let dqdb = -2.0 * yp * yp / (b2 * b);
                        let dqdth = 2.0 * xp * yp * (1.0 / b2 - 1.0 / a2);
                        part.x = part.x + sbeta * dqdcx;
                        part.y = part.y + sbeta * dqdcy;
                        part.z = part.z + sbeta * dqda;
                        part.w = part.w + sbeta * dqdb;
                        part2.x = part2.x + sbeta * dqdth;
                    }
                }
            }
            ry = ry + 64u;
        }
    }

    // 固定树归约（确定性）
    shared_red[lid.x] = part;
    shared_red2[lid.x] = part2;
    workgroupBarrier();
    var step_size: u32 = 32u;
    loop {
        if (step_size == 0u) { break; }
        if (lid.x < step_size) {
            shared_red[lid.x] = shared_red[lid.x] + shared_red[lid.x + step_size];
            shared_red2[lid.x] = shared_red2[lid.x] + shared_red2[lid.x + step_size];
        }
        workgroupBarrier();
        step_size = step_size / 2u;
    }
    if (lid.x == 0u) {
        grads[base] = shared_red[0].x;
        grads[base + 1u] = shared_red[0].y;
        grads[base + 2u] = shared_red[0].z;
        grads[base + 3u] = shared_red[0].w;
        grads[base + 4u] = shared_red2[0].x;
        grads[base + 5u] = shared_red2[0].y;
        grads[base + 6u] = shared_red2[0].z;
        grads[base + 7u] = shared_red2[0].w;
    }
}
