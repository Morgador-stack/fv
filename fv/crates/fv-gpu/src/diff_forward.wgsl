// M1.8 前向软光栅：双扫描单内核（契约见 M18_DESIGN.md §2/§3）。
// v2 快照量化（2026-10-27）：cbelow 每像素 1×u32（RGB|pad，4B/px）；
// tabove 每 4 像素 1×u32（1B/px，atomicOr 打包，渲染前 Rust 侧清零）。
// 每层 5B/px —— 2000 层 @328² ≈ 1.07GB，适配崩后驱动的 1.29GB buffer 上限。
// 快照平面长 = w*h（不是 h·(w+1)）。

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
};

@group(0) @binding(0) var<uniform> P: Uni;
@group(0) @binding(1) var<storage, read> shapes: array<f32>;        // n×8
@group(0) @binding(2) var<storage, read_write> cnv: array<f32>;     // w*h*3
@group(0) @binding(3) var<storage, read_write> cbelow: array<u32>;  // n*(w*h)*2 每像素 2 词（u16 定点×257）
@group(0) @binding(4) var<storage, read_write> tabove: array<f32>;     // n*(w*h)（f32 精确——u8 会碾碎深层小透射）

struct Layer {
    cx: f32, cy: f32, a: f32, b: f32, theta: f32, r: f32, g: f32, bl: f32,
};

fn load_layer(i: u32) -> Layer {
    let base = i * 8u;
    return Layer(
        shapes[base], shapes[base + 1u], shapes[base + 2u], shapes[base + 3u],
        shapes[base + 4u], shapes[base + 5u], shapes[base + 6u], shapes[base + 7u],
    );
}

// 每层尺度归一化 τ：带宽绝对于像素时细椭圆（b≈1.2px）的软目标近乎不可见
// （B1 栈 refine 失效根因）。τ_l = τ·min(a,b)/B_REF → 像素带宽 ≈ 4·B_REF/τ 恒定。
fn beta_at(l: Layer, px: f32, py: f32, tau: f32) -> f32 {
    let c = cos(l.theta);
    let s = sin(l.theta);
    let dx = px - l.cx;
    let dy = py - l.cy;
    let xp = dx * c - dy * s;
    let yp = dx * s + dy * c;
    let q = xp * xp / (l.a * l.a) + yp * yp / (l.b * l.b);
    let tl = clamp(tau * min(l.a, l.b) / 8.0, 4.0, 4096.0);
    return 1.0 / (1.0 + exp((q - 1.0) * tl));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(P.w);
    let h = u32(P.h);
    let n = u32(P.n);
    let tau = P.tau;
    if (gid.x >= w || gid.y >= h) { return; }
    let plane = w * h;
    let p = gid.y * w + gid.x;
    let px = f32(gid.x) + 0.5;
    let py = f32(gid.y) + 0.5;

    let snap_lo = u32(P.snap_start);
    let snap_hi = u32(P.snap_end);
    // ---- 扫 1：自底向上 over 合成，层前画布 u16 定点（×257）双词直存（仅窗口层）----
    // 每 2px 一词的打包在多通道下不可行——改每像素 2 词：[r16|g16<<16, b16]。
    // u16 = round(c*257)：步长 0.004，近收敛区梯度不被量化噪声劫持（u8 的教训）。
    var col = vec3f(P.bg_r, P.bg_g, P.bg_b);
    for (var i: u32 = 0u; i < n; i++) {
        if (i >= snap_lo && i < snap_hi) {
            let base = ((i - snap_lo) * plane + p) * 2u;
            let r16 = u32(clamp(col.r, 0.0, 255.0) * 257.0 + 0.5);
            let g16 = u32(clamp(col.g, 0.0, 255.0) * 257.0 + 0.5);
            let b16 = u32(clamp(col.b, 0.0, 255.0) * 257.0 + 0.5);
            cbelow[base] = r16 | (g16 << 16u);
            cbelow[base + 1u] = b16;
        }
        let l = load_layer(i);
        let beta = beta_at(l, px, py, tau);
        if (beta > 1e-12) {
            col = mix(col, vec3f(l.r, l.g, l.bl), beta);
        }
    }
    let o = p * 3u;
    cnv[o] = col.r;
    cnv[o + 1u] = col.g;
    cnv[o + 2u] = col.b;

    // ---- 扫 2：自顶向下透射，层前透射存 tabove（f32 精确，仅窗口层）----
    var t = 1.0;
    for (var i: u32 = n; i > 0u; i--) {
        let idx = i - 1u;
        if (idx >= snap_lo && idx < snap_hi) {
            tabove[(idx - snap_lo) * plane + p] = t;
        }
        let l = load_layer(idx);
        let beta = beta_at(l, px, py, tau);
        if (beta > 1e-12) {
            t = t * (1.0 - beta);
        }
    }
}
