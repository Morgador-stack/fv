// GPU 软椭圆（228）候选评估内核：一个 invocation 评估一个候选。
// 混合语义：new = C + β(k−C)，β = alpha228(rn)·opacity（固定 opacity=1）。
// 闭式最优：k*_ch = Σβg_ch/Σβ²，g = T − (1−β)C；
// delta = ΣΔ² − err(k_rounded)，Δ = T − C（f32 累加，仅用于候选排序；
// 精确接纳由 CPU 侧快照实测把关）。

struct Params {
    w: u32,
    h: u32,
    count: u32,
    stride: u32,
    maps_plane_len: u32, // 占位对齐（Rust Params 六字段布局；软内核不用 maps）
    plane_len: u32,      // = w*h
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tgt: array<f32>;
@group(0) @binding(2) var<storage, read> cnv: array<f32>;
@group(0) @binding(3) var<storage, read> cands: array<f32>; // 5/cand
@group(0) @binding(4) var<storage, read_write> out: array<i64>; // 4/cand

fn alpha228(rn: f32) -> f32 {
    if (rn >= 1.0) {
        return 0.0;
    }
    var a = 1.0 - rn;
    for (var k = 0; k < 5; k = k + 1) {
        let fa = 1.0 - 0.9132 * a - 0.1875 * a * a + 0.1007 * a * a * a;
        let fpa = -(0.9132 + 0.375 * a - 0.3021 * a * a);
        a = a - (fa - rn) / fpa;
    }
    return clamp(a, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.count) {
        return;
    }
    out[i * 4u + 3u] = i64(0);

    let cx = f32(cands[i * 5u]);
    let cy = f32(cands[i * 5u + 1u]);
    let a = f32(cands[i * 5u + 2u]);
    let b = f32(cands[i * 5u + 3u]);
    let th = f32(cands[i * 5u + 4u]);
    if (a <= 0.0 || b <= 0.0) {
        return;
    }

    let c = cos(th);
    let s = sin(th);
    let ia = 1.0 / (a * a);
    let ib = 1.0 / (b * b);
    let A = c * c * ia + s * s * ib;
    let Bk = c * s * (ia - ib);
    let Ck = s * s * ia + c * c * ib;
    let yext = sqrt(a * a * s * s + b * b * c * c);
    let y0 = max(i32(0), i32(ceil(cy - yext - 1.0)));
    let y1 = min(i32(P.h), i32(floor(cy + yext + 1.0)) + 1);

    // 两遍扫描（数值稳定）：单遍公式 err = k²Σβ² − 2kΣβg + Σg² 在贴合区
    //（canvas≈target）是巨数相减，f32 灾难性抵消会产生幻影 delta。
    // 第一遍：Σβ², Σβg → k*；第二遍：逐像素直接累加 Δ² − (βk−g)²（全是小量）。

    var sb2 = 0.0;
    var sbg = vec3<f32>(0.0);

    for (var y = y0; y < y1; y = y + 1) {
        let dy = f32(y) + 0.5 - cy;
        let q = Bk * Bk * dy * dy - A * (dy * dy * Ck - 1.0);
        if (q < 0.0) {
            continue;
        }
        let hw = sqrt(q) / A;
        let pm = -Bk * dy / A;
        let x0f = ceil(cx - 0.5 + pm - hw);
        let x1f = floor(cx - 0.5 + pm + hw) + 1.0;
        if (x1f <= 0.0 || x0f >= f32(P.w)) {
            continue;
        }
        let x0 = u32(max(0.0, x0f));
        let x1 = u32(min(f32(P.w), x1f));
        if (x1 <= x0) {
            continue;
        }
        for (var x = x0; x < x1; x = x + 1) {
            let dx = f32(x) + 0.5 - cx;
            let u = c * dx + s * dy;
            let v = -s * dx + c * dy;
            let r2 = u * u * ia + v * v * ib;
            if (r2 >= 1.0) {
                continue;
            }
            let beta = alpha228(sqrt(r2));
            if (beta <= 0.0) {
                continue;
            }
            let p = u32(y) * P.w + x;
            sb2 = sb2 + beta * beta;
            for (var ch = 0u; ch < 3u; ch = ch + 1u) {
                let t = tgt[ch * P.plane_len + p];
                let cv = cnv[ch * P.plane_len + p];
                sbg[ch] = sbg[ch] + beta * (t - (1.0 - beta) * cv);
            }
        }
    }

    if (sb2 <= 0.0) {
        return;
    }

    var packed: u32 = 0u;
    var k = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        k[ch] = round(clamp(sbg[ch] / sb2, 0.0, 255.0));
        packed = packed | (u32(k[ch]) << (ch * 8u));
    }

    var delta = 0.0;
    for (var y = y0; y < y1; y = y + 1) {
        let dy = f32(y) + 0.5 - cy;
        let q = Bk * Bk * dy * dy - A * (dy * dy * Ck - 1.0);
        if (q < 0.0) {
            continue;
        }
        let hw = sqrt(q) / A;
        let pm = -Bk * dy / A;
        let x0f = ceil(cx - 0.5 + pm - hw);
        let x1f = floor(cx - 0.5 + pm + hw) + 1.0;
        if (x1f <= 0.0 || x0f >= f32(P.w)) {
            continue;
        }
        let x0 = u32(max(0.0, x0f));
        let x1 = u32(min(f32(P.w), x1f));
        for (var x = x0; x < x1; x = x + 1) {
            let dx = f32(x) + 0.5 - cx;
            let u = c * dx + s * dy;
            let v = -s * dx + c * dy;
            let r2 = u * u * ia + v * v * ib;
            if (r2 >= 1.0) {
                continue;
            }
            let beta = alpha228(sqrt(r2));
            if (beta <= 0.0) {
                continue;
            }
            let p = u32(y) * P.w + x;
            for (var ch = 0u; ch < 3u; ch = ch + 1u) {
                let t = tgt[ch * P.plane_len + p];
                let cv = cnv[ch * P.plane_len + p];
                let d = t - cv;
                let e_new = beta * k[ch] - (t - (1.0 - beta) * cv);
                delta = delta + d * d - e_new * e_new;
            }
        }
    }

    // δ<1 视为无收益（避免 i64 截断产出 (k,0) 的伪正收益项）
    if (delta < 1.0) {
        return;
    }
    out[i * 4u] = i64(delta);
    out[i * 4u + 1u] = i64(packed);
    out[i * 4u + 2u] = i64(0);
    out[i * 4u + 3u] = i64(1);
}
