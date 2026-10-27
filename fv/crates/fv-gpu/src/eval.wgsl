// GPU 候选评估内核：一个 invocation 评估一个旋转椭圆。
// 数学与 fv-fit raster/engine 的闭式解一致（f32 精度），i64 累加（需 SHADER_INT64）。

struct Params {
    w: u32,
    h: u32,
    count: u32,
    stride: u32,   // w+1
    maps_plane_len: u32, // h*stride
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> maps: array<i64>;
@group(0) @binding(2) var<storage, read> cands: array<f32>; // 5/cand: cx,cy,rx,ry,angle
@group(0) @binding(3) var<storage, read_write> out: array<i64>; // 4/cand: delta, packed_rgb, area, ok

fn seg(plane: u32, y: u32, x0: u32, x1: u32) -> i64 {
    let base = plane * P.maps_plane_len + y * P.stride;
    return maps[base + x1] - maps[base + x0];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.count) { return; }
    out[i * 4u + 3u] = i64(0);

    let cx = cands[i * 5u];
    let cy = cands[i * 5u + 1u];
    let a = cands[i * 5u + 2u];
    let b = cands[i * 5u + 3u];
    let th = cands[i * 5u + 4u];
    if (a <= 0.0 || b <= 0.0) { return; }

    let c = cos(th);
    let s = sin(th);
    let ia = 1.0 / (a * a);
    let ib = 1.0 / (b * b);
    let A = c * c * ia + s * s * ib;
    let Bk = c * s * (ia - ib);   // px_mid = -Bk*dy/A
    let Ck = s * s * ia + c * c * ib; // C = dy^2*Ck - 1

    let yext = sqrt(a * a * s * s + b * b * c * c);
    let y0 = max(i32(0), i32(ceil(cy - yext - 1.0)));
    let y1 = min(i32(P.h), i32(floor(cy + yext + 1.0)) + 1);

    var t = vec3<i64>(i64(0));
    var t2 = vec3<i64>(i64(0));
    var tc = vec3<i64>(i64(0));
    var c2 = vec3<i64>(i64(0));
    var area: i64 = i64(0);

    for (var y = y0; y < y1; y = y + 1) {
        let dy = f32(y) + 0.5 - cy;
        let q = Bk * Bk * dy * dy - A * (dy * dy * Ck - 1.0);
        if (q < 0.0) { continue; }
        let hw = sqrt(q) / A;
        let pm = -Bk * dy / A;
        let x0f = ceil(cx - 0.5 + pm - hw);
        let x1f = floor(cx - 0.5 + pm + hw) + 1.0;
        if (x1f <= 0.0 || x0f >= f32(P.w)) { continue; }
        let x0 = u32(max(0.0, x0f));
        let x1 = u32(min(f32(P.w), x1f));
        if (x1 <= x0) { continue; }

        area = area + i64(x1 - x0);
        t[0] = t[0] + seg(0u, u32(y), x0, x1);
        t[1] = t[1] + seg(1u, u32(y), x0, x1);
        t[2] = t[2] + seg(2u, u32(y), x0, x1);
        t2[0] = t2[0] + seg(3u, u32(y), x0, x1);
        t2[1] = t2[1] + seg(4u, u32(y), x0, x1);
        t2[2] = t2[2] + seg(5u, u32(y), x0, x1);
        tc[0] = tc[0] + seg(6u, u32(y), x0, x1);
        tc[1] = tc[1] + seg(7u, u32(y), x0, x1);
        tc[2] = tc[2] + seg(8u, u32(y), x0, x1);
        c2[0] = c2[0] + seg(9u, u32(y), x0, x1);
        c2[1] = c2[1] + seg(10u, u32(y), x0, x1);
        c2[2] = c2[2] + seg(11u, u32(y), x0, x1);
    }

    if (area == i64(0)) { return; }

    var delta: i64 = i64(0);
    var packed: u32 = 0u;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        var mean = (t[ch] + area / i64(2)) / area;
        mean = clamp(mean, i64(0), i64(255));
        let new_err = t2[ch] - i64(2) * mean * t[ch] + area * mean * mean;
        let cur_err = t2[ch] - i64(2) * tc[ch] + c2[ch];
        delta = delta + cur_err - new_err;
        packed = packed | (u32(mean) << (ch * 8u));
    }
    out[i * 4u] = delta;
    out[i * 4u + 1u] = i64(packed);
    out[i * 4u + 2u] = area;
    out[i * 4u + 3u] = i64(1);
}
