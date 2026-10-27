//! 候选形状采样：xoshiro128++ 计数式 RNG（线程流独立、可重放、无外部依赖）。

use crate::raster::EllipseParams;

/// xoshiro128++。每个线程一个独立实例，种子派生自主种子。
pub struct Rng {
    s: [u32; 4],
}

impl Rng {
    /// 由 64 位种子经 splitmix64 初始化状态。
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let next = |z: &mut u64| {
            *z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = *z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Self {
            s: [
                next(&mut z) as u32,
                (next(&mut z) >> 32) as u32,
                next(&mut z) as u32,
                (next(&mut z) >> 32) as u32,
            ],
        }
    }

    /// xoshiro128++ 主输出。
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(7)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);
        result
    }

    /// [0,1) 均匀 f32（24 位精度）。
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

/// 采样参数（opt 分辨率下的绝对像素单位）。
pub struct CandidateSpec {
    /// 半轴下限
    pub min_radius: f32,
    /// 半轴上限
    pub max_radius: f32,
}

/// 采一个候选椭圆：中心均匀、半轴面积均匀（sqrt 分布）、角度 [0,π) 均匀。
#[inline]
pub fn sample(rng: &mut Rng, w: f32, h: f32, spec: &CandidateSpec) -> EllipseParams {
    let cx = rng.next_f32() * w;
    let cy = rng.next_f32() * h;
    sample_with_center(rng, spec, cx, cy)
}

/// 同上，但中心由调用方给定（供误差加权采样复用）。
#[inline]
pub fn sample_with_center(rng: &mut Rng, spec: &CandidateSpec, cx: f32, cy: f32) -> EllipseParams {
    let span = spec.max_radius - spec.min_radius;
    let rx = spec.min_radius + span * rng.next_f32().sqrt();
    let ry = spec.min_radius + span * rng.next_f32().sqrt();
    let angle_rad = rng.next_f32() * core::f32::consts::PI;
    EllipseParams {
        cx,
        cy,
        rx,
        ry,
        angle_rad,
    }
}

/// B1 结构导引采样（M19_PROPOSALS §B1）：对数均匀半轴 + 截断 Pareto 长宽比 +
/// 局部墨迹走向（±5° 高斯，低置信回退均匀）。thin/orig spec 各 50%。
/// 理想细长候选命中率从 ~10⁻⁸ 提到 ~2×10⁻²。RNG 流确定性不变。
#[inline]
pub fn sample_structured(
    rng: &mut Rng,
    guide: &crate::orient::StructureGuide,
    cx: f32,
    cy: f32,
) -> EllipseParams {
    let spec = if rng.next_f32() < 0.5 { &guide.thin } else { &guide.orig };
    // 对数均匀长半轴
    let lmin = spec.min_radius.max(1.0).ln();
    let lmax = spec.max_radius.max(spec.min_radius + 1.0).ln();
    let rx = (lmin + (lmax - lmin) * rng.next_f32()).exp();
    // 截断 Pareto(1) 长宽比 ∈ [1,12]：偏细
    let u = rng.next_f32();
    let aspect = 1.0 / (1.0 - u * (11.0 / 12.0)).clamp(1.0 / 12.0, 1.0);
    let ry = (rx / aspect).max(spec.min_radius);
    // 走向：高置信取局部角 ± 5°，否则均匀
    let xi = (cx.floor().clamp(0.0, (guide.w - 1) as f32)) as usize;
    let yi = (cy.floor().clamp(0.0, (guide.h - 1) as f32)) as usize;
    let p = yi * guide.w as usize + xi;
    let angle_rad = if guide.conf.get(p).is_some_and(|&c| c > guide.conf_gate) {
        // Box-Muller（确定性）
        let u1 = rng.next_f32().clamp(1e-6, 1.0);
        let u2 = rng.next_f32();
        let g = (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos();
        (guide.angle[p] + g * 0.0873).rem_euclid(core::f32::consts::PI)
    } else {
        rng.next_f32() * core::f32::consts::PI
    };
    EllipseParams {
        cx,
        cy,
        rx,
        ry,
        angle_rad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn f32_in_unit_range() {
        let mut r = Rng::new(7);
        for _ in 0..100_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn distinct_streams() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn sample_bounds() {
        let mut r = Rng::new(99);
        let spec = CandidateSpec {
            min_radius: 2.0,
            max_radius: 30.0,
        };
        for _ in 0..10_000 {
            let e = sample(&mut r, 100.0, 80.0, &spec);
            assert!((0.0..100.0).contains(&e.cx));
            assert!((0.0..80.0).contains(&e.cy));
            assert!((2.0..30.0).contains(&e.rx));
            assert!((2.0..30.0).contains(&e.ry));
            assert!((0.0..core::f32::consts::PI).contains(&e.angle_rad));
        }
    }
}
