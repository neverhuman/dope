//! Typed per-column meta of one training episode's FIT rows (`Nc2ColumnMetaV2`, plan v3 C4).
//!
//! Every episode has a fresh random split, so this meta is recomputed from that episode's FIT
//! rows every time and never cached per table. Only the declared group structure
//! ([`ColumnDecl`], from the NC1 schema) is split-independent.
//!
//! Nothing here is a source value or an extremum. The lattice step and origin are expressed in a
//! caller-chosen unit space (`unit = (raw - center) / scale`), and the origin only as a phase in
//! `[0, q0)`. Bound flags and the tail asymmetry use order statistics at rank `>= k = 10` from
//! either end. Atoms and point masses need at least `k` rows.
//!
//! The struct mirrors `Dse8ColumnMeta` field for field. Its float fields rule out `Eq`; a
//! public schema that needs `Eq` stores [`Nc2ColumnMetaV2::quantized`] instead.

use super::atoms::{detect_atoms, DEFAULT_SUPPORT_K, MAX_ATOMS_V2};
use super::lattice::detect_lattice;

/// Number of f32 values in [`Nc2ColumnMetaV2::to_f32s`].
pub const META_FIELDS: usize = 12;

pub const KIND_CONT: u8 = 0;
pub const KIND_INT: u8 = 1;
pub const KIND_ORD: u8 = 2;
pub const KIND_BOOL: u8 = 3;
pub const KIND_ONEHOT: u8 = 4;
pub const KIND_MISSING_IND: u8 = 5;

pub const SUPPORT_CONT: u8 = 0;
pub const SUPPORT_LATTICE: u8 = 1;
pub const SUPPORT_ATOMS: u8 = 2;
pub const SUPPORT_ZERO_INFLATED: u8 = 3;
pub const SUPPORT_CONST: u8 = 4;

/// Distinct values at most this many, with `>= k` rows each covering this share: atoms.
const ATOM_COVERAGE: f64 = 0.85;
/// Point-mass share that makes a non-atom column zero-inflated.
const SPIKE_SHARE: f64 = 0.12;
/// Largest off-grid share of an accepted lattice.
const MAX_OFFGRID: f64 = 0.20;
/// |unit| beyond which the scale card compresses tails; the guard needs this share of rows.
const TAIL_Z: f64 = 3.0;
const TAIL_SHARE: f64 = 0.01;
/// Stride subsample for the lattice search: the detector costs O(n) per candidate grid.
const LATTICE_PROBE: usize = 1024;

/// Split-independent declaration of a column's group (NC1 schema). `group_kind: None` derives
/// the kind from the FIT values (constant, boolean, integer or continuous only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnDecl {
    pub group_id: u16,
    pub group_kind: Option<u8>,
    pub group_width: u16,
    pub parent_col_idx: u16,
}

impl ColumnDecl {
    /// A column that is its own group, kind derived.
    pub fn single(col: usize) -> Self {
        Self {
            group_id: col.min(u16::MAX as usize) as u16,
            group_kind: None,
            group_width: 1,
            parent_col_idx: u16::MAX,
        }
    }
}

/// The affine unit space of q0 and origin: `unit = (raw - center) / scale`, `scale > 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitSpace {
    pub center: f64,
    pub scale: f64,
}

impl UnitSpace {
    /// FIT median and IQR / 1.349, falling back to the k-th order spread, then 1.
    pub fn robust(sorted: &[f64]) -> Self {
        let n = sorted.len();
        if n == 0 {
            return Self {
                center: 0.0,
                scale: 1.0,
            };
        }
        let center = quantile(sorted, 0.5);
        let iqr = (quantile(sorted, 0.75) - quantile(sorted, 0.25)) / 1.349;
        let k = DEFAULT_SUPPORT_K.min(n / 4);
        let spread = (sorted[n - 1 - k] - sorted[k]) / 4.0;
        let scale = [iqr, spread, 1.0]
            .into_iter()
            .find(|s| s.is_finite() && *s > 1e-12)
            .unwrap_or(1.0);
        Self { center, scale }
    }
}

/// Typed column meta from FIT rows; mirrors `Dse8ColumnMeta`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nc2ColumnMetaV2 {
    pub group_id: u16,
    /// 0 continuous, 1 integer, 2 ordinal, 3 boolean, 4 nominal one-hot, 5 missing indicator.
    pub group_kind: u8,
    pub group_width: u16,
    /// `u16::MAX` when the column has no parent.
    pub parent_col_idx: u16,
    /// 0 continuous, 1 lattice, 2 atoms, 3 zero-inflated, 4 constant.
    pub support_class: u8,
    /// 1 when at least 1 % (and at least k) of FIT rows lie beyond |unit| > 3.
    pub skew_guard_active: u8,
    /// Lattice step in unit space; 0 when the support is not a lattice.
    pub lattice_q0_unit: f32,
    /// Lattice phase in unit space, in `[0, lattice_q0_unit)`.
    pub lattice_origin_unit: f32,
    /// 1 when at least k FIT rows sit exactly at the FIT minimum (a floor or censoring mass).
    pub has_lower_bound: u8,
    /// 1 when at least k FIT rows sit exactly at the FIT maximum.
    pub has_upper_bound: u8,
    /// `(u - l) / (u + l)` = `tanh(ln(u / l) / 2)` of the spreads above and below the median,
    /// measured at rank `max(k, 1 %)` from each end, before any tail transform. In `[-1, 1]`.
    pub log_pre_tail_asym: f32,
    /// `log2(clamp(distinct, 1, 256)) / 8`.
    pub log_unique_bucket: f32,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

/// Run lengths of a sorted slice: (value, count).
fn runs(sorted: &[f64]) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = Vec::new();
    for &v in sorted {
        match out.last_mut() {
            Some((last, c)) if *last == v => *c += 1,
            _ => out.push((v, 1)),
        }
    }
    out
}

impl Nc2ColumnMetaV2 {
    /// Meta of one column from its FIT values (non-finite cells are missing and ignored).
    /// `unit: None` uses [`UnitSpace::robust`] of the FIT values.
    pub fn fit(fit: &[f64], decl: ColumnDecl, unit: Option<UnitSpace>) -> Self {
        let k = DEFAULT_SUPPORT_K;
        let mut sorted: Vec<f64> = fit.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let runs = runs(&sorted);
        let distinct = runs.len();
        let mut meta = Self {
            group_id: decl.group_id,
            group_kind: KIND_CONT,
            group_width: decl.group_width,
            parent_col_idx: decl.parent_col_idx,
            support_class: SUPPORT_CONST,
            skew_guard_active: 0,
            lattice_q0_unit: 0.0,
            lattice_origin_unit: 0.0,
            has_lower_bound: 0,
            has_upper_bound: 0,
            log_pre_tail_asym: 0.0,
            log_unique_bucket: ((distinct.clamp(1, 256) as f64).log2() / 8.0) as f32,
        };
        meta.group_kind = decl.group_kind.unwrap_or_else(|| {
            if distinct == 2 && runs.iter().all(|&(v, _)| v == 0.0 || v == 1.0) {
                KIND_BOOL
            } else if distinct >= 2 && runs.iter().all(|&(v, _)| v.fract() == 0.0) {
                KIND_INT
            } else {
                KIND_CONT
            }
        });
        if n < 2 || distinct < 2 {
            return meta;
        }
        let unit = unit
            .filter(|u| u.center.is_finite() && u.scale.is_finite() && u.scale > 0.0)
            .unwrap_or_else(|| UnitSpace::robust(&sorted));

        // Support class, in the order DSE8 tests it: atoms, lattice, zero-inflated, continuous.
        let supported: Vec<usize> = runs.iter().map(|r| r.1).filter(|&c| c >= k).collect();
        let covered = supported.iter().sum::<usize>() as f64 / n as f64;
        meta.support_class =
            if distinct <= MAX_ATOMS_V2 && supported.len() >= 2 && covered >= ATOM_COVERAGE {
                SUPPORT_ATOMS
            } else if let Some(lat) = lattice(&sorted, k) {
                let q0 = lat.step / unit.scale;
                let phase = ((lat.origin - unit.center) / unit.scale).rem_euclid(q0);
                meta.lattice_q0_unit = q0 as f32;
                meta.lattice_origin_unit = if (phase as f32) < q0 as f32 {
                    phase as f32
                } else {
                    0.0
                };
                SUPPORT_LATTICE
            } else if detect_atoms(&sorted, k).iter().map(|a| a.prob).sum::<f64>() >= SPIKE_SHARE {
                SUPPORT_ZERO_INFLATED
            } else {
                SUPPORT_CONT
            };

        let tail = sorted
            .iter()
            .filter(|&&v| ((v - unit.center) / unit.scale).abs() > TAIL_Z)
            .count();
        meta.skew_guard_active = (tail >= k && tail as f64 >= TAIL_SHARE * n as f64) as u8;
        meta.has_lower_bound = (runs[0].1 >= k) as u8;
        meta.has_upper_bound = (runs[distinct - 1].1 >= k) as u8;
        let r = k.max((0.01 * n as f64).ceil() as usize);
        if n > 2 * r {
            let med = quantile(&sorted, 0.5);
            let (lo, hi) = (sorted[r], sorted[n - 1 - r]);
            let (l, u) = (med - lo, hi - med);
            if l + u > 0.0 {
                meta.log_pre_tail_asym = ((u - l) / (u + l)).clamp(-1.0, 1.0) as f32;
            }
        }
        meta
    }

    /// The twelve fields as f32, in declaration order.
    pub fn to_f32s(&self) -> [f32; META_FIELDS] {
        [
            self.group_id as f32,
            self.group_kind as f32,
            self.group_width as f32,
            if self.parent_col_idx == u16::MAX {
                -1.0
            } else {
                self.parent_col_idx as f32
            },
            self.support_class as f32,
            self.skew_guard_active as f32,
            self.lattice_q0_unit,
            self.lattice_origin_unit,
            self.has_lower_bound as f32,
            self.has_upper_bound as f32,
            self.log_pre_tail_asym,
            self.log_unique_bucket,
        ]
    }

    /// Fixed-point form for a schema that derives `Eq`: floats as IEEE half bits of the value.
    pub fn quantized(&self) -> [u16; META_FIELDS] {
        let f = self.to_f32s();
        let mut out = [0u16; META_FIELDS];
        for (o, v) in out.iter_mut().zip(f) {
            *o = half_bits(v);
        }
        out
    }
}

/// Round-to-nearest IEEE 754 binary16 bits (saturating; NaN becomes 0x7e00).
fn half_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let a = v.abs();
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 6.103_515_6e-5 {
        // subnormal half: units of 2^-24
        return sign | (a * 16_777_216.0).round() as u16;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// Lattice search on a stride subsample, confirmed on every FIT value.
fn lattice(sorted: &[f64], k: usize) -> Option<super::lattice::LatticeSpec> {
    let n = sorted.len();
    let probe: Vec<f64> = if n > LATTICE_PROBE {
        (0..LATTICE_PROBE)
            .map(|i| sorted[i * (n - 1) / (LATTICE_PROBE - 1)])
            .collect()
    } else {
        sorted.to_vec()
    };
    let lat = detect_lattice(&probe, k)?;
    let off = sorted.iter().filter(|&&v| !lat.is_on_grid(v)).count();
    let off_ok = off as f64 <= MAX_OFFGRID * n as f64;
    (off_ok && lat.step.is_finite() && lat.step > 0.0).then_some(lat)
}

/// Meta of every column (`cols[j]` = FIT values of column j). `decls` / `units` default per
/// column to [`ColumnDecl::single`] and [`UnitSpace::robust`].
pub fn fit_columns(
    cols: &[Vec<f64>],
    decls: Option<&[ColumnDecl]>,
    units: Option<&[UnitSpace]>,
) -> Vec<Nc2ColumnMetaV2> {
    cols.iter()
        .enumerate()
        .map(|(j, col)| {
            let decl = decls
                .and_then(|d| d.get(j).copied())
                .unwrap_or_else(|| ColumnDecl::single(j));
            Nc2ColumnMetaV2::fit(col, decl, units.and_then(|u| u.get(j).copied()))
        })
        .collect()
}
