//! Tier T0 Evidence Extraction: per-column support, atoms, decimal lattice, heaping,
//! smoothed quantile curve residuals, L-moments, tails, multimodality, missingness,
//! knot uncertainty, and closed-loop packet-marginal PIT discrepancy.

use super::primitives::{
    clamp_f32, hartigan_dip_stat, l_moments_5, set_mask_bit, squash, squash_z, support_u_floor,
    SUPPORT_FLOOR_K,
};
use super::{ColumnMeta, PacketProbe, COL_DIM};
use crate::marginal::atoms::detect_atoms;
use crate::marginal::lattice::{detect_heaping, detect_lattice};
use crate::stats::{norm_cdf, norm_ppf, quantile_sorted};

/// Summary retained per column from T0 so T1/T2 reuse the exact sorted vector and metadata.
#[derive(Clone, Debug)]
pub(crate) struct ColT0Summary {
    pub is_const: bool,
    pub is_disc: bool,
    pub n_obs: usize,
    pub miss_rate: f64,
}

/// Fill slots `0..56` of `out_col` and its presence mask bits strictly from FIT values.
pub(crate) fn extract_column_t0(
    col_unit: &[f32],
    col_raw: Option<&[f64]>,
    meta: Option<&ColumnMeta>,
    probe: Option<&PacketProbe>,
    col_idx: usize,
    out_col: &mut [f32; COL_DIM],
    out_mask: &mut [u8; 16],
) -> ColT0Summary {
    let n_total = col_unit.len();
    let mut sorted_u: Vec<f64> = col_unit
        .iter()
        .map(|&v| v as f64)
        .filter(|v| v.is_finite())
        .collect();
    let n_obs = sorted_u.len();
    let miss_rate = if n_total > 0 {
        (n_total - n_obs) as f64 / n_total as f64
    } else {
        1.0
    };
    sorted_u.sort_by(f64::total_cmp);

    let mut sorted_raw: Vec<f64> = col_raw
        .map(|raw| raw.iter().copied().filter(|v| v.is_finite()).collect())
        .unwrap_or_else(|| sorted_u.clone());
    sorted_raw.sort_by(f64::total_cmp);

    // Block S_SUPPORT (0..16)
    let group_kind = meta.map(|m| m.group_kind).unwrap_or(0);
    out_col[0] = if group_kind == 0 { 1.0 } else { 0.0 };
    out_col[1] = if group_kind == 1 || group_kind == 2 {
        1.0
    } else {
        0.0
    };
    out_col[2] = if group_kind == 3 { 1.0 } else { 0.0 };
    out_col[3] = if group_kind == 4 { 1.0 } else { 0.0 };
    out_col[4] = if group_kind == 5 { 1.0 } else { 0.0 };
    out_col[5] = ((meta.map(|m| m.group_width).unwrap_or(1).max(1) as f64).log2() / 6.0) as f32;
    for s in 0..6 {
        set_mask_bit(out_mask, s);
    }

    if n_obs < 4 {
        out_col[6] = 1.0; // is_const / unavailable
        set_mask_bit(out_mask, 6);
        out_col[48] = miss_rate as f32;
        set_mask_bit(out_mask, 48);
        return ColT0Summary {
            is_const: true,
            is_disc: true,
            n_obs,
            miss_rate,
        };
    }

    let u_floor = support_u_floor(n_obs);
    let q_lo = quantile_sorted(&sorted_u, u_floor);
    let q_hi = quantile_sorted(&sorted_u, 1.0 - u_floor);
    let span = (q_hi - q_lo).max(0.0);
    let is_const = span < 1e-6;

    // Distinct run counts on raw (or unit) sorted values.
    let mut runs: Vec<(f64, usize)> = Vec::new();
    for &v in &sorted_raw {
        if let Some(last) = runs.last_mut() {
            if (last.0 - v).abs() <= 1e-9 * v.abs().max(1.0) {
                last.1 += 1;
                continue;
            }
        }
        runs.push((v, 1));
    }
    let n_distinct = runs.len();
    let max_run = runs.iter().map(|r| r.1).max().unwrap_or(0);
    let mode_mass = max_run as f64 / n_obs as f64;
    let is_disc = !is_const && n_distinct <= 16;

    // Atoms (k >= 10 floor) & decimal lattice + heaping (reusing dope-core::marginal!).
    let atoms = detect_atoms(&sorted_raw, SUPPORT_FLOOR_K);
    let total_atom_mass: f64 = atoms.iter().map(|a| a.prob).sum();
    let lattice = if !is_const {
        detect_lattice(&sorted_raw, SUPPORT_FLOOR_K)
    } else {
        None
    };
    let heaping = lattice
        .as_ref()
        .and_then(|spec| detect_heaping(&sorted_raw, spec));

    let lo_atom_mass = if runs[0].1 >= SUPPORT_FLOOR_K {
        runs[0].1 as f64 / n_obs as f64
    } else {
        0.0
    };
    let hi_atom_mass = if runs[n_distinct - 1].1 >= SUPPORT_FLOOR_K {
        runs[n_distinct - 1].1 as f64 / n_obs as f64
    } else {
        0.0
    };
    let zero_atom_mass = runs
        .iter()
        .find(|(v, c)| v.abs() <= 1e-9 && *c >= SUPPORT_FLOOR_K)
        .map(|(_, c)| *c as f64 / n_obs as f64)
        .unwrap_or(0.0);
    let inner_atom_mass = atoms
        .iter()
        .filter(|a| a.u_lo > 0.02 && a.u_hi < 0.98)
        .map(|a| a.prob)
        .fold(0.0f64, f64::max);

    let support_class = if is_const {
        4u8
    } else if is_disc || (!atoms.is_empty() && total_atom_mass >= 0.90) {
        2u8
    } else if lattice.is_some() {
        1u8
    } else if zero_atom_mass >= 0.10 || inner_atom_mass >= 0.10 {
        3u8
    } else {
        0u8
    };

    out_col[6] = if is_const { 1.0 } else { 0.0 };
    out_col[7] = if is_disc { 1.0 } else { 0.0 };
    out_col[8] = if lattice.is_some() { 1.0 } else { 0.0 };
    out_col[9] = ((n_distinct as f64).ln() / (n_obs as f64).ln().max(1.0)).clamp(0.0, 1.0) as f32;
    out_col[10] = mode_mass as f32;
    out_col[11] = clamp_f32(total_atom_mass, 0.0, 1.0);
    out_col[12] = clamp_f32(lo_atom_mass, 0.0, 1.0);
    out_col[13] = clamp_f32(hi_atom_mass, 0.0, 1.0);
    out_col[14] = clamp_f32(zero_atom_mass.max(inner_atom_mass), 0.0, 1.0);
    out_col[15] = heaping
        .as_ref()
        .map(|h| clamp_f32(h.excess, 0.0, 1.0))
        .unwrap_or(0.0);
    for s in 6..16 {
        set_mask_bit(out_mask, s);
    }

    // Robust scale & 17-knot quantile residuals M_QF_RESID (16..33).
    let mean = sorted_u.iter().sum::<f64>() / n_obs as f64;
    let var = sorted_u.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_obs as f64;
    let sd = var.sqrt();
    let med = quantile_sorted(&sorted_u, 0.50);
    let q25 = quantile_sorted(&sorted_u, 0.25);
    let q75 = quantile_sorted(&sorted_u, 0.75);
    let iqr = (q75 - q25).max(0.0);
    let robust_sd = (iqr / 1.349).max(sd).max(1e-6);

    let z_max = norm_ppf(1.0 - u_floor).clamp(1.2, 2.6);
    for k in 0..17 {
        let z_k = -z_max + (2.0 * z_max) * (k as f64 / 16.0);
        let q_k = quantile_sorted(&sorted_u, norm_cdf(z_k));
        let std_q = (q_k - med) / robust_sd;
        out_col[16 + k] = clamp_f32(std_q - z_k, -4.0, 4.0);
        set_mask_bit(out_mask, 16 + k);
    }

    // M_SHAPE_TAIL (33..48): L-moments, Bowley, Moors, Pickands/Hill, AD, Dip, KDE, entropy, NC2.
    let lm = l_moments_5(&sorted_u);
    let l_scale = if span > 1e-6 { lm[1] / span } else { 0.0 };
    let l_skew = if lm[1].abs() > 1e-9 {
        lm[2] / lm[1]
    } else {
        0.0
    };
    let l_kurt = if lm[1].abs() > 1e-9 {
        lm[3] / lm[1] - 0.1226
    } else {
        0.0
    };
    let l_tau5 = if lm[1].abs() > 1e-9 {
        lm[4] / lm[1]
    } else {
        0.0
    };

    let bowley = if iqr > 1e-9 {
        (q75 + q25 - 2.0 * med) / iqr
    } else {
        0.0
    };
    let o1 = quantile_sorted(&sorted_u, 0.125);
    let o3 = quantile_sorted(&sorted_u, 0.375);
    let o5 = quantile_sorted(&sorted_u, 0.625);
    let o7 = quantile_sorted(&sorted_u, 0.875);
    let moors = if iqr > 1e-9 {
        ((o7 - o5) + (o3 - o1)) / iqr - 1.233
    } else {
        0.0
    };

    // Pickands tail indices respecting u_floor (4*u < 0.45).
    let u_pick = u_floor.max(0.05);
    let (pick_hi, pick_lo) = if 4.0 * u_pick < 0.45 && !is_const {
        let q1 = quantile_sorted(&sorted_u, 1.0 - u_pick);
        let q2 = quantile_sorted(&sorted_u, 1.0 - 2.0 * u_pick);
        let q4 = quantile_sorted(&sorted_u, 1.0 - 4.0 * u_pick);
        let hi = (((q1 - q2).max(1e-6) / (q2 - q4).max(1e-6)).ln() / std::f64::consts::LN_2)
            .clamp(-4.0, 4.0);
        let l1 = quantile_sorted(&sorted_u, u_pick);
        let l2 = quantile_sorted(&sorted_u, 2.0 * u_pick);
        let l4 = quantile_sorted(&sorted_u, 4.0 * u_pick);
        let lo = (((l2 - l1).max(1e-6) / (l4 - l2).max(1e-6)).ln() / std::f64::consts::LN_2)
            .clamp(-4.0, 4.0);
        (hi, lo)
    } else {
        (0.0, 0.0)
    };

    // Anderson-Darling A^2 against N(mean, sd^2)
    let mut ad2 = 0.0f64;
    if sd > 1e-9 && !is_const {
        let nf = n_obs as f64;
        let mut s = 0.0;
        for (i, &v) in sorted_u.iter().enumerate() {
            let p1 = norm_cdf((v - mean) / sd).clamp(1e-12, 1.0 - 1e-12);
            let p2 = norm_cdf((sorted_u[n_obs - 1 - i] - mean) / sd).clamp(1e-12, 1.0 - 1e-12);
            s += (2.0 * (i as f64) + 1.0) * (p1.ln() + (1.0 - p2).ln());
        }
        ad2 = (-nf - s / nf).max(0.0);
    }

    // Hartigan Dip & 1D 32-pt KDE modes.
    let dip = if !is_const && !is_disc {
        hartigan_dip_stat(&sorted_u)
    } else {
        0.0
    };
    let (kde_modes, kde_pv_ratio) = if !is_const && !is_disc && n_obs >= 20 {
        estimate_kde_modes(&sorted_u, q_lo, q_hi, robust_sd)
    } else {
        (1, 0.0)
    };

    // 16-bin histogram Shannon & Renyi-2 entropy (normalized).
    let (h1_norm, h2_norm) = histogram_entropies(&sorted_u, q_lo, q_hi, 16);

    // Pre-tail / skew guard flags from NC2.1 metadata (or estimated from unit space).
    let skew_guard = meta.map(|m| m.skew_guard_active as f32).unwrap_or_else(|| {
        let frac_comp = sorted_u.iter().filter(|&&v| v.abs() > 3.0).count() as f64 / n_obs as f64;
        if frac_comp > 0.01 {
            1.0
        } else {
            0.0
        }
    });
    let pre_tail_asym = meta
        .map(|m| m.log_pre_tail_asym.clamp(-1.0, 1.0))
        .unwrap_or_else(|| {
            let num = (q_hi - med).max(1e-6);
            let den = (med - q_lo).max(1e-6);
            ((num / den).ln().clamp(-4.0, 4.0) / 4.0) as f32
        });

    out_col[33] = clamp_f32(l_scale, 0.0, 2.0);
    out_col[34] = clamp_f32(l_skew, -1.0, 1.0);
    out_col[35] = clamp_f32(l_kurt, -1.0, 1.0);
    out_col[36] = clamp_f32(l_tau5, -1.0, 1.0);
    out_col[37] = clamp_f32(bowley, -1.0, 1.0);
    out_col[38] = squash(moors);
    out_col[39] = squash(pick_hi);
    out_col[40] = squash(pick_lo);
    out_col[41] = squash(ad2 / 4.0);
    out_col[42] = squash(2.0 * (n_obs as f64).sqrt() * dip);
    out_col[43] = (kde_modes.min(4) as f32) / 4.0;
    out_col[44] = clamp_f32(kde_pv_ratio, 0.0, 1.0);
    out_col[45] = clamp_f32(h1_norm, 0.0, 1.0);
    out_col[46] = clamp_f32(h2_norm, 0.0, 1.0);
    out_col[47] = pre_tail_asym;
    for s in 33..48 {
        set_mask_bit(out_mask, s);
    }

    // M_MISS_UNC_PIT (48..56)
    let frac_clipped = sorted_u.iter().filter(|&&v| v.abs() >= 5.99).count() as f64 / n_obs as f64;
    let inv_sqrt_n = (1.0 / (n_obs as f64).sqrt()).clamp(0.0, 1.0);
    let sample_conf = n_obs as f64 / (n_obs as f64 + 20.0);
    out_col[48] = miss_rate as f32;
    out_col[49] = skew_guard;
    out_col[50] = clamp_f32(frac_clipped, 0.0, 1.0);
    out_col[51] = inv_sqrt_n as f32;
    out_col[52] = sample_conf as f32;
    for s in 48..53 {
        set_mask_bit(out_mask, s);
    }

    // M07 Packet-marginal PIT KS, Z, W1 when closed-loop PacketProbe is available (slots 53..56).
    if let Some(pp) = probe {
        if let Some(syn_col) = pp.x_syn_unit.get(col_idx) {
            let (ks, ks_z, w1) = two_sample_ks_w1(&sorted_u, syn_col);
            out_col[53] = clamp_f32(ks, 0.0, 1.0);
            out_col[54] = squash_z(ks_z);
            out_col[55] = squash(w1);
            set_mask_bit(out_mask, 53);
            set_mask_bit(out_mask, 54);
            set_mask_bit(out_mask, 55);
        }
    }

    let _ = support_class;
    ColT0Summary {
        is_const,
        is_disc,
        n_obs,
        miss_rate,
    }
}

fn estimate_kde_modes(sorted: &[f64], lo: f64, hi: f64, robust_sd: f64) -> (usize, f64) {
    let n = sorted.len();
    let span = hi - lo;
    if span <= 1e-6 {
        return (1, 0.0);
    }
    let h = (0.9 * robust_sd * (n as f64).powf(-0.2)).max(span / 32.0);
    let inv_h = 1.0 / h;
    let m = 32usize;
    let mut grid = [0.0f64; 32];
    // Evaluate KDE using 64 stratified quantiles for O(1) speed regardless of n.
    let q_pts = 64usize;
    let samples: Vec<f64> = (0..q_pts)
        .map(|k| quantile_sorted(sorted, (k as f64 + 0.5) / q_pts as f64))
        .collect();
    for g in 0..m {
        let xg = lo + span * (g as f64 / (m - 1) as f64);
        let mut d = 0.0;
        for &s in &samples {
            let z = (xg - s) * inv_h;
            d += (-0.5 * z * z).exp();
        }
        grid[g] = d / q_pts as f64;
    }
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for g in 1..(m - 1) {
        if grid[g] > grid[g - 1] && grid[g] >= grid[g + 1] {
            peaks.push((g, grid[g]));
        }
    }
    if peaks.is_empty() {
        return (1, 0.0);
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    if peaks.len() == 1 {
        return (1, 0.0);
    }
    let (i1, _) = peaks[0];
    let (i2, p2) = peaks[1];
    let (a, b) = if i1 < i2 { (i1, i2) } else { (i2, i1) };
    let valley = grid[a..=b].iter().copied().fold(f64::INFINITY, f64::min);
    let pv_ratio = if p2 > 1e-12 {
        (1.0 - valley / p2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (peaks.len(), pv_ratio)
}

fn histogram_entropies(sorted: &[f64], lo: f64, hi: f64, bins: usize) -> (f64, f64) {
    let n = sorted.len();
    let span = hi - lo;
    if n < 4 || span <= 1e-9 {
        return (0.0, 0.0);
    }
    let mut cnt = vec![0usize; bins];
    for &v in sorted {
        let b = (((v - lo) / span) * bins as f64)
            .floor()
            .clamp(0.0, (bins - 1) as f64) as usize;
        cnt[b] += 1;
    }
    let nf = n as f64;
    let mut h1 = 0.0;
    let mut sum_p2 = 0.0;
    for &c in &cnt {
        if c > 0 {
            let p = c as f64 / nf;
            h1 -= p * p.ln();
            sum_p2 += p * p;
        }
    }
    let max_h = (bins as f64).ln().max(1e-6);
    let h2 = if sum_p2 > 1e-12 { -sum_p2.ln() } else { 0.0 };
    ((h1 / max_h).clamp(0.0, 1.0), (h2 / max_h).clamp(0.0, 1.0))
}

/// Fast two-sample KS, KS-Z, and 1D Wasserstein-1 on 33 quantiles.
pub(crate) fn two_sample_ks_w1(sorted_real: &[f64], syn: &[f32]) -> (f64, f64, f64) {
    let mut s_syn: Vec<f64> = syn
        .iter()
        .map(|&v| v as f64)
        .filter(|v| v.is_finite())
        .collect();
    let n1 = sorted_real.len();
    let n2 = s_syn.len();
    if n1 < 4 || n2 < 4 {
        return (0.0, 0.0, 0.0);
    }
    s_syn.sort_by(f64::total_cmp);
    // Exact two-pointer KS.
    let (mut i, mut j, mut ks) = (0usize, 0usize, 0.0f64);
    while i < n1 && j < n2 {
        if sorted_real[i] <= s_syn[j] {
            i += 1;
        } else {
            j += 1;
        }
        let d = ((i as f64 / n1 as f64) - (j as f64 / n2 as f64)).abs();
        if d > ks {
            ks = d;
        }
    }
    let n_eff = (n1 as f64 * n2 as f64) / (n1 + n2) as f64;
    let ks_z = ks * n_eff.sqrt();
    // 33-point W1.
    let mut w1 = 0.0;
    for k in 0..33 {
        let u = (k as f64 + 0.5) / 33.0;
        w1 += (quantile_sorted(sorted_real, u) - quantile_sorted(&s_syn, u)).abs();
    }
    w1 /= 33.0;
    (ks, ks_z, w1)
}
