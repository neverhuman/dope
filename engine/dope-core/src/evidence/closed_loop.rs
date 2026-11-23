//! Tier T3 Ephemeral Closed-Loop Evidence against a candidate decoded packet (`PacketProbe`).
//!
//! Strictly uncached per table and masked `0` when `packet_probe` is `None`.

use super::primitives::{canonical_row_order, clamp_f32, set_mask_bit, squash};
use super::witness::fit_obliv_gbdt;
use super::{PacketProbe, COL_DIM, GLOB_DIM};
use crate::rng::Rng;
use crate::stats::pearson;

/// Extract Tier T3 closed-loop discrepancy against candidate packet samples (`PacketProbe`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_closed_loop_t3(
    x_unit: &[Vec<f32>],
    probe: &PacketProbe,
    seed: u64,
    out_col: &mut [[f32; COL_DIM]],
    out_col_mask: &mut [[u8; 16]],
    out_glob: &mut [f32; GLOB_DIM],
    out_glob_mask: &mut [u8; 20],
) {
    let p = x_unit.len();
    let n_real = x_unit.first().map(|c| c.len()).unwrap_or(0);
    let n_syn = probe.x_syn_unit.first().map(|c| c.len()).unwrap_or(0);
    if p == 0 || probe.x_syn_unit.len() != p || n_real < 20 || n_syn < 20 {
        return;
    }

    // Canonicalize synthetic probe row order too so T3 is 100% row-permutation invariant.
    let syn_order = canonical_row_order(&probe.x_syn_unit);
    let syn_canon: Vec<Vec<f32>> = (0..p)
        .map(|j| syn_order.iter().map(|&r| probe.x_syn_unit[j][r]).collect())
        .collect();

    let m = n_real.min(n_syn).min(256);
    let mut c2st_bins = vec![vec![0u8; 2 * m]; p];
    let mut y_disc = vec![0.0f64; 2 * m];
    for y in y_disc.iter_mut().take(m) {
        *y = 1.0;
    }
    let mut col_mean_gap = vec![0.0f64; p];
    let mut col_sd_gap = vec![0.0f64; p];

    for j in 0..p {
        let mut s_r: Vec<f64> = x_unit[j][..m]
            .iter()
            .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
            .collect();
        let s_s: Vec<f64> = syn_canon[j][..m]
            .iter()
            .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
            .collect();
        let mr = s_r.iter().sum::<f64>() / m as f64;
        let ms = s_s.iter().sum::<f64>() / m as f64;
        let sdr = (s_r.iter().map(|v| (v - mr).powi(2)).sum::<f64>() / m as f64).sqrt();
        let sds = (s_s.iter().map(|v| (v - ms).powi(2)).sum::<f64>() / m as f64).sqrt();
        col_mean_gap[j] = (mr - ms).abs();
        col_sd_gap[j] = (sdr - sds).abs();

        s_r.sort_by(f64::total_cmp);
        let edges: Vec<f64> = (1..16).map(|b| s_r[(b * m / 16).min(m - 1)]).collect();
        for r in 0..m {
            let vr = if x_unit[j][r].is_finite() {
                x_unit[j][r] as f64
            } else {
                0.0
            };
            let vs = if syn_canon[j][r].is_finite() {
                syn_canon[j][r] as f64
            } else {
                0.0
            };
            c2st_bins[j][r] = edges.partition_point(|&e| vr > e) as u8;
            c2st_bins[j][m + r] = edges.partition_point(|&e| vs > e) as u8;
        }
    }

    let mut gain_c2st = vec![0.0f64; p];
    let mut scores = vec![0.0f64; 2 * m];
    for fold in 0..2 {
        let tr: Vec<usize> = (0..(2 * m)).filter(|r| r % 2 != fold).collect();
        let va: Vec<usize> = (0..(2 * m)).filter(|r| r % 2 == fold).collect();
        let (base, trees) = fit_obliv_gbdt(&c2st_bins, &y_disc, &tr, true, 16, 0.25);
        for t in &trees {
            for d in 0..3 {
                gain_c2st[t.cols[d]] += t.gains[d];
            }
        }
        for &r in &va {
            let mut s = base;
            for t in &trees {
                s += t.predict(&c2st_bins, r);
            }
            scores[r] = s;
        }
    }
    let mut wins = 0.0;
    for i in 0..m {
        for j in m..(2 * m) {
            if scores[i] > scores[j] {
                wins += 1.0;
            } else if (scores[i] - scores[j]).abs() <= 1e-12 {
                wins += 0.5;
            }
        }
    }
    let c2st_auc = (wins / (m * m) as f64).clamp(0.0, 1.0);
    let tot_g: f64 = gain_c2st.iter().sum::<f64>().max(1e-12);

    let mut corr_gap = 0.0;
    let mut n_p = 0usize;
    for i in 0..p.min(32) {
        let xi_r: Vec<f64> = x_unit[i][..m]
            .iter()
            .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
            .collect();
        let xi_s: Vec<f64> = syn_canon[i][..m]
            .iter()
            .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
            .collect();
        for j in (i + 1)..p.min(32) {
            let xj_r: Vec<f64> = x_unit[j][..m]
                .iter()
                .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
                .collect();
            let xj_s: Vec<f64> = syn_canon[j][..m]
                .iter()
                .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
                .collect();
            corr_gap += (pearson(&xi_r, &xj_r) - pearson(&xi_s, &xj_s)).powi(2);
            n_p += 1;
        }
    }
    let corr_rms = if n_p > 0 {
        (corr_gap / n_p as f64).sqrt()
    } else {
        0.0
    };

    let mut rng = Rng::new(seed ^ 0xDC80_7300 ^ probe.probe_id);
    let m_dcr = (m / 2).min(64);
    let mut closer_a = 0usize;
    for _ in 0..m_dcr {
        let sr = (rng.next_u64() as usize) % n_syn;
        let mut da = f64::INFINITY;
        let mut db = f64::INFINITY;
        for a_idx in 0..m_dcr {
            let dist: f64 = (0..p)
                .map(|j| (syn_canon[j][sr] as f64 - x_unit[j][a_idx] as f64).powi(2))
                .sum();
            if dist < da {
                da = dist;
            }
        }
        for b_idx in m_dcr..(2 * m_dcr).min(n_real) {
            let dist: f64 = (0..p)
                .map(|j| (syn_canon[j][sr] as f64 - x_unit[j][b_idx] as f64).powi(2))
                .sum();
            if dist < db {
                db = dist;
            }
        }
        if da < db {
            closer_a += 1;
        }
    }
    let dcr_share = if m_dcr > 0 {
        closer_a as f64 / m_dcr as f64
    } else {
        0.5
    };

    for j in 0..p {
        out_col[j][124] = clamp_f32(gain_c2st[j] / tot_g, 0.0, 1.0);
        out_col[j][125] = squash(col_mean_gap[j]);
        out_col[j][126] = squash(col_sd_gap[j]);
        set_mask_bit(&mut out_col_mask[j], 124);
        set_mask_bit(&mut out_col_mask[j], 125);
        set_mask_bit(&mut out_col_mask[j], 126);
    }
    out_glob[140] = clamp_f32(c2st_auc, 0.0, 1.0);
    out_glob[141] = squash(corr_rms);
    out_glob[142] = clamp_f32(dcr_share, 0.0, 1.0);
    out_glob[143] = clamp_f32(probe.packet_bytes as f64 / 2048.0, 0.0, 4.0);
    out_glob[144] = 1.0;
    for s in 140..145 {
        set_mask_bit(out_glob_mask, s);
    }
}
