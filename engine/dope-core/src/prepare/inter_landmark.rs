//! Rust port of `compute_inter_landmark_summary_stats` (legacy, full-report
//! `inter_landmark_vec48`) and `compute_observed_inter_landmark_stats`
//! (optional `none`/`fast`/`progressive` observed-only `inter_landmark_vec48`)
//! from `research/levels/building_block_landmarks.py`.

use super::building_blocks::{
    gini, level_rank, np, safe_pair_corr, squash, BuildingBlockDescriptors,
};

/// The report-descriptor fields the inter-landmark block reads
/// (`extract_report_descriptors` output).
#[derive(Debug, Clone, Copy)]
pub struct ReportLandmarkView<'a> {
    /// `gains23`: SQUASHED gains, `f64` (zero where absent).
    pub gains23: &'a [f64; 23],
    /// `mask23`: present cells.
    pub mask23: &'a [bool; 23],
    /// `ses23`: squashed standard errors, `f64`.
    pub ses23: &'a [f64; 23],
    /// `lambdas23`: `mean(lambda_indices) / 8`.
    pub lambdas23: &'a [f64; 23],
    /// `column_spectrum`: `squash(spec_matrix).astype(float32)`, one row per
    /// report feature (at least one row).
    pub column_spectrum: &'a [[f32; 9]],
}

fn pick(v: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| v[i]).collect()
}

/// Python builtin `max(*vals)` on floats.
fn pmax(vals: &[f64]) -> f64 {
    vals[1..].iter().fold(vals[0], |a, &b| np::py_max(a, b))
}

/// `compute_inter_landmark_summary_stats(rep_desc, tab_desc, bb_desc, level=level)`.
pub fn compute_inter_landmark_summary_stats(
    rep: &ReportLandmarkView,
    bb: &BuildingBlockDescriptors,
    level: &str,
) -> [f32; 48] {
    let lvl_r = level_rank(level);
    let g23: &[f64; 23] = rep.gains23;
    let m23: &[bool; 23] = rep.mask23;
    let se23: &[f64; 23] = rep.ses23;
    let lam23: &[f64; 23] = rep.lambdas23;
    let bb16: Vec<f64> = bb.bb_gains16.iter().map(|&v| v as f64).collect();

    // 1. Category A: functional landmark family moments
    let cat = |a: &[usize], b: &[usize]| {
        let mut v = pick(g23, a);
        v.extend(pick(&bb16, b));
        v
    };
    let families: [Vec<f64>; 5] = [
        cat(&[0, 1, 2, 4], &[0, 1, 2, 3, 4]),
        cat(&[3, 5, 9], &[6, 7, 13]),
        if lvl_r >= 1 {
            cat(&[6, 7, 15, 16, 17], &[9, 10])
        } else {
            vec![0.0]
        },
        if lvl_r >= 2 {
            cat(&[8, 12], &[11, 12, 14])
        } else {
            vec![0.0]
        },
        if lvl_r >= 1 {
            cat(&[10, 11, 18, 19, 20], &[5, 15])
        } else {
            vec![bb16[5]]
        },
    ];
    let mut all_active: Vec<f64> = (0..23).filter(|&i| m23[i]).map(|i| g23[i]).collect();
    all_active.extend(bb16.iter().copied().filter(|&v| v > 0.0));
    let s_max = np::py_max(
        if all_active.is_empty() {
            0.0
        } else {
            np::max(&all_active)
        },
        0.02,
    );

    let mut fam_vec = [0.0f64; 20];
    for (f, vals) in families.iter().enumerate() {
        let pos: Vec<f64> = vals
            .iter()
            .copied()
            .filter(|v| v.is_finite() && v.abs() > 1e-9)
            .collect();
        if !pos.is_empty() {
            let (f_max, f_mean, f_std) = (np::max(&pos), np::mean(&pos), np::std(&pos));
            fam_vec[4 * f] = np::clip(f_max, -1.0, 1.0);
            fam_vec[4 * f + 1] = np::clip(f_mean, -1.0, 1.0);
            fam_vec[4 * f + 2] = np::clip(f_std / s_max, 0.0, 1.0);
            fam_vec[4 * f + 3] = np::clip(f_max / s_max, -1.0, 1.0);
        }
    }

    // 2. Category B: cross-family ratios
    let lin_g = pmax(&[g23[0], bb16[4] * 0.5, 0.0]);
    let best1_g = pmax(&[g23[14], lin_g, 0.01]);
    let tail_best = pmax(&[g23[1], g23[2], bb16[0], bb16[3]]);
    let sharp_best = pmax(&[g23[7], g23[15], bb16[9]]);
    let smooth_best = pmax(&[g23[4], bb16[1], bb16[2], bb16[6]]);
    let pair_polar_best = pmax(&[g23[11], bb16[13], bb16[14], bb16[15]]);
    let row_comb = pmax(&[g23[10], bb16[5]]);
    let per_best = pmax(&[g23[8], bb16[11], bb16[12]]);
    let d_lin = if m23[0] && m23[21] {
        g23[0] - g23[21]
    } else {
        0.0
    };
    let d_gbdt = if lvl_r >= 3 && m23[17] && m23[22] {
        g23[17] - g23[22]
    } else {
        0.0
    };

    let lm_entropy = if all_active.len() > 1 {
        let logits: Vec<f64> = all_active.iter().map(|&a| 8.0 * a / s_max).collect();
        let lmax = np::max(&logits);
        let logits: Vec<f64> = logits.iter().map(|&l| l - lmax).collect();
        let ex: Vec<f64> = logits.iter().map(|&l| l.exp()).collect();
        let den = np::py_max(np::sum(&ex), 1e-12);
        let probs: Vec<f64> = ex.iter().map(|&e| e / den).collect();
        let terms: Vec<f64> = probs
            .iter()
            .map(|&q| q * np::maximum(q, 1e-12).ln())
            .collect();
        -np::sum(&terms) / np::py_max((probs.len() as f64).ln(), 1e-9)
    } else {
        0.0
    };

    let c = |v: f64| np::clip(v, -1.0, 1.0);
    let gate = |ok: bool, v: f64| if ok { v } else { 0.0 };
    let ratios_vec: [f64; 12] = [
        c((tail_best - lin_g) / s_max),
        gate(lvl_r >= 2, c((sharp_best - smooth_best) / s_max)),
        gate(lvl_r >= 2, c((g23[16] - g23[15]) / s_max)),
        gate(lvl_r >= 3, c((g23[17] - g23[16]) / s_max)),
        gate(lvl_r >= 2, c((pair_polar_best - best1_g) / s_max)),
        c((row_comb - best1_g) / s_max),
        squash(row_comb / (lin_g + 0.02)),
        gate(lvl_r >= 2, c((per_best - fam_vec[0]) / s_max)),
        gate(
            lvl_r >= 3,
            (np::asinh(np::py_max(g23[20], 0.0) / (np::py_max(g23[17], 0.0) + 0.02))
                - np::asinh(1.0))
            .tanh(),
        ),
        gate(lvl_r >= 3, c((g23[18] - g23[19]) / s_max)),
        if lvl_r >= 3 {
            c((d_gbdt - d_lin) / (np::py_max(d_gbdt.abs(), d_lin.abs()) + 0.02))
        } else {
            c(d_lin / s_max)
        },
        np::clip(lm_entropy, 0.0, 1.0),
    ];

    // 3. Category C: confidence (gain/SE) & regularization
    let mut conf_vec = [0.0f64; 8];
    if m23.iter().any(|&m| m) {
        let act: Vec<usize> = (0..23).filter(|&i| m23[i]).collect();
        let g_act = pick(g23, &act);
        let se_act: Vec<f64> = act.iter().map(|&i| np::maximum(se23[i], 1e-3)).collect();
        let lam_act = pick(lam23, &act);
        let weights: Vec<f64> = se_act.iter().map(|&s| 1.0 / (s * s + 1e-4)).collect();
        let t_stats: Vec<f64> = g_act.iter().zip(&se_act).map(|(&g, &s)| g / s).collect();
        let wg: Vec<f64> = weights.iter().zip(&g_act).map(|(&w, &g)| w * g).collect();
        conf_vec[0] = c(np::sum(&wg) / np::py_max(np::sum(&weights), 1e-9));
        conf_vec[1] = squash(np::max(&t_stats) / 10.0);
        conf_vec[2] = squash(np::mean(&t_stats) / 10.0);
        conf_vec[3] = squash(np::std(&t_stats) / 10.0);
        let above = t_stats.iter().filter(|&&t| t > 2.0).count();
        conf_vec[4] = above as f64 / t_stats.len() as f64;
        conf_vec[5] = np::clip(np::mean(&lam_act), 0.0, 1.0);
        conf_vec[6] = np::clip(np::std(&lam_act), 0.0, 1.0);
        conf_vec[7] = safe_pair_corr(&g_act, &lam_act);
    }

    // 4. Category D: column × landmark spectrum alignment
    let mut align_vec = [0.0f64; 8];
    let col_spec = rep.column_spectrum;
    let col_bb: Vec<[f64; 7]> = bb
        .col_bb_probes7
        .iter()
        .map(|r| r.map(|v| v as f64))
        .collect();
    let p = col_bb.len();
    if p >= 1 {
        let spec_ok = col_spec.len() >= p;
        let spec = |j: usize, k: usize| {
            if spec_ok {
                col_spec[j][k] as f64
            } else {
                0.0
            }
        };
        let s_lin: Vec<f64> = (0..p)
            .map(|j| if spec_ok { spec(j, 0) } else { col_bb[j][0] })
            .collect();
        let s_step: Vec<f64> = (0..p)
            .map(|j| np::maximum(col_bb[j][3], spec(j, 1)))
            .collect();
        let s_xi: Vec<f64> = (0..p)
            .map(|j| np::maximum(col_bb[j][6], spec(j, 7)))
            .collect();
        let s_asinh: Vec<f64> = (0..p).map(|j| col_bb[j][0]).collect();
        let s_nonlin: Vec<f64> = (0..p).map(|j| np::max(&col_bb[j][1..])).collect();
        if lvl_r >= 2 {
            align_vec[0] = safe_pair_corr(&s_lin, &s_step);
            align_vec[1] = safe_pair_corr(&s_lin, &s_xi);
        }
        align_vec[2] = safe_pair_corr(&s_lin, &s_asinh);
        let orth: Vec<f64> = (0..p)
            .map(|j| np::maximum(s_nonlin[j] - s_lin[j], 0.0))
            .collect();
        let s_orth = np::sorted_desc(&orth);
        align_vec[3] = squash(s_orth[0]);
        align_vec[4] = squash(np::mean(&s_orth[..p.min(3)]));
        let ln7 = 7f64.ln();
        let col_ent: Vec<f64> = col_bb
            .iter()
            .map(|row| {
                let pos = row.map(|v| np::maximum(v, 1e-6));
                let total = np::sum(&pos);
                let terms = pos.map(|v| {
                    let q = v / total;
                    q * q.ln()
                });
                -np::sum(&terms) / ln7
            })
            .collect();
        align_vec[5] = np::clip(np::mean(&col_ent), 0.0, 1.0);
        align_vec[6] = squash(gini(&s_nonlin) / (gini(&s_lin) + 0.05));
        align_vec[7] = squash(
            bb16[5] / pmax(&[np::max(&s_nonlin), np::max(&s_lin), 0.02]),
        );
    }

    let mut out = [0.0f32; 48];
    let all = fam_vec
        .iter()
        .chain(&ratios_vec)
        .chain(&conf_vec)
        .chain(&align_vec);
    for (o, &v) in out.iter_mut().zip(all) {
        let f = v as f32;
        *o = if f.is_finite() { f } else { 0.0 };
    }
    out
}

/// Observed-only 48 inter-landmark stats (`compute_observed_inter_landmark_stats`
/// in `research/levels/building_block_landmarks.py`).
///
/// Uses ONLY measured landmark gains (`gains23` squashed `f64`, `mask23`
/// level-filtered). Building blocks, SE, lambdas and spectrum are unmeasured
/// on the `observed-only-v1` path (`none` / `fast` / `progressive`), so every
/// term involving them is dropped and every entry whose required gains are
/// absent is `0.0`. On an empty mask (`none` / `fast`), all 48 outputs are `0.0`.
pub fn compute_observed_inter_landmark_stats(
    gains23: &[f64; 23],
    mask23: &[bool; 23],
    level: &str,
) -> [f32; 48] {
    let lvl_r = level_rank(level);
    let g = gains23;
    let m = mask23;

    let mp = |idxs: &[usize]| -> Option<f64> {
        let mut iter = idxs.iter().filter(|&&i| m[i]).map(|&i| g[i]);
        let first = iter.next()?;
        Some(iter.fold(first, np::py_max))
    };

    let active: Vec<f64> = (0..23).filter(|&i| m[i]).map(|i| g[i]).collect();
    let s_max = np::py_max(
        if active.is_empty() {
            0.0
        } else {
            np::max(&active)
        },
        0.02,
    );
    let fam_idx: [&[usize]; 5] = [
        &[0, 1, 2, 4],
        &[3, 5, 9],
        if lvl_r >= 1 { &[6, 7, 15, 16, 17] } else { &[] },
        if lvl_r >= 2 { &[8, 12] } else { &[] },
        if lvl_r >= 1 {
            &[10, 11, 18, 19, 20]
        } else {
            &[]
        },
    ];
    let mut fam_vec = [0.0f64; 20];
    let mut fam_has = [false; 5];
    for (f_idx, idxs) in fam_idx.iter().enumerate() {
        let pos: Vec<f64> = idxs
            .iter()
            .copied()
            .filter(|&i| m[i] && g[i].is_finite() && g[i].abs() > 1e-9)
            .map(|i| g[i])
            .collect();
        if !pos.is_empty() {
            fam_has[f_idx] = true;
            let (f_max, f_mean, f_std) = (np::max(&pos), np::mean(&pos), np::std(&pos));
            fam_vec[4 * f_idx] = np::clip(f_max, -1.0, 1.0);
            fam_vec[4 * f_idx + 1] = np::clip(f_mean, -1.0, 1.0);
            fam_vec[4 * f_idx + 2] = np::clip(f_std / s_max, 0.0, 1.0);
            fam_vec[4 * f_idx + 3] = np::clip(f_max / s_max, -1.0, 1.0);
        }
    }

    let clip1 = |v: f64| np::clip(v, -1.0, 1.0);
    let mut r = [0.0f64; 12];
    let lin = if m[0] { Some(np::py_max(g[0], 0.0)) } else { None };
    let b1 = mp(&[0, 14]);
    let best1 = b1.map(|v| np::py_max(v, 0.01));
    let tail = mp(&[1, 2]);
    if let (Some(lin), Some(tail)) = (lin, tail) {
        r[0] = clip1((tail - lin) / s_max);
    }
    if lvl_r >= 2 {
        if let (Some(sharp), Some(smooth)) = (mp(&[7, 15]), mp(&[4])) {
            r[1] = clip1((sharp - smooth) / s_max);
        }
        if m[15] && m[16] {
            r[2] = clip1((g[16] - g[15]) / s_max);
        }
        if m[11] {
            if let Some(best1) = best1 {
                r[4] = clip1((g[11] - best1) / s_max);
            }
        }
        if m[8] && fam_has[0] {
            r[7] = clip1((g[8] - fam_vec[0]) / s_max);
        }
    }
    if lvl_r >= 3 && m[16] && m[17] {
        r[3] = clip1((g[17] - g[16]) / s_max);
    }
    if m[10] {
        if let Some(best1) = best1 {
            r[5] = clip1((g[10] - best1) / s_max);
        }
        if let Some(lin) = lin {
            r[6] = squash(g[10] / (lin + 0.02));
        }
    }
    if lvl_r >= 3 {
        if m[20] && m[17] {
            r[8] = (np::asinh(np::py_max(g[20], 0.0) / (np::py_max(g[17], 0.0) + 0.02))
                - np::asinh(1.0))
            .tanh();
        }
        if m[18] && m[19] {
            r[9] = clip1((g[18] - g[19]) / s_max);
        }
        if m[0] && m[21] && m[17] && m[22] {
            let d_lin = g[0] - g[21];
            let d_gbdt = g[17] - g[22];
            r[10] = clip1((d_gbdt - d_lin) / (np::py_max(d_gbdt.abs(), d_lin.abs()) + 0.02));
        }
    } else if m[0] && m[21] {
        r[10] = clip1((g[0] - g[21]) / s_max);
    }
    if active.len() > 1 {
        let logits: Vec<f64> = active.iter().map(|&a| 8.0 * a / s_max).collect();
        let lmax = np::max(&logits);
        let logits: Vec<f64> = logits.iter().map(|&l| l - lmax).collect();
        let ex: Vec<f64> = logits.iter().map(|&l| l.exp()).collect();
        let den = np::py_max(np::sum(&ex), 1e-12);
        let probs: Vec<f64> = ex.iter().map(|&e| e / den).collect();
        let terms: Vec<f64> = probs
            .iter()
            .map(|&q| q * np::maximum(q, 1e-12).ln())
            .collect();
        let ent = -np::sum(&terms) / np::py_max((probs.len() as f64).ln(), 1e-9);
        r[11] = np::clip(ent, 0.0, 1.0);
    }

    let mut out = [0.0f32; 48];
    for (o, &v) in out[..32].iter_mut().zip(fam_vec.iter().chain(&r)) {
        let f = v as f32;
        *o = if f.is_finite() { f } else { 0.0 };
    }
    out
}

/// Convenience wrapper over [`compute_observed_inter_landmark_stats`] taking a
/// [`ReportLandmarkView`].
pub fn compute_observed_inter_landmark_summary_stats(
    rep: &ReportLandmarkView,
    level: &str,
) -> [f32; 48] {
    compute_observed_inter_landmark_stats(rep.gains23, rep.mask23, level)
}
