//! Tier T2 Evidence: 2-fold 40-tree depth-3 oblivious GBDT witness on 16-bin inputs,
//! held-out grouped permutation importance, exact Walsh-Hadamard fANOVA, OOF ECE15,
//! and 1-NN LOO label noise floor.

use super::primitives::{clamp_f32, set_mask_bit, SUPPORT_FLOOR_K};
use super::target_interactions::TargetContext;
use super::{ColumnMeta, COL_DIM, EDGE_DIM, GLOB_DIM};

/// Lightweight depth-3 oblivious tree fitted on 16-bin features (`min_leaf >= 10`).
#[derive(Clone, Debug)]
pub(crate) struct OblivTree3 {
    pub cols: [usize; 3],
    pub thrs: [u8; 3],
    pub leaves: [f64; 8],
    pub leaf_counts: [usize; 8],
    pub gains: [f64; 3],
}

impl OblivTree3 {
    #[inline]
    pub fn leaf_idx(&self, bins16: &[Vec<u8>], r: usize) -> usize {
        let b0 = usize::from(bins16[self.cols[0]][r] > self.thrs[0]);
        let b1 = usize::from(bins16[self.cols[1]][r] > self.thrs[1]);
        let b2 = usize::from(bins16[self.cols[2]][r] > self.thrs[2]);
        b0 | (b1 << 1) | (b2 << 2)
    }

    #[inline]
    pub fn predict(&self, bins16: &[Vec<u8>], r: usize) -> f64 {
        self.leaves[self.leaf_idx(bins16, r)]
    }
}

/// Fit a 1D boosted ensemble of `n_trees` depth-3 oblivious trees on `train_rows`.
pub(crate) fn fit_obliv_gbdt(
    bins16: &[Vec<u8>],
    y: &[f64],
    train_rows: &[usize],
    is_binary: bool,
    n_trees: usize,
    lr: f64,
) -> (f64, Vec<OblivTree3>) {
    let p = bins16.len();
    let n = train_rows.len();
    if p == 0 || n < 2 * SUPPORT_FLOOR_K {
        return (0.0, Vec::new());
    }
    let y_mean = train_rows.iter().map(|&r| y[r]).sum::<f64>() / n as f64;
    let base = if is_binary {
        let pr = y_mean.clamp(0.02, 0.98);
        (pr / (1.0 - pr)).ln()
    } else {
        y_mean
    };
    let mut pred = vec![base; n];
    let mut trees = Vec::with_capacity(n_trees);

    for _ in 0..n_trees {
        let mut grad = vec![0.0f64; n];
        let mut hess = vec![1.0f64; n];
        for (i, &r) in train_rows.iter().enumerate() {
            if is_binary {
                let pr = 1.0 / (1.0 + (-pred[i].clamp(-10.0, 10.0)).exp());
                grad[i] = y[r] - pr;
                hess[i] = (pr * (1.0 - pr)).max(0.02);
            } else {
                grad[i] = y[r] - pred[i];
                hess[i] = 1.0;
            }
        }
        let mut leaf_id = vec![0usize; n];
        let mut cols = [0usize; 3];
        let mut thrs = [7u8; 3];
        let mut gains = [0.0f64; 3];
        let reg = 4.0f64;

        for depth in 0..3 {
            let n_leaves = 1usize << depth;
            let mut best_gain = 0.0f64;
            let mut best_col = 0usize;
            let mut best_thr = 7u8;

            let mut cur_g = [0.0f64; 4];
            let mut cur_h = [0.0f64; 4];
            let mut cur_c = [0usize; 4];
            for i in 0..n {
                let l = leaf_id[i];
                cur_g[l] += grad[i];
                cur_h[l] += hess[i];
                cur_c[l] += 1;
            }
            let base_score: f64 = (0..n_leaves)
                .map(|l| cur_g[l] * cur_g[l] / (cur_h[l] + reg))
                .sum();

            for col in 0..p {
                let mut hg = [[0.0f64; 16]; 4];
                let mut hh = [[0.0f64; 16]; 4];
                let mut hc = [[0usize; 16]; 4];
                let cb = &bins16[col];
                for (i, &r) in train_rows.iter().enumerate() {
                    let b = cb[r].min(15) as usize;
                    let l = leaf_id[i];
                    hg[l][b] += grad[i];
                    hh[l][b] += hess[i];
                    hc[l][b] += 1;
                }
                let mut lg = [0.0f64; 4];
                let mut lh = [0.0f64; 4];
                let mut lc = [0usize; 4];
                for t in 0..15u8 {
                    let tb = t as usize;
                    let mut score = 0.0;
                    let mut ok = false;
                    for l in 0..n_leaves {
                        lg[l] += hg[l][tb];
                        lh[l] += hh[l][tb];
                        lc[l] += hc[l][tb];
                        let rc = cur_c[l] - lc[l];
                        if lc[l] >= SUPPORT_FLOOR_K && rc >= SUPPORT_FLOOR_K {
                            ok = true;
                            let rg = cur_g[l] - lg[l];
                            let rh = cur_h[l] - lh[l];
                            score += lg[l] * lg[l] / (lh[l] + reg) + rg * rg / (rh + reg);
                        } else {
                            score += cur_g[l] * cur_g[l] / (cur_h[l] + reg);
                        }
                    }
                    let gain = score - base_score;
                    if ok && gain > best_gain {
                        best_gain = gain;
                        best_col = col;
                        best_thr = t;
                    }
                }
            }
            cols[depth] = best_col;
            thrs[depth] = best_thr;
            gains[depth] = best_gain.max(0.0);
            let cb = &bins16[best_col];
            for (i, &r) in train_rows.iter().enumerate() {
                if cb[r] > best_thr {
                    leaf_id[i] |= 1 << depth;
                }
            }
        }
        let mut g8 = [0.0f64; 8];
        let mut h8 = [0.0f64; 8];
        let mut c8 = [0usize; 8];
        for i in 0..n {
            let l = leaf_id[i];
            g8[l] += grad[i];
            h8[l] += hess[i];
            c8[l] += 1;
        }
        let mut leaves = [0.0f64; 8];
        for l in 0..8 {
            if c8[l] >= SUPPORT_FLOOR_K {
                leaves[l] = (lr * g8[l] / (h8[l] + reg)).clamp(-2.0, 2.0);
            }
        }
        for i in 0..n {
            pred[i] += leaves[leaf_id[i]];
        }
        trees.push(OblivTree3 {
            cols,
            thrs,
            leaves,
            leaf_counts: c8,
            gains,
        });
    }
    (base, trees)
}

fn eval_obliv_loss(
    bins16: &[Vec<u8>],
    y: &[f64],
    rows: &[usize],
    base: f64,
    trees: &[OblivTree3],
    is_binary: bool,
    perm_col: Option<(usize, usize)>,
) -> f64 {
    let n = rows.len();
    if n == 0 {
        return 0.0;
    }
    let mut loss = 0.0;
    for (ri, &r) in rows.iter().enumerate() {
        let mut s = base;
        for t in trees {
            let mut idx = 0usize;
            for d in 0..3 {
                let c = t.cols[d];
                let r_use = match perm_col {
                    Some((pc, shift)) if pc == c => rows[(ri + shift) % n],
                    _ => r,
                };
                if bins16[c][r_use] > t.thrs[d] {
                    idx |= 1 << d;
                }
            }
            s += t.leaves[idx];
        }
        if is_binary {
            let pr = 1.0 / (1.0 + (-s.clamp(-12.0, 12.0)).exp());
            let yr = if y[r] > 0.5 { 1.0 } else { 0.0 };
            loss -= yr * pr.max(1e-6).ln() + (1.0 - yr) * (1.0 - pr).max(1e-6).ln();
        } else {
            loss += (y[r] - s).powi(2);
        }
    }
    loss / n as f64
}

/// Extract Tier T2 cross-fitted GBDT witness, Walsh-Hadamard fANOVA, ECE15, and 1-NN noise floor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_witness_t2(
    bins16: &[Vec<u8>],
    z_tie: &[Vec<f64>],
    y: &[f64],
    task_id: u8,
    n_classes: usize,
    meta: Option<&[ColumnMeta]>,
    tgt: &TargetContext,
    out_col: &mut [[f32; COL_DIM]],
    out_col_mask: &mut [[u8; 16]],
    out_edge: &mut [[f32; EDGE_DIM]],
    out_edge_mask: &mut [u16],
    out_glob: &mut [f32; GLOB_DIM],
    out_glob_mask: &mut [u8; 20],
) {
    let p = bins16.len();
    let n = y.len();
    if p == 0 || n < 4 * SUPPORT_FLOOR_K {
        return;
    }
    let is_binary = task_id != 0;
    let k_eff = if task_id == 2 {
        n_classes.clamp(2, 8)
    } else {
        1
    };
    let n_trees = if task_id == 2 { 20 } else { 40 };

    let mut gain_imp = vec![0.0f64; p];
    let mut perm_imp = vec![0.0f64; p];
    let mut fanova_main = vec![0.0f64; p];
    let mut fanova_pair = vec![0.0f64; p * p];
    let mut fanova_tot_main = 0.0f64;
    let mut fanova_tot_pair = 0.0f64;
    let mut fanova_tot_high = 0.0f64;
    let mut gbdt_oof_loss = 0.0f64;
    let mut null_oof_loss = 0.0f64;

    let classes_to_fit = if task_id == 2 { k_eff.min(4) } else { 1 };
    for cls in 0..classes_to_fit {
        let y_use: Vec<f64> = if task_id == 2 {
            y.iter()
                .map(|&v| {
                    if (v.round() as isize) == cls as isize {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        } else if task_id == 1 {
            y.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect()
        } else {
            y.to_vec()
        };

        for fold in 0..2 {
            let tr: Vec<usize> = (0..n).filter(|&r| tgt.folds.of(r, 2) != fold).collect();
            let va: Vec<usize> = (0..n).filter(|&r| tgt.folds.of(r, 2) == fold).collect();
            let (base, trees) = fit_obliv_gbdt(bins16, &y_use, &tr, is_binary, n_trees, 0.20);
            let l0 = eval_obliv_loss(bins16, &y_use, &va, base, &[], is_binary, None);
            let l_base = eval_obliv_loss(bins16, &y_use, &va, base, &trees, is_binary, None);
            null_oof_loss += l0 / (2 * classes_to_fit) as f64;
            gbdt_oof_loss += l_base / (2 * classes_to_fit) as f64;

            let shift = (va.len() / 2).max(1);
            for j in 0..p {
                let lp = eval_obliv_loss(
                    bins16,
                    &y_use,
                    &va,
                    base,
                    &trees,
                    is_binary,
                    Some((j, shift)),
                );
                perm_imp[j] += (lp - l_base).max(0.0);
            }

            for t in &trees {
                for d in 0..3 {
                    gain_imp[t.cols[d]] += t.gains[d];
                }
                let tot_c: f64 = t.leaf_counts.iter().sum::<usize>().max(1) as f64;
                let w: [f64; 8] = std::array::from_fn(|l| t.leaf_counts[l] as f64 / tot_c);
                for s in 1usize..8 {
                    let mut c_s = 0.0;
                    for b in 0usize..8 {
                        let mut sign = 1.0;
                        for bit in 0..3 {
                            if (s & (1 << bit)) != 0 && (b & (1 << bit)) == 0 {
                                sign = -sign;
                            }
                        }
                        c_s += (w[b] * 8.0) * t.leaves[b] * sign;
                    }
                    c_s /= 8.0;
                    let e = c_s * c_s;
                    match s.count_ones() {
                        1 => {
                            let d = s.trailing_zeros() as usize;
                            fanova_main[t.cols[d]] += e;
                            fanova_tot_main += e;
                        }
                        2 => {
                            let mut bits = [0usize; 2];
                            let mut bi = 0;
                            for bit in 0..3 {
                                if (s & (1 << bit)) != 0 {
                                    bits[bi] = bit;
                                    bi += 1;
                                }
                            }
                            let c1 = t.cols[bits[0]];
                            let c2 = t.cols[bits[1]];
                            if c1 != c2 {
                                fanova_pair[c1 * p + c2] += e;
                                fanova_pair[c2 * p + c1] += e;
                                fanova_tot_pair += e;
                            } else {
                                fanova_main[c1] += e;
                                fanova_tot_main += e;
                            }
                        }
                        _ => fanova_tot_high += e,
                    }
                }
            }
        }
    }

    let sum_perm: f64 = perm_imp.iter().sum::<f64>().max(1e-12);
    let sum_gain: f64 = gain_imp.iter().sum::<f64>().max(1e-12);
    let sum_fanova = (fanova_tot_main + fanova_tot_pair + fanova_tot_high).max(1e-12);

    let mut tree_ord: Vec<usize> = (0..p).collect();
    tree_ord.sort_by(|&a, &b| perm_imp[b].total_cmp(&perm_imp[a]));
    let mut lin_ord: Vec<usize> = (0..p).collect();
    lin_ord.sort_by(|&a, &b| out_col[b][94].total_cmp(&out_col[a][94]));
    let mut rank_t = vec![0usize; p];
    let mut rank_l = vec![0usize; p];
    for (r, &c) in tree_ord.iter().enumerate() {
        rank_t[c] = r;
    }
    for (r, &c) in lin_ord.iter().enumerate() {
        rank_l[c] = r;
    }

    for j in 0..p {
        let p_share = perm_imp[j] / sum_perm;
        let g_share = gain_imp[j] / sum_gain;
        let fm_share = fanova_main[j] / sum_fanova;
        let fp_share: f64 = (0..p).map(|k| fanova_pair[j * p + k]).sum::<f64>() / sum_fanova;
        let grp_share = if let Some(ms) = meta {
            let gid = ms[j].group_id;
            (0..p)
                .filter(|&k| ms[k].group_id == gid)
                .map(|k| perm_imp[k] / sum_perm)
                .sum()
        } else {
            p_share
        };
        let rank_div = (rank_l[j] as f64 - rank_t[j] as f64) / p.max(1) as f64;
        out_col[j][116] = clamp_f32(p_share, 0.0, 1.0);
        out_col[j][117] = clamp_f32(g_share, 0.0, 1.0);
        out_col[j][118] = clamp_f32(grp_share, 0.0, 1.0);
        out_col[j][119] = clamp_f32(fm_share, 0.0, 1.0);
        out_col[j][120] = clamp_f32(fp_share, 0.0, 1.0);
        out_col[j][121] = clamp_f32(rank_div, -1.0, 1.0);
        out_col[j][122] = (rank_t[j] as f64 / p.max(1) as f64) as f32;
        out_col[j][123] = clamp_f32((p_share - out_col[j][94] as f64).abs(), 0.0, 1.0);
        for s in 116..124 {
            set_mask_bit(&mut out_col_mask[j], s);
        }
    }

    let mut max_pair_e = 0.0f64;
    for i in 0..p {
        for j in (i + 1)..p {
            let pe = (fanova_pair[i * p + j] / sum_fanova).clamp(0.0, 1.0);
            max_pair_e = max_pair_e.max(pe);
            out_edge[i * p + j][11] = pe as f32;
            out_edge[j * p + i][11] = pe as f32;
            out_edge_mask[i * p + j] |= 1 << 11;
            out_edge_mask[j * p + i] |= 1 << 11;
        }
    }

    let gbdt_gain = (1.0 - gbdt_oof_loss / null_oof_loss.max(1e-6)).clamp(-1.0, 1.0);
    let ece15 = compute_oof_ece15(y, &tgt.oof_pred, task_id);
    let nn1_noise = compute_1nn_noise_floor(z_tie, y, task_id);

    out_glob[104] = clamp_f32(gbdt_gain, -1.0, 1.0);
    out_glob[105] = clamp_f32(gbdt_gain - tgt.lin_gain, -1.0, 1.0);
    out_glob[106] = clamp_f32(fanova_tot_main / sum_fanova, 0.0, 1.0);
    out_glob[107] = clamp_f32(fanova_tot_pair / sum_fanova, 0.0, 1.0);
    out_glob[108] = clamp_f32(fanova_tot_high / sum_fanova, 0.0, 1.0);
    out_glob[109] = clamp_f32(max_pair_e, 0.0, 1.0);
    out_glob[110] = clamp_f32(ece15, 0.0, 1.0);
    out_glob[111] = clamp_f32(nn1_noise, 0.0, 1.0);
    for s in 104..112 {
        set_mask_bit(out_glob_mask, s);
    }
}

fn compute_oof_ece15(y: &[f64], oof_pred: &[Vec<f64>], task_id: u8) -> f64 {
    let n = y.len();
    if task_id == 0 || n < 15 {
        return 0.0;
    }
    let mut bin_p = [0.0f64; 15];
    let mut bin_acc = [0.0f64; 15];
    let mut bin_c = [0usize; 15];
    for r in 0..n {
        let (conf, hit) = if task_id == 1 {
            let p1 = oof_pred[0][r].clamp(0.0, 1.0);
            let pred = p1 >= 0.5;
            (if pred { p1 } else { 1.0 - p1 }, pred == (y[r] > 0.5))
        } else {
            let mut best_c = 0usize;
            let mut best_p = -1.0;
            for (c, col) in oof_pred.iter().enumerate() {
                if col[r] > best_p {
                    best_p = col[r];
                    best_c = c;
                }
            }
            (
                best_p.clamp(0.0, 1.0),
                best_c as isize == y[r].round() as isize,
            )
        };
        let b = ((conf * 15.0).floor() as usize).min(14);
        bin_p[b] += conf;
        bin_acc[b] += if hit { 1.0 } else { 0.0 };
        bin_c[b] += 1;
    }
    let mut ece = 0.0;
    for b in 0..15 {
        if bin_c[b] > 0 {
            ece += (bin_c[b] as f64 / n as f64) * ((bin_acc[b] - bin_p[b]).abs() / bin_c[b] as f64);
        }
    }
    ece.clamp(0.0, 1.0)
}

fn compute_1nn_noise_floor(z_tie: &[Vec<f64>], y: &[f64], task_id: u8) -> f64 {
    let p = z_tie.len();
    let n = y.len().min(192);
    if p == 0 || n < 12 {
        return 0.0;
    }
    let mut errs = 0.0;
    let var_y = if task_id == 0 {
        let ym = y[..n].iter().sum::<f64>() / n as f64;
        (y[..n].iter().map(|v| (v - ym).powi(2)).sum::<f64>() / n as f64).max(1e-6)
    } else {
        1.0
    };
    for i in 0..n {
        let mut best_j = if i == 0 { 1 } else { 0 };
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..p).map(|c| (z_tie[c][i] - z_tie[c][j]).powi(2)).sum();
            if d < best_d {
                best_d = d;
                best_j = j;
            }
        }
        if task_id == 0 {
            errs += 0.5 * (y[i] - y[best_j]).powi(2) / var_y;
        } else if (y[i].round() as isize) != (y[best_j].round() as isize) {
            errs += 1.0;
        }
    }
    (errs / n as f64).clamp(0.0, 1.0)
}
