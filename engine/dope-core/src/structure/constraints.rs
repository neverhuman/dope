//! Constraint rule miners (`BIN`, `INEQ`, `EXCL`, `MISS`) confirmed on `SELECT`.

use super::rules::StructuralRule;

pub(crate) fn try_bin(
    bx: &[Vec<f64>],
    sx: &[Vec<f64>],
    p: usize,
    c: usize,
    support_k: usize,
) -> Option<StructuralRule> {
    let mut pairs: Vec<(f64, f64)> = (0..bx[0].len())
        .filter_map(|r| {
            let (xp, xc) = (bx[p][r], bx[c][r]);
            (xp.is_finite() && xc.is_finite()).then_some((xp, xc))
        })
        .collect();
    if pairs.len() < support_k * 2 {
        return None;
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut edges = Vec::new();
    let mut bin_values = vec![pairs[0].1];
    let mut run_len = 1usize;
    for i in 1..pairs.len() {
        let cur = *bin_values.last().unwrap();
        if (pairs[i].1 - cur).abs() > 1e-6 {
            if run_len < support_k || bin_values.len() >= 16 {
                return None;
            }
            edges.push(0.5 * (pairs[i - 1].0 + pairs[i].0));
            bin_values.push(pairs[i].1);
            run_len = 1;
        } else {
            run_len += 1;
        }
    }
    if edges.is_empty() || run_len < support_k {
        return None;
    }
    let (mut ok_s, mut tot_s) = (0usize, 0usize);
    for r in 0..sx[0].len() {
        let (xp, xc) = (sx[p][r], sx[c][r]);
        if xp.is_finite() && xc.is_finite() {
            tot_s += 1;
            let b = edges.partition_point(|&e| e < xp).min(bin_values.len() - 1);
            if (bin_values[b] - xc).abs() <= 1e-4 {
                ok_s += 1;
            }
        }
    }
    if tot_s < 5 {
        return None;
    }
    let rs = ok_s as f64 / tot_s as f64;
    (rs >= 0.995).then_some(StructuralRule::Bin {
        parent: p,
        child: c,
        edges,
        bin_values,
        epsilon: (1.0 - rs).max(0.0),
    })
}

pub(crate) fn try_ineq(
    bx: &[Vec<f64>],
    sx: &[Vec<f64>],
    a: usize,
    b: usize,
) -> Option<StructuralRule> {
    let check = |xs: &[Vec<f64>]| -> (usize, usize, f64) {
        let (mut ok, mut tot, mut hi_cap) = (0, 0, f64::NEG_INFINITY);
        for r in 0..xs[0].len() {
            let (xa, xb) = (xs[a][r], xs[b][r]);
            if xa.is_finite() && xb.is_finite() {
                tot += 1;
                hi_cap = hi_cap.max(xb);
                if xa <= xb + 1e-9 {
                    ok += 1;
                }
            }
        }
        (ok, tot, hi_cap)
    };
    let (ok_b, tot_b, hi_cap) = check(bx);
    let (ok_s, tot_s, _) = check(sx);
    if tot_b < 20 || tot_s < 5 {
        return None;
    }
    let strict_lt = (0..bx[0].len())
        .filter(|&r| bx[a][r].is_finite() && bx[b][r].is_finite() && bx[a][r] + 1e-6 < bx[b][r])
        .count();
    if strict_lt < tot_b / 10 {
        return None;
    }
    let rb = ok_b as f64 / tot_b as f64;
    let rs = ok_s as f64 / tot_s as f64;
    (rb >= 0.995 && rs >= 0.995).then_some(StructuralRule::Ineq {
        lo_col: a,
        hi_col: b,
        hi_cap,
        epsilon: (1.0 - rs).max(0.0),
    })
}

pub(crate) fn try_excl(
    bx: &[Vec<f64>],
    sx: &[Vec<f64>],
    a: usize,
    b: usize,
    support_k: usize,
) -> Option<StructuralRule> {
    let (mut nz_a, mut nz_b, mut both, mut tot_b) = (0usize, 0usize, 0usize, 0usize);
    for r in 0..bx[0].len() {
        let (xa, xb) = (bx[a][r], bx[b][r]);
        if xa.is_finite() && xb.is_finite() {
            tot_b += 1;
            let fa = xa.abs() > 1e-9;
            let fb = xb.abs() > 1e-9;
            if fa {
                nz_a += 1;
            }
            if fb {
                nz_b += 1;
            }
            if fa && fb {
                both += 1;
            }
        }
    }
    if nz_a < support_k || nz_b < support_k || tot_b < 20 {
        return None;
    }
    let rb = 1.0 - both as f64 / tot_b as f64;
    let (mut both_s, mut tot_s) = (0usize, 0usize);
    for r in 0..sx[0].len() {
        let (xa, xb) = (sx[a][r], sx[b][r]);
        if xa.is_finite() && xb.is_finite() {
            tot_s += 1;
            if xa.abs() > 1e-9 && xb.abs() > 1e-9 {
                both_s += 1;
            }
        }
    }
    if tot_s < 5 {
        return None;
    }
    let rs = 1.0 - both_s as f64 / tot_s as f64;
    (rb >= 0.995 && rs >= 0.995).then_some(StructuralRule::Excl {
        col_a: a,
        col_b: b,
        prob_a_nonzero: nz_a as f64 / (nz_a + nz_b) as f64,
        epsilon: (1.0 - rs).max(0.0),
    })
}

pub(crate) fn try_miss(
    bx: &[Vec<f64>],
    sx: &[Vec<f64>],
    a: usize,
    b: usize,
    support_k: usize,
) -> Option<StructuralRule> {
    let (mut trig_b, mut hit_b) = (0usize, 0usize);
    for r in 0..bx[0].len() {
        if !bx[a][r].is_finite() {
            trig_b += 1;
            if !bx[b][r].is_finite() {
                hit_b += 1;
            }
        }
    }
    if trig_b < support_k {
        return None;
    }
    let rb = hit_b as f64 / trig_b as f64;
    let (mut trig_s, mut hit_s) = (0usize, 0usize);
    for r in 0..sx[0].len() {
        if !sx[a][r].is_finite() {
            trig_s += 1;
            if !sx[b][r].is_finite() {
                hit_s += 1;
            }
        }
    }
    if trig_s < 2 {
        return None;
    }
    let rs = hit_s as f64 / trig_s as f64;
    (rb >= 0.995 && rs >= 0.995).then_some(StructuralRule::Miss {
        parent: a,
        child: b,
        trigger_value: None,
        epsilon: (1.0 - rs).max(0.0),
    })
}
