//! Structural rule definitions and mining (`FD-1`, `FD-2`, `ARITH`, `BIN`, `INEQ`, `EXCL`, `MISS`).
//!
//! All rules are mined on `BUILD`, confirmed on `SELECT` (`>= 99.5%` support, `>= 99%` for
//! `ARITH` within tolerance, support `>= k = 10` per O-3), and store explicit exception rate
//! `epsilon` so direct sampling never relies on rejection loops.

use super::constraints::{try_bin, try_excl, try_ineq, try_miss};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleMode {
    None,
    #[default]
    Structural,
    All,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StructuralRule {
    /// Single-key functional dependency: `x[child] = map[round(x[parent]*1e4)]`.
    Fd1 {
        parent: usize,
        child: usize,
        map: BTreeMap<i64, f64>,
        epsilon: f64,
    },
    /// Two-key functional dependency: `x[child] = map[(p1, p2)]`.
    Fd2 {
        parent_a: usize,
        parent_b: usize,
        child: usize,
        map: BTreeMap<(i64, i64), f64>,
        epsilon: f64,
    },
    /// Linear arithmetic identity: `x[child] = a * x[pa] + b * x[pb] + c` (`pb == pa` for unary).
    Arith {
        pa: usize,
        pb: usize,
        child: usize,
        a: f64,
        b: f64,
        c: f64,
        tol: f64,
        epsilon: f64,
    },
    /// Monotone binning rule: `x[child] = bin_values[partition_point(edges, x[parent])]`.
    Bin {
        parent: usize,
        child: usize,
        edges: Vec<f64>,
        bin_values: Vec<f64>,
        epsilon: f64,
    },
    /// Inequality `x[lo_col] <= x[hi_col]` enforced via truncated upper projection.
    Ineq {
        lo_col: usize,
        hi_col: usize,
        hi_cap: f64,
        epsilon: f64,
    },
    /// Mutual exclusion: `!(x[a] != 0 && x[b] != 0)` enforced by joint zeroing of lower-priority col.
    Excl {
        col_a: usize,
        col_b: usize,
        prob_a_nonzero: f64,
        epsilon: f64,
    },
    /// Gated missingness: `x[child]` is NaN whenever `x[parent]` is NaN (or equals trigger).
    Miss {
        parent: usize,
        child: usize,
        /// `None` => missing-triggers-missing; `Some(v)` => `x[parent] == v` triggers missing.
        trigger_value: Option<f64>,
        epsilon: f64,
    },
}

fn qkey(v: f64) -> Option<i64> {
    v.is_finite().then_some((v * 1e4).round() as i64)
}

fn col_span(col: &[f64]) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in col {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if hi > lo {
        hi - lo
    } else {
        1.0
    }
}

/// Mine `FD-1`, `FD-2`, `ARITH`, `BIN`, `INEQ`, `EXCL`, `MISS` on `build_x` and confirm on `select_x`.
pub fn mine_structural_rules(
    build_x: &[Vec<f64>],
    select_x: &[Vec<f64>],
    is_discrete: &[bool],
    mode: RuleMode,
    support_k: usize,
) -> Vec<StructuralRule> {
    if mode == RuleMode::None {
        return Vec::new();
    }
    let d = build_x.len();
    let nb = build_x.first().map_or(0, Vec::len);
    let ns = select_x.first().map_or(0, Vec::len);
    if d < 2 || nb < support_k.max(20) || ns < 5 {
        return Vec::new();
    }
    let mut rules = Vec::new();
    let mut locked_child = vec![false; d];
    let mut is_parent = vec![false; d];
    let max_d = d.min(256);

    // 1. FD-1 (supports copied / deterministic discrete mappings, `T26`).
    for c in 1..max_d {
        if !is_discrete[c] {
            continue;
        }
        for p in 0..c {
            if !is_discrete[p] || locked_child[p] {
                continue;
            }
            if let Some(rule) = try_fd1(build_x, select_x, p, c, support_k) {
                rules.push(rule);
                locked_child[c] = true;
                is_parent[p] = true;
                break;
            }
        }
    }

    // 2. ARITH (`x[c] = a * x[pa] + b * x[pb] + c0`, confirmed >= 99% on SELECT).
    // Order candidate children by descending span (then descending column index) so sums
    // like `total = a + b` always derive `total` instead of rewriting a base input!
    let scan_d = d.min(48);
    let mut arith_order: Vec<usize> = (0..scan_d).collect();
    arith_order.sort_by(|&i, &j| {
        col_span(&build_x[j])
            .total_cmp(&col_span(&build_x[i]))
            .then(j.cmp(&i))
    });
    for c in arith_order {
        if locked_child[c] || is_parent[c] {
            continue;
        }
        let tol = (0.002 * col_span(&build_x[c])).max(1e-4);
        let mut found = None;
        'pairs: for pa in 0..scan_d {
            if pa == c || locked_child[pa] {
                continue;
            }
            for pb in pa..scan_d {
                if pb == c || locked_child[pb] {
                    continue;
                }
                if let Some(rule) = try_arith(build_x, select_x, pa, pb, c, tol) {
                    found = Some((pa, pb, rule));
                    break 'pairs;
                }
            }
        }
        if let Some((pa, pb, rule)) = found {
            rules.push(rule);
            locked_child[c] = true;
            is_parent[pa] = true;
            is_parent[pb] = true;
        }
    }

    // 3. FD-2 (pair discrete key -> discrete child, for remaining unlocked discrete cols).
    let fd2_d = d.min(24);
    for c in 2..fd2_d {
        if locked_child[c] || !is_discrete[c] {
            continue;
        }
        'fd2: for pa in 0..c {
            if !is_discrete[pa] {
                continue;
            }
            for pb in (pa + 1)..c {
                if !is_discrete[pb] {
                    continue;
                }
                if let Some(rule) = try_fd2(build_x, select_x, pa, pb, c, support_k) {
                    rules.push(rule);
                    locked_child[c] = true;
                    break 'fd2;
                }
            }
        }
    }

    // 4. BIN (continuous parent -> discrete monotone bin child).
    for c in 0..scan_d {
        if locked_child[c] || !is_discrete[c] {
            continue;
        }
        for p in 0..scan_d {
            if p == c || is_discrete[p] {
                continue;
            }
            if let Some(rule) = try_bin(build_x, select_x, p, c, support_k) {
                rules.push(rule);
                locked_child[c] = true;
                break;
            }
        }
    }

    // 5. INEQ (`x[a] <= x[b]` confirmed >= 99.5% on BUILD and SELECT, on unlocked base cols).
    for a in 0..scan_d {
        if is_discrete[a] || locked_child[a] {
            continue;
        }
        for b in 0..scan_d {
            if a == b || is_discrete[b] || locked_child[b] {
                continue;
            }
            if let Some(rule) = try_ineq(build_x, select_x, a, b) {
                rules.push(rule);
            }
        }
    }

    // 6. EXCL (`!(x[a] != 0 && x[b] != 0)` confirmed >= 99.5% on BUILD and SELECT).
    for a in 0..scan_d {
        for b in (a + 1)..scan_d {
            if let Some(rule) = try_excl(build_x, select_x, a, b, support_k) {
                rules.push(rule);
            }
        }
    }

    // 7. MISS (missing-triggers-missing confirmed >= 99.5%).
    for a in 0..scan_d {
        for b in 0..scan_d {
            if a == b {
                continue;
            }
            if let Some(rule) = try_miss(build_x, select_x, a, b, support_k) {
                rules.push(rule);
            }
        }
    }
    rules
}

fn try_fd1(
    bx: &[Vec<f64>],
    sx: &[Vec<f64>],
    p: usize,
    c: usize,
    support_k: usize,
) -> Option<StructuralRule> {
    let mut counts: BTreeMap<i64, BTreeMap<i64, usize>> = BTreeMap::new();
    for r in 0..bx[0].len() {
        if let (Some(kp), Some(kc)) = (qkey(bx[p][r]), qkey(bx[c][r])) {
            *counts.entry(kp).or_default().entry(kc).or_insert(0) += 1;
        }
    }
    if counts.is_empty() || counts.len() > 64 {
        return None;
    }
    let mut map = BTreeMap::new();
    let (mut ok_b, mut tot_b) = (0usize, 0usize);
    for (kp, ch) in counts {
        let tot: usize = ch.values().sum();
        if tot < support_k {
            continue;
        }
        let (&best_k, &best_n) = ch.iter().max_by_key(|&(_, &cnt)| cnt)?;
        if best_n as f64 / tot as f64 >= 0.995 {
            map.insert(kp, best_k as f64 / 1e4);
            ok_b += best_n;
            tot_b += tot;
        } else {
            return None;
        }
    }
    if map.is_empty() || tot_b < support_k {
        return None;
    }
    let (mut ok_s, mut tot_s) = (0usize, 0usize);
    for r in 0..sx[0].len() {
        if let (Some(kp), Some(kc)) = (qkey(sx[p][r]), qkey(sx[c][r])) {
            if let Some(&v) = map.get(&kp) {
                tot_s += 1;
                if (kc as f64 / 1e4 - v).abs() <= 1e-3 {
                    ok_s += 1;
                }
            }
        }
    }
    if tot_s == 0 {
        return None;
    }
    let rate_s = ok_s as f64 / tot_s as f64;
    let rate_b = ok_b as f64 / tot_b as f64;
    (rate_b >= 0.995 && rate_s >= 0.995).then_some(StructuralRule::Fd1 {
        parent: p,
        child: c,
        map,
        epsilon: (1.0 - rate_s).max(0.0),
    })
}

fn try_fd2(
    bx: &[Vec<f64>],
    sx: &[Vec<f64>],
    pa: usize,
    pb: usize,
    c: usize,
    support_k: usize,
) -> Option<StructuralRule> {
    let mut counts: BTreeMap<(i64, i64), BTreeMap<i64, usize>> = BTreeMap::new();
    for r in 0..bx[0].len() {
        if let (Some(ka), Some(kb), Some(kc)) = (qkey(bx[pa][r]), qkey(bx[pb][r]), qkey(bx[c][r])) {
            *counts.entry((ka, kb)).or_default().entry(kc).or_insert(0) += 1;
        }
    }
    if counts.len() < 2 || counts.len() > 128 {
        return None;
    }
    let mut map = BTreeMap::new();
    for (kpair, ch) in counts {
        let tot: usize = ch.values().sum();
        if tot < support_k {
            continue;
        }
        let (&best_k, &best_n) = ch.iter().max_by_key(|&(_, &cnt)| cnt)?;
        if best_n as f64 / tot as f64 >= 0.995 {
            map.insert(kpair, best_k as f64 / 1e4);
        } else {
            return None;
        }
    }
    if map.len() < 2 {
        return None;
    }
    let (mut ok_s, mut tot_s) = (0usize, 0usize);
    for r in 0..sx[0].len() {
        if let (Some(ka), Some(kb), Some(kc)) = (qkey(sx[pa][r]), qkey(sx[pb][r]), qkey(sx[c][r])) {
            if let Some(&v) = map.get(&(ka, kb)) {
                tot_s += 1;
                if (kc as f64 / 1e4 - v).abs() <= 1e-3 {
                    ok_s += 1;
                }
            }
        }
    }
    if tot_s == 0 {
        return None;
    }
    let rate_s = ok_s as f64 / tot_s as f64;
    (rate_s >= 0.995).then_some(StructuralRule::Fd2 {
        parent_a: pa,
        parent_b: pb,
        child: c,
        map,
        epsilon: (1.0 - rate_s).max(0.0),
    })
}

fn try_arith(
    bx: &[Vec<f64>],
    sx: &[Vec<f64>],
    pa: usize,
    pb: usize,
    c: usize,
    tol: f64,
) -> Option<StructuralRule> {
    // Fit exact 3x3 OLS [xa, xb, 1] -> xc (or 2x2 when pa == pb).
    let mut s = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    let mut n = 0usize;
    for r in 0..bx[0].len() {
        let (xa, xb, xc) = (bx[pa][r], bx[pb][r], bx[c][r]);
        if !xa.is_finite() || !xb.is_finite() || !xc.is_finite() {
            continue;
        }
        let v = [xa, if pa == pb { 0.0 } else { xb }, 1.0];
        for i in 0..3 {
            rhs[i] += v[i] * xc;
            for j in 0..3 {
                s[i][j] += v[i] * v[j];
            }
        }
        n += 1;
    }
    if n < 20 {
        return None;
    }
    if pa == pb {
        s[1][1] = 1.0;
    }
    let coef = solve_3x3(s, rhs)?;
    let (a, b, c0) = (coef[0], if pa == pb { 0.0 } else { coef[1] }, coef[2]);
    if a.abs() < 1e-4 && b.abs() < 1e-4 {
        return None;
    }
    let check = |xs: &[Vec<f64>]| -> (usize, usize) {
        let (mut ok, mut tot) = (0, 0);
        for r in 0..xs[0].len() {
            let (xa, xb, xc) = (xs[pa][r], xs[pb][r], xs[c][r]);
            if xa.is_finite() && xb.is_finite() && xc.is_finite() {
                tot += 1;
                if (a * xa + b * xb + c0 - xc).abs() <= tol {
                    ok += 1;
                }
            }
        }
        (ok, tot)
    };
    let (ok_b, tot_b) = check(bx);
    let (ok_s, tot_s) = check(sx);
    if tot_b < 20 || tot_s < 5 {
        return None;
    }
    let rb = ok_b as f64 / tot_b as f64;
    let rs = ok_s as f64 / tot_s as f64;
    (rb >= 0.99 && rs >= 0.99).then_some(StructuralRule::Arith {
        pa,
        pb,
        child: c,
        a,
        b,
        c: c0,
        tol,
        epsilon: (1.0 - rs).max(0.0),
    })
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let mut piv = i;
        for r in (i + 1)..3 {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        if a[piv][i].abs() < 1e-10 {
            return None;
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let d = a[i][i];
        for j in i..3 {
            a[i][j] /= d;
        }
        b[i] /= d;
        for r in 0..3 {
            if r != i {
                let f = a[r][i];
                for j in i..3 {
                    a[r][j] -= f * a[i][j];
                }
                b[r] -= f * b[i];
            }
        }
    }
    b.iter().all(|v| v.is_finite()).then_some(b)
}
