//! Mined structural rules (`structure/`): direct post-transport enforcement and audit.

mod constraints;
pub mod rules;

pub use rules::{mine_structural_rules, RuleMode, StructuralRule};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuleSet {
    pub mode: RuleMode,
    pub rules: Vec<StructuralRule>,
}

impl RuleSet {
    pub fn mine(
        build_x: &[Vec<f64>],
        select_x: &[Vec<f64>],
        is_discrete: &[bool],
        mode: RuleMode,
        support_k: usize,
    ) -> Self {
        Self {
            mode,
            rules: mine_structural_rules(build_x, select_x, is_discrete, mode, support_k),
        }
    }

    /// Return the set of columns deterministically derived by high-support functional rules
    /// (`FD-1`, `FD-2`, `ARITH`, `BIN` with `epsilon <= 0.01`) so CATF skips wasting tree bytes on them.
    pub fn derived_columns(&self) -> std::collections::BTreeSet<usize> {
        let mut out = std::collections::BTreeSet::new();
        for r in &self.rules {
            match *r {
                StructuralRule::Fd1 { child, epsilon, .. }
                | StructuralRule::Fd2 { child, epsilon, .. }
                | StructuralRule::Arith { child, epsilon, .. }
                | StructuralRule::Bin { child, epsilon, .. }
                    if epsilon <= 0.01 =>
                {
                    out.insert(child);
                }
                _ => {}
            }
        }
        out
    }

    /// Direct stateless rule enforcement on a single feature row (`u_exc`, `u_renorm` in `(0, 1)`).
    /// Zero rejection loops: applies base constraints (`INEQ`/`EXCL`/`MISS`) first, then derived
    /// functional rules (`ARITH`/`BIN`/`FD-1`/`FD-2`) with independent per-rule hashed uniforms.
    pub fn apply_row(&self, x: &mut [f64], u_exc: f64, u_renorm: f64) {
        let mix_u = |base: f64, idx: usize, salt: u64| -> f64 {
            let mut z = base.to_bits() ^ ((idx as u64 + 1).wrapping_mul(salt));
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (((z ^ (z >> 31)) >> 11) as f64) / ((1u64 << 53) as f64)
        };
        // Pass 1: base domain constraints (INEQ, EXCL, MISS).
        for (ri, rule) in self.rules.iter().enumerate() {
            let ue = if ri == 0 {
                u_exc
            } else {
                mix_u(u_exc, ri, 0x9e37_79b9_7f4a_7c15)
            };
            let ur = if ri == 0 {
                u_renorm
            } else {
                mix_u(u_renorm, ri, 0x517c_c1b7_2722_0a95)
            };
            match rule {
                StructuralRule::Ineq {
                    lo_col,
                    hi_col,
                    hi_cap,
                    epsilon,
                } => {
                    if ue >= *epsilon
                        && x[*lo_col].is_finite()
                        && x[*hi_col].is_finite()
                        && x[*lo_col] > x[*hi_col]
                    {
                        let upper = hi_cap.max(x[*lo_col]);
                        x[*hi_col] = x[*lo_col] + ur.clamp(0.0, 1.0) * (upper - x[*lo_col]);
                    }
                }
                StructuralRule::Excl {
                    col_a,
                    col_b,
                    prob_a_nonzero,
                    epsilon,
                } => {
                    if ue >= *epsilon
                        && x[*col_a].is_finite()
                        && x[*col_b].is_finite()
                        && x[*col_a].abs() > 1e-9
                        && x[*col_b].abs() > 1e-9
                    {
                        if ur < *prob_a_nonzero {
                            x[*col_b] = 0.0;
                        } else {
                            x[*col_a] = 0.0;
                        }
                    }
                }
                StructuralRule::Miss {
                    parent,
                    child,
                    trigger_value,
                    epsilon,
                } => {
                    if ue >= *epsilon {
                        let trig = match trigger_value {
                            None => !x[*parent].is_finite(),
                            Some(tv) => x[*parent].is_finite() && (x[*parent] - tv).abs() <= 1e-9,
                        };
                        if trig {
                            x[*child] = f64::NAN;
                        }
                    }
                }
                _ => {}
            }
        }
        // Pass 2: derived functional dependencies (ARITH, BIN, FD-1, FD-2).
        for (ri, rule) in self.rules.iter().enumerate() {
            let ue = if ri == 0 {
                u_exc
            } else {
                mix_u(u_exc, ri, 0x9e37_79b9_7f4a_7c15)
            };
            match rule {
                StructuralRule::Fd1 {
                    parent,
                    child,
                    map,
                    epsilon,
                } => {
                    if ue >= *epsilon && x[*parent].is_finite() {
                        let k = (x[*parent] * 1e4).round() as i64;
                        if let Some(&v) = map.get(&k) {
                            x[*child] = v;
                        }
                    }
                }
                StructuralRule::Fd2 {
                    parent_a,
                    parent_b,
                    child,
                    map,
                    epsilon,
                } => {
                    if ue >= *epsilon && x[*parent_a].is_finite() && x[*parent_b].is_finite() {
                        let ka = (x[*parent_a] * 1e4).round() as i64;
                        let kb = (x[*parent_b] * 1e4).round() as i64;
                        if let Some(&v) = map.get(&(ka, kb)) {
                            x[*child] = v;
                        }
                    }
                }
                StructuralRule::Arith {
                    pa,
                    pb,
                    child,
                    a,
                    b,
                    c,
                    epsilon,
                    ..
                } => {
                    if ue >= *epsilon && x[*pa].is_finite() && x[*pb].is_finite() {
                        x[*child] = a * x[*pa] + b * x[*pb] + c;
                    }
                }
                StructuralRule::Bin {
                    parent,
                    child,
                    edges,
                    bin_values,
                    epsilon,
                } => {
                    if ue >= *epsilon && x[*parent].is_finite() {
                        let idx = edges
                            .partition_point(|&e| e < x[*parent])
                            .min(bin_values.len() - 1);
                        x[*child] = bin_values[idx];
                    }
                }
                _ => {}
            }
        }
    }

    /// Measure fraction of rows violating at least one mined rule.
    pub fn violation_rate(&self, x_cols: &[Vec<f64>]) -> f64 {
        let n = x_cols.first().map_or(0, Vec::len);
        if n == 0 || self.rules.is_empty() {
            return 0.0;
        }
        let mut viol = 0usize;
        for r in 0..n {
            let mut bad = false;
            for rule in &self.rules {
                match rule {
                    StructuralRule::Fd1 {
                        parent, child, map, ..
                    } => {
                        if x_cols[*parent][r].is_finite() && x_cols[*child][r].is_finite() {
                            let k = (x_cols[*parent][r] * 1e4).round() as i64;
                            if let Some(&v) = map.get(&k) {
                                if (x_cols[*child][r] - v).abs() > 1e-3 {
                                    bad = true;
                                }
                            }
                        }
                    }
                    StructuralRule::Arith {
                        pa,
                        pb,
                        child,
                        a,
                        b,
                        c,
                        tol,
                        ..
                    } => {
                        let (xa, xb, xc) = (x_cols[*pa][r], x_cols[*pb][r], x_cols[*child][r]);
                        if xa.is_finite()
                            && xb.is_finite()
                            && xc.is_finite()
                            && (a * xa + b * xb + c - xc).abs() > *tol * 1.5
                        {
                            bad = true;
                        }
                    }
                    StructuralRule::Ineq { lo_col, hi_col, .. } => {
                        let (xa, xb) = (x_cols[*lo_col][r], x_cols[*hi_col][r]);
                        if xa.is_finite() && xb.is_finite() && xa > xb + 1e-6 {
                            bad = true;
                        }
                    }
                    StructuralRule::Excl { col_a, col_b, .. } => {
                        let (xa, xb) = (x_cols[*col_a][r], x_cols[*col_b][r]);
                        if xa.is_finite() && xb.is_finite() && xa.abs() > 1e-9 && xb.abs() > 1e-9 {
                            bad = true;
                        }
                    }
                    StructuralRule::Miss {
                        parent,
                        child,
                        trigger_value: None,
                        ..
                    } => {
                        if !x_cols[*parent][r].is_finite() && x_cols[*child][r].is_finite() {
                            bad = true;
                        }
                    }
                    _ => {}
                }
                if bad {
                    break;
                }
            }
            if bad {
                viol += 1;
            }
        }
        viol as f64 / n as f64
    }
}
