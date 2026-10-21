//! Gaussian dependence between features in normal-score space.
//!
//! Every family compiles to a *mixing matrix* `M` (d × m) so that a draw is a
//! single matrix product `z = ε Mᵀ` with ε ~ N(0, I_m) — one GEMM on any
//! device. Families, in increasing byte cost:
//!
//! * `Independent` (0 parameters),
//! * `Tree` — Chow–Liu maximum-likelihood tree, a linear Gaussian SEM
//!   `z = B z + D ε`, hence `M = (I − B)⁻¹ D` (≈ 2d bytes),
//! * `Factor(r)` — `Σ = ΛΛᵀ + Ψ` by principal-axis factoring (d·r bytes),
//! * `Full` — the whole latent correlation matrix (d(d−1)/2 bytes).

use crate::stats::{cholesky, nearest_correlation, sym_eigen};

#[derive(Clone, Debug, PartialEq)]
pub enum Dependence {
    Independent,
    /// `parent[i]` is `None` for roots; `rho[i]` is corr(z_i, z_parent).
    Tree {
        parent: Vec<Option<usize>>,
        rho: Vec<f64>,
    },
    /// Row-major d × r loadings.
    Factor {
        loadings: Vec<f64>,
        rank: usize,
    },
    /// Row-major d × d correlation.
    Full {
        corr: Vec<f64>,
    },
}

impl Dependence {
    pub fn name(&self) -> String {
        match self {
            Self::Independent => "independent".into(),
            Self::Tree { .. } => "tree".into(),
            Self::Factor { rank, .. } => format!("factor{rank}"),
            Self::Full { .. } => "full".into(),
        }
    }

    pub fn fit_tree(r: &[f64], d: usize) -> Self {
        // Prim's maximum spanning tree on Gaussian mutual information.
        let w = |i: usize, j: usize| -0.5 * (1.0 - r[i * d + j].powi(2)).max(1e-12).ln();
        let mut parent = vec![None; d];
        let mut rho = vec![0.0; d];
        if d == 0 {
            return Self::Tree { parent, rho };
        }
        let mut in_tree = vec![false; d];
        let mut best = vec![(f64::NEG_INFINITY, 0usize); d];
        in_tree[0] = true;
        for j in 1..d {
            best[j] = (w(0, j), 0);
        }
        for _ in 1..d {
            let next = (0..d)
                .filter(|&j| !in_tree[j])
                .max_by(|&a, &b| best[a].0.total_cmp(&best[b].0))
                .expect("a node remains");
            in_tree[next] = true;
            let p = best[next].1;
            if best[next].0 > 1e-6 {
                parent[next] = Some(p);
                rho[next] = r[p * d + next];
            }
            for j in 0..d {
                if !in_tree[j] && w(next, j) > best[j].0 {
                    best[j] = (w(next, j), next);
                }
            }
        }
        Self::Tree { parent, rho }
    }

    pub fn fit_factor(r: &[f64], d: usize, rank: usize) -> Self {
        let rank = rank.min(d.saturating_sub(1)).max(1);
        // Principal-axis factoring with communality iteration.
        let mut h: Vec<f64> = (0..d)
            .map(|i| {
                (0..d)
                    .filter(|&j| j != i)
                    .map(|j| r[i * d + j].abs())
                    .fold(0.0, f64::max)
            })
            .collect();
        let mut loadings = vec![0.0; d * rank];
        for _ in 0..25 {
            let mut m = r.to_vec();
            for i in 0..d {
                m[i * d + i] = h[i];
            }
            let (vals, vecs) = sym_eigen(&m, d);
            for k in 0..rank {
                let s = vals[k].max(0.0).sqrt();
                for i in 0..d {
                    loadings[i * rank + k] = vecs[i * d + k] * s;
                }
            }
            for i in 0..d {
                h[i] = (0..rank)
                    .map(|k| loadings[i * rank + k].powi(2))
                    .sum::<f64>()
                    .min(0.995);
            }
        }
        clamp_rows(&mut loadings, d, rank);
        Self::Factor { loadings, rank }
    }

    pub fn fit_full(r: &[f64], d: usize) -> Self {
        Self::Full {
            corr: nearest_correlation(r, d, 1e-3),
        }
    }

    /// Mixing matrix `M` (row-major d × m) and its width `m`.
    pub fn mixing(&self, d: usize) -> (Vec<f64>, usize) {
        match self {
            Self::Independent => (identity(d), d),
            Self::Tree { parent, rho } => {
                // Topological order: roots first, then children.
                let order = topo_order(parent);
                // z_i = rho_i z_p + sqrt(1 - rho_i^2) eps_i  =>  rows of M.
                let mut m = vec![0.0; d * d];
                for &i in &order {
                    let s = (1.0 - rho[i] * rho[i]).max(0.0).sqrt();
                    match parent[i] {
                        None => m[i * d + i] = 1.0,
                        Some(p) => {
                            for k in 0..d {
                                m[i * d + k] = rho[i] * m[p * d + k];
                            }
                            m[i * d + i] += s;
                        }
                    }
                }
                (m, d)
            }
            Self::Factor { loadings, rank } => {
                let w = d + rank;
                let mut m = vec![0.0; d * w];
                for i in 0..d {
                    let mut comm = 0.0;
                    for k in 0..*rank {
                        m[i * w + k] = loadings[i * rank + k];
                        comm += loadings[i * rank + k].powi(2);
                    }
                    m[i * w + rank + i] = (1.0 - comm).max(0.0).sqrt();
                }
                (m, w)
            }
            Self::Full { corr } => {
                let fixed = nearest_correlation(corr, d, 1e-3);
                let l = cholesky(&fixed, d).unwrap_or_else(|| identity(d));
                (l, d)
            }
        }
    }

    /// Mean Gaussian log-likelihood of standardized scores (row-major n × d,
    /// NaN treated as 0) under this model.
    pub fn log_likelihood(&self, scores: &[f64], n: usize, d: usize) -> f64 {
        let (m, w) = self.mixing(d);
        let mut sigma = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..=i {
                let s: f64 = (0..w).map(|k| m[i * w + k] * m[j * w + k]).sum();
                sigma[i * d + j] = s;
                sigma[j * d + i] = s;
            }
            sigma[i * d + i] += 1e-6;
        }
        let Some(l) = cholesky(&sigma, d) else {
            return f64::NEG_INFINITY;
        };
        let logdet: f64 = (0..d).map(|i| 2.0 * l[i * d + i].ln()).sum();
        let mut total = 0.0;
        let mut y = vec![0.0; d];
        for r in 0..n {
            let row = &scores[r * d..(r + 1) * d];
            let mut q = 0.0;
            for i in 0..d {
                let x = if row[i].is_finite() { row[i] } else { 0.0 };
                let mut s = x;
                for k in 0..i {
                    s -= l[i * d + k] * y[k];
                }
                y[i] = s / l[i * d + i];
                q += y[i] * y[i];
            }
            total += -0.5 * (q + logdet + d as f64 * (2.0 * std::f64::consts::PI).ln());
        }
        total / n.max(1) as f64
    }
}

pub fn clamp_rows(loadings: &mut [f64], d: usize, rank: usize) {
    for i in 0..d {
        let row = &mut loadings[i * rank..(i + 1) * rank];
        let s: f64 = row.iter().map(|v| v * v).sum();
        if s > 0.995 {
            let f = (0.995 / s).sqrt();
            row.iter_mut().for_each(|v| *v *= f);
        }
    }
}

fn identity(d: usize) -> Vec<f64> {
    let mut m = vec![0.0; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

pub fn topo_order(parent: &[Option<usize>]) -> Vec<usize> {
    let d = parent.len();
    let mut depth = vec![usize::MAX; d];
    fn depth_of(i: usize, parent: &[Option<usize>], depth: &mut [usize], guard: usize) -> usize {
        if depth[i] != usize::MAX {
            return depth[i];
        }
        let v = match parent[i] {
            None => 0,
            Some(_) if guard > parent.len() => 0, // cycle guard for corrupt input
            Some(p) => depth_of(p, parent, depth, guard + 1) + 1,
        };
        depth[i] = v;
        v
    }
    for i in 0..d {
        depth_of(i, parent, &mut depth, 0);
    }
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by_key(|&i| (depth[i], i));
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr3() -> Vec<f64> {
        vec![1.0, 0.8, 0.4, 0.8, 1.0, 0.5, 0.4, 0.5, 1.0]
    }

    fn implied(dep: &Dependence, d: usize) -> Vec<f64> {
        let (m, w) = dep.mixing(d);
        let mut s = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                s[i * d + j] = (0..w).map(|k| m[i * w + k] * m[j * w + k]).sum();
            }
        }
        s
    }

    #[test]
    fn full_reproduces_correlation() {
        let s = implied(&Dependence::fit_full(&corr3(), 3), 3);
        for (a, b) in s.iter().zip(corr3()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn tree_keeps_strong_edges_and_unit_variance() {
        let t = Dependence::fit_tree(&corr3(), 3);
        let s = implied(&t, 3);
        assert!((s[1] - 0.8).abs() < 1e-9);
        for i in 0..3 {
            assert!((s[i * 3 + i] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn factor_has_unit_diagonal_and_beats_independence() {
        let f = Dependence::fit_factor(&corr3(), 3, 1);
        let s = implied(&f, 3);
        for i in 0..3 {
            assert!((s[i * 3 + i] - 1.0).abs() < 1e-9);
        }
        assert!(s[1] > 0.5);
    }
}
