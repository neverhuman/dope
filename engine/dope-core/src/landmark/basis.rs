//! Basis choices are fitted on outer-training rows only. A fixed maximum
//! width bounds Gram memory and cubic solve work before any allocation.
use super::frame::Encoded;
use crate::{
    eval::Matrix,
    rng::Rng,
    stats::{chatterjee_xi, quantile_sorted},
};
use rayon::prelude::*;

#[derive(Clone)]
enum Term {
    Raw(usize),
    Shape(usize, u8),
    Hinge(usize, f64),
    Step(usize, f64),
    Periodic(usize, Vec<f64>, usize, bool),
    Pair(usize, usize),
    Bump(usize, f64),
    Summary(u8),
    Fourier(Vec<f64>, f64),
}
pub(crate) struct Basis {
    terms: Vec<Term>,
    numeric: Vec<usize>,
}

impl Basis {
    pub fn fit(id: usize, z: &Encoded, y: &[f64], seed: u64) -> Option<Self> {
        let mut numeric = z.numeric.clone();
        if numeric.len() > 64 || id == 12 {
            // One xi per column (pure, so caching keeps the comparator's order):
            // descending xi, ties by column index.
            let xi: Vec<(f64, usize)> = numeric
                .par_iter()
                .map(|&j| {
                    let col: Vec<f64> = (0..z.x.rows).map(|r| z.x.at(r, j)).collect();
                    (chatterjee_xi(&col, y), j)
                })
                .collect();
            let mut ranked = xi;
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            numeric = ranked.into_iter().map(|(_, j)| j).collect();
        }
        numeric.truncate(if id == 12 { 8 } else { 64 });
        let mut terms: Vec<Term> = (0..z.x.cols).map(Term::Raw).collect();
        if id == 14 {
            terms = z.missing.iter().copied().map(Term::Raw).collect();
        }
        if id == 13 {
            terms.clear();
            let mut rng = Rng::new(seed);
            let mut distances = Vec::new();
            for _ in 0..512.min(z.x.rows) {
                let (a, b) = (rng.below(z.x.rows), rng.below(z.x.rows));
                let d =
                    z.x.row(a)
                        .iter()
                        .zip(z.x.row(b))
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>();
                if d > 0.0 {
                    distances.push(d);
                }
            }
            distances.sort_by(f64::total_cmp);
            let bw = if distances.is_empty() {
                1.0
            } else {
                quantile_sorted(&distances, 0.5).sqrt().max(1e-6)
            };
            for _ in 0..256 {
                terms.push(Term::Fourier(
                    (0..z.x.cols).map(|_| rng.normal() / bw).collect(),
                    rng.uniform() * std::f64::consts::TAU,
                ));
            }
        } else if id == 11 {
            terms.extend((0..7).map(Term::Summary));
        } else if id == 12 {
            for a in 0..numeric.len() {
                for b in a + 1..numeric.len() {
                    terms.push(Term::Pair(numeric[a], numeric[b]));
                }
            }
        } else {
            for &j in &numeric {
                match id {
                    2..=6 => {
                        terms.push(Term::Shape(j, id as u8));
                        if id == 4 {
                            terms.push(Term::Shape(j, 24));
                        }
                        if id == 5 {
                            terms.push(Term::Shape(j, 25));
                        }
                    }
                    7 | 8 => {
                        let mut v: Vec<f64> = (0..z.x.rows).map(|r| z.x.at(r, j)).collect();
                        v.sort_by(f64::total_cmp);
                        let knots = if id == 7 { 7 } else { 15 };
                        for k in 1..=knots {
                            let q = quantile_sorted(&v, k as f64 / (knots + 1) as f64);
                            terms.push(if id == 7 {
                                Term::Hinge(j, q)
                            } else {
                                Term::Step(j, q)
                            });
                        }
                    }
                    9 => {
                        let mut v: Vec<f64> = (0..z.x.rows).map(|r| z.x.at(r, j)).collect();
                        v.sort_by(f64::total_cmp);
                        v.dedup();
                        if v.len() > 2 {
                            for k in 1..=3 {
                                for cos in [false, true] {
                                    terms.push(Term::Periodic(j, v.clone(), k, cos));
                                }
                            }
                        }
                    }
                    10 => terms.extend([-1.0, 0.0, 1.0].map(|c| Term::Bump(j, c))),
                    _ => (),
                }
            }
        }
        if terms.is_empty() || terms.len() > 511 {
            return None;
        }
        Some(Self { terms, numeric })
    }

    pub fn transform(&self, x: &Matrix) -> Matrix {
        let cols = self.terms.len();
        let mut data = Vec::with_capacity(x.rows * cols);
        for r in 0..x.rows {
            let row = x.row(r);
            for term in &self.terms {
                data.push(match term {
                    Term::Raw(j) => row[*j],
                    Term::Shape(j, id) => {
                        let v = row[*j];
                        match id {
                            2 => v.asinh(),
                            3 => v.signum() * v.abs().sqrt(),
                            4 => v * v,
                            24 => v * v * v / 3.0,
                            5 => v.tanh(),
                            25 => (v / 3.0).tanh(),
                            6 => v.abs(),
                            _ => unreachable!(),
                        }
                    }
                    Term::Hinge(j, q) => (row[*j] - q).max(0.0),
                    Term::Step(j, q) => (row[*j] > *q) as u8 as f64,
                    Term::Periodic(j, v, k, cos) => {
                        let u =
                            2.0 * v.partition_point(|v| *v < row[*j]) as f64 / v.len() as f64 - 1.0;
                        let t = *k as f64 * std::f64::consts::PI * u;
                        if *cos {
                            t.cos()
                        } else {
                            t.sin()
                        }
                    }
                    Term::Pair(a, b) => row[*a] * row[*b],
                    Term::Bump(j, c) => (-0.5 * (row[*j] - c).powi(2)).exp(),
                    Term::Summary(which) => {
                        let v: Vec<f64> = self.numeric.iter().map(|&j| row[j]).collect();
                        let n = v.len().max(1) as f64;
                        let sum: f64 = v.iter().sum();
                        let m = sum / n;
                        match which {
                            0 => m,
                            1 => sum,
                            2 => v.iter().map(|v| v.abs()).sum(),
                            3 => v.iter().map(|v| v * v).sum::<f64>().sqrt(),
                            4 => (v.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt(),
                            5 => v.iter().copied().reduce(f64::min).unwrap_or(0.0),
                            _ => v.iter().copied().reduce(f64::max).unwrap_or(0.0),
                        }
                    }
                    Term::Fourier(w, b) => {
                        (row.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + b).cos()
                            * (2.0 / 256.0f64).sqrt()
                    }
                });
            }
        }
        Matrix {
            rows: x.rows,
            cols,
            data,
        }
    }
}

pub(crate) fn select(x: &Matrix, rows: &[usize]) -> Matrix {
    Matrix {
        rows: rows.len(),
        cols: x.cols,
        data: rows
            .iter()
            .flat_map(|&r| x.row(r).iter().copied())
            .collect(),
    }
}

pub(crate) fn feature(z: &Encoded, j: usize) -> Encoded {
    let indices: Vec<usize> = (0..z.x.cols).filter(|&c| z.groups[c] == j).collect();
    Encoded {
        numeric: indices
            .iter()
            .enumerate()
            .filter_map(|(i, c)| z.numeric.contains(c).then_some(i))
            .collect(),
        missing: indices
            .iter()
            .enumerate()
            .filter_map(|(i, c)| z.missing.contains(c).then_some(i))
            .collect(),
        groups: vec![j; indices.len()],
        x: Matrix {
            rows: z.x.rows,
            cols: indices.len(),
            data: (0..z.x.rows)
                .flat_map(|r| indices.iter().map(move |&c| z.x.at(r, c)))
                .collect(),
        },
    }
}
