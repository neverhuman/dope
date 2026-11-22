//! Deterministic fixtures and small statistics shared by the unit tests.

use super::fit::{fit_features, FitInput, Labels};
use super::packet::{
    f64_to_f16, AtomTable, AtomValues, ColumnMarginal, Packet, PairTerm, Task, MAIN_BINS, MAX_RANK,
    PAIRS,
};
use super::scale::ScaleCard;
use dope_core::rng::{hash3, normal, uniform};

/// Standard normal fixture cell.
pub fn n01(seed: u64, col: usize, row: usize) -> f64 {
    normal(seed, 1000 + col as u64, row as u64)
}

/// Uniform(0, 1) fixture cell.
pub fn u01(seed: u64, col: usize, row: usize) -> f64 {
    uniform(seed, 5000 + col as u64, row as u64)
}

/// Row-major `n x p` matrix from a cell function `(row, col)`.
pub fn table(n: usize, p: usize, cell: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    (0..n * p).map(|i| cell(i / p, i % p)).collect()
}

/// Scale card and unit-space rows of a raw table.
pub fn to_unit(raw: &[f64], n: usize, p: usize) -> (ScaleCard, Vec<f64>) {
    let card = ScaleCard::fit(raw, n, p).unwrap();
    let unit = card.to_unit_rows(raw).unwrap();
    (card, unit)
}

/// Level-E regression packet of unit-space rows.
pub fn fit_regression(unit: &[f64], n: usize, p: usize) -> Packet {
    fit_features(&FitInput {
        x: unit,
        rows: n,
        cols: p,
        task: Task::Regression,
        k: 0,
        labels: Labels::None,
    })
    .unwrap()
}

pub fn column<T: Copy + Into<f64>>(x: &[T], p: usize, j: usize) -> Vec<f64> {
    x.iter().skip(j).step_by(p).map(|&v| v.into()).collect()
}

pub fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v
}

/// Type-7 quantile of an ascending slice.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let h = q * (sorted.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

/// Largest absolute difference between the pairwise correlations of two tables.
pub fn max_corr_gap(a: &[f64], b: &[f64], p: usize) -> f64 {
    let (ca, cb): (Vec<_>, Vec<_>) = (0..p).map(|j| (column(a, p, j), column(b, p, j))).unzip();
    let mut worst = 0.0f64;
    for j in 0..p {
        for k in j + 1..p {
            worst = worst.max((pearson(&ca[j], &ca[k]) - pearson(&cb[j], &cb[k])).abs());
        }
    }
    worst
}

/// Column kinds for [`synthetic_packet`].
#[derive(Clone, Copy)]
pub enum Kind {
    Continuous,
    Constant,
    /// Explicit atoms.
    Explicit(usize),
    /// Lattice atoms.
    Lattice(usize),
}

fn byte(seed: u64, stream: u64, i: usize) -> u8 {
    (hash3(seed, stream, i as u64) >> 24) as u8
}

/// A structurally valid level-E packet with pseudo-random codes in every field.
pub fn synthetic_packet(
    task: Task,
    k: u8,
    p: usize,
    kind: impl Fn(usize) -> Kind,
    seed: u64,
) -> Packet {
    let columns = (0..p)
        .map(|j| {
            let code = |i: usize| byte(seed, 10 + j as u64, i) % 64;
            match kind(j) {
                Kind::Constant => ColumnMarginal::Constant,
                Kind::Continuous => ColumnMarginal::Continuous {
                    lo: (hash3(seed, 10 + j as u64, 90) % 4096) as u16,
                    hi: (hash3(seed, 10 + j as u64, 91) % 4096) as u16,
                    split: std::array::from_fn(|i| byte(seed, 10 + j as u64, 92 + i)),
                    mid9: std::array::from_fn(|i| code(4 + i)),
                    mid17: std::array::from_fn(|i| code(8 + i)),
                },
                Kind::Explicit(a) => ColumnMarginal::Discrete(AtomTable {
                    values: AtomValues::Explicit(
                        (0..a).map(|i| f64_to_f16(-2.0 + 0.37 * i as f64)).collect(),
                    ),
                    prob: (0..a).map(|i| 1 + code(i) % 63).collect(),
                }),
                Kind::Lattice(a) => ColumnMarginal::Discrete(AtomTable {
                    values: AtomValues::Lattice {
                        base: f64_to_f16(-1.5),
                        step: f64_to_f16(0.25),
                        index: (0..a).map(|i| (3 * i) as u8).collect(),
                    },
                    prob: (0..a).map(|i| 1 + code(i) % 63).collect(),
                }),
            }
        })
        .collect();
    let mut out = Packet::skeleton(task, k, 11, columns);
    let o = out.outputs();
    let prior_len = out.prior.len();
    for (i, c) in out.prior.iter_mut().enumerate() {
        *c = if task == Task::Multilabel {
            byte(seed, 1, i)
        } else {
            byte(seed, 1, i) % (255 / prior_len.max(1)) as u8
        };
    }
    out.resid_sd = byte(seed, 2, 0);
    out.temperature = byte(seed, 2, 1);
    let fill = |v: &mut Vec<i8>, stream: u64| {
        v.iter_mut()
            .enumerate()
            .for_each(|(i, c)| *c = byte(seed, stream, i) as i8)
    };
    fill(&mut out.target_weight, 3);
    fill(&mut out.loading, 4);
    fill(&mut out.class_mean, 5);
    fill(&mut out.main_effect, 6);
    for (i, bits) in out.target_scale.iter_mut().enumerate() {
        *bits = f64_to_f16(0.01 + byte(seed, 7, i) as f64 / 64.0);
    }
    for (i, bits) in out.target_intercept.iter_mut().enumerate() {
        *bits = f64_to_f16(byte(seed, 8, i) as f64 / 32.0 - 4.0);
    }
    out.pairs = (0..PAIRS)
        .map(|i| PairTerm {
            a: (hash3(seed, 20, i as u64) % p as u64) as u16,
            b: (hash3(seed, 21, i as u64) % p as u64) as u16,
            coef: (0..o).map(|q| byte(seed, 22 + q as u64, i) as i8).collect(),
        })
        .collect();
    assert_eq!(out.loading.len(), p * MAX_RANK);
    assert_eq!(out.main_effect.len(), p * o * MAIN_BINS);
    out.validate().unwrap();
    out
}
