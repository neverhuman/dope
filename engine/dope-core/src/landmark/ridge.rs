//! One Gram per design and one Cholesky per lambda, shared across outputs.
//! Lambda selection is nested inside the outer training fold. Outer query
//! labels never choose a ridge strength or a basis.
use super::basis::select;
use crate::{eval::Matrix, stats::cholesky, target::Head};
use rayon::prelude::*;

/// Gram tiles: TILE_I design rows by TILE_J columns of the upper triangle
/// advance together over a block of ROW_BLOCK rows held in L2. Every entry is
/// still one running sum over rows in ascending order (multiply, then add; no
/// fused multiply-add), so tiling and threading change no rounding.
const TILE_I: usize = 4;
const TILE_J: usize = 8;
const ROW_BLOCK: usize = 64;
/// Below this design width the bands run on the calling thread.
const PARALLEL_WIDTH: usize = 48;

pub(crate) fn targets(head: Head, y: &[f64]) -> Vec<f64> {
    match head {
        Head::Regression => y.to_vec(),
        Head::Binary => y.iter().flat_map(|&v| [1.0 - v, v]).collect(),
        Head::Multiclass(k) => y
            .iter()
            .flat_map(|&v| (0..k).map(move |j| (j == v as usize) as u8 as f64))
            .collect(),
    }
}
pub(crate) fn outputs(head: Head) -> usize {
    match head {
        Head::Regression => 1,
        Head::Binary => 2,
        Head::Multiclass(k) => k,
    }
}

struct Gram {
    g: Vec<f64>,
    b: Vec<f64>,
    p: usize,
    o: usize,
}
impl Gram {
    /// Gram of `[1, x_r]` and cross-products with targets `t` (`o` per row)
    /// over `rows`, which must be ascending.
    fn new(x: &Matrix, t: &[f64], o: usize, rows: &[usize]) -> Self {
        let p = x.cols + 1;
        let mut g = vec![0.0; p * p];
        let mut b = vec![0.0; p * o];
        let mut block = vec![1.0; ROW_BLOCK * p];
        // Fixed row order: no floating reduction depends on Rayon scheduling.
        for chunk in rows.chunks(ROW_BLOCK) {
            for (slot, &r) in chunk.iter().enumerate() {
                block[slot * p + 1..(slot + 1) * p].copy_from_slice(x.row(r));
            }
            let block = &block[..chunk.len() * p];
            let band = |(n, (gb, bb)): (usize, (&mut [f64], &mut [f64]))| {
                accumulate_band(gb, bb, block, chunk, t, n * TILE_I)
            };
            if p < PARALLEL_WIDTH {
                g.chunks_mut(TILE_I * p)
                    .zip(b.chunks_mut(TILE_I * o))
                    .enumerate()
                    .for_each(band);
            } else {
                g.par_chunks_mut(TILE_I * p)
                    .zip(b.par_chunks_mut(TILE_I * o))
                    .enumerate()
                    .for_each(band);
            }
        }
        for i in 0..p {
            for j in 0..i {
                g[i * p + j] = g[j * p + i];
            }
        }
        Self { g, b, p, o }
    }
    fn solve(&self, lambda: f64) -> Option<Vec<f64>> {
        let (p, o) = (self.p, self.o);
        let mut g = self.g.clone();
        let scale = (1..p).map(|i| g[i * p + i]).sum::<f64>() / (p - 1).max(1) as f64;
        for i in 1..p {
            g[i * p + i] += lambda * scale.max(1e-12) + 1e-9;
        }
        g[0] += 1e-12;
        let l = cholesky(&g, p)?;
        let mut w = vec![0.0; p * o];
        for k in 0..o {
            for i in 0..p {
                let mut v = self.b[i * o + k];
                for j in 0..i {
                    v -= l[i * p + j] * w[j * o + k];
                }
                w[i * o + k] = v / l[i * p + i];
            }
            for i in (0..p).rev() {
                let mut v = w[i * o + k];
                for j in i + 1..p {
                    v -= l[j * p + i] * w[j * o + k];
                }
                w[i * o + k] = v / l[i * p + i];
            }
        }
        w.iter().all(|v| v.is_finite()).then_some(w)
    }
}

/// Adds one block of design rows (`block`, row-major, leading 1) into the
/// Gram band starting at design row `i0` (`gb`, full rows) and its target
/// cross-products (`bb`). Only entries on or right of the band's diagonal
/// tile are meaningful; the mirror in `Gram::new` overwrites the rest.
fn accumulate_band(
    gb: &mut [f64],
    bb: &mut [f64],
    block: &[f64],
    chunk: &[usize],
    t: &[f64],
    i0: usize,
) {
    let p = block.len() / chunk.len();
    let h = gb.len() / p;
    let o = bb.len() / h;
    let mut j0 = i0;
    while j0 < p {
        let w = TILE_J.min(p - j0);
        if h == TILE_I && w == TILE_J {
            let mut acc = [[0.0; TILE_J]; TILE_I];
            for (ii, a) in acc.iter_mut().enumerate() {
                a.copy_from_slice(&gb[ii * p + j0..ii * p + j0 + TILE_J]);
            }
            for row in block.chunks_exact(p) {
                let xj = &row[j0..j0 + TILE_J];
                for (ii, a) in acc.iter_mut().enumerate() {
                    let xi = row[i0 + ii];
                    for (v, &xv) in a.iter_mut().zip(xj) {
                        *v += xi * xv;
                    }
                }
            }
            for (ii, a) in acc.iter().enumerate() {
                gb[ii * p + j0..ii * p + j0 + TILE_J].copy_from_slice(a);
            }
        } else {
            for ii in 0..h {
                for j in j0..j0 + w {
                    let mut v = gb[ii * p + j];
                    for row in block.chunks_exact(p) {
                        v += row[i0 + ii] * row[j];
                    }
                    gb[ii * p + j] = v;
                }
            }
        }
        j0 += TILE_J;
    }
    for ii in 0..h {
        for k in 0..o {
            let mut v = bb[ii * o + k];
            for (row, &r) in block.chunks_exact(p).zip(chunk) {
                v += row[i0 + ii] * t[r * o + k];
            }
            bb[ii * o + k] = v;
        }
    }
}

fn predict(x: &Matrix, w: &[f64], o: usize, head: Head) -> Vec<f64> {
    let mut out = vec![0.0; x.rows * o];
    for r in 0..x.rows {
        for k in 0..o {
            out[r * o + k] = w[k]
                + (0..x.cols)
                    .map(|c| x.at(r, c) * w[(c + 1) * o + k])
                    .sum::<f64>();
        }
    }
    probabilities(head, &mut out);
    out
}
pub(crate) fn probabilities(head: Head, p: &mut [f64]) {
    if head == Head::Regression {
        return;
    }
    for row in p.chunks_mut(outputs(head)) {
        for v in row.iter_mut() {
            *v = v.clamp(1e-4, 1.0);
        }
        let sum: f64 = row.iter().sum();
        for v in row {
            *v /= sum;
        }
    }
}
pub(crate) fn losses(head: Head, p: &[f64], y: &[f64]) -> Vec<f64> {
    let o = outputs(head);
    let t = targets(head, y);
    (0..y.len())
        .map(|r| (0..o).map(|k| (p[r * o + k] - t[r * o + k]).powi(2)).sum())
        .collect()
}
pub(crate) fn null(head: Head, y: &[f64], rows: usize) -> Vec<f64> {
    let o = outputs(head);
    let t = targets(head, y);
    let mut mean = vec![0.0; o];
    for row in t.chunks(o) {
        for k in 0..o {
            mean[k] += row[k] / y.len().max(1) as f64;
        }
    }
    (0..rows).flat_map(|_| mean.iter().copied()).collect()
}

pub(crate) fn fit_predict(
    x: &Matrix,
    y: &[f64],
    head: Head,
    query: &Matrix,
) -> Option<(Vec<f64>, u8)> {
    let all: Vec<usize> = (0..x.rows).collect();
    let train: Vec<usize> = (0..x.rows).filter(|r| r % 5 != 0).collect();
    let val: Vec<usize> = (0..x.rows).filter(|r| r % 5 == 0).collect();
    let xv = select(x, &val);
    let yv: Vec<f64> = val.iter().map(|&r| y[r]).collect();
    let (t, o) = (targets(head, y), outputs(head));
    let inner = Gram::new(x, &t, o, &train);
    let outer = Gram::new(x, &t, o, &all);
    let lambdas = [1e-4, 1e-2, 1.0];
    let mut best: Option<(f64, usize)> = None;
    for (i, &lambda) in lambdas.iter().enumerate() {
        if let Some(w) = inner.solve(lambda) {
            let p = predict(&xv, &w, inner.o, head);
            let loss: f64 = losses(head, &p, &yv).iter().sum();
            if best.is_none_or(|(b, _)| loss < b) {
                best = Some((loss, i));
            }
        }
    }
    let (_, i) = best?;
    let w = outer.solve(lambdas[i])?;
    Some((predict(query, &w, outer.o, head), i as u8))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_linear_and_singular_designs() {
        let x = Matrix {
            rows: 100,
            cols: 2,
            data: (0..100).flat_map(|i| [i as f64 / 100.0, 1.0]).collect(),
        };
        let y: Vec<f64> = (0..100).map(|i| 3.0 + 2.0 * i as f64 / 100.0).collect();
        let (p, index) = fit_predict(&x, &y, Head::Regression, &x).unwrap();
        let m = (0..100).map(|i| i as f64 / 100.0).sum::<f64>() / 100.0;
        let variance = (0..100)
            .map(|i| (i as f64 / 100.0 - m).powi(2))
            .sum::<f64>();
        let scale = ((0..100).map(|i| (i as f64 / 100.0).powi(2)).sum::<f64>() + 100.0) / 2.0;
        let lambda = [1e-4, 1e-2, 1.0][index as usize] * scale + 1e-9;
        let beta = 2.0 * variance / (variance + lambda);
        for (i, pred) in p.iter().enumerate() {
            let expected = 3.0 + 2.0 * m + beta * (i as f64 / 100.0 - m);
            assert!((pred - expected).abs() < 1e-9);
        }
    }
    /// The row-by-row sweep the tiled Gram replaced; kept as the bitwise oracle.
    fn reference(x: &Matrix, y: &[f64], head: Head) -> (Vec<f64>, Vec<f64>) {
        let (p, o) = (x.cols + 1, outputs(head));
        let t = targets(head, y);
        let mut g = vec![0.0; p * p];
        let mut b = vec![0.0; p * o];
        let mut row = vec![1.0; p];
        for r in 0..x.rows {
            row[1..].copy_from_slice(x.row(r));
            for i in 0..p {
                for k in 0..o {
                    b[i * o + k] += row[i] * t[r * o + k];
                }
                for j in i..p {
                    g[i * p + j] += row[i] * row[j];
                }
            }
        }
        for i in 0..p {
            for j in 0..i {
                g[i * p + j] = g[j * p + i];
            }
        }
        (g, b)
    }

    #[test]
    fn tiled_gram_is_bitwise_identical_to_row_sweep() {
        let mut rng = crate::rng::Rng::new(7);
        for &(rows, cols) in &[
            (0, 3),
            (1, 0),
            (5, 1),
            (63, 4),
            (64, 7),
            (65, 8),
            (130, 9),
            (200, 47),
            (321, 60),
        ] {
            let data = (0..rows * cols)
                .map(|i| match i % 11 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => rng.normal() * 1e12,
                    3 => rng.normal() * 1e-12,
                    _ => rng.normal(),
                })
                .collect();
            let x = Matrix { rows, cols, data };
            for head in [Head::Regression, Head::Binary, Head::Multiclass(3)] {
                let y: Vec<f64> = (0..rows)
                    .map(|r| match head {
                        Head::Regression => rng.normal(),
                        _ => (r % outputs(head)) as f64,
                    })
                    .collect();
                let (t, o) = (targets(head, &y), outputs(head));
                let train: Vec<usize> = (0..rows).filter(|r| r % 5 != 0).collect();
                let yt: Vec<f64> = train.iter().map(|&r| y[r]).collect();
                let cases = [
                    (
                        Gram::new(&x, &t, o, &(0..rows).collect::<Vec<_>>()),
                        reference(&x, &y, head),
                    ),
                    (
                        Gram::new(&x, &t, o, &train),
                        reference(&select(&x, &train), &yt, head),
                    ),
                ];
                for (gram, (g, b)) in cases {
                    let bits = |v: &[f64]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&gram.g), bits(&g), "gram {rows}x{cols}");
                    assert_eq!(bits(&gram.b), bits(&b), "targets {rows}x{cols}");
                }
            }
        }
    }
}
