//! One Gram per design and one Cholesky per lambda, shared across outputs.
//! Lambda selection is nested inside the outer training fold. Outer query
//! labels never choose a ridge strength or a basis.
use super::basis::select;
use crate::{eval::Matrix, stats::cholesky, target::Head};

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
    fn new(x: &Matrix, y: &[f64], head: Head) -> Self {
        let (p, o) = (x.cols + 1, outputs(head));
        let t = targets(head, y);
        let mut g = vec![0.0; p * p];
        let mut b = vec![0.0; p * o];
        let mut row = vec![1.0; p];
        // Fixed row order: no floating reduction depends on Rayon scheduling.
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
    let train: Vec<usize> = (0..x.rows).filter(|r| r % 5 != 0).collect();
    let val: Vec<usize> = (0..x.rows).filter(|r| r % 5 == 0).collect();
    let xt = select(x, &train);
    let xv = select(x, &val);
    let yt: Vec<f64> = train.iter().map(|&r| y[r]).collect();
    let yv: Vec<f64> = val.iter().map(|&r| y[r]).collect();
    let inner = Gram::new(&xt, &yt, head);
    let outer = Gram::new(x, y, head);
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
}
