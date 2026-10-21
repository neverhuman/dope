//! Linear auditor: ridge regression, ridge logistic (Newton, iteratively reweighted) for binary
//! targets; multiclass uses a softmax layer trained in `nn`.

use super::Matrix;
use crate::stats::solve_spd;
use crate::target::Head;

fn design(x: &Matrix) -> (Vec<f64>, usize) {
    let p = x.cols + 1;
    let mut a = vec![0.0; x.rows * p];
    for r in 0..x.rows {
        a[r * p] = 1.0;
        a[r * p + 1..(r + 1) * p].copy_from_slice(x.row(r));
    }
    (a, p)
}

fn penalize(g: &mut [f64], p: usize, rel: f64) {
    let avg = (1..p).map(|i| g[i * p + i]).sum::<f64>() / (p - 1).max(1) as f64;
    for i in 1..p {
        g[i * p + i] += rel * avg.max(1e-12) + 1e-9;
    }
}

/// Returns predictions (regression) or P(class 1) (binary) for `test`.
pub fn fit_predict(x: &Matrix, y: &[f64], head: Head, test: &Matrix) -> Vec<f64> {
    let (a, p) = design(x);
    let n = x.rows;
    let mut theta = vec![0.0; p];
    match head {
        Head::Regression => {
            let mut g = vec![0.0; p * p];
            let mut rhs = vec![0.0; p];
            for r in 0..n {
                let row = &a[r * p..(r + 1) * p];
                for i in 0..p {
                    rhs[i] += row[i] * y[r];
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
            penalize(&mut g, p, 1e-3);
            theta = solve_spd(&g, &rhs, p);
        }
        Head::Binary => {
            let prior = (y.iter().sum::<f64>() / n.max(1) as f64).clamp(1e-4, 1.0 - 1e-4);
            theta[0] = (prior / (1.0 - prior)).ln();
            for _ in 0..25 {
                let mut g = vec![0.0; p * p];
                let mut grad = vec![0.0; p];
                for r in 0..n {
                    let row = &a[r * p..(r + 1) * p];
                    let z: f64 = row.iter().zip(&theta).map(|(u, v)| u * v).sum();
                    let mu = 1.0 / (1.0 + (-z).exp());
                    let w = (mu * (1.0 - mu)).max(1e-6);
                    for i in 0..p {
                        grad[i] += (mu - y[r]) * row[i];
                        for j in i..p {
                            g[i * p + j] += w * row[i] * row[j];
                        }
                    }
                }
                for i in 0..p {
                    for j in 0..i {
                        g[i * p + j] = g[j * p + i];
                    }
                }
                let avg = (1..p).map(|i| g[i * p + i]).sum::<f64>() / (p - 1).max(1) as f64;
                for i in 1..p {
                    let lam = 1e-3 * avg.max(1e-12) + 1e-9;
                    g[i * p + i] += lam;
                    grad[i] += lam * theta[i];
                }
                let step = solve_spd(&g, &grad, p);
                let size: f64 = step.iter().map(|s| s * s).sum::<f64>().sqrt();
                for i in 0..p {
                    theta[i] -= step[i];
                }
                if size < 1e-8 {
                    break;
                }
            }
        }
        Head::Multiclass(_) => unreachable!("multiclass linear auditor is the softmax layer in nn"),
    }
    (0..test.rows)
        .map(|r| {
            let z = theta[0]
                + test
                    .row(r)
                    .iter()
                    .zip(&theta[1..])
                    .map(|(u, v)| u * v)
                    .sum::<f64>();
            match head {
                Head::Binary => 1.0 / (1.0 + (-z).exp()),
                _ => z,
            }
        })
        .collect()
}
