use super::{
    basis::{self, Basis},
    frame::Encoded,
    ridge, LandmarkOptions,
};
use crate::{
    eval::{gbdt, Matrix},
    forest::{boost, predict_columns, BoostOptions, Split},
    rng::Rng,
    target::{log_sum_exp, Head},
    train::TrainOptions,
    train_cpu::train_cpu,
};

pub(crate) fn run(
    id: usize,
    z: &Encoded,
    y: &[f64],
    head: Head,
    q: &Encoded,
    opts: &LandmarkOptions,
    seed: u64,
) -> Option<(Vec<f64>, Option<u8>)> {
    if id <= 14 || id == 22 {
        let b = Basis::fit(if id == 22 { 1 } else { id }, z, y, seed)?;
        let (x, q) = (b.transform(&z.x), b.transform(&q.x));
        return ridge::fit_predict(&x, y, head, &q).map(|(p, l)| (p, Some(l)));
    }
    if id == 15 {
        let mut best: Option<(f64, Vec<f64>, u8)> = None;
        for j in z
            .groups
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
        {
            let (f, fq) = (basis::feature(z, j), basis::feature(q, j));
            let b = Basis::fit(8, &f, y, seed)?;
            let (x, xq) = (b.transform(&f.x), b.transform(&fq.x));
            // Pick a feature inside the outer fold, using an inner holdout.
            let tr: Vec<usize> = (0..x.rows).filter(|r| r % 5 != 0).collect();
            let va: Vec<usize> = (0..x.rows).filter(|r| r % 5 == 0).collect();
            let yt: Vec<f64> = tr.iter().map(|&r| y[r]).collect();
            let yv: Vec<f64> = va.iter().map(|&r| y[r]).collect();
            let (p, _) =
                ridge::fit_predict(&basis::select(&x, &tr), &yt, head, &basis::select(&x, &va))?;
            let loss: f64 = ridge::losses(head, &p, &yv).iter().sum();
            if best.as_ref().is_none_or(|(b, _, _)| loss < *b) {
                let (p, l) = ridge::fit_predict(&x, y, head, &xq)?;
                best = Some((loss, p, l));
            }
        }
        return best.map(|(_, p, l)| (p, Some(l)));
    }
    if id == 18 || id == 23 {
        let params = gbdt::Params {
            rounds: 64,
            max_depth: 4,
            learning_rate: 0.15,
            bins: 32,
            patience: 64,
            seed,
            ..Default::default()
        };
        return Some((
            expanded(head, gbdt::fit_predict(&z.x, y, head, &q.x, &params)),
            None,
        ));
    }
    if id == 16 || id == 17 {
        let codes = |x: &Matrix| -> Vec<u8> {
            (0..x.cols)
                .flat_map(|j| {
                    (0..x.rows).map(move |r| {
                        (((x.at(r, j) + 5.0) / 10.0).clamp(0.0, 1.0) * 255.0).round() as u8
                    })
                })
                .collect()
        };
        let fit_codes = codes(&z.x);
        let query_codes = codes(&q.x);
        let mean = ridge::null(head, y, 1);
        let base_row = match head {
            Head::Regression => mean,
            Head::Binary => vec![(mean[1].clamp(1e-6, 1.0 - 1e-6) / mean[0].clamp(1e-6, 1.0)).ln()],
            Head::Multiclass(_) => mean.iter().map(|v| v.max(1e-6).ln()).collect(),
        };
        let base: Vec<f64> = (0..z.x.rows)
            .flat_map(|_| base_row.iter().copied())
            .collect();
        let fit = Split {
            codes: &fit_codes,
            base: &base,
            y,
        };
        // Outer query labels are deliberately absent. The single tree is fit
        // against the training loss, never selected on its reported query.
        let forest = boost(
            head,
            z.x.cols,
            &fit,
            &fit,
            &BoostOptions {
                depth: if id == 16 { 1 } else { 3 },
                bits: 8,
                eta: 1.0,
                max_trees: 1,
                min_rows: 5,
                seed,
                ..Default::default()
            },
        )
        .forest;
        let mut p = predict_columns(&forest, &query_codes, q.x.cols, q.x.rows);
        for row in p.chunks_mut(head.outputs()) {
            for (v, b) in row.iter_mut().zip(&base_row) {
                *v += b;
            }
        }
        return Some((from_logits(head, p), None));
    }
    if id == 19 || id == 20 {
        return Some((
            knn(
                &z.x,
                y,
                head,
                &q.x,
                if id == 19 { 1 } else { 16 },
                opts.knn_reference,
                seed,
            ),
            None,
        ));
    }
    if id == 21 {
        let cols = z.x.cols.min(32);
        let trim = |x: &Matrix| -> Vec<f64> {
            (0..x.rows)
                .flat_map(|r| x.row(r)[..cols].iter().copied())
                .collect()
        };
        let (x, xq) = (trim(&z.x), trim(&q.x));
        let net = train_cpu(
            head,
            cols,
            (&x, y),
            (&[], &[]),
            &TrainOptions {
                hidden: 16,
                max_steps: 300,
                patience: 300,
                seed,
                ..Default::default()
            },
        );
        return Some((from_logits(head, net.forward(&xq, q.x.rows)), None));
    }
    None
}

pub(crate) fn expanded(head: Head, p: Vec<f64>) -> Vec<f64> {
    if head == Head::Binary {
        p.into_iter().flat_map(|v| [1.0 - v, v]).collect()
    } else {
        p
    }
}
pub(crate) fn from_logits(head: Head, mut p: Vec<f64>) -> Vec<f64> {
    match head {
        Head::Regression => p,
        Head::Binary => expanded(
            head,
            p.into_iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        ),
        Head::Multiclass(k) => {
            for row in p.chunks_mut(k) {
                let l = log_sum_exp(row);
                for v in row {
                    *v = (*v - l).exp();
                }
            }
            p
        }
    }
}

fn knn(x: &Matrix, y: &[f64], head: Head, q: &Matrix, k: usize, cap: usize, seed: u64) -> Vec<f64> {
    let dims = x.cols.min(32);
    let mut rng = Rng::new(seed);
    let weights: Vec<f64> = if x.cols > 32 {
        (0..x.cols * dims)
            .map(|_| rng.normal() / (dims as f64).sqrt())
            .collect()
    } else {
        Vec::new()
    };
    let project = |x: &Matrix| -> Matrix {
        if weights.is_empty() {
            return x.clone();
        }
        Matrix {
            rows: x.rows,
            cols: dims,
            data: (0..x.rows)
                .flat_map(|r| (0..dims).map(move |c| (r, c)))
                .map(|(r, c)| {
                    (0..x.cols)
                        .map(|j| x.at(r, j) * weights[j * dims + c])
                        .sum()
                })
                .collect(),
        }
    };
    let (x, q) = (project(x), project(q));
    let refs = x.rows.min(cap);
    let o = ridge::outputs(head);
    let t = ridge::targets(head, y);
    let mut out = vec![0.0; q.rows * o];
    let k = k.min(refs);
    for r in 0..q.rows {
        let mut distances: Vec<(f64, usize)> = (0..refs)
            .map(|i| {
                (
                    (0..dims).map(|c| (x.at(i, c) - q.at(r, c)).powi(2)).sum(),
                    i,
                )
            })
            .collect();
        distances.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        distances[..k].sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(_, i) in &distances[..k] {
            for c in 0..o {
                out[r * o + c] += t[i * o + c] / k as f64;
            }
        }
    }
    ridge::probabilities(head, &mut out);
    out
}
