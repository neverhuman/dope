use super::{
    cost, folds,
    frame::{Encoded, Representation},
    job_seed, learners,
    report::contrasts,
    ridge, CellState, FeatureSpectrum, LandmarkCell, LandmarkFrame, LandmarkOptions, Signature,
    PANEL,
};
use crate::{
    stats::{chatterjee_xi, mean},
    target::Head,
    Result,
};
use rayon::prelude::*;

struct Fold {
    z: Encoded,
    q: Encoded,
    y: Vec<f64>,
    qy: Vec<f64>,
}

fn prepare(
    frame: &LandmarkFrame,
    assigned: &[usize],
    fold: usize,
    cap: usize,
    query_cap: usize,
    seed: u64,
    quarter: bool,
) -> Result<Fold> {
    let (mut train, query) = folds::fold_rows(frame, assigned, fold, cap, seed);
    if quarter {
        train.truncate((train.len() / 4).max(4));
    }
    let query = folds::capped(frame, &query, query_cap, job_seed(seed, 0, fold));
    let repr = Representation::fit(frame, &train);
    let (z, q) = (repr.encode(frame, &train)?, repr.encode(frame, &query)?);
    let y: Vec<f64> = train.iter().map(|&r| frame.y[r]).collect();
    let qy: Vec<f64> = query.iter().map(|&r| frame.y[r]).collect();
    let (y, qy) = standardize(frame.head, &y, &qy);
    Ok(Fold { z, q, y, qy })
}

pub(crate) fn standardize(head: Head, y: &[f64], q: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if head != Head::Regression {
        return (y.to_vec(), q.to_vec());
    }
    let m = mean(y);
    let s = (y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / y.len().max(1) as f64)
        .sqrt()
        .max(1e-12);
    (
        y.iter().map(|v| (v - m) / s).collect(),
        q.iter().map(|v| (v - m) / s).collect(),
    )
}

fn evaluate(id: usize, head: Head, folds: &[Fold], opts: &LandmarkOptions) -> LandmarkCell {
    let recipe = PANEL[id - 1];
    let mut cell = LandmarkCell {
        id: recipe.id,
        name: recipe.name,
        family: recipe.family,
        state: CellState::Complete,
        gain: None,
        se: None,
        stable: false,
        lambda_indices: Vec::new(),
        train_rows: Vec::new(),
        query_rows: 0,
    };
    let mut losses = 0.0;
    let mut nulls = 0.0;
    let mut gains = Vec::new();
    // Schedule both landmarks and folds. A tightly budgeted panel may contain
    // only linear, so landmark-level parallelism alone would leave cores idle.
    // Collect in fold order and reduce serially for bit-identical statistics.
    let measurements: Vec<_> = folds
        .par_iter()
        .enumerate()
        .map(|(k, f)| {
            let (p, lambda) =
                learners::run(id, &f.z, &f.y, head, &f.q, opts, job_seed(opts.seed, id, k))
                    .ok_or(CellState::Unsupported)?;
            let loss: f64 = ridge::losses(head, &p, &f.qy).iter().sum();
            let null: f64 = ridge::losses(head, &ridge::null(head, &f.y, f.qy.len()), &f.qy)
                .iter()
                .sum();
            if !loss.is_finite() || !null.is_finite() {
                return Err(CellState::Failed);
            }
            Ok((loss, null, lambda, f.y.len(), f.qy.len()))
        })
        .collect();
    for measurement in measurements {
        let (pred_loss, null_loss, lambda, train_rows, query_rows) = match measurement {
            Ok(v) => v,
            Err(state) => {
                cell.state = state;
                return cell;
            }
        };
        losses += pred_loss;
        nulls += null_loss;
        if null_loss > 1e-12 {
            gains.push(1.0 - pred_loss / null_loss);
        }
        cell.train_rows.push(train_rows);
        cell.query_rows += query_rows;
        if let Some(l) = lambda {
            cell.lambda_indices.push(l);
        }
    }
    if nulls <= 1e-12 || gains.len() < 2 {
        cell.state = CellState::Uninformative;
        return cell;
    }
    let gain = 1.0 - losses / nulls;
    let m = mean(&gains);
    let se = (gains.iter().map(|v| (v - m).powi(2)).sum::<f64>()
        / (gains.len() - 1) as f64
        / gains.len() as f64)
        .sqrt();
    if !gain.is_finite() || !se.is_finite() {
        cell.state = CellState::Failed;
        return cell;
    }
    cell.gain = Some(gain);
    cell.se = Some(se);
    cell.stable = se <= 0.02f64.max(0.25 * gain.abs());
    if gain <= 0.01 {
        cell.state = CellState::Uninformative;
    }
    cell
}

pub fn signature(frame: &LandmarkFrame, opts: &LandmarkOptions) -> Result<Signature> {
    opts.validate()?;
    frame.validate()?;
    let assigned = folds::assign_folds(frame, opts.folds, opts.seed)?;
    let (enabled, mut work) = cost::plan(frame, opts)?;
    let ridge_folds: Vec<Fold> = (0..opts.folds)
        .into_par_iter()
        .map(|k| {
            prepare(
                frame,
                &assigned,
                k,
                opts.ridge_rows,
                opts.ridge_rows,
                opts.seed,
                false,
            )
        })
        .collect::<Result<_>>()?;
    let small_folds: Vec<Fold> = (0..if enabled[15..21].iter().any(|v| *v) {
        opts.folds
    } else {
        0
    })
        .into_par_iter()
        .map(|k| {
            prepare(
                frame,
                &assigned,
                k,
                opts.small_rows,
                opts.small_rows.div_ceil(opts.folds),
                opts.seed,
                false,
            )
        })
        .collect::<Result<_>>()?;
    let cells: Vec<LandmarkCell> = PANEL
        .par_iter()
        .enumerate()
        .map(|(i, r)| {
            if !enabled[i] {
                return LandmarkCell {
                    id: r.id,
                    name: r.name,
                    family: r.family,
                    state: CellState::OverBudget,
                    gain: None,
                    se: None,
                    stable: false,
                    lambda_indices: Vec::new(),
                    train_rows: Vec::new(),
                    query_rows: 0,
                };
            }
            if i + 1 == 22 || i + 1 == 23 {
                let cap = if i + 1 == 22 {
                    opts.ridge_rows
                } else {
                    opts.small_rows
                };
                match (0..opts.folds)
                    .map(|k| {
                        prepare(
                            frame,
                            &assigned,
                            k,
                            cap,
                            if i + 1 == 22 {
                                opts.ridge_rows
                            } else {
                                opts.small_rows.div_ceil(opts.folds)
                            },
                            opts.seed,
                            true,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                {
                    Ok(f) => evaluate(i + 1, frame.head, &f, opts),
                    Err(_) => LandmarkCell {
                        id: r.id,
                        name: r.name,
                        family: r.family,
                        state: CellState::Failed,
                        gain: None,
                        se: None,
                        stable: false,
                        lambda_indices: Vec::new(),
                        train_rows: Vec::new(),
                        query_rows: 0,
                    },
                }
            } else {
                evaluate(
                    i + 1,
                    frame.head,
                    if i < 15 { &ridge_folds } else { &small_folds },
                    opts,
                )
            }
        })
        .collect();
    let target_stats = target_stats(frame);
    let contrasts = contrasts(&cells, target_stats);
    let mut spectrum = Vec::new();
    for slot in 0..frame.features().min(opts.spectrum_features) {
        // Charge every univariate fit; do not hide feature spectra outside the
        // work budget. Missing coordinates are explicit, not zero gain.
        let price =
            cost::cost(8, frame.rows(), 2, ridge::outputs(frame.head), opts).saturating_mul(8);
        if work.estimated_units.saturating_add(price) > opts.work_budget {
            break;
        }
        work.estimated_units += price;
        work.spectrum_run += 1;
        let ff: Vec<Fold> = ridge_folds
            .iter()
            .map(|f| Fold {
                z: super::basis::feature(&f.z, slot),
                q: super::basis::feature(&f.q, slot),
                y: f.y.clone(),
                qy: f.qy.clone(),
            })
            .collect();
        let mut values = [None; 9];
        for (j, id) in [1, 8, 7, 2, 5, 9, 6, 16].iter().enumerate() {
            if frame.categories[slot].is_some() && [2, 3, 4, 5, 6].contains(&j) {
                continue;
            }
            let c = evaluate(*id, frame.head, &ff, opts);
            if c.stable && c.state == CellState::Complete {
                values[if j == 7 { 8 } else { j }] = c.gain;
            }
        }
        // Xi is an in-role descriptive statistic, not an out-of-fold score.
        let pairs: Vec<(f64, f64)> = frame.columns[slot]
            .iter()
            .copied()
            .zip(frame.y.iter().copied())
            .filter(|(x, _)| x.is_finite())
            .collect();
        if pairs.len() > 2 {
            values[7] = Some(chatterjee_xi(
                &pairs.iter().map(|p| p.0).collect::<Vec<_>>(),
                &pairs.iter().map(|p| p.1).collect::<Vec<_>>(),
            ));
        }
        spectrum.push(FeatureSpectrum { slot, values });
    }
    let classes = match frame.head {
        Head::Regression => 0,
        Head::Binary => 2,
        Head::Multiclass(k) => k,
    };
    Ok(Signature {
        panel: opts.panel.clone(),
        kind: frame.kind,
        task: if classes == 0 {
            "regression"
        } else {
            "classification"
        },
        classes,
        cells,
        target_stats,
        contrasts,
        spectrum,
        work,
        options: opts.clone(),
        rows: frame.rows(),
        features: frame.features(),
    })
}

fn target_stats(f: &LandmarkFrame) -> [Option<f64>; 3] {
    if f.head == Head::Regression {
        let m = mean(&f.y);
        let v = f.y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / f.rows() as f64;
        if v <= 1e-24 {
            return [None, None, None];
        }
        [
            Some(f.y.iter().map(|v| (v - m).powi(3)).sum::<f64>() / f.rows() as f64 / v.powf(1.5)),
            Some(
                f.y.iter().map(|v| (v - m).powi(4)).sum::<f64>() / f.rows() as f64 / v.powi(2)
                    - 3.0,
            ),
            None,
        ]
    } else {
        let mut counts = vec![0.0; ridge::outputs(f.head)];
        for &y in &f.y {
            counts[y as usize] += 1.0 / f.rows() as f64;
        }
        [
            Some(
                -counts
                    .iter()
                    .filter(|v| **v > 0.0)
                    .map(|p| p * p.ln())
                    .sum::<f64>()
                    / (counts.len() as f64).ln(),
            ),
            counts.iter().copied().reduce(f64::min),
            None,
        ]
    }
}
