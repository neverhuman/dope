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

pub(super) struct Fold {
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
    signature_requested(frame, opts, None)
}

/// Additive: compute only the requested landmark ids (1..=23). `None` is the
/// full panel and is byte-identical to [`signature`]. Unrequested cells are
/// reported as [`CellState::NotRequested`], cost nothing and are never run.
/// Row subsampling is NOT an option here: apply it to the frame beforehand
/// (`LandmarkFrame::subsample`), keeping `LandmarkOptions` unchanged.
pub fn signature_requested(
    frame: &LandmarkFrame,
    opts: &LandmarkOptions,
    ids: Option<&[u8]>,
) -> Result<Signature> {
    opts.validate()?;
    frame.validate()?;
    let requested = requested_mask(ids)?;
    let full_panel = requested == [true; 23];
    let assigned = folds::assign_folds(frame, opts.folds, opts.seed)?;
    let (enabled, mut work) = cost::plan_requested(frame, opts, &requested)?;
    let ridge_folds: Vec<Fold> = if full_panel || enabled[..15].iter().any(|&v| v) {
        ridge_folds(frame, &assigned, opts)?
    } else {
        Vec::new()
    };
    let small_folds: Vec<Fold> = if needs_small_folds(&enabled) {
        small_folds(frame, &assigned, opts)?
    } else {
        Vec::new()
    };
    let cells: Vec<LandmarkCell> = PANEL
        .par_iter()
        .enumerate()
        .map(|(i, _)| {
            if !enabled[i] {
                return idle_cell(
                    i,
                    if requested[i] {
                        CellState::OverBudget
                    } else {
                        CellState::NotRequested
                    },
                );
            }
            run_cell(i, frame, &assigned, opts, &ridge_folds, &small_folds)
        })
        .collect();
    let spectrum = if full_panel {
        spectrum(frame, opts, &ridge_folds, &mut work)
    } else {
        Vec::new()
    };
    Ok(assemble(frame, opts, cells, spectrum, work))
}

pub(super) fn needs_small_folds(enabled: &[bool; 23]) -> bool {
    enabled[15..21].iter().any(|v| *v)
}

pub(super) fn ridge_folds(
    frame: &LandmarkFrame,
    assigned: &[usize],
    opts: &LandmarkOptions,
) -> Result<Vec<Fold>> {
    (0..opts.folds)
        .into_par_iter()
        .map(|k| {
            prepare(
                frame,
                assigned,
                k,
                opts.ridge_rows,
                opts.ridge_rows,
                opts.seed,
                false,
            )
        })
        .collect::<Result<_>>()
}

pub(super) fn small_folds(
    frame: &LandmarkFrame,
    assigned: &[usize],
    opts: &LandmarkOptions,
) -> Result<Vec<Fold>> {
    (0..opts.folds)
        .into_par_iter()
        .map(|k| {
            prepare(
                frame,
                assigned,
                k,
                opts.small_rows,
                opts.small_rows.div_ceil(opts.folds),
                opts.seed,
                false,
            )
        })
        .collect::<Result<_>>()
}

pub(super) fn idle_cell(i: usize, state: CellState) -> LandmarkCell {
    let r = PANEL[i];
    LandmarkCell {
        id: r.id,
        name: r.name,
        family: r.family,
        state,
        gain: None,
        se: None,
        stable: false,
        lambda_indices: Vec::new(),
        train_rows: Vec::new(),
        query_rows: 0,
    }
}

/// One enabled panel cell (0-based slot `i`). `small` must be prepared when
/// `i` is in 15..21.
pub(super) fn run_cell(
    i: usize,
    frame: &LandmarkFrame,
    assigned: &[usize],
    opts: &LandmarkOptions,
    ridge: &[Fold],
    small: &[Fold],
) -> LandmarkCell {
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
                    assigned,
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
            Err(_) => idle_cell(i, CellState::Failed),
        }
    } else {
        evaluate(i + 1, frame.head, if i < 15 { ridge } else { small }, opts)
    }
}

pub(super) fn spectrum(
    frame: &LandmarkFrame,
    opts: &LandmarkOptions,
    ridge_folds: &[Fold],
    work: &mut cost::WorkReport,
) -> Vec<FeatureSpectrum> {
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
    spectrum
}

pub(super) fn assemble(
    frame: &LandmarkFrame,
    opts: &LandmarkOptions,
    cells: Vec<LandmarkCell>,
    spectrum: Vec<FeatureSpectrum>,
    work: cost::WorkReport,
) -> Signature {
    let target_stats = target_stats(frame);
    let contrasts = contrasts(&cells, target_stats);
    let classes = match frame.head {
        Head::Regression => 0,
        Head::Binary => 2,
        Head::Multiclass(k) => k,
    };
    Signature {
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
    }
}

pub(super) fn requested_mask(ids: Option<&[u8]>) -> Result<[bool; 23]> {
    let Some(ids) = ids else {
        return Ok([true; 23]);
    };
    let mut requested = [false; 23];
    for &id in ids {
        if !(1..=23).contains(&id) || requested[id as usize - 1] {
            return Err(crate::DopeError::Input(
                "requested landmark ids must be unique and in 1..=23".into(),
            ));
        }
        requested[id as usize - 1] = true;
    }
    if ids.is_empty() {
        return Err(crate::DopeError::Input(
            "at least one landmark must be requested".into(),
        ));
    }
    Ok(requested)
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
