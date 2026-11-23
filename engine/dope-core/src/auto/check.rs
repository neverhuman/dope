//! Internal selection check of `--format auto`, on the input rows only.
//!
//! Identical rows (features and target, as written) form groups that stay on one side of a
//! seeded 75/25 split. Both encoders are fitted on the 75% part with the final settings,
//! each draws twice that many rows, and two quick auditors are trained on each synthetic
//! set and on the real 75% part, then scored on the 25% part:
//!
//! ```text
//! R = clip((L0 − L_syn) / (L0 − L_real), −1, 1.25)     L0: training mean / class prior
//! score = min over auditors with L0 − L_real > 1e-9    (MSE or log-loss)
//! ```
//!
//! A candidate that cannot be fitted or sampled on the split scores −1 on every auditor.

use super::auditors::{fit_predict, null_predictions, Featurizer, AUDITORS};
use super::v7::{self, V7Decode, V7Model};
use crate::compile::{fit, FitOptions};
use crate::error::{DopeError, Result};
use crate::eval::{prediction_loss, Matrix};
use crate::program::Program;
use crate::rng::Rng;
use crate::sample::{sample, SampleOptions};
use crate::schema::{infer, IntakeOptions, Schema};
use crate::table::RawTable;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Fewer distinct rows than this: no check, Format 1 is kept.
pub const MIN_CHECK_GROUPS: usize = 80;
/// Rows (whole groups) the check may use, so large tables stay quick.
pub const MAX_CHECK_ROWS: usize = 20_000;
pub const CHECK_TEST_SHARE: f64 = 0.25;
pub const CHECK_SYNTH_FACTOR: usize = 2;
/// Auditor training rows (each set) and scored rows, thinned by a fixed stride.
const TRAIN_CAP: usize = 8192;
const TEST_CAP: usize = 4096;
const RETENTION_MIN: f64 = -1.0;
const RETENTION_MAX: f64 = 1.25;
const MIN_GAIN: f64 = 1e-9;
const SPLIT_SALT: u64 = 0xA070_5E1E_C7ED_0075;
const F1_SAMPLE_SALT: u64 = 0xA070_F1F1_5A3B_0001;
const V7_SAMPLE_SALT: u64 = 0xA070_D5E7_5A3B_0007;
const AUDIT_SALT: u64 = 0xA070_A0D1_7000_0002;

/// One auditor family's retention for each candidate (`None` when it is not usable).
#[derive(Clone, Debug, Serialize)]
pub struct AuditorRetention {
    pub auditor: String,
    pub usable: bool,
    pub format1: Option<f64>,
    pub v7: Option<f64>,
}

/// What the check did. Counts and retentions only: never cells, headers or extrema.
#[derive(Clone, Debug, Serialize)]
pub struct CheckReport {
    pub ran: bool,
    pub note: String,
    pub groups: usize,
    pub train_rows: usize,
    pub test_rows: usize,
    pub score_format1: Option<f64>,
    pub score_v7: Option<f64>,
    pub auditors: Vec<AuditorRetention>,
}

impl CheckReport {
    pub(crate) fn skipped(note: impl Into<String>, groups: usize) -> Self {
        Self {
            ran: false,
            note: note.into(),
            groups,
            train_rows: 0,
            test_rows: 0,
            score_format1: None,
            score_v7: None,
            auditors: Vec::new(),
        }
    }
}

/// Inputs shared with the full-table fits.
pub(crate) struct CheckInput<'a> {
    pub(crate) raw: &'a RawTable,
    pub(crate) target: usize,
    /// Intake schema of the whole table.
    pub(crate) schema: &'a Schema,
    /// Rows with an observed target.
    pub(crate) rows: &'a [usize],
    /// Format 1 settings, budget already capped, task fixed to the whole table's.
    pub(crate) format1: &'a FitOptions,
    pub(crate) room: usize,
    pub(crate) model: &'a V7Model,
}

/// Retention clipped to `[−1, 1.25]`; NaN counts as the worst value.
pub(crate) fn retention(l0: f64, l_real: f64, l_syn: f64) -> f64 {
    let r = (l0 - l_syn) / (l0 - l_real);
    if r.is_nan() {
        RETENTION_MIN
    } else {
        r.clamp(RETENTION_MIN, RETENTION_MAX)
    }
}

/// Exact-duplicate groups of `rows` over the columns `cols`, in a content-defined order.
pub(crate) fn duplicate_groups(raw: &RawTable, cols: &[usize], rows: &[usize]) -> Vec<Vec<usize>> {
    let mut map: BTreeMap<[u8; 16], Vec<usize>> = BTreeMap::new();
    for &r in rows {
        let mut h = Sha256::new();
        for &c in cols {
            let cell = raw.columns[c][r].as_bytes();
            h.update((cell.len() as u64).to_le_bytes());
            h.update(cell);
        }
        let mut key = [0u8; 16];
        key.copy_from_slice(&h.finalize()[..16]);
        map.entry(key).or_default().push(r);
    }
    map.into_values().collect()
}

/// Seeded grouped split: whole groups, at most [`MAX_CHECK_ROWS`] rows, a test part of at
/// most a quarter of them. Returns sorted `(train, test)` rows.
pub(crate) fn grouped_split(mut groups: Vec<Vec<usize>>, seed: u64) -> (Vec<usize>, Vec<usize>) {
    Rng::new(seed ^ SPLIT_SALT).shuffle(&mut groups);
    let mut kept = Vec::new();
    let mut total = 0;
    for g in groups {
        if total + g.len() <= MAX_CHECK_ROWS || total == 0 {
            total += g.len();
            kept.push(g);
        }
    }
    let test_target = (CHECK_TEST_SHARE * total as f64).round() as usize;
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for g in kept {
        if test.len() + g.len() <= test_target {
            test.extend(g);
        } else {
            train.extend(g);
        }
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

/// Format 1 samples decoded to cells with the intake schema the sub-fit used.
fn format1_rows(program: &Program, schema: &Schema, n: usize, seed: u64) -> Result<RawTable> {
    let d = schema.features.len();
    if program.features() != d {
        return Err(DopeError::Artifact("format1 check schema mismatch".into()));
    }
    let s = sample(
        program,
        &SampleOptions {
            rows: n,
            seed,
            ..Default::default()
        },
    )?;
    let mut columns: Vec<Vec<String>> = (0..d)
        .map(|j| {
            s.x[j]
                .iter()
                .map(|&v| schema.features[j].to_raw(v))
                .collect()
        })
        .collect();
    columns.push(s.y.iter().map(|&v| schema.target_to_raw(v)).collect());
    let mut headers: Vec<String> = schema.features.iter().map(|f| f.name.clone()).collect();
    headers.push(schema.target.name.clone());
    Ok(RawTable { headers, columns })
}

/// Run the check. Errors mean "no evidence" and keep Format 1.
pub(crate) fn run(input: &CheckInput<'_>, seed: u64) -> CheckReport {
    let mut cols: Vec<usize> = input
        .schema
        .features
        .iter()
        .filter_map(|f| input.raw.headers.iter().position(|h| *h == f.name))
        .collect();
    cols.push(input.target);
    let groups = duplicate_groups(input.raw, &cols, input.rows);
    let n_groups = groups.len();
    if n_groups < MIN_CHECK_GROUPS {
        let note = format!("skipped: fewer than {MIN_CHECK_GROUPS} distinct rows");
        return CheckReport::skipped(note, n_groups);
    }
    let (train_rows, test_rows) = grouped_split(groups, seed);
    match scored(input, &train_rows, &test_rows, seed) {
        Ok(mut report) => {
            report.groups = n_groups;
            report
        }
        Err(_) => CheckReport::skipped("skipped: the split could not be scored", n_groups),
    }
}

fn scored(
    input: &CheckInput<'_>,
    train: &[usize],
    test: &[usize],
    seed: u64,
) -> Result<CheckReport> {
    let (train, test) = (input.raw.select_rows(train), input.raw.select_rows(test));
    let intake = IntakeOptions {
        task: Some(input.schema.task),
        ..Default::default()
    };
    let schema = infer(&train, input.target, &intake)?;
    let real = schema.to_model(&train)?;
    let featurizer = Featurizer::fit(&schema, &real);
    let head = Program::head_for(&schema);
    let (xr, yr) = featurizer.matrix(&real, TRAIN_CAP);
    let (xt, yt) = featurizer.matrix(&schema.to_model(&test)?, TEST_CAP);
    if yr.len() < 20 || yt.len() < 10 {
        return Err(DopeError::Input("split too small".into()));
    }
    let n_syn = CHECK_SYNTH_FACTOR * train.rows();
    let as_matrix = |t: Result<RawTable>| -> Option<(Matrix, Vec<f64>)> {
        let data = schema.to_model(&t.ok()?).ok()?;
        let (x, y) = featurizer.matrix(&data, TRAIN_CAP);
        (!y.is_empty()).then_some((x, y))
    };
    let syn_f1 = as_matrix(super::guarded("format1", || {
        let fitted = fit(&train, input.target, input.format1)?;
        format1_rows(&fitted.program, &schema, n_syn, seed ^ F1_SAMPLE_SALT)
    }));
    let syn_v7 = as_matrix(super::guarded("v7", || {
        let f = v7::fit(
            &train,
            input.target,
            Some(input.schema.task),
            input.format1.anonymize,
            input.room,
        )?;
        f.decoder()?
            .table(V7Decode::Trained(input.model), n_syn, seed ^ V7_SAMPLE_SALT)
    }));
    let l0 = prediction_loss(head, &null_predictions(head, &yr, yt.len()), &yt);
    let mut auditors = Vec::new();
    for name in AUDITORS {
        let s = seed ^ AUDIT_SALT;
        let l_real = prediction_loss(head, &fit_predict(name, head, &xr, &yr, &xt, s), &yt);
        let usable = l_real.is_finite() && l0 - l_real > MIN_GAIN;
        let r = |syn: &Option<(Matrix, Vec<f64>)>| {
            usable.then(|| match syn {
                Some((xs, ys)) => {
                    let l_syn =
                        prediction_loss(head, &fit_predict(name, head, xs, ys, &xt, s), &yt);
                    retention(l0, l_real, l_syn)
                }
                None => RETENTION_MIN,
            })
        };
        auditors.push(AuditorRetention {
            auditor: name.to_string(),
            usable,
            format1: r(&syn_f1),
            v7: r(&syn_v7),
        });
    }
    let min = |pick: fn(&AuditorRetention) -> Option<f64>| {
        auditors.iter().filter_map(pick).reduce(f64::min)
    };
    let (score_format1, score_v7) = (min(|a| a.format1), min(|a| a.v7));
    let note = if score_format1.is_some() {
        "scored"
    } else {
        "scored: no auditor beat the training mean or prior"
    };
    Ok(CheckReport {
        ran: true,
        note: note.into(),
        groups: 0,
        train_rows: train.rows(),
        test_rows: test.rows(),
        score_format1,
        score_v7,
        auditors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_clips_and_treats_nan_as_worst() {
        assert_eq!(retention(1.0, 0.5, 0.75), 0.5);
        assert_eq!(retention(1.0, 0.5, 9.0), -1.0);
        assert_eq!(retention(1.0, 0.5, 0.0), 1.25);
        assert_eq!(retention(1.0, 1.0, 1.0), -1.0);
    }

    #[test]
    fn grouped_split_keeps_duplicates_together_and_is_seeded() {
        let mut cells = Vec::new();
        for r in 0..400 {
            cells.push(format!("{}", r % 120));
        }
        let raw = RawTable {
            headers: vec!["a".into(), "y".into()],
            columns: vec![cells.clone(), cells],
        };
        let rows: Vec<usize> = (0..400).collect();
        let groups = duplicate_groups(&raw, &[0, 1], &rows);
        assert_eq!(groups.len(), 120);
        let (train, test) = grouped_split(groups.clone(), 7);
        assert_eq!(train.len() + test.len(), 400);
        assert!(test.len() <= 100 && test.len() >= 90, "{}", test.len());
        let key = |r: usize| raw.columns[0][r].clone();
        let test_keys: std::collections::BTreeSet<String> = test.iter().map(|&r| key(r)).collect();
        assert!(train.iter().all(|&r| !test_keys.contains(&key(r))));
        assert_eq!(
            grouped_split(groups.clone(), 7),
            (train.clone(), test.clone())
        );
        assert_ne!(grouped_split(groups, 8).1, test);
    }
}
