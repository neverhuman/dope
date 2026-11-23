//! `dope fit --format auto`: per table, the better of two encoders that both respect the
//! byte budget — Format 1 and the SingleEncV7 DSE7 v2 packet — in one `DPA1` container.
//!
//! Both candidates are fitted on all rows within `budget − 13` bytes. If exactly one fits,
//! it is used. If both fit and the table has at least [`MIN_CHECK_GROUPS`] distinct rows,
//! the internal check ([`CheckReport`]) scores both on a grouped split of the input rows,
//! and V7 is chosen only when its score beats Format 1's by more than [`SELECTION_MARGIN`];
//! in every other case Format 1 is kept. The stored artifact is the candidate fitted on all
//! rows. [`AutoOptions::encoder`] can instead ask for one encoder only (`--format v7`).
//! Fits, split, sampling and auditors are deterministic for a seed and do not depend on the
//! thread count.
//!
//! Candidate errors are caught and reported. Release builds abort on panic, so the
//! candidates are written not to panic; the unwind guard only covers unwinding builds.

mod auditors;
mod check;
mod container;
mod v7;
mod v7_schema;
pub mod v8;
pub mod v8_dpa;
mod v8_guard;

pub use check::{
    AuditorRetention, CheckReport, CHECK_SYNTH_FACTOR, CHECK_TEST_SHARE, MAX_CHECK_ROWS,
    MIN_CHECK_GROUPS,
};
pub use container::{
    fingerprint_hex, is_dpa1, parse_dpa1, wrap_dpa1, Dpa1, Encoder, DPA1_HEADER_BYTES, DPA1_MAGIC,
    NO_FINGERPRINT,
};
pub use v7::{V7Decode, V7Decoder, V7Model};
pub use v8_dpa::{
    fit_dpa1_v8, fit_dpa1_v8_with_v7, V8Decoder, V8DpaFitted, V8DpaMode, V8DpaReport, V8Model,
};

use crate::compile::{fit, FitOptions};
use crate::error::{DopeError, Result};
use crate::schema::{infer, IntakeOptions};
use crate::table::RawTable;
use check::CheckInput;
use serde::Serialize;
use std::time::Instant;

/// Whole-file cap of `--format auto` when neither `--budget` nor `--size` is given.
pub const DEFAULT_AUTO_BUDGET: usize = 2048;
/// V7 must beat Format 1's internal score by more than this to be chosen.
pub const SELECTION_MARGIN: f64 = 0.02;

#[derive(Clone, Debug)]
pub struct AutoOptions {
    /// Whole-file cap of the `DPA1` artifact, header included.
    pub budget: usize,
    /// Format 1 settings. `budget` is replaced by the room left after the header; `seed`,
    /// `task` and `anonymize` also apply to the V7 candidate and the internal check.
    pub format1: FitOptions,
    /// `None` picks per table; `Some` fits only that encoder (no internal check).
    pub encoder: Option<Encoder>,
}

/// Outcome of one candidate fitted on all rows. Errors never quote cells.
#[derive(Clone, Debug, Serialize)]
pub struct CandidateReport {
    /// False when [`AutoOptions::encoder`] asked for the other encoder only.
    pub requested: bool,
    pub fitted: bool,
    /// Payload bytes (the file adds the 13-byte header).
    pub bytes: Option<usize>,
    pub error: Option<String>,
}

/// Value-free summary of an auto fit.
#[derive(Clone, Debug, Serialize)]
pub struct AutoReport {
    /// `auto`, or the only encoder that was requested.
    pub mode: String,
    /// `format1` or `v7`.
    pub encoder: String,
    /// Whole file, header included.
    pub bytes: usize,
    pub budget: usize,
    pub task: String,
    /// Rows with an observed target.
    pub rows: usize,
    pub features: usize,
    pub format1: CandidateReport,
    pub v7: CandidateReport,
    pub check: CheckReport,
    pub margin: f64,
    /// Fingerprint of the SingleEncV7 weights the V7 candidate used.
    pub v7_weights: String,
    pub seconds: f64,
}

pub struct AutoFitted {
    /// The complete `DPA1` file.
    pub bytes: Vec<u8>,
    pub encoder: Encoder,
    pub report: AutoReport,
}

/// Run a candidate step, turning an unwinding panic into a value-free error.
pub(crate) fn guarded<T>(what: &str, step: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(step)).unwrap_or_else(|_| {
        Err(DopeError::Artifact(format!(
            "the {what} encoder stopped on an internal error"
        )))
    })
}

fn candidate(result: Option<&Result<Vec<u8>>>) -> CandidateReport {
    CandidateReport {
        requested: result.is_some(),
        fitted: matches!(result, Some(Ok(_))),
        bytes: result.and_then(|r| r.as_ref().ok()).map(Vec::len),
        error: result
            .and_then(|r| r.as_ref().err())
            .map(ToString::to_string),
    }
}

fn skipped(encoder: &str, other_requested: bool) -> CheckReport {
    let note = if other_requested {
        format!("not run: only {encoder} fits")
    } else {
        format!("not run: {encoder} requested")
    };
    CheckReport::skipped(note, 0)
}

/// Selection rule when both candidates fit: V7 only on evidence that it is better by more
/// than the margin; no scores (check skipped or uninformative) keeps Format 1.
pub fn prefer_v7(score_format1: Option<f64>, score_v7: Option<f64>) -> bool {
    matches!((score_format1, score_v7), (Some(f), Some(v)) if v > f + SELECTION_MARGIN)
}

fn both_failed(f1: &DopeError, v7: &DopeError) -> DopeError {
    let msg = format!("no encoder fits this table (format1: {f1}; v7: {v7})");
    match (f1, v7) {
        (DopeError::Budget(_), DopeError::Budget(_)) => DopeError::Budget(msg),
        (DopeError::Input(_), _) | (_, DopeError::Input(_)) => DopeError::Input(msg),
        _ => DopeError::Artifact(msg),
    }
}

/// Fit both candidates (or only `opts.encoder`) and keep the better one as a `DPA1` artifact
/// of at most `budget` bytes.
pub fn fit_auto(
    raw: &RawTable,
    target: usize,
    opts: &AutoOptions,
    model: &V7Model,
) -> Result<AutoFitted> {
    let start = Instant::now();
    let budget = opts.budget;
    let room = budget
        .checked_sub(DPA1_HEADER_BYTES)
        .filter(|&r| r > 0)
        .ok_or_else(|| {
            DopeError::Budget(format!(
                "{budget} bytes cannot hold the {DPA1_HEADER_BYTES}-byte DPA1 header and a payload"
            ))
        })?;
    let intake = IntakeOptions {
        task: opts.format1.task,
        ..Default::default()
    };
    let schema = infer(raw, target, &intake)?;
    let data = schema.to_model(raw)?;
    let rows: Vec<usize> = (0..data.rows())
        .filter(|&r| data.y[r].is_finite())
        .collect();
    let mut format1 = opts.format1.clone();
    format1.budget = Some(room);
    let wanted = |e: Encoder| opts.encoder.is_none_or(|only| only == e);
    let f1 = wanted(Encoder::Format1)
        .then(|| guarded("format1", || fit(raw, target, &format1).map(|f| f.bytes)));
    let (task, anonymize) = (opts.format1.task, opts.format1.anonymize);
    let v7 = wanted(Encoder::V7).then(|| {
        guarded("v7", || {
            v7::fit(raw, target, task, anonymize, room).map(|f| f.payload())
        })
    });
    let reports = (candidate(f1.as_ref()), candidate(v7.as_ref()));
    let (encoder, payload, check) = match (f1, v7) {
        (Some(Ok(p1)), Some(Ok(p7))) => {
            format1.task = Some(schema.task);
            let input = CheckInput {
                raw,
                target,
                schema: &schema,
                rows: &rows,
                format1: &format1,
                room,
                model,
            };
            let check = check::run(&input, opts.format1.seed);
            if prefer_v7(check.score_format1, check.score_v7) {
                (Encoder::V7, p7, check)
            } else {
                (Encoder::Format1, p1, check)
            }
        }
        (Some(Ok(p)), other) => (Encoder::Format1, p, skipped("Format 1", other.is_some())),
        (other, Some(Ok(p))) => (Encoder::V7, p, skipped("V7", other.is_some())),
        (Some(Err(a)), Some(Err(b))) => return Err(both_failed(&a, &b)),
        (Some(Err(e)), None) | (None, Some(Err(e))) => return Err(e),
        (None, None) => return Err(DopeError::Input("no encoder was requested".into())),
    };
    let fingerprint = match encoder {
        Encoder::V7 => model.fingerprint(),
        Encoder::Format1 | Encoder::V8 => NO_FINGERPRINT,
    };
    let bytes = wrap_dpa1(encoder, fingerprint, &payload);
    if bytes.len() > budget {
        return Err(DopeError::Budget(format!(
            "the {} artifact exceeds {budget} bytes",
            encoder.name()
        )));
    }
    let report = AutoReport {
        mode: opts.encoder.map_or("auto", Encoder::name).into(),
        encoder: encoder.name().into(),
        bytes: bytes.len(),
        budget,
        task: format!("{:?}", schema.task).to_lowercase(),
        rows: rows.len(),
        features: schema.features.len(),
        format1: reports.0,
        v7: reports.1,
        check,
        margin: SELECTION_MARGIN,
        v7_weights: fingerprint_hex(&model.fingerprint()),
        seconds: start.elapsed().as_secs_f64(),
    };
    Ok(AutoFitted {
        bytes,
        encoder,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v7_needs_a_margin_over_format1() {
        assert!(prefer_v7(Some(0.50), Some(0.53)));
        assert!(!prefer_v7(Some(0.50), Some(0.52)));
        assert!(!prefer_v7(Some(0.50), Some(0.40)));
        assert!(prefer_v7(Some(-1.0), Some(0.0)));
        assert!(!prefer_v7(None, None));
        assert!(!prefer_v7(None, Some(1.0)));
    }

    #[test]
    fn both_failed_keeps_the_most_actionable_kind() {
        let b = || DopeError::Budget("x".into());
        let i = || DopeError::Input("y".into());
        assert!(matches!(both_failed(&b(), &b()), DopeError::Budget(_)));
        assert!(matches!(both_failed(&b(), &i()), DopeError::Input(_)));
        let msg = both_failed(&b(), &i()).to_string();
        assert!(
            msg.contains("format1: budget: x") && msg.contains("v7: input: y"),
            "{msg}"
        );
    }
}
