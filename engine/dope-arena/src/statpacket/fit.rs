//! Fit: [`fit_packet`] runs scale card -> marginals -> dependence -> target ->
//! budget -> pack on raw FIT rows. [`fit_features`] is the feature part alone
//! (class priors, marginals and dependence from unit-space rows; the target
//! sections stay zero).

use super::codec::{fit_to_budget, pack, packed_len, BudgetFit};
use super::dependence::fit_dependence;
use super::marginals::{fit_column, knot_plan, UNIT_CLIP};
use super::packet::{Level, Packet, Task, LOG2_N_MAX, SUPPORT_K};
use super::scale::ScaleCard;
use super::target::{fit_target, refresh_noise, FitRows, Response, TargetPlan};
use super::StatError;

/// Fewest FIT rows a packet can be built from.
pub const MIN_FIT_ROWS: usize = 16;
/// Widest table the factor fit is run for; wider tables keep zero loadings
/// (their dependence sections alone would exceed 4 KiB).
pub const MAX_DEPENDENCE_COLS: usize = 1024;

/// Labels of the FIT rows; they must match the task.
#[derive(Clone, Copy, Debug)]
pub enum Labels<'a> {
    /// Regression.
    None,
    /// Binary or multiclass: one class id in `0..K` per row.
    Classes(&'a [u16]),
    /// Multilabel: row-major `rows x K` cells, each 0 or 1.
    Multilabel(&'a [u8]),
}

#[derive(Clone, Copy, Debug)]
pub struct FitInput<'a> {
    /// Unit-space features, row-major `rows x cols`, every cell finite in [-6, 6].
    pub x: &'a [f64],
    pub rows: usize,
    pub cols: usize,
    pub task: Task,
    /// Header K: 0 regression, 2 binary, classes, or labels.
    pub k: u8,
    pub labels: Labels<'a>,
}

/// `K` u8 codes that sum to 255, by largest remainder; every class that has
/// rows gets at least 1 (ties go to the lower class index).
pub fn class_prior_codes(counts: &[usize]) -> Vec<u8> {
    let n: usize = counts.iter().sum();
    let exact: Vec<f64> = counts
        .iter()
        .map(|&c| 255.0 * c as f64 / n.max(1) as f64)
        .collect();
    let mut codes: Vec<i32> = exact
        .iter()
        .zip(counts)
        .map(|(&e, &c)| if c > 0 { (e.floor() as i32).max(1) } else { 0 })
        .collect();
    loop {
        let sum: i32 = codes.iter().sum();
        if sum == 255 || n == 0 {
            break;
        }
        let mut pick: Option<usize> = None;
        for c in 0..codes.len() {
            let eligible = if sum < 255 {
                counts[c] > 0
            } else {
                codes[c] > 1
            };
            let gap = |i: usize| (exact[i] - codes[i] as f64) * if sum < 255 { 1.0 } else { -1.0 };
            if eligible && pick.is_none_or(|best| gap(c) > gap(best)) {
                pick = Some(c);
            }
        }
        match pick {
            Some(c) => codes[c] += if sum < 255 { 1 } else { -1 },
            None => break,
        }
    }
    codes.into_iter().map(|c| c as u8).collect()
}

fn check(input: &FitInput) -> Result<(), StatError> {
    let bad = |why: &'static str| Err(StatError::Invalid(why));
    if input.rows < MIN_FIT_ROWS {
        return bad("fewer than 16 FIT rows");
    }
    if input.cols == 0 || input.cols > u16::MAX as usize {
        return bad("column count outside 1..=65535");
    }
    if input.x.len() != input.rows * input.cols {
        return bad("feature matrix shape does not match its dimensions");
    }
    if input
        .x
        .iter()
        .any(|v| !v.is_finite() || v.abs() > UNIT_CLIP)
    {
        return bad("features are not finite unit-space values");
    }
    if !input.task.accepts_k(input.k) {
        return bad("class count is not valid for the task");
    }
    let k = input.k as usize;
    let labels_ok = match (input.task, input.labels) {
        (Task::Regression, Labels::None) => true,
        (Task::Binary | Task::Multiclass, Labels::Classes(ids)) => {
            ids.len() == input.rows && ids.iter().all(|&c| (c as usize) < k)
        }
        (Task::Multilabel, Labels::Multilabel(cells)) => {
            cells.len() == input.rows * k && cells.iter().all(|&c| c <= 1)
        }
        _ => false,
    };
    if !labels_ok {
        return bad("labels do not match the task");
    }
    Ok(())
}

/// Fits a level-E packet whose feature sections are complete and whose target
/// sections are zero.
pub fn fit_features(input: &FitInput) -> Result<Packet, StatError> {
    fit(input, None)
}

/// As [`fit_features`], for a caller that will pack at most `max_bytes`: when
/// that budget cannot hold any dependence section, the factor fit (cubic in
/// the column count) is skipped. Section lengths do not depend on the factor
/// values, so `fit_to_budget` returns the same bytes either way.
pub fn fit_features_for_budget(input: &FitInput, max_bytes: usize) -> Result<Packet, StatError> {
    fit(input, Some(max_bytes))
}

fn fit(input: &FitInput, max_bytes: Option<usize>) -> Result<Packet, StatError> {
    check(input)?;
    let (n, p, k) = (input.rows, input.cols, input.k as usize);
    let log2_n = (n.ilog2() as u8).min(LOG2_N_MAX);
    let plan = knot_plan(log2_n);
    let columns = (0..p)
        .map(|j| {
            let col: Vec<f64> = (0..n).map(|i| input.x[i * p + j]).collect();
            fit_column(&col, &plan)
        })
        .collect();
    let mut packet = Packet::skeleton(input.task, input.k, log2_n, columns);
    let classes = match input.labels {
        Labels::Classes(ids) => {
            let mut counts = vec![0usize; k];
            ids.iter().for_each(|&c| counts[c as usize] += 1);
            // Support floor: a class with fewer than 10 rows gets no prior mass.
            // If no class reaches it, all the mass falls on the last class.
            for count in &mut counts {
                if *count < SUPPORT_K {
                    *count = 0;
                }
            }
            packet.prior = class_prior_codes(&counts)[..k - 1].to_vec();
            Some((ids, k))
        }
        Labels::Multilabel(cells) => {
            for (label, code) in packet.prior.iter_mut().enumerate() {
                let ones = (0..n).filter(|&i| cells[i * k + label] == 1).count();
                // A side with fewer than 10 rows is never emitted: code 0 or 255.
                *code = if ones < SUPPORT_K {
                    0
                } else if n - ones < SUPPORT_K {
                    255
                } else {
                    (255.0 * ones as f64 / n as f64).round().clamp(1.0, 254.0) as u8
                };
            }
            None
        }
        Labels::None => None,
    };
    let dependence_fits = max_bytes.is_none_or(|max| fit_to_budget(&packet, max).level > Level::A);
    if dependence_fits && p <= MAX_DEPENDENCE_COLS {
        let dependence = fit_dependence(input.x, n, p, classes);
        packet.loading = dependence.loading;
        packet.class_mean = dependence.class_mean;
    }
    Ok(packet)
}

/// Target of the FIT rows as the caller holds it.
#[derive(Clone, Copy, Debug)]
pub enum Target<'a> {
    /// Regression: one raw value per row.
    Regression(&'a [f64]),
    /// Binary or multiclass: one class id in `0..K` per row.
    Classes(&'a [u16]),
    /// Multilabel: row-major `rows x K` cells, each 0 or 1.
    Multilabel(&'a [u8]),
}

/// Size request: the highest level to keep and an optional byte budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    pub level: Level,
    pub max_bytes: Option<usize>,
}

impl Size {
    pub fn level(level: Level) -> Self {
        Self {
            level,
            max_bytes: None,
        }
    }
    pub fn budget(max_bytes: usize) -> Self {
        Self {
            level: Level::E,
            max_bytes: Some(max_bytes),
        }
    }
}

/// A fitted packet: the wire bytes, the owner-side scale card and how the
/// byte budget was met (`budget.fits` is false when even the smallest packet
/// is larger than the budget).
#[derive(Clone, Debug)]
pub struct Fitted {
    pub bytes: Vec<u8>,
    pub card: ScaleCard,
    pub budget: BudgetFit,
}

/// Fits a packet from raw FIT rows: `x` is row-major `n_fit x P` in raw units,
/// `y` the target of the same rows. Order: scale card, marginals, dependence,
/// target (S3, temperature, S8, S10), level cap and byte budget
/// (`codec::fit_to_budget`), noise codes of the final shape, pack.
pub fn fit_packet(
    x: &[f64],
    y: Target,
    task: Task,
    k: u8,
    size: Size,
    n_fit: usize,
) -> Result<Fitted, StatError> {
    if n_fit == 0 || x.is_empty() || x.len() % n_fit != 0 {
        return Err(StatError::Invalid(
            "feature matrix shape does not match its dimensions",
        ));
    }
    let p = x.len() / n_fit;
    let mut card = ScaleCard::fit(x, n_fit, p)?;
    let unit = card.to_unit_rows(x)?;
    let standardised: Vec<f64>;
    let (labels, response) = match y {
        Target::Regression(raw) => {
            if raw.len() != n_fit {
                return Err(StatError::Invalid("labels do not match the task"));
            }
            card = card.with_regression_target(raw)?;
            standardised = raw.iter().map(|&v| card.target_to_unit(v)).collect();
            (Labels::None, Response::Regression(&standardised))
        }
        Target::Classes(ids) => {
            card = card.with_classes((0..k).map(f64::from).collect());
            (Labels::Classes(ids), Response::Classes(ids))
        }
        Target::Multilabel(cells) => (Labels::Multilabel(cells), Response::Multilabel(cells)),
    };
    let input = FitInput {
        x: &unit,
        rows: n_fit,
        cols: p,
        task,
        k,
        labels,
    };
    let max_bytes = size.max_bytes.unwrap_or(usize::MAX);
    let mut packet = fit(&input, size.max_bytes)?;
    // A residual section is fitted only if the budget could hold it when empty.
    let mut probe = packet.clone();
    probe.main_mask.fill(false);
    let room = |level: Level| size.level >= level && packed_len(&probe, level) <= max_bytes;
    let plan = TargetPlan {
        main_effects: room(Level::D),
        pairs: room(Level::E),
    };
    let rows = FitRows {
        x: &unit,
        n: n_fit,
        p,
    };
    fit_target(&mut packet, &rows, response, plan)?;
    let mut budget = fit_to_budget(&packet.at_level(size.level)?, max_bytes);
    refresh_noise(&mut budget.packet, &rows, response)?;
    let bytes = pack(&budget.packet, budget.level)?;
    Ok(Fitted {
        bytes,
        card,
        budget,
    })
}
