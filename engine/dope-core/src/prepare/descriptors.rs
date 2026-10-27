//! Rust port of `research/levels/descriptors.py`: level masks, structural
//! landmark contrasts, report descriptors (`vector116`, column spectrum) and
//! train-only table descriptors (`table_vec24`, `column_desc`).

use super::np::{clip, finite_or_zero, mean1d, np_max, pairwise_sum, squash};
use super::report_view::NativeReportView;
use crate::error::{DopeError, Result};

pub use super::descriptors_table::*;

pub const LANDMARK_NAMES: [&str; 23] = [
    "lin",
    "log",
    "sqrt",
    "pow",
    "sat",
    "abs",
    "hinge",
    "step",
    "periodic",
    "bump",
    "rows",
    "pairs",
    "rff",
    "missing",
    "best1",
    "stump",
    "otree3",
    "gbdt",
    "knn1",
    "knn16",
    "net",
    "lin_quarter",
    "gbdt_quarter",
];
pub const MACRO_INDICES: [usize; 9] = [0, 1, 2, 3, 4, 5, 13, 14, 21];
pub const LARGER_ADDED_INDICES: [usize; 9] = [6, 7, 8, 9, 10, 11, 12, 15, 16];
pub const FINE_ADDED_INDICES: [usize; 5] = [17, 18, 19, 20, 22];

/// MLC-v3 level (`descriptors.LEVEL_LANDMARK_INDICES` keys).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    Low,
    Small,
    Medium,
    High,
    Ultra,
}

impl Level {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "low" => Ok(Level::Low),
            "small" => Ok(Level::Small),
            "medium" => Ok(Level::Medium),
            "high" => Ok(Level::High),
            "ultra" => Ok(Level::Ultra),
            _ => Err(DopeError::Input(format!("unknown_level_{s}"))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Low => "low",
            Level::Small => "small",
            Level::Medium => "medium",
            Level::High => "high",
            Level::Ultra => "ultra",
        }
    }
    /// `level in ("high", "ultra")`
    pub fn includes_larger(self) -> bool {
        matches!(self, Level::High | Level::Ultra)
    }
    /// `level == "ultra"`
    pub fn includes_fine(self) -> bool {
        self == Level::Ultra
    }
}

/// `descriptors.level_landmark_mask`.
pub fn level_landmark_mask(level: Level) -> [bool; 23] {
    let mut m = [false; 23];
    match level {
        Level::Low | Level::Small => MACRO_INDICES[..6].iter().for_each(|&i| m[i] = true),
        Level::Medium => MACRO_INDICES.iter().for_each(|&i| m[i] = true),
        Level::High => MACRO_INDICES
            .iter()
            .chain(LARGER_ADDED_INDICES.iter())
            .for_each(|&i| m[i] = true),
        Level::Ultra => m = [true; 23],
    }
    m
}

/// `descriptors.compute_level_contrasts`.
pub fn compute_level_contrasts(gains: &[f64; 23], mask: &[bool; 23], level: Level) -> [f64; 16] {
    let lm = level_landmark_mask(level);
    let allowed: Vec<bool> = (0..23).map(|i| lm[i] && mask[i]).collect();
    let g: Vec<f64> = (0..23)
        .map(|i| if allowed[i] { gains[i] } else { f64::NAN })
        .collect();
    let get = |i: usize| -> f64 {
        if allowed[i] && g[i].is_finite() {
            g[i]
        } else {
            f64::NAN
        }
    };
    let maxf = |idxs: &[usize]| -> f64 {
        let vals: Vec<f64> = idxs
            .iter()
            .filter(|&&i| allowed[i] && g[i].is_finite())
            .map(|&i| g[i])
            .collect();
        if vals.is_empty() {
            f64::NAN
        } else {
            let mut m = vals[0];
            for &v in &vals[1..] {
                if v > m {
                    m = v;
                }
            }
            m
        }
    };
    let sub = |a: f64, b: f64, denom: f64| -> f64 {
        if a.is_finite() && b.is_finite() && denom >= 0.01 {
            clip((a - b) / denom, -1.0, 1.0)
        } else {
            0.0
        }
    };
    let all: Vec<usize> = (0..23).collect();
    let s = maxf(&all);
    if !s.is_finite() || s < 0.01 {
        return [0.0; 16];
    }
    let lin = get(0);
    let add_macro = maxf(&[0, 1, 2, 3, 4, 5, 14]);
    let add_all = maxf(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]);
    let mono_excess = sub(maxf(&[1, 2, 4]), lin, s);
    let sym_excess = sub(maxf(&[3, 5, 9]), lin, s);
    let best1_share = if get(14).is_finite() {
        clip(get(14) / s, -1.0, 1.0)
    } else {
        0.0
    };
    let miss_share = if get(13).is_finite() {
        clip(get(13) / s, -1.0, 1.0)
    } else {
        0.0
    };
    let lin_lc = sub(lin, get(21), s);
    let hinge_step = sub(get(7), get(6), s);
    let periodic_ex = sub(get(8), maxf(&[0, 4]), s);
    let pairs_ex = sub(get(11), lin, s);
    let rff_ex = sub(get(12), add_all, s);
    let otree_ex = sub(maxf(&[15, 16]), add_all, s);
    let nonadd_ex = sub(maxf(&[17, 12, 20]), add_all, s);
    let mut knn_ratio = 0.0;
    if get(18).is_finite() && get(19).is_finite() && get(19).abs() > 1e-6 {
        knn_ratio = clip(get(18) / get(19).max(1e-6), -1.0, 1.0);
    }
    let gbdt_lc = sub(get(17), get(22), s);
    let net_vs_gbdt = sub(get(20), maxf(&[17, 12]), s);
    let out = [
        clip(s, -1.0, 1.0),
        if lin.is_finite() {
            clip(lin / s, -1.0, 1.0)
        } else {
            0.0
        },
        if add_macro.is_finite() {
            clip(add_macro / s, -1.0, 1.0)
        } else {
            0.0
        },
        mono_excess,
        sym_excess,
        best1_share,
        miss_share,
        lin_lc,
        hinge_step,
        periodic_ex,
        pairs_ex,
        rff_ex,
        otree_ex,
        nonadd_ex,
        knn_ratio,
        gbdt_lc + 0.5 * net_vs_gbdt,
    ];
    out.map(finite_or_zero)
}

/// Explicit table shape and target-stat overrides for
/// `descriptors.extract_report_descriptors`: an explicit value wins over the
/// report field, which wins over the default (`1` for `rows`/`features`, `0`
/// for `classes`/`target_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShapeOverride {
    pub rows: Option<usize>,
    pub features: Option<usize>,
    pub classes: Option<usize>,
    pub target_stats: Option<[f64; 2]>,
}

impl ShapeOverride {
    pub fn shape_only(rows: usize, features: usize) -> Self {
        Self {
            rows: Some(rows),
            features: Some(features),
            classes: None,
            target_stats: None,
        }
    }

    pub fn from_table(table: &super::scaling::PreparedTable) -> Self {
        Self {
            rows: Some(table.rows),
            features: Some(table.features),
            classes: Some(table.classes),
            target_stats: Some(table.target_stats),
        }
    }
}

/// Output of `descriptors.extract_report_descriptors`.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportDescriptors {
    pub level: Level,
    /// Squashed gains.
    pub gains23: [f64; 23],
    /// Squashed standard errors.
    pub ses23: [f64; 23],
    pub lambdas23: [f64; 23],
    pub mask23: [bool; 23],
    pub midranks23: [f64; 23],
    pub contrasts16: [f64; 16],
    pub target_stats2: [f64; 2],
    pub shape4: [f64; 4],
    pub spectrum_summary16: [f64; 16],
    pub group_summary9: [f64; 9],
    pub vector116: [f32; 116],
    /// `squash(spec_matrix)` as float32, `feats` rows.
    pub column_spectrum: Vec<[f32; 9]>,
    pub column_prior_importance: Vec<f32>,
}

/// `descriptors.extract_report_descriptors(report, level=level, rows=, features=)`.
pub fn extract_report_descriptors(
    report: &NativeReportView,
    level: Level,
    shape: ShapeOverride,
) -> ReportDescriptors {
    let lvl_mask = level_landmark_mask(level);
    let mut gains = [0.0f64; 23];
    let mut ses = [0.0f64; 23];
    let mut present = [false; 23];
    let mut stable = [false; 23];
    let mut lambdas = [0.0f64; 23];
    for c in &report.cells {
        let idx = c.id - 1;
        if !(0..23).contains(&idx) || !lvl_mask[idx as usize] {
            continue;
        }
        let idx = idx as usize;
        if (c.state == "complete" || c.state == "uninformative")
            && c.gain.is_some_and(f64::is_finite)
        {
            gains[idx] = c.gain.expect("gain");
            present[idx] = true;
            if let Some(se) = c.se.filter(|s| s.is_finite()) {
                ses[idx] = se;
            }
            stable[idx] = c.stable;
            if !c.lambda_indices.is_empty() {
                lambdas[idx] = mean1d(&c.lambda_indices) / 8.0;
            }
        }
    }

    let mut midranks = [0.0f64; 23];
    let valid: Vec<usize> = (0..23).filter(|&i| present[i]).collect();
    if valid.len() > 1 {
        for &i in &valid {
            let gi = gains[i];
            let less = valid.iter().filter(|&&k| gains[k] < gi).count() as f64;
            let eq = valid.iter().filter(|&&k| gains[k] == gi).count() as f64;
            midranks[i] = (less + 0.5 * (eq - 1.0)) / (valid.len() - 1) as f64;
        }
    } else if valid.len() == 1 {
        midranks[valid[0]] = 0.5;
    }

    let contrasts = compute_level_contrasts(&gains, &present, level);

    let ts = report.target_stats.clone().unwrap_or_default();
    let t_at = |k: usize| -> f64 {
        if let Some(ovr) = shape.target_stats {
            return if ovr[k].is_finite() { ovr[k] } else { 0.0 };
        }
        ts.get(k)
            .copied()
            .flatten()
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    };
    let t_stats = [squash(t_at(0)), squash(t_at(1))];

    let rows = shape
        .rows
        .map(|r| r as i64)
        .or(report.rows)
        .unwrap_or(1)
        .max(1);
    let feats = shape
        .features
        .map(|r| r as i64)
        .or(report.features)
        .unwrap_or(1)
        .max(1);
    let classes = shape
        .classes
        .map(|c| c as i64)
        .or(report.classes)
        .unwrap_or(0);
    let is_clf = if classes >= 2 || report.task.as_deref() == Some("classification") {
        1.0
    } else {
        0.0
    };
    let shape_vec = [
        is_clf,
        (rows as f64).log10() / 6.0,
        (feats as f64).log2() / 12.0,
        (feats as f64 / rows as f64).log10() / 4.0,
    ];

    let feats_u = feats as usize;
    let mut spec = vec![[0.0f64; 9]; feats_u];
    let mut spec_mask = vec![[false; 9]; feats_u];
    for item in &report.spectrum {
        let slot = item.slot;
        if slot < 0 || slot >= feats {
            continue;
        }
        let slot = slot as usize;
        let vals: Vec<Option<f64>> = item.values.clone().unwrap_or_else(|| vec![None; 9]);
        for k in 0..vals.len().min(9) {
            if matches!(level, Level::Low | Level::Small | Level::Medium)
                && matches!(k, 1 | 2 | 5 | 7 | 8)
            {
                continue;
            }
            if level == Level::High && k == 7 {
                continue;
            }
            if let Some(v) = vals[k].filter(|v| v.is_finite()) {
                spec[slot][k] = v;
                spec_mask[slot][k] = true;
            }
        }
    }
    let spec_summary = summarize_feature_spectrum(&spec, &spec_mask);
    let sg: [f64; 23] = gains.map(squash);
    let sse: [f64; 23] = ses.map(squash);

    let any_present = |idxs: &[usize]| idxs.iter().any(|&i| present[i]);
    let max_of = |idxs: &[usize]| np_max(&idxs.iter().map(|&i| sg[i]).collect::<Vec<_>>());
    let mean_of = |idxs: &[usize]| mean1d(&idxs.iter().map(|&i| sg[i]).collect::<Vec<_>>());
    let macro_gain_max = if any_present(&MACRO_INDICES) {
        max_of(&MACRO_INDICES)
    } else {
        0.0
    };
    let larger_gain_max = if level.includes_larger() && any_present(&LARGER_ADDED_INDICES) {
        max_of(&LARGER_ADDED_INDICES)
    } else {
        0.0
    };
    let fine_gain_max = if level.includes_fine() && any_present(&FINE_ADDED_INDICES) {
        max_of(&FINE_ADDED_INDICES)
    } else {
        0.0
    };
    let pres_mean = |vals: &[f64; 23]| -> f64 {
        if valid.is_empty() {
            0.0
        } else {
            mean1d(&valid.iter().map(|&i| vals[i]).collect::<Vec<_>>())
        }
    };
    let stable_f: [f64; 23] = stable.map(|b| if b { 1.0 } else { 0.0 });
    let group_summary = [
        macro_gain_max,
        larger_gain_max,
        fine_gain_max,
        mean_of(&MACRO_INDICES),
        if level.includes_larger() {
            mean_of(&LARGER_ADDED_INDICES)
        } else {
            0.0
        },
        if level.includes_fine() {
            mean_of(&FINE_ADDED_INDICES)
        } else {
            0.0
        },
        pres_mean(&sse),
        pres_mean(&stable_f),
        pres_mean(&lambdas),
    ];

    let mut vector116 = [0.0f32; 116];
    {
        let mut k = 0;
        let mut push = |v: f64| {
            vector116[k] = v as f32;
            k += 1;
        };
        sg.iter().for_each(|&v| push(v));
        present
            .iter()
            .for_each(|&b| push(if b { 1.0 } else { 0.0 }));
        midranks.iter().for_each(|&v| push(v));
        contrasts.iter().for_each(|&v| push(v));
        t_stats.iter().for_each(|&v| push(v));
        shape_vec.iter().for_each(|&v| push(v));
        spec_summary.iter().for_each(|&v| push(v));
        group_summary.iter().for_each(|&v| push(v));
    }

    let col_prior_raw: Vec<f64> = spec
        .iter()
        .map(|r| np_max(&r.iter().map(|v| v.abs()).collect::<Vec<_>>()))
        .collect();
    let total = pairwise_sum(&col_prior_raw);
    let col_prior: Vec<f32> = if total > 1e-9 {
        col_prior_raw.iter().map(|v| (v / total) as f32).collect()
    } else {
        vec![(1.0 / feats as f64) as f32; feats_u]
    };
    let column_spectrum: Vec<[f32; 9]> = spec.iter().map(|r| r.map(|v| squash(v) as f32)).collect();

    ReportDescriptors {
        level,
        gains23: sg,
        ses23: sse,
        lambdas23: lambdas,
        mask23: present,
        midranks23: midranks,
        contrasts16: contrasts,
        target_stats2: t_stats,
        shape4: shape_vec,
        spectrum_summary16: spec_summary,
        group_summary9: group_summary,
        vector116,
        column_spectrum,
        column_prior_importance: col_prior,
    }
}
