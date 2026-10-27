//! Rust port of `research/levels/descriptors.py`: level masks, structural
//! landmark contrasts, report descriptors (`vector116`, column spectrum) and
//! train-only table descriptors (`table_vec24`, `column_desc`).

use super::np::{
    argsort, clip, col_mean, col_std, eigvalsh_sym, finite_or_zero, finite_or_zero32,
    gini_coefficient, mean1d, np_max, pairwise_sum, quantile_sorted, ranks_double_argsort, row_sum,
    sorted, squash, std1d,
};
use super::report_view::NativeReportView;
use crate::error::{DopeError, Result};

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
            // Python max(): first maximal element
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
/// for `classes`/`target_stats`). Used both when a report lacks shape fields
/// and when a report was measured on a row subsample so `vector116[85:91]`
/// reflects the full training table.
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

/// `descriptors._summarize_feature_spectrum`.
pub fn summarize_feature_spectrum(spec: &[[f64; 9]], mask: &[[bool; 9]]) -> [f64; 16] {
    if spec.is_empty() || !mask.iter().any(|r| r.iter().any(|&b| b)) {
        return [0.0; 16];
    }
    let s: Vec<[f64; 9]> = spec.iter().map(|r| r.map(squash)).collect();
    let pos = |v: f64| v.max(0.0);
    let rmax =
        |r: &[f64; 9], ks: &[usize]| np_max(&ks.iter().map(|&k| pos(r[k])).collect::<Vec<_>>());
    let lin_col: Vec<f64> = s.iter().map(|r| pos(r[0])).collect();
    let nonlin_macro: Vec<f64> = s.iter().map(|r| rmax(r, &[3, 4, 6])).collect();
    let nonlin_larger: Vec<f64> = s.iter().map(|r| rmax(r, &[1, 2, 5, 8])).collect();
    let xi_col: Vec<f64> = s.iter().map(|r| pos(r[7])).collect();
    let best: Vec<f64> = s
        .iter()
        .map(|r| rmax(r, &[0, 1, 2, 3, 4, 5, 6, 7, 8]))
        .collect();
    let mut sb = sorted(&best);
    sb.reverse();
    let top1 = sb[0];
    let top3 = mean1d(&sb[..sb.len().min(3)]);
    let diff =
        |a: &[f64], b: &[f64]| np_max(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>());
    [
        np_max(&lin_col),
        mean1d(&lin_col),
        gini_coefficient(&lin_col),
        np_max(&nonlin_macro),
        mean1d(&nonlin_macro),
        np_max(&nonlin_larger),
        mean1d(&nonlin_larger),
        np_max(&xi_col),
        mean1d(&xi_col),
        gini_coefficient(&xi_col),
        top1,
        top3,
        gini_coefficient(&best),
        diff(&nonlin_macro, &lin_col),
        diff(&nonlin_larger, &lin_col),
        diff(&xi_col, &lin_col),
    ]
}

/// Output of `descriptors.compute_table_descriptors`.
#[derive(Clone, Debug, PartialEq)]
pub struct TableDescriptors {
    pub table_vec24: [f32; 24],
    /// `max(p, 1)` rows of 8 descriptors.
    pub column_desc: Vec<[f32; 8]>,
    pub empirical_driver_weights: Vec<f32>,
}

/// `descriptors.compute_table_descriptors(X_train_scaled, y_train_scaled, level=level)`;
/// `x` is row-major `n × p`.
pub fn compute_table_descriptors(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    level: Level,
) -> TableDescriptors {
    assert_eq!(x.len(), n * p, "x must be n*p");
    if n < 4 || p == 0 {
        let pp = p.max(1);
        return TableDescriptors {
            table_vec24: [0.0; 24],
            column_desc: vec![[0.0; 8]; pp],
            empirical_driver_weights: vec![(1.0 / pp as f64) as f32; pp],
        };
    }
    let cm = col_mean(x, n, p);
    let cs = col_std(x, n, p);
    let valid: Vec<bool> = cs.iter().map(|&s| s > 1e-9).collect();
    let mut z = vec![0.0f64; n * p];
    let mut cols: Vec<Vec<f64>> = vec![vec![0.0f64; n]; p];
    for i in 0..n {
        for j in 0..p {
            if valid[j] {
                let v = (x[i * p + j] - cm[j]) / cs[j];
                z[i * p + j] = v;
                cols[j][i] = v;
            }
        }
    }
    let col_mean_1d = |buf: &[f64]| -> f64 {
        if p == 1 {
            pairwise_sum(buf) / n as f64
        } else {
            let mut s = 0.0;
            for &v in buf {
                s += v;
            }
            s / n as f64
        }
    };
    let mut tmp_col = vec![0.0f64; n];
    let mut skew = vec![0.0f64; p];
    let mut kurt = vec![0.0f64; p];
    let mut tail_ratio = vec![0.0f64; p];
    let mut mode_conc = vec![0.0f64; p];
    let y_std = std1d(y);
    let mut col_orders: Vec<Vec<usize>> = if y_std > 1e-9 {
        Vec::with_capacity(p)
    } else {
        Vec::new()
    };
    for j in 0..p {
        let cj = &cols[j];
        for i in 0..n {
            tmp_col[i] = cj[i].powf(3.0);
        }
        skew[j] = col_mean_1d(&tmp_col);
        for i in 0..n {
            tmp_col[i] = cj[i].powf(4.0);
        }
        kurt[j] = clip(col_mean_1d(&tmp_col) - 3.0, -3.0, 50.0);
        let s = if y_std > 1e-9 {
            let ord = argsort(cj);
            for (k, &idx) in ord.iter().enumerate() {
                tmp_col[k] = cj[idx];
            }
            col_orders.push(ord);
            &tmp_col[..]
        } else {
            let sorted_vec = sorted(cj);
            tmp_col.copy_from_slice(&sorted_vec);
            &tmp_col[..]
        };
        let q05 = quantile_sorted(s, 0.05);
        let q25 = quantile_sorted(s, 0.25);
        let q75 = quantile_sorted(s, 0.75);
        let q95 = quantile_sorted(s, 0.95);
        let iqr = (q75 - q25).max(1e-6);
        tail_ratio[j] = clip((q95 - q05) / iqr, 0.0, 20.0) / 10.0;
        let med_j = if s[n - 1].is_nan() {
            f64::NAN
        } else if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        for i in 0..n {
            tmp_col[i] = if (cj[i] - med_j).abs() < 0.05 {
                1.0
            } else {
                0.0
            };
        }
        mode_conc[j] = col_mean_1d(&tmp_col);
    }

    let mut y_z = vec![0.0; n];
    let (pearson_r, spearman_r) = if y_std > 1e-9 {
        let ym = mean1d(y);
        for i in 0..n {
            y_z[i] = (y[i] - ym) / y_std;
        }
        let mut pr = vec![0.0f64; p];
        for j in 0..p {
            let cj = &cols[j];
            for i in 0..n {
                tmp_col[i] = cj[i] * y_z[i];
            }
            pr[j] = col_mean_1d(&tmp_col).abs();
        }
        let yr = ranks_double_argsort(y);
        let yrm = mean1d(&yr);
        let yrs = std1d(&yr).max(1e-9);
        let yr: Vec<f64> = yr.iter().map(|v| (v - yrm) / yrs).collect();
        let mut sr = vec![0.0f64; p];
        let mut r_col = vec![0.0f64; n];
        for j in 0..p {
            for (rank, &idx) in col_orders[j].iter().enumerate() {
                r_col[idx] = rank as f64;
            }
            let xrm = col_mean_1d(&r_col);
            for i in 0..n {
                let d = r_col[i] - xrm;
                tmp_col[i] = d * d;
            }
            let xrs = col_mean_1d(&tmp_col).sqrt().max(1e-9);
            for i in 0..n {
                tmp_col[i] = ((r_col[i] - xrm) / xrs) * yr[i];
            }
            sr[j] = col_mean_1d(&tmp_col).abs();
        }
        (pr, sr)
    } else {
        (vec![0.0; p], vec![0.0; p])
    };

    let (col_max_corr, col_mean_corr, eff_rank, spec_entropy, top1_eig, top3_eig, top5_eig);
    if p > 1 {
        let mut corr = vec![0.0f64; p * p];
        for a in 0..p {
            let ca = &cols[a];
            for b in a..p {
                let cb = &cols[b];
                let mut s = 0.0;
                for i in 0..n {
                    s += ca[i] * cb[i];
                }
                let v = s / n as f64;
                corr[a * p + b] = v;
                corr[b * p + a] = v;
            }
        }
        for a in 0..p {
            corr[a * p + a] = 1.0;
        }
        for v in &mut corr {
            *v = finite_or_zero(*v);
        }
        let off: Vec<f64> = (0..p * p)
            .map(|k| {
                let eye = if k / p == k % p { 1.0 } else { 0.0 };
                (corr[k] - eye).abs()
            })
            .collect();
        col_max_corr = (0..p)
            .map(|a| np_max(&off[a * p..(a + 1) * p]))
            .collect::<Vec<_>>();
        col_mean_corr = row_sum(&off, p, p)
            .into_iter()
            .map(|s| s / (p - 1) as f64)
            .collect::<Vec<_>>();
        let mut ev = eigvalsh_sym(&corr, p);
        ev.reverse();
        let ev: Vec<f64> = ev.into_iter().map(|v| v.max(0.0)).collect();
        let eig_sum = pairwise_sum(&ev).max(1e-9);
        let shares: Vec<f64> = ev.iter().map(|v| v / eig_sum).collect();
        let sq: Vec<f64> = ev.iter().map(|v| v * v).collect();
        eff_rank = ((eig_sum * eig_sum) / pairwise_sum(&sq).max(1e-9)) / p as f64;
        let ent: Vec<f64> = shares
            .iter()
            .filter(|&&s| s > 1e-12)
            .map(|&s| s * s.ln())
            .collect();
        spec_entropy = -pairwise_sum(&ent) / (p as f64).ln().max(1e-9);
        top1_eig = shares[0];
        top3_eig = pairwise_sum(&shares[..p.min(3)]);
        top5_eig = pairwise_sum(&shares[..p.min(5)]);
    } else {
        col_max_corr = vec![0.0];
        col_mean_corr = vec![0.0];
        eff_rank = 1.0;
        spec_entropy = 1.0;
        top1_eig = 1.0;
        top3_eig = 1.0;
        top5_eig = 1.0;
    }

    let quad_r: Vec<f64> = if y_std > 1e-9 && level == Level::Ultra {
        let z2: Vec<f64> = z.iter().map(|&v| v * v - 1.0).collect();
        let z2s: Vec<f64> = col_std(&z2, n, p)
            .into_iter()
            .map(|s| s.max(1e-6))
            .collect();
        let prod: Vec<f64> = (0..n * p)
            .map(|k| (z2[k] / z2s[k % p]) * y_z[k / p])
            .collect();
        col_mean(&prod, n, p).into_iter().map(f64::abs).collect()
    } else {
        vec![0.0; p]
    };

    let raw_driver: Vec<f64> = (0..p)
        .map(|j| 0.45 * pearson_r[j] + 0.35 * spearman_r[j] + 0.20 * quad_r[j])
        .collect();
    let rd_sum = pairwise_sum(&raw_driver);
    let driver_weights: Vec<f64> = if rd_sum > 1e-9 {
        raw_driver.iter().map(|v| v / rd_sum).collect()
    } else {
        vec![1.0 / p as f64; p]
    };
    let mut sorted_r = sorted(&pearson_r);
    sorted_r.reverse();
    let mut sorted_sp = sorted(&spearman_r);
    sorted_sp.reverse();

    let inc_l = if level.includes_larger() { 1.0 } else { 0.0 };
    let inc_f = if level.includes_fine() { 1.0 } else { 0.0 };
    let abs_skew: Vec<f64> = skew.iter().map(|v| v.abs()).collect();
    let mas = mean1d(&abs_skew);
    let xas = np_max(&abs_skew);
    let kpos: Vec<f64> = kurt.iter().map(|v| v.max(0.0)).collect();
    let mk = mean1d(&kpos);
    let sp_minus_p: Vec<f64> = (0..p).map(|j| spearman_r[j] - pearson_r[j]).collect();
    let tv: [f64; 24] = [
        eff_rank,
        spec_entropy,
        mas / (1.0 + mas),
        mean1d(&mode_conc),
        sorted_r[0],
        mean1d(&pearson_r),
        gini_coefficient(&pearson_r),
        (n.max(1) as f64).log10() / 6.0,
        inc_l * top1_eig,
        inc_l * top3_eig,
        inc_l * top5_eig,
        inc_l * np_max(&col_max_corr),
        inc_l * mean1d(&col_mean_corr),
        inc_l * mean1d(&tail_ratio),
        inc_l * sorted_sp[0],
        inc_l * np_max(&sp_minus_p),
        inc_f * (xas / (1.0 + xas)),
        inc_f * (mk / (1.0 + mk)),
        inc_f * np_max(&tail_ratio),
        inc_f * mean1d(&sorted_r[..p.min(3)]),
        inc_f * mean1d(&sorted_sp[..p.min(3)]),
        inc_f * np_max(&quad_r),
        inc_f * gini_coefficient(&raw_driver),
        inc_f * np_max(&raw_driver),
    ];
    let table_vec24 = tv.map(|v| finite_or_zero32(v as f32));
    // col_max_corr has length 1 when p == 1 (matches numpy broadcasting for p == 1).
    let column_desc: Vec<[f32; 8]> = (0..p)
        .map(|j| {
            [
                squash(skew[j]),
                squash(kurt[j]) * inc_l,
                tail_ratio[j] * inc_l,
                mode_conc[j],
                col_max_corr[j.min(col_max_corr.len() - 1)] * inc_l,
                pearson_r[j],
                spearman_r[j] * inc_l,
                quad_r[j] * inc_f,
            ]
            .map(|v| finite_or_zero32(v as f32))
        })
        .collect();
    let empirical_driver_weights: Vec<f32> = driver_weights
        .iter()
        .map(|&v| {
            let f = v as f32;
            if f.is_nan() {
                (1.0 / p as f64) as f32
            } else if f == f32::INFINITY {
                f32::MAX
            } else if f == f32::NEG_INFINITY {
                f32::MIN
            } else {
                f
            }
        })
        .collect();
    TableDescriptors {
        table_vec24,
        column_desc,
        empirical_driver_weights,
    }
}
