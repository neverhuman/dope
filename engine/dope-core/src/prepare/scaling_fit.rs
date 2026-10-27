//! Column and target fitting helpers for `prepare::scaling`.

use super::np::{
    clip, linspace, mean1d, pairwise_sum_map, quantiles, sorted, std1d, std1d_with_mean,
};
use super::scaling::{
    err, CategoricalColumnEncoder, NumericColumnScaler, PrepTask, TargetScaler, TargetValue,
    TransformKind, QUANTILE_GRID,
};
use crate::error::Result;
use std::collections::HashMap;

const MISSING_TOKENS: [&str; 5] = ["nan", "null", "none", "?", "na"];

#[inline]
pub(crate) fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// Python `str.strip()` (whitespace per `str.isspace` for ASCII + Unicode).
#[inline]
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

#[inline]
pub(crate) fn is_missing_token(s: &str) -> bool {
    let _ = MISSING_TOKENS;
    match s.len() {
        0 => true,
        1 => s == "?",
        2 => s.eq_ignore_ascii_case("na"),
        3 => s.eq_ignore_ascii_case("nan"),
        4 => s.eq_ignore_ascii_case("null") || s.eq_ignore_ascii_case("none"),
        _ => false,
    }
}

/// Python `float(s)` on an already-stripped string.
#[inline]
pub(crate) fn py_float(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    let body = match b[0] {
        b'+' | b'-' => &b[1..],
        _ => b,
    };
    if body.is_empty() || matches!(body[0], b'+' | b'-') {
        return None;
    }
    let valid = body
        .iter()
        .all(|&c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        || body.eq_ignore_ascii_case(b"inf")
        || body.eq_ignore_ascii_case(b"infinity")
        || body.eq_ignore_ascii_case(b"nan");
    if !valid {
        return None;
    }
    s.parse::<f64>().ok()
}

/// `scaling._parse_float` for a string cell.
pub fn parse_float_cell(cell: &str) -> f64 {
    let s = py_strip(cell);
    if is_missing_token(s) {
        return f64::NAN;
    }
    py_float(s).unwrap_or(f64::NAN)
}

pub(crate) fn regression_array(y: &[TargetValue<'_>]) -> Result<Vec<f64>> {
    y.iter()
        .map(|v| match v {
            TargetValue::Num(x) => Ok(*x),
            TargetValue::Str(s) => {
                py_float(py_strip(s)).ok_or_else(|| err("prepare_target_not_numeric"))
            }
        })
        .collect()
}

fn safe_skew_kurt_with_stats(x: &[f64], m: f64, s: f64) -> (f64, f64) {
    if x.len() < 4 || !s.is_finite() || s <= 1e-12 {
        return (0.0, 0.0);
    }
    let n_f = x.len() as f64;
    let skew = pairwise_sum_map(x, |v| ((v - m) / s).powf(3.0)) / n_f;
    let kurt = pairwise_sum_map(x, |v| ((v - m) / s).powf(4.0)) / n_f - 3.0;
    let skew = if skew.is_finite() {
        clip(skew, -50.0, 50.0)
    } else {
        0.0
    };
    let kurt = if kurt.is_finite() {
        clip(kurt, -50.0, 200.0)
    } else {
        0.0
    };
    (skew, kurt)
}

pub(super) fn safe_skew_kurt(x: &[f64]) -> (f64, f64) {
    if x.len() < 4 {
        return (0.0, 0.0);
    }
    let m = mean1d(x);
    let s = std1d_with_mean(x, m);
    safe_skew_kurt_with_stats(x, m, s)
}

fn py_max(vals: &[f64]) -> f64 {
    let mut m = vals[0];
    for &v in &vals[1..] {
        if v > m {
            m = v;
        }
    }
    m
}

pub(crate) fn fit_numeric_column(slot: usize, col: &[f64]) -> NumericColumnScaler {
    let finite: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    let miss_rate = 1.0 - (finite.len() as f64 / col.len().max(1) as f64);
    if finite.is_empty() {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Constant,
            missing_rate: 1.0,
            center: 0.0,
            scale: 1.0,
            post_mean: 0.0,
            post_std: 1.0,
            quantiles: [0.0; 9],
            skewness: 0.0,
            kurtosis: 0.0,
            blend_quantile_weight: 0.35,
        };
    }
    let s = sorted(&finite);
    let mut uniq_count = 1usize;
    for w in s.windows(2) {
        if w[0] != w[1] {
            uniq_count += 1;
            if uniq_count > 2 {
                break;
            }
        }
    }
    let q_vals: [f64; 9] = QUANTILE_GRID.map(|q| super::np::quantile_sorted(&s, q));
    if uniq_count <= 1 {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Constant,
            missing_rate: miss_rate,
            center: s[0],
            scale: 1.0,
            post_mean: 0.0,
            post_std: 1.0,
            quantiles: q_vals,
            skewness: 0.0,
            kurtosis: 0.0,
            blend_quantile_weight: 0.35,
        };
    }
    let mean_f = mean1d(&finite);
    let std_f = std1d_with_mean(&finite, mean_f);
    let (skew, kurt) = safe_skew_kurt_with_stats(&finite, mean_f, std_f);
    if uniq_count == 2 {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Binary,
            missing_rate: miss_rate,
            center: mean_f,
            scale: std_f.max(1e-6),
            post_mean: 0.0,
            post_std: 1.0,
            quantiles: q_vals,
            skewness: skew,
            kurtosis: kurt,
            blend_quantile_weight: 0.0,
        };
    }
    let med = q_vals[4];
    let iqr = (q_vals[5] - q_vals[3]).max(0.0);
    let span90 = (q_vals[7] - q_vals[1]).max(0.0);
    let tail_ratio = if iqr > 1e-9 {
        span90 / iqr.max(1e-9)
    } else {
        1.0
    };
    let k = s.len();
    let split = s.partition_point(|&v| v < med);
    let mut left = split;
    let mut right = split;
    let mut prev = 0.0f64;
    let mut curr = 0.0f64;
    for _ in 0..=(k / 2) {
        prev = curr;
        curr = if left > 0 && right < k {
            let dl = (s[left - 1] - med).abs();
            let dr = (s[right] - med).abs();
            if dl <= dr {
                left -= 1;
                dl
            } else {
                right += 1;
                dr
            }
        } else if left > 0 {
            left -= 1;
            (s[left] - med).abs()
        } else {
            let v = (s[right] - med).abs();
            right += 1;
            v
        };
    }
    let mad = if k % 2 == 1 {
        curr
    } else {
        (prev + curr) / 2.0
    };
    let robust_scale = py_max(&[iqr / 1.3489795, mad * 1.4826, std_f * 0.25, 1e-6]);
    if skew.abs() > 1.0 || kurt > 4.0 || tail_ratio > 4.2 {
        let logged: Vec<f64> = finite
            .iter()
            .map(|&v| {
                let sh = v - med;
                np_sign(sh) * (sh.abs() / robust_scale).ln_1p()
            })
            .collect();
        let lm = mean1d(&logged);
        let ls = std1d_with_mean(&logged, lm).max(1e-6);
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::SignedLog1pHybrid,
            missing_rate: miss_rate,
            center: med,
            scale: robust_scale,
            post_mean: lm,
            post_std: ls,
            quantiles: q_vals,
            skewness: skew,
            kurtosis: kurt,
            blend_quantile_weight: 0.35,
        };
    }
    NumericColumnScaler {
        slot,
        transform_kind: TransformKind::QuantileRobustHybrid,
        missing_rate: miss_rate,
        center: med,
        scale: robust_scale,
        post_mean: mean_f,
        post_std: std_f.max(1e-6),
        quantiles: q_vals,
        skewness: skew,
        kurtosis: kurt,
        blend_quantile_weight: 0.30,
    }
}

pub(crate) fn fit_categorical_column(
    slot: usize,
    col: &[&str],
    y_num: &[f64],
    y_global_mean: f64,
    min_level_count: usize,
    prior_smoothing: f64,
) -> CategoricalColumnEncoder {
    let n = col.len().max(1);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut miss_count = 0usize;
    let mut cleaned: Vec<&str> = Vec::with_capacity(col.len());
    for val in col {
        let s = py_strip(val);
        if is_missing_token(s) {
            miss_count += 1;
            cleaned.push("");
        } else {
            *counts.entry(s.to_string()).or_insert(0) += 1;
            cleaned.push(s);
        }
    }
    let mut retained: Vec<String> = counts
        .iter()
        .filter(|(_, &c)| c >= min_level_count)
        .map(|(t, _)| t.clone())
        .collect();
    retained.sort_by(|a, b| counts[b].cmp(&counts[a]).then_with(|| a.cmp(b)));
    retained.truncate(127);
    let mut vocab = vec!["__other__".to_string()];
    vocab.extend(retained.iter().cloned());
    let token_to_id: HashMap<String, usize> = retained
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i + 1))
        .collect();
    let v = vocab.len();
    let mut id_counts = vec![0.0f64; v];
    let mut id_y_sums = vec![0.0f64; v];
    for (i, s) in cleaned.iter().enumerate() {
        let idx = token_to_id.get(*s).copied().unwrap_or(0);
        id_counts[idx] += 1.0;
        id_y_sums[idx] += y_num[i];
    }
    let lin = linspace(-0.25, 0.25, v);
    let combined: Vec<f64> = (0..v)
        .map(|k| {
            let shrunk =
                (id_y_sums[k] + prior_smoothing * y_global_mean) / (id_counts[k] + prior_smoothing);
            shrunk + 0.15 * lin[k]
        })
        .collect();
    let freqs: Vec<f64> = id_counts.iter().map(|c| c / n as f64).collect();
    let train_enc: Vec<f64> = cleaned
        .iter()
        .map(|s| combined[token_to_id.get(*s).copied().unwrap_or(0)])
        .collect();
    CategoricalColumnEncoder {
        slot,
        missing_rate: miss_count as f64 / n as f64,
        vocab_tokens: vocab,
        token_to_id,
        id_target_encoding: combined,
        id_frequencies: freqs,
        post_mean: mean1d(&train_enc),
        post_std: std1d(&train_enc).max(1e-6),
    }
}

/// Python `repr(float)` for the classification positive-label heuristic.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').expect("exp");
    let exp: i32 = exp.parse().expect("exp int");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let es = if exp < 0 { "-" } else { "+" };
        return format!("{sign}{m}e{es}{:02}", exp.abs());
    }
    let nd = digits.len() as i32;
    let s = if exp >= 0 {
        if nd <= exp + 1 {
            format!("{}{}.0", digits, "0".repeat((exp + 1 - nd) as usize))
        } else {
            let k = (exp + 1) as usize;
            format!("{}.{}", &digits[..k], &digits[k..])
        }
    } else {
        format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
    };
    format!("{sign}{s}")
}

pub(crate) fn fit_target_scaler_with_policy(
    y: &[TargetValue<'_>],
    task: PrepTask,
    policy: super::scaling::ClassVocabPolicy,
) -> Result<TargetScaler> {
    match task {
        PrepTask::Multiclass => Ok(super::scaling::fit_multiclass_target_scaler(y, policy)),
        PrepTask::Classification => {
            let tokens: Vec<String> = y
                .iter()
                .map(|v| match v {
                    TargetValue::Num(x) => py_float_repr(*x),
                    TargetValue::Str(s) => py_strip(s).to_string(),
                })
                .collect();
            let mut uniq = tokens.clone();
            uniq.sort();
            uniq.dedup();
            let pos = if uniq.iter().any(|u| u == "1") {
                "1".to_string()
            } else if uniq.iter().any(|u| u == "1.0") {
                "1.0".to_string()
            } else if let Some(t) = uniq.iter().find(|u| u.to_lowercase() == "true") {
                t.clone()
            } else {
                uniq.last().cloned().unwrap_or_else(|| "1".to_string())
            };
            let bin: Vec<f64> = tokens.iter().map(|t| f64::from(*t == pos)).collect();
            let prior = if bin.is_empty() { 0.5 } else { mean1d(&bin) };
            let (s, med) = (
                (prior * (1.0 - prior)).sqrt().max(1e-6),
                f64::from(prior >= 0.5),
            );
            let mut ts = TargetScaler::base(task, prior, s, med, 1.0, 0.0, 0.0);
            ts.positive_label = Some(pos);
            ts.class_prior = prior;
            Ok(ts)
        }
        PrepTask::Regression => {
            let finite: Vec<f64> = regression_array(y)?
                .into_iter()
                .filter(|v| v.is_finite())
                .collect();
            if finite.is_empty() {
                return Ok(TargetScaler::base(task, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));
            }
            let (m, s) = (mean1d(&finite), std1d(&finite).max(1e-6));
            let q = quantiles(&finite, &[0.25, 0.5, 0.75]);
            let (skew, kurt) = safe_skew_kurt(&finite);
            Ok(TargetScaler::base(
                task,
                m,
                s,
                q[1],
                (q[2] - q[0]).max(1e-6),
                skew,
                kurt,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_repr_matches_python() {
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(1234.5), "1234.5");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(0.00001), "1e-05");
        assert_eq!(py_float_repr(-2.5e-7), "-2.5e-07");
        assert_eq!(py_float_repr(123456789012345.0), "123456789012345.0");
    }

    #[test]
    fn parse_float_semantics() {
        assert!(parse_float_cell(" NA ").is_nan());
        assert_eq!(parse_float_cell(" 1e3 "), 1000.0);
        assert!(parse_float_cell("abc").is_nan());
        assert_eq!(parse_float_cell("-inf"), f64::NEG_INFINITY);
    }
}
