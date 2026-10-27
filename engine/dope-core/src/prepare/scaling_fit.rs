//! Column and target fitting helpers for `prepare::scaling`.

use super::np::{clip, linspace, mean1d, sorted, std1d};
use super::scaling::{
    err, CategoricalColumnEncoder, ColumnKind, NumericColumnScaler, TargetValue, TransformKind,
    QUANTILE_GRID,
};
use crate::error::Result;
use std::collections::HashMap;

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
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

#[inline]
pub(crate) fn is_missing_token(s: &str) -> bool {
    matches!(s.len(), 0..=4)
        && (s.is_empty()
            || s == "?"
            || s.eq_ignore_ascii_case("na")
            || s.eq_ignore_ascii_case("nan")
            || s.eq_ignore_ascii_case("null")
            || s.eq_ignore_ascii_case("none"))
}

/// Python `float(s)` on an already-stripped string.
#[inline]
pub(crate) fn py_float(s: &str) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    let b = s.as_bytes();
    let body = match b[0] {
        b'+' | b'-' => &s[1..],
        _ => s,
    };
    if body.is_empty() || matches!(body.as_bytes()[0], b'+' | b'-') {
        return None;
    }
    if !(body.eq_ignore_ascii_case("inf")
        || body.eq_ignore_ascii_case("infinity")
        || body.eq_ignore_ascii_case("nan")
        || body
            .bytes()
            .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-')))
    {
        return None;
    }
    s.parse::<f64>().ok()
}

/// `scaling._parse_float` for a string cell.
#[inline]
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

pub(crate) fn infer_feature_kinds_and_parse(
    rows: &[Vec<String>],
    p: usize,
) -> (Vec<ColumnKind>, Vec<Option<Vec<f64>>>) {
    let n = rows.len();
    let mut kinds = Vec::with_capacity(p);
    let mut parsed_cols = Vec::with_capacity(p);
    for j in 0..p {
        let mut col = Vec::with_capacity(n);
        let mut is_numeric = true;
        for r in rows {
            let s = py_strip(&r[j]);
            if is_missing_token(s) {
                col.push(f64::NAN);
                continue;
            }
            if let Some(v) = py_float(s) {
                col.push(v);
            } else {
                is_numeric = false;
                break;
            }
        }
        if is_numeric {
            kinds.push(ColumnKind::Numeric);
            parsed_cols.push(Some(col));
        } else {
            kinds.push(ColumnKind::Categorical);
            parsed_cols.push(None);
        }
    }
    (kinds, parsed_cols)
}

fn safe_skew_kurt(x: &[f64]) -> (f64, f64) {
    if x.len() < 4 {
        return (0.0, 0.0);
    }
    let m = mean1d(x);
    let s = std1d(x);
    if !s.is_finite() || s <= 1e-12 {
        return (0.0, 0.0);
    }
    let z3: Vec<f64> = x.iter().map(|&v| ((v - m) / s).powf(3.0)).collect();
    let z4: Vec<f64> = x.iter().map(|&v| ((v - m) / s).powf(4.0)).collect();
    let skew = mean1d(&z3);
    let kurt = mean1d(&z4) - 3.0;
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
    let mut uniq = s.clone();
    uniq.dedup();
    let q_vals: [f64; 9] = QUANTILE_GRID.map(|q| super::np::quantile_sorted(&s, q));
    let (skew, kurt) = safe_skew_kurt(&finite);
    if uniq.len() <= 1 {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Constant,
            missing_rate: miss_rate,
            center: uniq[0],
            scale: 1.0,
            post_mean: 0.0,
            post_std: 1.0,
            quantiles: q_vals,
            skewness: 0.0,
            kurtosis: 0.0,
            blend_quantile_weight: 0.35,
        };
    }
    let std_f = std1d(&finite);
    if uniq.len() == 2 {
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::Binary,
            missing_rate: miss_rate,
            center: mean1d(&finite),
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
    let mut abs_sorted = Vec::with_capacity(k);
    let mut left = split;
    let mut right = split;
    while left > 0 && right < k {
        let dl = (s[left - 1] - med).abs();
        let dr = (s[right] - med).abs();
        if dl <= dr {
            abs_sorted.push(dl);
            left -= 1;
        } else {
            abs_sorted.push(dr);
            right += 1;
        }
    }
    while left > 0 {
        abs_sorted.push((s[left - 1] - med).abs());
        left -= 1;
    }
    while right < k {
        abs_sorted.push((s[right] - med).abs());
        right += 1;
    }
    let mad = if k % 2 == 1 {
        abs_sorted[k / 2]
    } else {
        (abs_sorted[k / 2 - 1] + abs_sorted[k / 2]) / 2.0
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
        return NumericColumnScaler {
            slot,
            transform_kind: TransformKind::SignedLog1pHybrid,
            missing_rate: miss_rate,
            center: med,
            scale: robust_scale,
            post_mean: mean1d(&logged),
            post_std: std1d(&logged).max(1e-6),
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
        post_mean: mean1d(&finite),
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
