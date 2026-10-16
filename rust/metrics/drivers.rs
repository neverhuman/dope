//! Evaluator-computed grouped importance consumer; no cross-model SHAP (T-032).
//!
//! Ceilings and stable signs come from independent real-real controls on the
//! same real test rows. Required components are combined by minimum, never a
//! weighted blend.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use super::{
    MetricError, MetricResult, failure_with_status, number, require, sum_checked, unit, vector,
};

pub fn mass_agreement(real: &[f64], synthetic: &[f64]) -> MetricResult<f64> {
    let rtotal = sum_checked(real)?;
    let stotal = sum_checked(synthetic)?;
    require(rtotal > 0.0, "degenerate_real_importance")?;
    if stotal <= 0.0 {
        return Ok(0.0);
    }
    let mut l1 = 0.0_f64;
    for (&r, &s) in real.iter().zip(synthetic.iter()) {
        let diff = (r / rtotal - s / stotal).abs();
        require(diff.is_finite(), "nonfinite_or_invalid_number")?;
        l1 += diff;
        require(l1.is_finite(), "nonfinite_or_invalid_number")?;
    }
    Ok((1.0 - 0.5 * l1).max(0.0))
}

pub fn rank_agreement(real: &[f64], synthetic: &[f64]) -> f64 {
    let mut pairs = 0usize;
    let mut score = 0.0_f64;
    for i in 0..real.len() {
        for j in (i + 1)..real.len() {
            let r = (real[i] > real[j]) as i32 - (real[i] < real[j]) as i32;
            let s = (synthetic[i] > synthetic[j]) as i32 - (synthetic[i] < synthetic[j]) as i32;
            pairs += 1;
            score += if r == s {
                1.0
            } else if r == 0 || s == 0 {
                0.5
            } else {
                0.0
            };
        }
    }
    if pairs > 0 {
        score / (pairs as f64)
    } else {
        1.0
    }
}

fn parse_signs(value: &Value, expected_len: usize) -> MetricResult<Vec<i64>> {
    let arr = value
        .as_array()
        .ok_or(MetricError::new("sign_shape_mismatch"))?;
    require(arr.len() == expected_len, "sign_shape_mismatch")?;
    let mut signs = Vec::with_capacity(arr.len());
    for item in arr {
        let n = item.as_number().ok_or(MetricError::new("invalid_sign"))?;
        require(!n.is_f64(), "invalid_sign")?;
        let s = n.as_i64().ok_or(MetricError::new("invalid_sign"))?;
        require(matches!(s, -1..=1), "invalid_sign")?;
        signs.push(s);
    }
    Ok(signs)
}

pub fn evaluate(data: &Value) -> Value {
    match evaluate_inner(data) {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

fn evaluate_inner(data: &Value) -> MetricResult<Value> {
    let obj = data.as_object().ok_or(MetricError::new("invalid_input"))?;
    let required: Vec<String> = match obj.get("required_components") {
        Some(v) => {
            let arr = v.as_array().ok_or(MetricError::new("invalid_components"))?;
            require(!arr.is_empty(), "invalid_components")?;
            let mut seen = BTreeSet::new();
            let mut list = Vec::with_capacity(arr.len());
            for item in arr {
                let s = item
                    .as_str()
                    .ok_or(MetricError::new("invalid_components"))?;
                require(
                    matches!(s, "mass" | "rank" | "sign" | "interaction"),
                    "invalid_components",
                )?;
                require(seen.insert(s.to_owned()), "invalid_components")?;
                list.push(s.to_owned());
            }
            list
        }
        None => vec![
            "mass".to_owned(),
            "rank".to_owned(),
            "sign".to_owned(),
            "interaction".to_owned(),
        ],
    };

    if [
        "real_importances",
        "synthetic_importances",
        "real_real_ceiling",
    ]
    .iter()
    .any(|k| obj.get(*k).is_none_or(Value::is_null))
    {
        return Err(MetricError::with_status(
            "missing_driver_component",
            "incomplete",
        ));
    }

    let real = vector(&obj["real_importances"], true)?;
    let syn = vector(&obj["synthetic_importances"], true)?;
    require(real.len() == syn.len(), "importance_shape_mismatch")?;
    if sum_checked(&real)? <= 0.0 {
        return Err(MetricError::with_status(
            "degenerate_real_importance",
            "uninformative",
        ));
    }

    let mut raw: BTreeMap<String, f64> = BTreeMap::new();
    raw.insert("mass".to_owned(), mass_agreement(&real, &syn)?);
    raw.insert("rank".to_owned(), rank_agreement(&real, &syn));

    if required.iter().any(|c| c == "sign") {
        if obj.get("real_signs").is_none_or(Value::is_null)
            || obj.get("synthetic_signs").is_none_or(Value::is_null)
        {
            return Err(MetricError::with_status(
                "missing_driver_component",
                "incomplete",
            ));
        }
        let rs = parse_signs(&obj["real_signs"], real.len())?;
        let ss = parse_signs(&obj["synthetic_signs"], real.len())?;
        let active: Vec<usize> = (0..real.len())
            .filter(|&i| rs[i] != 0 && real[i] > 0.0)
            .collect();
        if active.is_empty() {
            return Err(MetricError::with_status(
                "no_stable_direction",
                "uninformative",
            ));
        }
        let matches = active
            .iter()
            .filter(|&&i| rs[i] == ss[i] && syn[i] > 0.0)
            .count();
        raw.insert("sign".to_owned(), (matches as f64) / (active.len() as f64));
    }

    if required.iter().any(|c| c == "interaction") {
        if obj.get("real_interactions").is_none_or(Value::is_null)
            || obj.get("synthetic_interactions").is_none_or(Value::is_null)
        {
            return Err(MetricError::with_status(
                "missing_driver_component",
                "incomplete",
            ));
        }
        let ri = vector(&obj["real_interactions"], true)?;
        let si = vector(&obj["synthetic_interactions"], true)?;
        require(ri.len() == si.len(), "interaction_shape_mismatch")?;
        let rt = sum_checked(&ri)?;
        let st = sum_checked(&si)?;
        if rt <= 0.0 {
            return Err(MetricError::with_status(
                "no_stable_interaction",
                "uninformative",
            ));
        }
        let strength = rt.min(st) / rt.max(st);
        let inter = mass_agreement(&ri, &si)? * strength;
        raw.insert("interaction".to_owned(), inter);
    }

    let ceilings = obj["real_real_ceiling"]
        .as_object()
        .ok_or(MetricError::new("invalid_ceiling"))?;
    let floor = match obj.get("stability_floor") {
        Some(v) if !v.is_null() => unit(v)?,
        _ => 0.70,
    };

    let mut raw_out = Map::new();
    let mut normalized_out = Map::new();
    let mut min_score = f64::INFINITY;

    for comp in &required {
        let ceil_val = match ceilings.get(comp) {
            None | Some(Value::Null) => {
                return Err(MetricError::with_status(
                    "missing_driver_component",
                    "incomplete",
                ));
            }
            Some(v) => v,
        };
        let ceiling = unit(ceil_val)?;
        if ceiling <= 0.0 || ceiling < floor {
            return Err(MetricError::with_status(
                "unstable_real_drivers",
                "uninformative",
            ));
        }
        let r_val = raw[comp];
        let norm_val = (r_val / ceiling).clamp(0.0, 1.0);
        min_score = min_score.min(norm_val);
        raw_out.insert(comp.clone(), json!(r_val));
        normalized_out.insert(comp.clone(), json!(norm_val));
    }

    let threshold = match obj.get("driver_threshold") {
        Some(v) if !v.is_null() => unit(v)?,
        _ => 0.80,
    };
    let _ = number(&json!(min_score))?;

    Ok(json!({
        "status": "complete",
        "reason": Value::Null,
        "raw_components": Value::Object(raw_out),
        "normalized_components": Value::Object(normalized_out),
        "driver_fidelity": min_score,
        "driver_pass": min_score >= threshold,
    }))
}
