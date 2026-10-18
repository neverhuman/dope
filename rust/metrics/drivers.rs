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
    match data.get("protocol_version") {
        None | Some(Value::String(_)) if data.get("protocol_version").is_none_or(|v| v == "v1") => {
        }
        Some(v) if v == "v2" => {
            let mut out =
                evaluate_v2(data).unwrap_or_else(|e| failure_with_status(e.reason, e.status));
            out["format"] = json!("driver-result-v1");
            out["protocol_version"] = json!("v2");
            return out;
        }
        _ => return failure_with_status("unsupported_driver_protocol", "failed"),
    }
    match evaluate_inner(data) {
        Ok(val) => val,
        Err(err) => failure_with_status(err.reason, err.status),
    }
}

/// Candidate reducer consumes evaluator-owned full-family C02 bounds. The
/// estimates are checked against these paired endpoints. This is fixture
/// qualification only; D05/D19 calibration and release authorization are separate.
fn evaluate_v2(data: &Value) -> MetricResult<Value> {
    use crate::transfer_metrics::drivers::{mass_agreement, paired_margin, positive_importances};
    let missing = || MetricError::with_status("missing_driver_component", "incomplete");
    let obj = data.as_object().ok_or(MetricError::new("invalid_input"))?;
    let get = |key: &str| obj.get(key).filter(|v| !v.is_null());
    require(
        get("perturbation_status").is_none_or(Value::is_string),
        "invalid_perturbation_status",
    )?;
    match obj.get("perturbation_status").and_then(Value::as_str) {
        Some("complete") => {}
        Some("failed") => return Err(MetricError::new("perturbation_failed")),
        Some("timed_out") => {
            return Err(MetricError::with_status(
                "perturbation_timed_out",
                "timed_out",
            ));
        }
        Some("unsupported") => {
            return Err(MetricError::with_status(
                "perturbation_unsupported",
                "unsupported",
            ));
        }
        None | Some("missing") => {
            return Err(MetricError::with_status(
                "missing_perturbation",
                "incomplete",
            ));
        }
        _ => return Err(MetricError::new("invalid_perturbation_status")),
    }
    // Common evaluator pins bind the aligned arms; never project their values.
    for key in [
        "query_sha256",
        "groups_sha256",
        "perturbation_sha256",
        "prepared_view_sha256",
    ] {
        let pin = get(key)
            .ok_or_else(missing)?
            .as_str()
            .ok_or(MetricError::new("invalid_driver_pin"))?;
        require(
            pin.len() == 64
                && pin
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid_driver_pin",
        )?;
    }
    let list = get("required_components")
        .ok_or_else(missing)?
        .as_array()
        .ok_or(MetricError::new("invalid_components"))?;
    let mut required = BTreeSet::new();
    for item in list {
        let name = item
            .as_str()
            .ok_or(MetricError::new("invalid_components"))?;
        require(
            matches!(name, "mass" | "rank" | "sign" | "interaction") && required.insert(name),
            "invalid_components",
        )?;
    }
    require(required.contains("mass"), "invalid_components")?;
    let floor = |key| -> MetricResult<f64> {
        let value = number(get(key).ok_or_else(missing)?)?;
        require(value > 0.0, "invalid_driver_floor")?;
        Ok(value)
    };
    let stability_floor = floor("stability_floor")?;
    let importance_floor = floor("importance_floor")?;
    let strength_floor = floor("synthetic_strength_floor")?;
    require(
        stability_floor <= 1.0 && strength_floor <= 1.0,
        "invalid_driver_floor",
    )?;
    let read = |key: &str| -> MetricResult<Vec<f64>> {
        let values = vector(get(key).ok_or_else(missing)?, false)?;
        positive_importances(&values).map_err(|_| MetricError::new("invalid_driver_evidence"))
    };
    let real = read("real_importances")?;
    let syn = read("synthetic_importances")?;
    let control = read("real_control_importances")?;
    require(
        real.len() == syn.len() && real.len() == control.len(),
        "importance_shape_mismatch",
    )?;
    let inference = get("inference")
        .ok_or_else(|| MetricError::with_status("missing_simultaneous_bounds", "incomplete"))?;
    require(inference.is_object(), "invalid_driver_bounds")?;
    if inference["status"] != "complete" {
        return Err(MetricError::with_status(
            "incomplete_driver_inference",
            "incomplete",
        ));
    }
    require(
        inference["bound_method"] == "full_family_max_t",
        "non_simultaneous_driver_bounds",
    )?;
    let bounds = inference["bounds"]
        .as_object()
        .ok_or(MetricError::new("invalid_driver_bounds"))?;
    let mut components = Map::new();
    let mut pass = true;
    let mut fidelity = 1.0_f64;
    for name in required {
        let (r, s, c) = if name == "interaction" {
            (
                read("real_interactions")?,
                read("synthetic_interactions")?,
                read("real_control_interactions")?,
            )
        } else {
            (real.clone(), syn.clone(), control.clone())
        };
        require(
            r.len() == s.len() && r.len() == c.len(),
            "importance_shape_mismatch",
        )?;
        let (rt, st, ct) = (sum_checked(&r)?, sum_checked(&s)?, sum_checked(&c)?);
        if rt < importance_floor || ct < importance_floor {
            return Err(MetricError::with_status(
                "degenerate_real_importance",
                "uninformative",
            ));
        }
        let numeric = |result: Result<f64, crate::transfer_metrics::MetricError>| {
            result.map_err(|_| MetricError::new("invalid_driver_evidence"))
        };
        let (rs, rr) = match name {
            "rank" => (rank_agreement(&r, &s), rank_agreement(&r, &c)),
            "sign" => {
                let read_sign = |key| parse_signs(get(key).ok_or_else(missing)?, r.len());
                let (a, b, d) = (
                    read_sign("real_signs")?,
                    read_sign("synthetic_signs")?,
                    read_sign("real_control_signs")?,
                );
                let active: Vec<_> = (0..r.len()).filter(|&i| a[i] != 0 && r[i] > 0.0).collect();
                if active.is_empty() {
                    return Err(MetricError::with_status(
                        "no_stable_direction",
                        "uninformative",
                    ));
                }
                let agreement = |signs: &[i64], imp: &[f64]| {
                    active
                        .iter()
                        .filter(|&&i| a[i] == signs[i] && imp[i] > 0.0)
                        .count() as f64
                        / active.len() as f64
                };
                (agreement(&b, &s), agreement(&d, &c))
            }
            _ => (
                numeric(mass_agreement(&r, &s))?,
                numeric(mass_agreement(&r, &c))?,
            ),
        };
        if rr < stability_floor {
            return Err(MetricError::with_status(
                "unstable_real_drivers",
                "uninformative",
            ));
        }
        let margin = numeric(paired_margin(rs, rr))?;
        let strength = number(&json!(st / rt))?;
        let endpoints = [
            ("margin", margin),
            ("stability", rr),
            ("real_strength", rt),
            ("control_strength", ct),
            ("synthetic_strength", strength),
        ];
        let mut lcbs = Map::new();
        for (endpoint, estimate) in endpoints {
            let coordinate = format!("{name}.{endpoint}");
            let bound = bounds.get(&coordinate).ok_or_else(|| {
                MetricError::with_status("missing_simultaneous_bounds", "incomplete")
            })?;
            require(bound.is_object(), "invalid_driver_bounds")?;
            require(
                (number(&bound["estimate"])? - estimate).abs() <= 1e-12,
                "driver_bound_estimate_mismatch",
            )?;
            let lower = number(&bound["lower"])?;
            require(lower <= estimate + 1e-12, "invalid_driver_bounds")?;
            lcbs.insert(endpoint.to_owned(), json!(lower));
        }
        pass &= lcbs["margin"].as_f64().unwrap() >= 0.0
            && lcbs["stability"].as_f64().unwrap() >= stability_floor
            && lcbs["real_strength"].as_f64().unwrap() >= importance_floor
            && lcbs["control_strength"].as_f64().unwrap() >= importance_floor
            && lcbs["synthetic_strength"].as_f64().unwrap() >= strength_floor;
        let ratio = (rs / rr).clamp(0.0, 1.0);
        fidelity = fidelity.min(ratio);
        components.insert(name.to_owned(), json!({"a_rs": rs, "a_rr": rr, "margin": margin, "descriptive_ratio": ratio, "lcbs": lcbs}));
    }
    Ok(
        json!({"status": "complete", "reason": Value::Null, "components": components, "driver_fidelity": fidelity, "driver_pass": pass}),
    )
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

    let mut parsed_ceilings: BTreeMap<String, f64> = BTreeMap::new();
    for comp in &required {
        if let Some(v) = ceilings.get(comp)
            && !v.is_null()
        {
            parsed_ceilings.insert(comp.clone(), unit(v)?);
        }
    }
    for &ceiling in parsed_ceilings.values() {
        if ceiling <= 0.0 || ceiling < floor {
            return Err(MetricError::with_status(
                "unstable_real_drivers",
                "uninformative",
            ));
        }
    }
    if required
        .iter()
        .any(|comp| ceilings.get(comp).is_none_or(Value::is_null))
    {
        return Err(MetricError::with_status(
            "missing_driver_component",
            "incomplete",
        ));
    }

    let mut raw_out = Map::new();
    let mut normalized_out = Map::new();
    let mut min_score = f64::INFINITY;

    for comp in &required {
        let ceiling = parsed_ceilings[comp];
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
