//! Product port of the canonical C02 reducer at DOPE 59b1e051.
//! This preserves full-family simultaneous bounds, positive floors and pins;
//! descriptive ratios never substitute for those bounds. Shared legacy-v2
//! golden fixtures verify the port. The informative-rank successor is explicit.
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricError {
    pub reason: &'static str,
    pub status: &'static str,
}
impl MetricError {
    pub const fn new(reason: &'static str) -> Self {
        Self {
            reason,
            status: "failed",
        }
    }
    pub const fn with_status(reason: &'static str, status: &'static str) -> Self {
        Self { reason, status }
    }
}
pub type MetricResult<T> = Result<T, MetricError>;
const MAX_SAFE_INTEGER: i64 = (1i64 << 53) - 1;
pub fn require(condition: bool, reason: &'static str) -> MetricResult<()> {
    if condition {
        Ok(())
    } else {
        Err(MetricError::new(reason))
    }
}

pub fn failure_with_status(reason: &str, status: &str) -> Value {
    json!({
        "status": status,
        "reason": reason,
    })
}

pub fn number(value: &Value) -> MetricResult<f64> {
    match value {
        Value::Number(n) => {
            if !n.is_f64() {
                let i = n
                    .as_i64()
                    .ok_or(MetricError::new("nonfinite_or_invalid_number"))?;
                require(
                    (-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&i),
                    "nonfinite_or_invalid_number",
                )?;
            }
            let v = n
                .as_f64()
                .ok_or(MetricError::new("nonfinite_or_invalid_number"))?;
            require(v.is_finite(), "nonfinite_or_invalid_number")?;
            Ok(v)
        }
        _ => Err(MetricError::new("nonfinite_or_invalid_number")),
    }
}

pub fn unit(value: &Value) -> MetricResult<f64> {
    let v = number(value)?;
    require((0.0..=1.0).contains(&v), "invalid_probability")?;
    Ok(v)
}

pub fn vector(value: &Value, nonnegative: bool) -> MetricResult<Vec<f64>> {
    let arr = value.as_array().ok_or(MetricError::new("invalid_vector"))?;
    require(!arr.is_empty(), "invalid_vector")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let v = number(item)?;
        out.push(v);
    }
    require(
        !nonnegative || out.iter().all(|v| *v >= 0.0),
        "negative_value",
    )?;
    Ok(out)
}

pub fn sum_checked(values: &[f64]) -> MetricResult<f64> {
    let mut total = 0.0_f64;
    for &v in values {
        require(v.is_finite(), "nonfinite_or_invalid_number")?;
        total += v;
        require(total.is_finite(), "nonfinite_or_invalid_number")?;
    }
    Ok(total)
}

fn positive_importances(values: &[f64]) -> MetricResult<Vec<f64>> {
    require(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "invalid_driver_evidence",
    )?;
    Ok(values.iter().map(|v| v.max(0.0)).collect())
}
fn paired_margin(real_synthetic: f64, real_real: f64) -> MetricResult<f64> {
    require(
        (0.0..=1.0).contains(&real_synthetic) && (0.0..=1.0).contains(&real_real),
        "invalid_driver_evidence",
    )?;
    Ok(real_synthetic - 0.8 * real_real)
}
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

pub fn rank_agreement_informative(real: &[f64], synthetic: &[f64]) -> MetricResult<Value> {
    require(
        !real.is_empty() && real.len() == synthetic.len(),
        "invalid_rank_evidence",
    )?;
    require(
        real.iter()
            .chain(synthetic)
            .all(|v| v.is_finite() && *v >= 0.0),
        "invalid_rank_evidence",
    )?;
    let total = real
        .len()
        .checked_mul(real.len() - 1)
        .ok_or(MetricError::new("invalid_rank_evidence"))?
        / 2;
    let (mut pairs, mut score) = (0usize, 0.0_f64);
    for i in 0..real.len() {
        for j in (i + 1)..real.len() {
            let r = (real[i] > real[j]) as i32 - (real[i] < real[j]) as i32;
            if r == 0 {
                continue;
            }
            let s = (synthetic[i] > synthetic[j]) as i32 - (synthetic[i] < synthetic[j]) as i32;
            pairs += 1;
            score += if r == s {
                1.0
            } else if s == 0 {
                0.5
            } else {
                0.0
            };
        }
    }
    Ok(
        json!({"status": if pairs > 0 { "complete" } else { "uninformative" },
              "reason": if pairs > 0 { Value::Null } else { json!("no_informative_rank_pairs") },
              "value": if pairs > 0 { json!(score / pairs as f64) } else { Value::Null },
              "informative_pairs": pairs, "total_pairs": total}),
    )
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

fn evaluate_informative(data: &Value) -> MetricResult<Value> {
    require(
        data["importance_unit"] == "declared_group",
        "grouped_importance_required",
    )?;
    let groups = data["group_ids"]
        .as_array()
        .ok_or(MetricError::new("invalid_driver_groups"))?;
    let mut seen = BTreeSet::new();
    require(!groups.is_empty(), "invalid_driver_groups")?;
    for group in groups {
        let id = group
            .as_u64()
            .ok_or(MetricError::new("invalid_driver_groups"))?;
        require(
            id <= MAX_SAFE_INTEGER as u64 && seen.insert(id),
            "invalid_driver_groups",
        )?;
    }
    require(
        data["real_importances"]
            .as_array()
            .is_some_and(|a| a.len() == groups.len()),
        "importance_shape_mismatch",
    )?;
    evaluate_v2_with_rank(data, true)
}

fn evaluate_v2_with_rank(data: &Value, informative_rank: bool) -> MetricResult<Value> {
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
        let numeric = |result: MetricResult<f64>| {
            result.map_err(|_| MetricError::new("invalid_driver_evidence"))
        };
        let mut rank_diagnostics = None;
        let (rs, rr) = match name {
            "rank" if informative_rank => {
                let rs = rank_agreement_informative(&r, &s)?;
                let rr = rank_agreement_informative(&r, &c)?;
                let diagnostics = json!({"real_synthetic": rs, "real_control": rr});
                if diagnostics["real_synthetic"]["status"] != "complete"
                    || diagnostics["real_control"]["status"] != "complete"
                {
                    return Ok(
                        json!({"status": "uninformative", "reason": "no_informative_rank_pairs",
                        "driver_fidelity": Value::Null, "driver_pass": Value::Null,
                        "rank_diagnostics": diagnostics}),
                    );
                }
                let scores = (
                    number(&diagnostics["real_synthetic"]["value"])?,
                    number(&diagnostics["real_control"]["value"])?,
                );
                rank_diagnostics = Some(diagnostics);
                scores
            }
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
        let mut component = json!({"a_rs": rs, "a_rr": rr, "margin": margin, "descriptive_ratio": ratio, "lcbs": lcbs});
        if let Some(diagnostics) = rank_diagnostics {
            component["rank_diagnostics"] = diagnostics;
        }
        components.insert(name.to_owned(), component);
    }
    Ok(
        json!({"status": "complete", "reason": Value::Null, "components": components, "driver_fidelity": fidelity, "driver_pass": pass}),
    )
}

/// Explicit legacy C02 replay used for golden parity, never silently promoted.
pub fn evaluate_c02_legacy_v2(data: &Value) -> Value {
    let mut out = match evaluate_v2_with_rank(data, false) {
        Ok(v) => v,
        Err(e) => failure_with_status(e.reason, e.status),
    };
    out["format"] = json!("driver-result-v1");
    out["protocol_version"] = json!("v2");
    out
}

/// Prospective grouped-rank/C02 endpoint. No default protocol coercion.
pub fn evaluate(data: &Value) -> Value {
    if data["protocol_version"] != "driver-rank-informative-v1" {
        return failure_with_status("unsupported_driver_protocol", "failed");
    }
    let mut out = match evaluate_informative(data) {
        Ok(v) => v,
        Err(e) => failure_with_status(e.reason, e.status),
    };
    out["format"] = json!("driver-result-informative-v1");
    out["protocol_version"] = json!("driver-rank-informative-v1");
    out
}
