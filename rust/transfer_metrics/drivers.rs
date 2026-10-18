//! Paired driver endpoints. Inputs are evaluator-private, aligned perturbation
//! summaries on the same query rows, groups and draws. Negative noisy loss
//! increases are truncated to zero before normalization. Inference belongs to
//! C02; a descriptive ratio alone never qualifies D80.

use super::{MetricError, finite};

pub fn positive_importances(values: &[f64]) -> Result<Vec<f64>, MetricError> {
    if values.is_empty() {
        return Err(MetricError::InvalidEvidence);
    }
    values.iter().map(|v| Ok(finite(*v)?.max(0.0))).collect()
}

pub fn mass_agreement(real: &[f64], other: &[f64]) -> Result<f64, MetricError> {
    if real.len() != other.len() || real.is_empty() {
        return Err(MetricError::InvalidEvidence);
    }
    let real = positive_importances(real)?;
    let other = positive_importances(other)?;
    let rt = finite(real.iter().sum())?;
    let ot = finite(other.iter().sum())?;
    if rt <= 0.0 {
        return Err(MetricError::InvalidEvidence);
    }
    if ot == 0.0 {
        return Ok(0.0);
    }
    let l1 = finite(
        real.iter()
            .zip(&other)
            .map(|(r, o)| (r / rt - o / ot).abs())
            .sum(),
    )?;
    Ok((1.0 - 0.5 * l1).clamp(0.0, 1.0))
}

/// Arms must share query/group/perturbation identities before calling this.
pub fn paired_margin(real_synthetic: f64, real_real: f64) -> Result<f64, MetricError> {
    if !(0.0..=1.0).contains(&finite(real_synthetic)?) || !(0.0..=1.0).contains(&finite(real_real)?)
    {
        return Err(MetricError::InvalidEvidence);
    }
    finite(real_synthetic - 0.8 * real_real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{drivers::evaluate, strict_parse_json, values_close};
    use serde_json::{Value, json};

    #[test]
    fn paired_d80_boundary_and_noisy_mass() {
        let rs = mass_agreement(&[8.0, 2.0, -1.0], &[6.0, 4.0, -5.0]).unwrap();
        let rr = mass_agreement(&[8.0, 2.0, 0.0], &[8.0, 2.0, 0.0]).unwrap();
        assert!((rs - 0.8).abs() < 1e-12);
        assert!(paired_margin(rs, rr).unwrap().abs() < 1e-12);
        assert_eq!(mass_agreement(&[1.0], &[0.0]), Ok(0.0));
        for (r, s) in [
            (vec![], vec![]),
            (vec![1.0], vec![1.0, 2.0]),
            (vec![0.0], vec![1.0]),
            (vec![f64::NAN], vec![1.0]),
        ] {
            assert!(mass_agreement(&r, &s).is_err());
        }
        assert!(paired_margin(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn paired_failure_corpus_and_explicit_historical_dispatch() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/drivers");
        let mut candidates = 0;
        let paths = std::fs::read_dir(&root)
            .unwrap()
            .chain(std::fs::read_dir(root.join("v2")).unwrap());
        for path in paths.flatten().map(|e| e.path()) {
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }
            let case = strict_parse_json(&std::fs::read_to_string(path).unwrap()).unwrap();
            let mut input = case["input"].clone();
            let result = evaluate(&input);
            assert!(
                values_close(&result, &case["expected"], 1e-12),
                "{}: {result}",
                case["case_id"]
            );
            if input["protocol_version"] == "v2" {
                candidates += 1;
                assert_eq!(
                    crate::metrics::golden::run_case(&case).unwrap_err().reason,
                    "invalid_golden_schema"
                );
            } else {
                input["protocol_version"] = json!("v1");
                assert!(values_close(&evaluate(&input), &case["expected"], 1e-12));
            }
        }
        assert!(candidates >= 34);
    }

    #[test]
    fn every_simultaneous_coordinate_is_required_and_errors_are_source_free() {
        let case: Value = serde_json::from_str(include_str!(
            "../../tests/golden/drivers/v2/v2_additive.json"
        ))
        .unwrap();
        let data = &case["input"];
        for key in data["inference"]["bounds"].as_object().unwrap().keys() {
            let mut changed = data.clone();
            changed["inference"]["bounds"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert_eq!(evaluate(&changed)["status"], "incomplete");
        }
        for key in [
            "real_importances",
            "synthetic_importances",
            "real_control_importances",
            "groups_sha256",
        ] {
            let mut changed = data.clone();
            changed[key] = json!("CANARY_PRIVATE_HEADER_extremum_197");
            let result = evaluate(&changed);
            assert_eq!(result["status"], "failed");
            assert!(!result.to_string().contains("CANARY"));
        }
        for version in [Value::Null, json!(false), json!(2), json!("v3")] {
            assert_eq!(
                evaluate(&json!({"protocol_version": version}))["reason"],
                "unsupported_driver_protocol"
            );
        }
    }
}
