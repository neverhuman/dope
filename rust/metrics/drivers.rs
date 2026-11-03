//! Grouped-driver agreement primitives and stable evaluator facade.
use super::{MetricError, MetricResult, require, sum_checked};
use serde_json::{Value, json};

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

/// Informative grouped-rank successor. Real ties do not manufacture evidence.
/// Synthetic ties on strict real-order pairs receive half credit; no strict
/// real-order pairs is an explicit null/uninformative result.
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

mod evaluation;
pub use evaluation::evaluate;

#[cfg(test)]
mod rank_successor_tests {
    use super::super::vector;
    use super::*;

    #[test]
    fn rank_successor_shared_analytic_cases() {
        let cases: Value = serde_json::from_str(include_str!(
            "../../tests/golden/drivers/informative-v1/rank-cases.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            let real = vector(&case["real"], true).unwrap();
            let syn = vector(&case["synthetic"], true).unwrap();
            assert_eq!(
                rank_agreement_informative(&real, &syn).unwrap(),
                case["expected"],
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn rank_successor_preserves_legacy_and_refuses_invalid() {
        assert_eq!(rank_agreement(&[2., 2.], &[2., 2.]), 1.);
        for (r, s) in [
            (vec![], vec![]),
            (vec![1.], vec![1., 2.]),
            (vec![f64::NAN], vec![1.]),
            (vec![-1.], vec![1.]),
        ] {
            assert_eq!(
                rank_agreement_informative(&r, &s).unwrap_err().reason,
                "invalid_rank_evidence"
            );
        }
        assert_eq!(
            evaluate(
                &json!({"protocol_version": "driver-rank-informative-v1", "importance_unit": "columns"})
            )["reason"],
            "grouped_importance_required"
        );
    }

    #[test]
    fn rank_successor_reuses_simultaneous_bound_contract() {
        let case: Value = serde_json::from_str(include_str!(
            "../../tests/golden/drivers/v2/v2_additive.json"
        ))
        .unwrap();
        assert_eq!(evaluate(&case["input"]), case["expected"]);
        let mut input = case["input"].clone();
        input["protocol_version"] = json!("driver-rank-informative-v1");
        input["importance_unit"] = json!("declared_group");
        let width = input["real_importances"].as_array().unwrap().len();
        input["group_ids"] = json!((0..width).collect::<Vec<_>>());
        assert_eq!(evaluate(&input)["status"], "complete");
        input["inference"]["bound_method"] = json!("ratio_only");
        assert_eq!(evaluate(&input)["reason"], "non_simultaneous_driver_bounds");
    }
}
