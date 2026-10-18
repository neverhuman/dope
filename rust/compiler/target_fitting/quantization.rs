fn quantize(value: f32, bits: u8, unit: bool) -> f32 {
    let levels = ((1u32 << bits.min(16)) - 1) as f32;
    if unit {
        (value.clamp(0.0, 1.0) * levels).round() / levels
    } else {
        let step = 2.0f32.powi(-(i32::from(bits) - 2));
        (value / step).round() * step
    }
}

fn quantize_marginal(marginal: &mut Marginal, bits: u8) {
    match marginal {
        Marginal::Constant { value } => *value = quantize(*value, bits, true),
        Marginal::Bernoulli { probability } => *probability = quantize(*probability, bits, true),
        Marginal::Grid {
            values,
            probabilities,
        } => {
            values
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, true));
            probabilities.iter_mut().for_each(|value| {
                let positive = *value > 0.0;
                *value = quantize(*value, bits, true);
                if positive && *value == 0.0 {
                    *value = 1.0 / ((1u32 << bits.min(16)) - 1) as f32;
                }
            });
        }
        Marginal::QuantileSpline { values } => values
            .iter_mut()
            .for_each(|value| *value = quantize(*value, bits, true)),
        Marginal::Beta { alpha, beta } => {
            *alpha = quantize(*alpha, bits, false).max(1e-4);
            *beta = quantize(*beta, bits, false).max(1e-4);
        }
        Marginal::Gaussian { mean, sigma } => {
            *mean = quantize(*mean, bits, true);
            *sigma = quantize(*sigma, bits, false).max(1e-4);
        }
        Marginal::Histogram {
            edges,
            probabilities,
        } => {
            edges
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, true));
            probabilities.iter_mut().for_each(|value| {
                let positive = *value > 0.0;
                *value = quantize(*value, bits, true);
                if positive && *value == 0.0 {
                    *value = 1.0 / ((1u32 << bits.min(16)) - 1) as f32;
                }
            });
        }
        Marginal::ZeroInflated {
            point,
            probability,
            base,
        } => {
            *point = quantize(*point, bits, true);
            *probability = quantize(*probability, bits, true);
            quantize_marginal(base, bits);
        }
    }
}

fn quantize_edges(edges: &mut [CopulaEdge], bits: u8) {
    edges.iter_mut().for_each(|edge| {
        edge.correlation = quantize(edge.correlation, bits, false).clamp(-0.995, 0.995)
    });
}

fn quantize_strict_unit(values: &mut [f32], bits: u8) {
    let maximum = (1u32 << bits.min(16)) - 1;
    debug_assert!(values.len() <= maximum as usize + 1);
    let length = values.len();
    for (index, value) in values.iter_mut().enumerate() {
        let lower = index as u32;
        let upper = maximum - (length - index - 1) as u32;
        let level = (*value * maximum as f32).round() as u32;
        *value = level.clamp(lower, upper) as f32 / maximum as f32;
    }
    for index in 1..values.len() {
        let minimum = values[index - 1] + 1.0 / maximum as f32;
        values[index] = values[index].max(minimum);
    }
}

fn quantize_dependence(dependence: &mut Dependence, bits: u8) {
    match dependence {
        Dependence::Independent => {}
        Dependence::ChowLiu { edges, .. } | Dependence::SparseGraph { edges } => {
            quantize_edges(edges, bits)
        }
        Dependence::GaussianCopula { cholesky } => cholesky
            .iter_mut()
            .for_each(|value| *value = quantize(*value, bits, false).clamp(-1.0, 1.0)),
        Dependence::Poet {
            loadings,
            residual_edges,
            ..
        } => {
            loadings.iter_mut().for_each(|loading| {
                loading.loading = quantize(loading.loading, bits, false).clamp(-1.0, 1.0)
            });
            quantize_edges(residual_edges, bits);
        }
        Dependence::Vine { edges } => edges.iter_mut().for_each(|edge| {
            edge.parameter = quantize(edge.parameter, bits, false).clamp(-0.995, 0.995)
        }),
        Dependence::Mixture {
            weights,
            components,
        } => {
            weights
                .iter_mut()
                .for_each(|weight| *weight = quantize(*weight, bits, true));
            components
                .iter_mut()
                .for_each(|component| quantize_dependence(component, bits));
        }
        Dependence::Triangular { terms } => terms.iter_mut().for_each(|term| {
            term.coefficient = quantize(term.coefficient, bits, false).clamp(-0.995, 0.995)
        }),
    }
}

fn quantize_additive(terms: &mut [crate::model::AdditiveTerm], bits: u8) {
    for term in terms {
        quantize_strict_unit(&mut term.knots, bits);
        term.values
            .iter_mut()
            .for_each(|value| *value = quantize(*value, bits, false));
    }
}

#[cfg(test)]
mod precision_allocation_pilot {
    use super::quantize;
    use serde_json::Value;

    fn values(case: &Value, key: &str) -> Vec<f64> {
        case[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    }

    fn allocate(scores: &[f64]) -> Vec<u8> {
        let mut bits = vec![4; scores.len()];
        let nominal = scores.len() * 8;
        while bits.iter().map(|v| usize::from(*v)).sum::<usize>() < nominal {
            let selected = (0..scores.len()).filter(|&i| bits[i] < 16)
                .max_by(|&a, &b| (scores[a] / f64::from(bits[a] - 3))
                    .total_cmp(&(scores[b] / f64::from(bits[b] - 3))))
                .unwrap();
            bits[selected] += 1;
        }
        bits
    }

    fn weighted_error(coefficients: &[f64], sensitivity: &[f64], bits: &[u8]) -> f64 {
        coefficients.iter().zip(sensitivity).zip(bits).map(|((&value, &weight), &precision)| {
            let rounded = f64::from(quantize(value as f32, precision, false));
            (value - rounded).powi(2) * weight.powi(2)
        }).sum()
    }

    #[test]
    fn preregistered_precision_pilot_compares_fixed_and_equal_nominal_cost_allocations() {
        let manifest: Value = serde_json::from_str(include_str!(
            "../../../research/transfer/ablations/precision.json"
        )).unwrap();
        let pilot = &manifest["source_model_holdouts"]["pilot"];
        assert_eq!(pilot["qualification"], "fixture_only_not_a_source_or_model_holdout");
        assert_eq!(manifest["production_wiring"], "disabled_until_all_admission_gates_pass");
        let coefficients = values(pilot, "coefficients");
        let se = values(pilot, "standard_errors");
        let fisher = values(pilot, "fisher_information");
        let sensitivity = values(pilot, "measured_sensitivity");
        let bootstrap = pilot["bootstrap_sensitivity_draws"].as_array().unwrap();
        assert_eq!(coefficients.len(), 6);
        assert_eq!(coefficients.len(), se.len());
        assert_eq!(coefficients.len(), fisher.len());
        assert_eq!(coefficients.len(), sensitivity.len());
        let nominal_bits = coefficients.len() * 8;
        let mut outcomes = serde_json::Map::new();
        for fixed in [4u8, 8, 16] {
            let bits = vec![fixed; coefficients.len()];
            outcomes.insert(format!("fixed_q{fixed}"), serde_json::json!({
                "nominal_bits": bits.iter().map(|v| u64::from(*v)).sum::<u64>(),
                "sensitivity_weighted_squared_error": weighted_error(&coefficients, &sensitivity, &bits),
                "rare_weak_coefficient_erased": quantize(coefficients[5] as f32, fixed, false) == 0.0
            }));
        }
        let bootstrap_mean: Vec<f64> = (0..coefficients.len()).map(|i| {
            bootstrap.iter().map(|draw| draw[i].as_f64().unwrap()).sum::<f64>() / bootstrap.len() as f64
        }).collect();
        let strategies = [
            ("delta_kappa_se_kappa_1_equal_nominal_bits", coefficients.iter().zip(&se).map(|(c, e)| c.abs() / e).collect::<Vec<_>>()),
            ("fisher_information_equal_nominal_bits", sensitivity.iter().zip(&fisher).map(|(s, i)| s * s * i).collect()),
            ("bootstrap_sensitivity_equal_nominal_bits", bootstrap_mean.iter().map(|s| s * s).collect()),
            ("measured_sensitivity_equal_nominal_bits", sensitivity.iter().map(|s| s * s).collect()),
        ];
        for (name, scores) in strategies {
            let bits = allocate(&scores);
            assert_eq!(bits.iter().map(|v| usize::from(*v)).sum::<usize>(), nominal_bits);
            outcomes.insert(name.into(), serde_json::json!({
                "nominal_bits": nominal_bits,
                "sensitivity_weighted_squared_error": weighted_error(&coefficients, &sensitivity, &bits),
                "bits_by_coefficient": bits,
                "rare_weak_coefficient_erased": quantize(coefficients[5] as f32, 4, false) == 0.0
            }));
        }
        eprintln!("R02 pilot outcomes: {}", Value::Object(outcomes));
        assert_eq!(manifest["cost_and_bytes"]["pilot_nominal_coefficient_bit_budget"], nominal_bits);
        assert!(manifest["cost_and_bytes"]["complete_encoded_bytes_required_for_admission"].as_bool().unwrap());
    }
}
