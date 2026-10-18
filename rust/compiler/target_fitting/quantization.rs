/// Explicit DPK4 unit-parameter fitting boundary. The returned packet stores
/// actual q4 nibbles or q8 bytes when passed to `codec::dpk4::encode`.
/// Legacy DPK3 encoding and its historical precision ladder remain unchanged.
pub fn quantized_unit_packet(values: &[f32], bits: u8) -> Result<crate::codec::dpk4::Packet> {
    use crate::codec::dpk4::{MAX_VALUES, Packet};

    if !matches!(bits, 4 | 8) {
        return Err(DopeError::Codec("DPK4 unsupported precision".into()));
    }
    if values.len() > MAX_VALUES {
        return Err(DopeError::Codec(
            "DPK4 element count exceeds its bound".into(),
        ));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(DopeError::Codec("DPK4 invalid unit parameter".into()));
    }
    let maximum = ((1u32 << bits) - 1) as f32;
    let levels = values
        .iter()
        .map(|&value| (quantize(value, bits, true) * maximum).round() as u8)
        .collect();
    Ok(if bits == 4 {
        Packet::Q4(levels)
    } else {
        Packet::Q8(levels)
    })
}

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
