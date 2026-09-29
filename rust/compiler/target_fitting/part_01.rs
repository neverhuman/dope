
fn fit_gam_target(table: &Table, completed: &[Vec<f32>], screen: &[(usize, f32)]) -> Target {
    const KNOTS: usize = 9;
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let base_rate =
        (table.target.iter().sum::<f32>() / table.rows.max(1) as f32).clamp(1e-6, 1.0 - 1e-6);
    let intercept = if logistic {
        (base_rate / (1.0 - base_rate)).ln()
    } else {
        base_rate
    };
    let selected = spline_features(completed, screen, 12);
    let knots: Vec<_> = selected
        .iter()
        .map(|feature| feature_knots(&completed[*feature], KNOTS))
        .collect();
    let mut effects = vec![vec![0.0f32; KNOTS]; selected.len()];
    let mut linear = vec![intercept; table.rows];
    for _ in 0..10 {
        for (term, &feature) in selected.iter().enumerate() {
            let previous_effect = effects[term].clone();
            for (row, value) in linear.iter_mut().enumerate() {
                *value -= previous_effect[nearest_knot(&knots[term], completed[feature][row])];
            }
            let mut sums = [0.0f64; KNOTS];
            let mut weights = [0.0f64; KNOTS];
            for (row, &linear_value) in linear.iter().enumerate() {
                let slot = nearest_knot(&knots[term], completed[feature][row]);
                let (weight, residual) = if logistic {
                    let probability = sigmoid(f64::from(linear_value));
                    let weight = (probability * (1.0 - probability)).max(0.02);
                    (
                        weight,
                        (f64::from(table.target[row]) - probability) / weight,
                    )
                } else {
                    (1.0, f64::from(table.target[row] - linear_value))
                };
                sums[slot] += weight * residual;
                weights[slot] += weight;
            }
            for slot in 0..KNOTS {
                if weights[slot] > 0.0 {
                    effects[term][slot] = (0.5 * sums[slot] / weights[slot]) as f32;
                }
            }
            let center = effects[term]
                .iter()
                .zip(weights)
                .map(|(value, weight)| f64::from(*value) * weight)
                .sum::<f64>()
                / weights.iter().sum::<f64>().max(1e-12);
            effects[term]
                .iter_mut()
                .for_each(|value| *value -= center as f32);
            for (row, value) in linear.iter_mut().enumerate() {
                *value += effects[term][nearest_knot(&knots[term], completed[feature][row])];
            }
        }
    }
    Target::SparseGam {
        intercept,
        logistic,
        terms: selected
            .into_iter()
            .zip(knots)
            .zip(effects)
            .map(|((feature, knots), values)| AdditiveTerm {
                feature: feature as u32,
                knots,
                values,
            })
            .collect(),
    }
}

fn fit_ga2m_target(table: &Table, completed: &[Vec<f32>], screen: &[(usize, f32)]) -> Target {
    let gam = fit_gam_target(table, completed, screen);
    let Target::SparseGam {
        intercept,
        logistic,
        terms,
    } = gam
    else {
        unreachable!()
    };
    let mut linear = Vec::with_capacity(table.rows);
    for row in 0..table.rows {
        let values: Vec<_> = completed.iter().map(|column| column[row]).collect();
        linear.push(
            evaluate_target(
                &Target::SparseGam {
                    intercept,
                    logistic,
                    terms: terms.clone(),
                },
                &values,
            )
            .0,
        );
    }
    let selected: Vec<_> = terms
        .iter()
        .take(5)
        .map(|term| term.feature as usize)
        .collect();
    let mut interactions = Vec::new();
    'pairs: for left_index in 0..selected.len() {
        for right_index in left_index + 1..selected.len() {
            const KNOTS: usize = 5;
            let left = selected[left_index];
            let right = selected[right_index];
            let left_knots = feature_knots(&completed[left], KNOTS);
            let right_knots = feature_knots(&completed[right], KNOTS);
            let mut sums = [0.0f64; KNOTS * KNOTS];
            let mut weights = [0.0f64; KNOTS * KNOTS];
            for (row, &linear_value) in linear.iter().enumerate() {
                let left_slot = nearest_knot(&left_knots, completed[left][row]);
                let right_slot = nearest_knot(&right_knots, completed[right][row]);
                let slot = left_slot * KNOTS + right_slot;
                let (weight, residual) = if logistic {
                    let probability = sigmoid(f64::from(linear_value));
                    let weight = (probability * (1.0 - probability)).max(0.02);
                    (
                        weight,
                        (f64::from(table.target[row]) - probability) / weight,
                    )
                } else {
                    (1.0, f64::from(table.target[row] - linear_value))
                };
                sums[slot] += weight * residual;
                weights[slot] += weight;
            }
            let mut values: Vec<f32> = sums
                .iter()
                .zip(&weights)
                .map(|(sum, weight)| {
                    if *weight > 0.0 {
                        (0.5 * sum / weight) as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            let center = values
                .iter()
                .zip(&weights)
                .map(|(value, weight)| f64::from(*value) * weight)
                .sum::<f64>()
                / weights.iter().sum::<f64>().max(1e-12);
            values.iter_mut().for_each(|value| *value -= center as f32);
            for (row, linear_value) in linear.iter_mut().enumerate() {
                let slot = nearest_knot(&left_knots, completed[left][row]) * KNOTS
                    + nearest_knot(&right_knots, completed[right][row]);
                *linear_value += values[slot];
            }
            let (left, right, left_knots, right_knots, values) = if left < right {
                (left, right, left_knots, right_knots, values)
            } else {
                let transposed = (0..KNOTS)
                    .flat_map(|new_left| {
                        let values = &values;
                        (0..KNOTS).map(move |new_right| values[new_right * KNOTS + new_left])
                    })
                    .collect();
                (right, left, right_knots, left_knots, transposed)
            };
            interactions.push(InteractionTerm {
                left: left as u32,
                right: right as u32,
                left_knots,
                right_knots,
                values,
            });
            if interactions.len() == 4 {
                break 'pairs;
            }
        }
    }
    Target::Ga2m {
        intercept,
        logistic,
        main_terms: terms,
        interactions,
    }
}

fn fit_mars_target(table: &Table, completed: &[Vec<f32>], screen: &[(usize, f32)]) -> Target {
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let mut factors = Vec::new();
    let mut basis = Vec::new();
    for &(feature, _) in screen.iter().take(8) {
        let knot = feature_knots(&completed[feature], 3)[1];
        for direction in [-1, 1] {
            factors.push(MarsFactor {
                feature: feature as u32,
                knot,
                direction,
            });
            basis.push(
                completed[feature]
                    .iter()
                    .map(|value| {
                        if direction > 0 {
                            (*value - knot).max(0.0)
                        } else {
                            (knot - *value).max(0.0)
                        }
                    })
                    .collect(),
            );
        }
    }
    let (intercept, coefficients) = fit_basis_coefficients(&table.target, &basis, logistic);
    Target::Mars {
        intercept,
        logistic,
        terms: factors
            .into_iter()
            .zip(coefficients)
            .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
            .map(|(factor, coefficient)| MarsTerm {
                coefficient,
                factors: vec![factor],
            })
            .collect(),
    }
}

fn fit_oblivious_tree_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> Target {
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let features: Vec<_> = screen.iter().take(4).map(|(feature, _)| *feature).collect();
    let thresholds: Vec<_> = features
        .iter()
        .map(|feature| feature_knots(&completed[*feature], 3)[1])
        .collect();
    let mut sums = vec![0.0f64; 1usize << features.len()];
    let mut counts = vec![0usize; sums.len()];
    for (row, actual) in table.target.iter().enumerate() {
        let leaf = features.iter().zip(&thresholds).enumerate().fold(
            0usize,
            |leaf, (depth, (feature, threshold))| {
                leaf | (((completed[*feature][row] > *threshold) as usize) << depth)
            },
        );
        sums[leaf] += f64::from(*actual);
        counts[leaf] += 1;
    }
    let base = f64::from(table.target.iter().sum::<f32>() / table.rows.max(1) as f32);
    let leaves = sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            let value = if count == 0 { base } else { sum / count as f64 };
            if logistic {
                let probability = value.clamp(1e-4, 1.0 - 1e-4);
                (probability / (1.0 - probability)).ln() as f32
            } else {
                value as f32
            }
        })
        .collect();
    Target::ObliviousTree {
        logistic,
        features: features.into_iter().map(|feature| feature as u32).collect(),
        thresholds,
        leaves,
    }
}

#[cfg(not(feature = "gpu-training"))]
fn compact_neural_weight(unit: usize, input: usize) -> f32 {
    let mut value = (unit as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (input as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 30;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    let magnitude = 1.25 + ((value >> 8) & 3) as f32 * 0.5;
    if value & 1 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[cfg(not(feature = "gpu-training"))]
fn fit_compact_neural_target_native(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> Target {
    let logistic = table
        .target
        .iter()
        .all(|value| *value == 0.0 || *value == 1.0);
    let selected: Vec<_> = screen
        .iter()
        .take(12)
        .map(|(feature, _)| *feature)
        .collect();
    let hidden_width = selected.len().clamp(4, 12);
    let input_weights: Vec<_> = (0..hidden_width)
        .flat_map(|unit| (0..selected.len()).map(move |input| compact_neural_weight(unit, input)))
        .collect();
    let hidden_biases: Vec<_> = (0..hidden_width)
        .map(|unit| {
            let center = input_weights[unit * selected.len()..(unit + 1) * selected.len()]
                .iter()
                .sum::<f32>();
            let offset = match unit % 3 {
                0 => -0.5,
                1 => 0.0,
                _ => 0.5,
            };
            -0.5 * center + offset
        })
        .collect();
    let mut basis: Vec<Vec<f32>> = selected
        .iter()
        .map(|feature| completed[*feature].clone())
        .collect();
    basis.extend((0..hidden_width).map(|unit| {
        (0..table.rows)
            .map(|row| {
                (hidden_biases[unit]
                    + selected
                        .iter()
                        .enumerate()
                        .map(|(input, feature)| {
                            input_weights[unit * selected.len() + input] * completed[*feature][row]
                        })
                        .sum::<f32>())
                .tanh()
            })
            .collect()
    }));
    let (intercept, coefficients) = fit_basis_coefficients(&table.target, &basis, logistic);
    let (linear_coefficients, output_weights) = coefficients.split_at(selected.len());
    Target::CompactNeuralResidual {
        intercept,
        logistic,
        linear_terms: selected
            .iter()
            .zip(linear_coefficients)
            .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
            .map(|(&feature, &coefficient)| LinearTerm {
                feature: feature as u32,
                coefficient,
            })
            .collect(),
        hidden_features: selected.into_iter().map(|feature| feature as u32).collect(),
        hidden_width: hidden_width as u8,
        input_weights,
        hidden_biases,
        output_weights: output_weights.to_vec(),
    }
}

#[cfg(feature = "gpu-training")]
fn tensor_values(tensor: tch::Tensor, context: &str) -> Result<Vec<f32>> {
    Vec::<f32>::try_from(tensor.to_device(tch::Device::Cpu).view([-1]))
        .map_err(|error| DopeError::Data(format!("{context}: {error}")))
}