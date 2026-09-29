

fn solve_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>, size: usize) -> Vec<f64> {
    for pivot in 0..size {
        let best = (pivot..size)
            .max_by(|&left, &right| {
                matrix[left * size + pivot]
                    .abs()
                    .total_cmp(&matrix[right * size + pivot].abs())
            })
            .unwrap();
        if best != pivot {
            for column in pivot..size {
                matrix.swap(pivot * size + column, best * size + column);
            }
            rhs.swap(pivot, best);
        }
        let diagonal = matrix[pivot * size + pivot];
        if diagonal.abs() <= 1e-12 {
            continue;
        }
        for row in pivot + 1..size {
            let factor = matrix[row * size + pivot] / diagonal;
            for column in pivot..size {
                matrix[row * size + column] -= factor * matrix[pivot * size + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let remainder = (row + 1..size)
            .map(|column| matrix[row * size + column] * solution[column])
            .sum::<f64>();
        let diagonal = matrix[row * size + row];
        if diagonal.abs() > 1e-12 {
            solution[row] = (rhs[row] - remainder) / diagonal;
        }
    }
    solution
}

fn target_screen(completed: &[Vec<f32>], target: &[f32]) -> Vec<(usize, f32)> {
    let target_rank = rank_normalize(target);
    let mut scores: Vec<(usize, f32)> = completed
        .par_iter()
        .enumerate()
        .map(|(index, column)| (index, dot(&rank_normalize(column), &target_rank).abs()))
        .collect();
    scores.sort_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    scores.truncate(MAX_TARGET_TERMS.min(scores.len()));
    scores
}

fn fit_linear_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> (Target, Noise, f64) {
    let selected: Vec<usize> = screen.iter().map(|(index, _)| *index).collect();
    let terms = selected.len();
    let rows = table.rows;
    let target_mean = table
        .target
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>()
        / rows as f64;
    let means: Vec<f64> = selected
        .iter()
        .map(|&column| {
            completed[column]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / rows as f64
        })
        .collect();
    let mut gram = vec![0.0; terms * terms];
    let mut rhs = vec![0.0; terms];
    for left in 0..terms {
        for (row, actual) in table.target.iter().enumerate() {
            let x = f64::from(completed[selected[left]][row]) - means[left];
            rhs[left] += x * (f64::from(*actual) - target_mean);
        }
        for right in 0..=left {
            let value = (0..rows)
                .map(|row| {
                    (f64::from(completed[selected[left]][row]) - means[left])
                        * (f64::from(completed[selected[right]][row]) - means[right])
                })
                .sum::<f64>();
            gram[left * terms + right] = value;
            gram[right * terms + left] = value;
        }
        gram[left * terms + left] += 1.0;
    }
    let coefficients = solve_system(gram, rhs, terms);
    let intercept = target_mean
        - coefficients
            .iter()
            .zip(&means)
            .map(|(coefficient, mean)| coefficient * mean)
            .sum::<f64>();
    let model_terms: Vec<LinearTerm> = selected
        .iter()
        .zip(&coefficients)
        .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
        .map(|(&feature, &coefficient)| LinearTerm {
            feature: feature as u32,
            coefficient: coefficient as f32,
        })
        .collect();
    let mut squared = 0.0;
    let mut null_squared = 0.0;
    for (row, actual) in table.target.iter().enumerate() {
        let prediction = (intercept
            + selected
                .iter()
                .zip(&coefficients)
                .map(|(&feature, coefficient)| coefficient * f64::from(completed[feature][row]))
                .sum::<f64>())
        .clamp(0.0, 1.0);
        squared += (f64::from(*actual) - prediction).powi(2);
        null_squared += (f64::from(*actual) - target_mean).powi(2);
    }
    let sigma = (squared / rows as f64).sqrt() as f32;
    let utility = if null_squared <= 1e-12 {
        1.0 - squared / rows as f64
    } else {
        (1.0 - squared / null_squared).clamp(0.0, 1.0)
    };
    (
        Target::SparseLinear {
            intercept: intercept as f32,
            terms: model_terms,
        },
        Noise::Homoscedastic { sigma },
        utility,
    )
}

fn fit_logistic_target(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
) -> (Target, Noise, f64) {
    let selected: Vec<usize> = screen.iter().map(|(index, _)| *index).collect();
    let size = selected.len() + 1;
    let base_rate = (table.target.iter().sum::<f32>() / table.rows as f32).clamp(1e-6, 1.0 - 1e-6);
    let mut coefficients = vec![0.0f64; size];
    coefficients[0] = f64::from((base_rate / (1.0 - base_rate)).ln());
    for _ in 0..8 {
        let mut hessian = vec![0.0; size * size];
        let mut gradient = vec![0.0; size];
        for (row, actual) in table.target.iter().enumerate() {
            let linear = coefficients[0]
                + selected
                    .iter()
                    .enumerate()
                    .map(|(index, &feature)| {
                        coefficients[index + 1] * f64::from(completed[feature][row])
                    })
                    .sum::<f64>();
            let probability = 1.0 / (1.0 + (-linear.clamp(-30.0, 30.0)).exp());
            let residual = f64::from(*actual) - probability;
            let weight = (probability * (1.0 - probability)).max(1e-6);
            for left in 0..size {
                let x_left = if left == 0 {
                    1.0
                } else {
                    f64::from(completed[selected[left - 1]][row])
                };
                gradient[left] += x_left * residual;
                for right in 0..=left {
                    let x_right = if right == 0 {
                        1.0
                    } else {
                        f64::from(completed[selected[right - 1]][row])
                    };
                    hessian[left * size + right] += weight * x_left * x_right;
                    hessian[right * size + left] = hessian[left * size + right];
                }
            }
        }
        for index in 1..size {
            hessian[index * size + index] += 0.5;
            gradient[index] -= 0.5 * coefficients[index];
        }
        let delta = solve_system(hessian, gradient, size);
        let max_delta = delta
            .iter()
            .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
        for (coefficient, update) in coefficients.iter_mut().zip(delta) {
            *coefficient = (*coefficient + update).clamp(-30.0, 30.0);
        }
        if max_delta < 1e-5 {
            break;
        }
    }
    let terms = selected
        .iter()
        .enumerate()
        .filter(|(index, _)| coefficients[index + 1].abs() >= 1e-5)
        .map(|(index, &feature)| LinearTerm {
            feature: feature as u32,
            coefficient: coefficients[index + 1] as f32,
        })
        .collect();
    let mut model_loss = 0.0;
    let null_loss = -(f64::from(base_rate) * f64::from(base_rate).ln()
        + f64::from(1.0 - base_rate) * f64::from(1.0 - base_rate).ln());
    for (row, actual) in table.target.iter().enumerate() {
        let linear = coefficients[0]
            + selected
                .iter()
                .enumerate()
                .map(|(index, &feature)| {
                    coefficients[index + 1] * f64::from(completed[feature][row])
                })
                .sum::<f64>();
        let probability =
            (1.0 / (1.0 + (-linear.clamp(-30.0, 30.0)).exp())).clamp(1e-9, 1.0 - 1e-9);
        let y = f64::from(*actual);
        model_loss += -(y * probability.ln() + (1.0 - y) * (1.0 - probability).ln());
    }
    model_loss /= table.rows as f64;
    let utility = if null_loss <= 1e-12 {
        1.0
    } else {
        (1.0 - model_loss / null_loss).clamp(0.0, 1.0)
    };
    (
        Target::SparseLogistic {
            intercept: coefficients[0] as f32,
            terms,
        },
        Noise::BernoulliCalibration { base_rate },
        utility,
    )
}

fn nearest_knot(knots: &[f32], value: f32) -> usize {
    knots
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - value).abs().total_cmp(&(*right - value).abs()))
        .map_or(0, |(index, _)| index)
}

fn feature_knots(values: &[f32], count: usize) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted.dedup_by(|left, right| (*left - *right).abs() <= 1e-7);
    (0..count)
        .map(|index| quantile(&sorted, index as f32 / (count - 1).max(1) as f32))
        .collect()
}

fn spline_features(completed: &[Vec<f32>], screen: &[(usize, f32)], limit: usize) -> Vec<usize> {
    screen
        .iter()
        .filter_map(|(feature, _)| {
            let mut unique = completed[*feature].clone();
            unique.sort_by(f32::total_cmp);
            unique.dedup_by(|left, right| (*left - *right).abs() <= 1e-7);
            (unique.len() >= 3).then_some(*feature)
        })
        .take(limit)
        .collect()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}

fn fit_basis_coefficients(target: &[f32], basis: &[Vec<f32>], logistic: bool) -> (f32, Vec<f32>) {
    let rows = target.len();
    let size = basis.len() + 1;
    let mean = target.iter().sum::<f32>() / rows.max(1) as f32;
    let mut coefficients = vec![0.0f64; size];
    coefficients[0] = if logistic {
        let probability = mean.clamp(1e-6, 1.0 - 1e-6);
        f64::from((probability / (1.0 - probability)).ln())
    } else {
        f64::from(mean)
    };
    let iterations = if logistic { 12 } else { 1 };
    for _ in 0..iterations {
        let mut gram = vec![0.0; size * size];
        let mut rhs = vec![0.0; size];
        for row in 0..rows {
            let linear = coefficients[0]
                + basis
                    .iter()
                    .zip(&coefficients[1..])
                    .map(|(column, coefficient)| f64::from(column[row]) * coefficient)
                    .sum::<f64>();
            let (weight, working) = if logistic {
                let probability = sigmoid(linear);
                let weight = (probability * (1.0 - probability)).max(1e-4);
                (
                    weight,
                    linear + (f64::from(target[row]) - probability) / weight,
                )
            } else {
                (1.0, f64::from(target[row]))
            };
            for left in 0..size {
                let x_left = if left == 0 {
                    1.0
                } else {
                    f64::from(basis[left - 1][row])
                };
                rhs[left] += weight * x_left * working;
                for right in 0..=left {
                    let x_right = if right == 0 {
                        1.0
                    } else {
                        f64::from(basis[right - 1][row])
                    };
                    gram[left * size + right] += weight * x_left * x_right;
                    gram[right * size + left] = gram[left * size + right];
                }
            }
        }
        for index in 1..size {
            gram[index * size + index] += 0.5;
        }
        coefficients = solve_system(gram, rhs, size)
            .into_iter()
            .map(|value| value.clamp(-30.0, 30.0))
            .collect();
    }
    (
        coefficients[0] as f32,
        coefficients[1..]
            .iter()
            .map(|value| *value as f32)
            .collect(),
    )
}

include!("target_fitting.rs");
include!("candidate_search.rs");
include!("neural_candidates.rs");
