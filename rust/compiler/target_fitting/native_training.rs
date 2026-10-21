/// Native (pure Rust, CPU) training profile for the compact neural residual.
///
/// The profile names match the bounded GPU research profiles, so a default
/// build can train the same architecture as the published headline profile
/// (`features12_steps2048`) without libtorch. The optimizer mirrors the GPU
/// path: full-batch mean squared error (or logistic loss), AdamW with learning
/// rate 2e-3, betas (0.9, 0.999), epsilon 1e-8, decoupled weight decay 0.01,
/// element-wise gradient clipping at 5.0, and a final ridge refit of the
/// readout on the learned hidden basis. Arithmetic is f64 and the initial
/// weights come from a seeded splitmix64 stream, so results are deterministic
/// for a given seed but are not bitwise equal to libtorch's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeTargetProfile {
    pub feature_limit: usize,
    pub hidden_width: usize,
    pub optimizer_steps: usize,
}

/// Rows used for the gradient phase. Larger fit views are thinned with a fixed
/// stride; the readout refit always uses every fit row.
const NATIVE_TRAINING_ROW_CAP: usize = 4096;

impl NativeTargetProfile {
    /// The architecture of the published headline profile.
    pub const HEADLINE: Self = Self {
        feature_limit: 12,
        hidden_width: 16,
        optimizer_steps: 2048,
    };

    /// Parse `features{F}_steps{S}` or `features{F}_width{W}_steps{S}`.
    pub fn parse(name: &str) -> Result<Self> {
        let invalid = || DopeError::Unsupported(format!("unknown native target profile {name:?}"));
        let fields = name.split('_').collect::<Vec<_>>();
        let number = |field: &str, prefix: &str| {
            field
                .strip_prefix(prefix)
                .and_then(|digits| digits.parse::<usize>().ok())
                .ok_or_else(invalid)
        };
        let (features, width, steps) = match fields.as_slice() {
            [features, steps] => (number(features, "features")?, 16, number(steps, "steps")?),
            [features, width, steps] => (
                number(features, "features")?,
                number(width, "width")?,
                number(steps, "steps")?,
            ),
            _ => return Err(invalid()),
        };
        if !(1..=24).contains(&features) || !(1..=16).contains(&width) || !(1..=16384).contains(&steps)
        {
            return Err(invalid());
        }
        Ok(Self {
            feature_limit: features,
            hidden_width: width,
            optimizer_steps: steps,
        })
    }

    pub fn name(&self) -> String {
        if self.hidden_width == 16 {
            format!("features{}_steps{}", self.feature_limit, self.optimizer_steps)
        } else {
            format!(
                "features{}_width{}_steps{}",
                self.feature_limit, self.hidden_width, self.optimizer_steps
            )
        }
    }
}

struct AdamW {
    first: Vec<f64>,
    second: Vec<f64>,
    step: i32,
}

impl AdamW {
    const RATE: f64 = 2e-3;
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPSILON: f64 = 1e-8;
    const DECAY: f64 = 0.01;
    const CLIP: f64 = 5.0;

    fn new(size: usize) -> Self {
        Self {
            first: vec![0.0; size],
            second: vec![0.0; size],
            step: 0,
        }
    }

    fn update(&mut self, parameters: &mut [f64], gradients: &[f64]) {
        self.step += 1;
        let correction1 = 1.0 - Self::BETA1.powi(self.step);
        let correction2 = 1.0 - Self::BETA2.powi(self.step);
        for index in 0..parameters.len() {
            let gradient = gradients[index].clamp(-Self::CLIP, Self::CLIP);
            parameters[index] *= 1.0 - Self::RATE * Self::DECAY;
            self.first[index] = Self::BETA1 * self.first[index] + (1.0 - Self::BETA1) * gradient;
            self.second[index] =
                Self::BETA2 * self.second[index] + (1.0 - Self::BETA2) * gradient * gradient;
            let first = self.first[index] / correction1;
            let second = self.second[index] / correction2;
            parameters[index] -= Self::RATE * first / (second.sqrt() + Self::EPSILON);
        }
    }
}

/// Parameter layout: hidden weights (H x P, row major), hidden biases (H),
/// output weights (H), skip weights (P), intercept (1).
struct ResidualNetwork {
    inputs: usize,
    width: usize,
    parameters: Vec<f64>,
}

impl ResidualNetwork {
    fn initialize(inputs: usize, width: usize, seed: u64) -> Self {
        let mut state = splitmix64(seed ^ 0x6e61_7469_7665_5f6e);
        let mut uniform = |bound: f64| {
            state = splitmix64(state);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            (2.0 * unit - 1.0) * bound
        };
        // libtorch Linear defaults: Kaiming-uniform weights (ReLU gain) and
        // U(-1/sqrt(fan_in), 1/sqrt(fan_in)) biases.
        let weight_bound = |fan: usize| (6.0 / fan as f64).sqrt();
        let bias_bound = |fan: usize| 1.0 / (fan as f64).sqrt();
        let mut parameters = Vec::with_capacity(width * inputs + 2 * width + inputs + 1);
        parameters.extend((0..width * inputs).map(|_| uniform(weight_bound(inputs))));
        parameters.extend((0..width).map(|_| uniform(bias_bound(inputs))));
        parameters.extend((0..width).map(|_| uniform(weight_bound(width))));
        parameters.extend((0..inputs).map(|_| uniform(weight_bound(inputs))));
        parameters.push(uniform(bias_bound(inputs)) + uniform(bias_bound(width)));
        Self {
            inputs,
            width,
            parameters,
        }
    }

    /// One full-batch loss gradient. `rows` is row-major (n x P).
    fn gradient(&self, rows: &[f64], target: &[f64], logistic: bool, out: &mut [f64]) {
        let (p, h) = (self.inputs, self.width);
        let (weights, rest) = self.parameters.split_at(h * p);
        let (biases, rest) = rest.split_at(h);
        let (outputs, rest) = rest.split_at(h);
        let (skip, intercept) = rest.split_at(p);
        out.iter_mut().for_each(|value| *value = 0.0);
        let count = target.len() as f64;
        let mut activations = vec![0.0; h];
        for (row, &label) in rows.chunks_exact(p).zip(target) {
            let mut prediction = intercept[0];
            for (unit, activation) in activations.iter_mut().enumerate() {
                let mut sum = biases[unit];
                for (weight, value) in weights[unit * p..(unit + 1) * p].iter().zip(row) {
                    sum += weight * value;
                }
                *activation = sum.tanh();
                prediction += outputs[unit] * *activation;
            }
            for (weight, value) in skip.iter().zip(row) {
                prediction += weight * value;
            }
            let delta = if logistic {
                (sigmoid(prediction) - label) / count
            } else {
                2.0 * (prediction - label) / count
            };
            let (grad_weights, grad_rest) = out.split_at_mut(h * p);
            let (grad_biases, grad_rest) = grad_rest.split_at_mut(h);
            let (grad_outputs, grad_rest) = grad_rest.split_at_mut(h);
            let (grad_skip, grad_intercept) = grad_rest.split_at_mut(p);
            grad_intercept[0] += delta;
            for (gradient, value) in grad_skip.iter_mut().zip(row) {
                *gradient += delta * value;
            }
            for unit in 0..h {
                grad_outputs[unit] += delta * activations[unit];
                let back = delta * outputs[unit] * (1.0 - activations[unit] * activations[unit]);
                grad_biases[unit] += back;
                for (gradient, value) in grad_weights[unit * p..(unit + 1) * p].iter_mut().zip(row)
                {
                    *gradient += back * value;
                }
            }
        }
    }
}

/// Train the compact neural residual natively and refit its readout.
fn fit_compact_neural_target_trained(
    table: &Table,
    completed: &[Vec<f32>],
    screen: &[(usize, f32)],
    seed: u64,
    task: Task,
    profile: NativeTargetProfile,
) -> Target {
    let logistic = task == Task::Binary;
    let selected = screen
        .iter()
        .take(profile.feature_limit)
        .map(|(feature, _)| *feature)
        .collect::<Vec<_>>();
    if selected.is_empty() || table.rows == 0 {
        return fit_compact_neural_target_native_bounded(
            table,
            completed,
            screen,
            task,
            profile.hidden_width,
        );
    }
    let stride = table.rows.div_ceil(NATIVE_TRAINING_ROW_CAP).max(1);
    let training_rows = (0..table.rows).step_by(stride).collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(training_rows.len() * selected.len());
    for &row in &training_rows {
        rows.extend(selected.iter().map(|&feature| f64::from(completed[feature][row])));
    }
    let target = training_rows
        .iter()
        .map(|&row| f64::from(table.target[row]))
        .collect::<Vec<_>>();
    let width = profile.hidden_width;
    let mut network = ResidualNetwork::initialize(selected.len(), width, seed);
    let mut optimizer = AdamW::new(network.parameters.len());
    let mut gradients = vec![0.0; network.parameters.len()];
    for _ in 0..profile.optimizer_steps {
        network.gradient(&rows, &target, logistic, &mut gradients);
        optimizer.update(&mut network.parameters, &gradients);
    }
    let p = selected.len();
    let input_weights = network.parameters[..width * p]
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let hidden_biases = network.parameters[width * p..width * p + width]
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    // Same readout refit as the GPU profile: an unfinished intercept can
    // otherwise dominate targets with small variance.
    let mut basis: Vec<Vec<f32>> = selected.iter().map(|&feature| completed[feature].clone()).collect();
    basis.extend((0..width).map(|unit| {
        (0..table.rows)
            .map(|row| {
                (hidden_biases[unit]
                    + selected
                        .iter()
                        .enumerate()
                        .map(|(input, &feature)| input_weights[unit * p + input] * completed[feature][row])
                        .sum::<f32>())
                .tanh()
            })
            .collect()
    }));
    let (intercept, coefficients) = fit_basis_coefficients(&table.target, &basis, logistic);
    let (linear_weights, output_weights) = coefficients.split_at(p);
    Target::CompactNeuralResidual {
        intercept,
        logistic,
        linear_terms: selected
            .iter()
            .zip(linear_weights)
            .filter(|(_, coefficient)| coefficient.abs() >= 1e-5)
            .map(|(&feature, &coefficient)| LinearTerm {
                feature: feature as u32,
                coefficient,
            })
            .collect(),
        hidden_features: selected.into_iter().map(|feature| feature as u32).collect(),
        hidden_width: width as u8,
        input_weights,
        hidden_biases,
        output_weights: output_weights.to_vec(),
    }
}

#[cfg(test)]
mod native_training_tests {
    use super::*;

    fn nonlinear_table(rows: usize) -> Table {
        let mut features = Vec::with_capacity(rows * 3);
        let mut target = Vec::with_capacity(rows);
        for row in 0..rows {
            let a = (splitmix64(row as u64) >> 40) as f32 / (1u64 << 24) as f32;
            let b = (splitmix64(row as u64 ^ 0x55) >> 40) as f32 / (1u64 << 24) as f32;
            let c = (splitmix64(row as u64 ^ 0xaa) >> 40) as f32 / (1u64 << 24) as f32;
            features.extend([a, b, c]);
            target.push((0.5 + 0.35 * (6.0 * a).sin() * b + 0.1 * c).clamp(0.0, 1.0));
        }
        Table::from_arrays(&features, &target, rows, 3, Task::Regression).unwrap()
    }

    fn mse(target: &Target, table: &Table) -> f64 {
        (0..table.rows)
            .map(|row| {
                let values = table.columns.iter().map(|column| column[row]).collect::<Vec<_>>();
                f64::from(evaluate_target(target, &values).0 - table.target[row]).powi(2)
            })
            .sum::<f64>()
            / table.rows as f64
    }

    #[test]
    fn profiles_parse_and_render_the_research_names() {
        assert_eq!(
            NativeTargetProfile::parse("features12_steps2048").unwrap(),
            NativeTargetProfile::HEADLINE
        );
        let narrow = NativeTargetProfile::parse("features6_width8_steps512").unwrap();
        assert_eq!((narrow.feature_limit, narrow.hidden_width, narrow.optimizer_steps), (6, 8, 512));
        assert_eq!(narrow.name(), "features6_width8_steps512");
        for bad in ["features12", "features99_steps2048", "features12_width17_steps8", "x"] {
            assert!(NativeTargetProfile::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn trained_head_is_deterministic_and_beats_random_features() {
        let table = nonlinear_table(600);
        let screen = target_screen(&table.columns, &table.target);
        let profile = NativeTargetProfile::parse("features12_steps512").unwrap();
        let trained = fit_compact_neural_target_trained(
            &table, &table.columns, &screen, 11, Task::Regression, profile,
        );
        let again = fit_compact_neural_target_trained(
            &table, &table.columns, &screen, 11, Task::Regression, profile,
        );
        assert_eq!(trained, again);
        let random = fit_compact_neural_target_native_bounded(
            &table, &table.columns, &screen, Task::Regression, 12,
        );
        assert!(mse(&trained, &table) < mse(&random, &table));
        let Target::CompactNeuralResidual { hidden_width, input_weights, .. } = &trained else {
            panic!("compact residual expected");
        };
        assert_eq!((*hidden_width, input_weights.len()), (16, 16 * 3));
    }

    #[test]
    fn gaussian_copula_factor_has_unit_rows_and_validates() {
        let matrix = vec![1.0, 0.6, 0.3, 0.6, 1.0, 0.5, 0.3, 0.5, 1.0];
        let Dependence::GaussianCopula { cholesky } = gaussian_copula_dependence(Some(&matrix), &[], 3)
        else {
            panic!("dense factor expected");
        };
        for row in 0..3 {
            let norm = (0..3).map(|c| f64::from(cholesky[row * 3 + c]).powi(2)).sum::<f64>();
            assert!((norm - 1.0).abs() < 1e-5);
            assert!((row + 1..3).all(|c| cholesky[row * 3 + c] == 0.0));
        }
        let wide = vec![0.0; 40 * 40];
        assert!(matches!(
            gaussian_copula_dependence(Some(&wide), &[], 40),
            Dependence::ChowLiu { .. }
        ));
        assert_eq!(parse_symbolic_profile("symbolic_k2_d7_t5_q6"), Some((2, 7, 5, 6)));
        assert_eq!(parse_symbolic_profile("symbolic_k9_d7_t5_q6"), None);
        assert_eq!(parse_symbolic_profile("symbolic_k2_d8_t5_q6"), None);
        assert_eq!(backend_profile("compact_neural_gaussian_copula"), Some((2, 7, 5, 8)));
    }
}
