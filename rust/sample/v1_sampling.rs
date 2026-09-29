
fn parse_v1_marginals(value: &Value) -> Result<Vec<V1Marginal>> {
    value
        .get("marginals")
        .and_then(Value::as_array)
        .ok_or_else(|| DopeError::Codec("V1 marginals are missing".into()))?
        .iter()
        .map(|marginal| {
            let kind = marginal.get("kind").and_then(Value::as_str).unwrap_or("");
            let params = marginal
                .get("params")
                .ok_or_else(|| DopeError::Codec("V1 marginal params are missing".into()))?;
            match kind {
                "bern" => Ok(V1Marginal::Bernoulli(number(params.get("p"), "p")?)),
                "spike_grid" | "ordinal" => {
                    let values = params
                        .get("values")
                        .and_then(Value::as_array)
                        .ok_or_else(|| DopeError::Codec("V1 grid values missing".into()))?
                        .iter()
                        .map(|value| value.as_f64().unwrap_or(0.0))
                        .collect();
                    let probabilities = params
                        .get("probs")
                        .and_then(Value::as_array)
                        .ok_or_else(|| DopeError::Codec("V1 grid probabilities missing".into()))?
                        .iter()
                        .map(|value| value.as_f64().unwrap_or(0.0))
                        .collect();
                    Ok(V1Marginal::Grid(values, probabilities))
                }
                "beta" => Ok(V1Marginal::Beta(
                    Beta::new(
                        number(params.get("alpha"), "alpha")?,
                        number(params.get("beta"), "beta")?,
                    )
                    .map_err(|error| DopeError::Codec(error.to_string()))?,
                )),
                "zi_beta" => Ok(V1Marginal::ZeroInflatedBeta(
                    number(params.get("zero_prob"), "zero_prob")?,
                    Beta::new(
                        number(params.get("alpha"), "alpha")?,
                        number(params.get("beta"), "beta")?,
                    )
                    .map_err(|error| DopeError::Codec(error.to_string()))?,
                )),
                "empq" => {
                    let qs = params
                        .get("qs")
                        .and_then(Value::as_array)
                        .unwrap()
                        .iter()
                        .map(|value| value.as_f64().unwrap())
                        .collect();
                    let values = params
                        .get("values")
                        .and_then(Value::as_array)
                        .unwrap()
                        .iter()
                        .map(|value| value.as_f64().unwrap())
                        .collect();
                    Ok(V1Marginal::Quantile(qs, values))
                }
                _ => Err(DopeError::Unsupported(format!("V1 marginal {kind}"))),
            }
        })
        .collect()
}

fn parse_v1_dependence(value: &Value, width: usize) -> Result<V1Dependence> {
    let dependence = &value["dependence"];
    if dependence["kind"].as_str() != Some("gauss_copula") {
        return Ok(V1Dependence::Independent);
    }
    let rows = dependence["params"]["corr"]
        .as_array()
        .ok_or_else(|| DopeError::Codec("V1 Gaussian correlation missing".into()))?;
    if rows.len() != width {
        return Err(DopeError::Codec("V1 correlation width mismatch".into()));
    }
    let mut matrix = vec![0.0f64; width * width];
    for (row, values) in rows.iter().enumerate() {
        let values = values
            .as_array()
            .ok_or_else(|| DopeError::Codec("V1 correlation row invalid".into()))?;
        if values.len() != width {
            return Err(DopeError::Codec("V1 correlation is not square".into()));
        }
        for (column, value) in values.iter().enumerate() {
            matrix[row * width + column] =
                value
                    .as_f64()
                    .unwrap_or(if row == column { 1.0 } else { 0.0 });
        }
    }
    let mut cholesky = vec![0.0f64; width * width];
    for row in 0..width {
        for column in 0..=row {
            let remainder = matrix[row * width + column]
                - (0..column)
                    .map(|index| cholesky[row * width + index] * cholesky[column * width + index])
                    .sum::<f64>();
            cholesky[row * width + column] = if row == column {
                remainder.max(1e-9).sqrt()
            } else {
                remainder / cholesky[column * width + column].max(1e-9)
            };
        }
    }
    Ok(V1Dependence::Gaussian { cholesky, width })
}

fn array_i64(value: &Value, name: &str) -> Result<Vec<i64>> {
    value
        .as_array()
        .ok_or_else(|| DopeError::Codec(format!("V1 tree {name} missing")))?
        .iter()
        .map(|entry| {
            entry
                .as_i64()
                .ok_or_else(|| DopeError::Codec(format!("V1 tree {name} invalid")))
        })
        .collect()
}

fn array_f32(value: &Value, name: &str) -> Result<Vec<f32>> {
    value
        .as_array()
        .ok_or_else(|| DopeError::Codec(format!("V1 tree {name} missing")))?
        .iter()
        .map(|entry| {
            entry
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| DopeError::Codec(format!("V1 tree {name} invalid")))
        })
        .collect()
}

fn parse_v1_target(value: &Value, task: Task) -> Result<V1Target> {
    let target = &value["target"];
    let params = &target["params"];
    match target["kind"].as_str().unwrap_or("") {
        "linear_sparse" | "lift_logit" => {
            let terms = params["terms"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_array)
                .map(|pair| {
                    (
                        pair[0].as_u64().unwrap_or(0) as usize,
                        pair[1].as_f64().unwrap_or(0.0) as f32,
                    )
                })
                .collect();
            Ok(V1Target::Linear {
                intercept: params["intercept"].as_f64().unwrap_or(0.0) as f32,
                terms,
                logistic: task == Task::Binary,
            })
        }
        "tree_piecewise" if params.get("constant_prob").is_some() => Ok(V1Target::Constant(
            params["constant_prob"].as_f64().unwrap_or(0.0) as f32,
        )),
        "tree_piecewise" => {
            let left = array_i64(&params["children_left"], "children_left")?;
            let right = array_i64(&params["children_right"], "children_right")?;
            let feature = array_i64(&params["feature"], "feature")?;
            let threshold = array_f32(&params["threshold"], "threshold")?;
            let values = array_f32(&params["value"], "value")?;
            let leaf_sigma = params
                .get("leaf_sigma")
                .map(|entry| array_f32(entry, "leaf_sigma"))
                .transpose()?
                .unwrap_or_else(|| vec![0.0; left.len()]);
            if [
                right.len(),
                feature.len(),
                threshold.len(),
                values.len(),
                leaf_sigma.len(),
            ]
            .iter()
            .any(|length| *length != left.len())
            {
                return Err(DopeError::Codec("V1 tree arrays differ in length".into()));
            }
            Ok(V1Target::Tree {
                left,
                right,
                feature,
                threshold,
                value: values,
                leaf_sigma,
            })
        }
        kind => Err(DopeError::Unsupported(format!("V1 target {kind}"))),
    }
}

fn interpolate(xs: &[f64], ys: &[f64], value: f64) -> f64 {
    let upper = xs.partition_point(|point| *point < value).min(xs.len() - 1);
    if upper == 0 {
        return ys[0];
    }
    let lower = upper - 1;
    let fraction = (value - xs[lower]) / (xs[upper] - xs[lower]).max(f64::EPSILON);
    ys[lower] + fraction * (ys[upper] - ys[lower])
}

fn sample_v1(value: &Value, options: SampleOptions) -> Result<Vec<f32>> {
    let shell = value
        .get("shell")
        .ok_or_else(|| DopeError::Codec("V1 shell missing".into()))?;
    let width = shell
        .get("p")
        .and_then(Value::as_u64)
        .ok_or_else(|| DopeError::Codec("V1 width missing".into()))? as usize;
    let task = Task::parse(shell.get("task").and_then(Value::as_str).unwrap_or(""))
        .ok_or_else(|| DopeError::Codec("V1 task invalid".into()))?;
    let seed = options
        .seed
        .unwrap_or(shell.get("seed").and_then(Value::as_u64).unwrap_or(1729));
    let marginals = parse_v1_marginals(value)?;
    if marginals.len() != width {
        return Err(DopeError::Codec("V1 marginal width mismatch".into()));
    }
    let dependence = parse_v1_dependence(value, width)?;
    let target = parse_v1_target(value, task)?;
    let residual = value.get("residual").cloned().unwrap_or_default();
    let stride = width + 1;
    let mut output = vec![0.0f32; options.rows * stride];
    output
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(row_index, row)| {
            let mut probabilities = vec![0.0f64; width];
            match &dependence {
                V1Dependence::Independent => {
                    for (feature, slot) in probabilities.iter_mut().enumerate() {
                        *slot = uniform(seed, row_index, feature, 0x1234);
                    }
                }
                V1Dependence::Gaussian { cholesky, width } => {
                    let normals: Vec<f64> = (0..*width)
                        .map(|feature| inverse_normal(uniform(seed, row_index, feature, 0x1234)))
                        .collect();
                    for feature in 0..*width {
                        probabilities[feature] = normal_cdf(
                            (0..=feature)
                                .map(|index| cholesky[feature * *width + index] * normals[index])
                                .sum(),
                        );
                    }
                }
            }
            for (feature, marginal) in marginals.iter().enumerate() {
                let probability = probabilities[feature];
                row[feature] = match marginal {
                    V1Marginal::Bernoulli(threshold) => (probability <= *threshold) as u8 as f32,
                    V1Marginal::Grid(values, masses) => {
                        let threshold = probability * masses.iter().sum::<f64>();
                        let mut cumulative = 0.0;
                        let mut selected = *values.last().unwrap_or(&0.0);
                        for (value, mass) in values.iter().zip(masses) {
                            cumulative += mass;
                            if threshold <= cumulative {
                                selected = *value;
                                break;
                            }
                        }
                        selected as f32
                    }
                    V1Marginal::Beta(beta) => {
                        beta.inverse_cdf(probability.clamp(1e-9, 1.0 - 1e-9)) as f32
                    }
                    V1Marginal::ZeroInflatedBeta(zero, beta) => {
                        if probability <= *zero {
                            0.0
                        } else {
                            beta.inverse_cdf(
                                ((probability - zero) / (1.0 - zero)).clamp(1e-9, 1.0 - 1e-9),
                            ) as f32
                        }
                    }
                    V1Marginal::Quantile(qs, values) => interpolate(qs, values, probability) as f32,
                };
            }
            let (prediction, leaf_noise) = match &target {
                V1Target::Linear {
                    intercept,
                    terms,
                    logistic,
                } => {
                    let mut prediction = *intercept
                        + terms
                            .iter()
                            .map(|(feature, coefficient)| coefficient * row[*feature])
                            .sum::<f32>();
                    if *logistic {
                        prediction = 1.0 / (1.0 + (-prediction).exp());
                    }
                    (prediction, None)
                }
                V1Target::Constant(probability) => (*probability, None),
                V1Target::Tree {
                    left,
                    right,
                    feature,
                    threshold,
                    value,
                    leaf_sigma,
                } => {
                    let mut node = 0usize;
                    while left[node] != -1 {
                        node = if row[feature[node] as usize] <= threshold[node] {
                            left[node] as usize
                        } else {
                            right[node] as usize
                        };
                    }
                    (value[node], Some(leaf_sigma[node]))
                }
            };
            row[width] = if task == Task::Binary {
                (uniform(seed, row_index, width, 0x9876) as f32 <= prediction) as u8 as f32
            } else {
                let sigma = leaf_noise.unwrap_or_else(|| {
                    residual
                        .get("params")
                        .and_then(|params| params.get("sigma"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0) as f32
                });
                (prediction
                    + sigma * inverse_normal(uniform(seed, row_index, width, 0x9876)) as f32)
                    .clamp(0.0, 1.0)
            };
        });
    Ok(output)
}

pub fn sample_kernel(kernel: &LoadedKernel, options: SampleOptions) -> Result<Vec<f32>> {
    match kernel {
        LoadedKernel::V3(kernel) | LoadedKernel::V2(kernel) => sample_v2(kernel, options),
        LoadedKernel::V1(value) => sample_v1(value, options),
    }
}

pub fn sample_kernel_to_csv(
    kernel: &LoadedKernel,
    options: SampleOptions,
    path: &Path,
) -> Result<()> {
    let data = sample_kernel(kernel, options)?;
    let width = match kernel {
        LoadedKernel::V3(kernel) | LoadedKernel::V2(kernel) => kernel.features as usize + 1,
        LoadedKernel::V1(value) => value["shell"]["p"].as_u64().unwrap_or(0) as usize + 1,
    };
    let file = File::create(path).map_err(|error| io_error(path, error))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    for row in data.chunks_exact(width) {
        for (column, value) in row.iter().enumerate() {
            if column > 0 {
                writer
                    .write_all(b",")
                    .map_err(|error| io_error(path, error))?;
            }
            if value.is_nan() {
                writer
                    .write_all(b"nan")
                    .map_err(|error| io_error(path, error))?;
            } else {
                write!(writer, "{value:.8}").map_err(|error| io_error(path, error))?;
            }
        }
        writer
            .write_all(b"\n")
            .map_err(|error| io_error(path, error))?;
    }
    writer.flush().map_err(|error| io_error(path, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompileOptions, compile_kernel_from_arrays};

    #[test]
    fn seeded_sampling_is_exact_and_target_last() {
        let rows = 50;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| [0.2, row as f32 / rows as f32])
            .collect();
        let target: Vec<f32> = (0..rows).map(|row| row as f32 / rows as f32).collect();
        let compiled = compile_kernel_from_arrays(
            &features,
            &target,
            rows,
            2,
            Task::Regression,
            &CompileOptions::default(),
        )
        .unwrap();
        let loaded = LoadedKernel::V2(Box::new(
            crate::codec::decode_kernel(&compiled.artifact).unwrap(),
        ));
        let first = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 100,
                seed: Some(8),
            },
        )
        .unwrap();
        let second = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 100,
                seed: Some(8),
            },
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 300);
        assert!(first.chunks_exact(3).all(|row| row[2].is_finite()));
    }

    #[test]
    fn binary_targets_remain_binary() {
        let rows = 80;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| [row as f32 / rows as f32, (row % 3) as f32 / 2.0])
            .collect();
        let target: Vec<f32> = (0..rows).map(|row| (row > 40) as u8 as f32).collect();
        let compiled = compile_kernel_from_arrays(
            &features,
            &target,
            rows,
            2,
            Task::Binary,
            &CompileOptions::default(),
        )
        .unwrap();
        let loaded = LoadedKernel::V2(Box::new(
            crate::codec::decode_kernel(&compiled.artifact).unwrap(),
        ));
        let sampled = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 200,
                seed: Some(11),
            },
        )
        .unwrap();
        assert!(
            sampled
                .chunks_exact(3)
                .all(|row| row[2] == 0.0 || row[2] == 1.0)
        );
    }

    #[test]
    fn missingness_is_explicit_and_deterministic() {
        let rows = 100;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| {
                [
                    if row % 4 == 0 {
                        f32::NAN
                    } else {
                        row as f32 / rows as f32
                    },
                    0.3,
                ]
            })
            .collect();
        let target: Vec<f32> = (0..rows).map(|row| row as f32 / rows as f32).collect();
        let compiled = compile_kernel_from_arrays(
            &features,
            &target,
            rows,
            2,
            Task::Regression,
            &CompileOptions::default(),
        )
        .unwrap();
        let loaded = LoadedKernel::V2(Box::new(
            crate::codec::decode_kernel(&compiled.artifact).unwrap(),
        ));
        let first = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 1_000,
                seed: Some(12),
            },
        )
        .unwrap();
        let second = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 1_000,
                seed: Some(12),
            },
        )
        .unwrap();
        assert!(
            first
                .iter()
                .zip(&second)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        let missing = first.chunks_exact(3).filter(|row| row[0].is_nan()).count();
        assert!((180..=320).contains(&missing));
        assert!(
            first
                .chunks_exact(3)
                .all(|row| row[2].is_finite() && (0.0..=1.0).contains(&row[2]))
        );
    }
}
