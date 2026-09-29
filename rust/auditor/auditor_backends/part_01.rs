
#[cfg(feature = "gpu-training")]
struct GpuFeatureTransformer;

fn bounded_auditor_tables(
    train: &Table,
    tests: &[&Table],
    seed: u64,
    row_limit: usize,
    feature_limit: usize,
) -> (Table, Vec<Table>, Vec<usize>) {
    debug_assert!(row_limit > 0);
    debug_assert!(feature_limit > 0);
    let fit_rows = if train.rows > row_limit {
        let mut ranked_rows = (0..train.rows)
            .map(|row| {
                let mut generator =
                    SplitMix64(seed ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                (generator.next(), row)
            })
            .collect::<Vec<_>>();
        ranked_rows.sort_unstable();
        ranked_rows.truncate(row_limit);
        ranked_rows
            .into_iter()
            .map(|(_, row)| row)
            .collect::<Vec<_>>()
    } else {
        (0..train.rows).collect::<Vec<_>>()
    };
    let means = train
        .columns
        .iter()
        .map(|column| {
            let (sum, count) = fit_rows.iter().fold((0.0, 0usize), |(sum, count), &row| {
                let value = column[row];
                if value.is_finite() {
                    (sum + f64::from(value), count + 1)
                } else {
                    (sum, count)
                }
            });
            sum / count.max(1) as f64
        })
        .collect::<Vec<_>>();
    let target_mean = fit_rows
        .iter()
        .map(|&row| f64::from(train.target[row]))
        .sum::<f64>()
        / fit_rows.len().max(1) as f64;
    let mut ranked = (0..train.features)
        .map(|feature| {
            let (covariance, variance) =
                fit_rows
                    .iter()
                    .fold((0.0f64, 0.0f64), |(covariance, variance), &row| {
                        let centered = value(train, feature, row, &means) - means[feature];
                        (
                            covariance + centered * (f64::from(train.target[row]) - target_mean),
                            variance + centered * centered,
                        )
                    });
            (covariance.abs() + variance.sqrt() * 1e-9, feature)
        })
        .collect::<Vec<_>>();
    if train.features > feature_limit {
        ranked.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
    } else {
        ranked.sort_by_key(|(_, feature)| *feature);
    }
    ranked.truncate(feature_limit);
    let selected = ranked
        .into_iter()
        .map(|(_, feature)| feature)
        .collect::<Vec<_>>();
    let project_test = |table: &Table| Table {
        rows: table.rows,
        features: selected.len(),
        columns: selected
            .iter()
            .map(|&feature| table.columns[feature].clone())
            .collect(),
        target: table.target.clone(),
    };
    let bounded_train = Table {
        rows: fit_rows.len(),
        features: selected.len(),
        columns: selected
            .iter()
            .map(|&feature| {
                fit_rows
                    .iter()
                    .map(|&row| train.columns[feature][row])
                    .collect()
            })
            .collect(),
        target: fit_rows.iter().map(|&row| train.target[row]).collect(),
    };
    (
        bounded_train,
        tests.iter().map(|table| project_test(table)).collect(),
        selected,
    )
}

#[cfg(feature = "gpu-training")]
fn feature_transformer_predictions(
    train: &Table,
    tests: &[&Table],
    task: Task,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    use tch::nn::Module;

    let device = gpu_device()?;
    crate::libtorch::with_seeded_libtorch(seed, || {
        let (x_fit, y_fit, _x_validation, _y_validation, x_tests) =
            gpu_tensors(train, tests, seed, device)?;
        let store = tch::nn::VarStore::new(device);
        let root = store.root();
        let embedding = 32i64;
        let token = tch::nn::linear(&root / "token", 1, embedding, Default::default());
        let feature_embedding = root.var(
            "feature_embedding",
            &[1, train.features as i64, embedding],
            tch::nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let query = root.var(
            "query",
            &[1, 1, embedding],
            tch::nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let keys = tch::nn::linear(&root / "keys", embedding, embedding, Default::default());
        let values = tch::nn::linear(&root / "values", embedding, embedding, Default::default());
        let feed1 = tch::nn::linear(&root / "feed1", embedding, 64, Default::default());
        let feed2 = tch::nn::linear(&root / "feed2", 64, embedding, Default::default());
        let representation = |xs: &tch::Tensor| {
            let tokens = (token.forward(&xs.unsqueeze(-1)) + &feature_embedding).relu();
            let key = keys.forward(&tokens);
            let attention =
                ((key * &query).sum_dim_intlist([-1].as_slice(), false, tch::Kind::Float)
                    / (embedding as f64).sqrt())
                .softmax(-1, tch::Kind::Float);
            let pooled = (values.forward(&tokens) * attention.unsqueeze(-1)).sum_dim_intlist(
                [1].as_slice(),
                false,
                tch::Kind::Float,
            );
            let transformed = feed2.forward(&feed1.forward(&pooled).gelu("none"));
            pooled + transformed
        };
        // Wide tables otherwise materialize rows × features × embedding for the
        // entire table. Batching only the row-independent representation keeps
        // exact model weights and row order while bounding CUDA working memory.
        const ROW_BATCH: i64 = 128;
        let batched_representation =
            |xs: &tch::Tensor| transform_row_batches(xs, ROW_BATCH, representation);
        let head = ridge_head(&batched_representation(&x_fit), &y_fit, task)?;
        x_tests
            .into_iter()
            .zip(tests)
            .map(|(test_tensor, test)| {
                tensor_predictions(
                    ridge_predict(&batched_representation(&test_tensor), &head),
                    task,
                    test.rows,
                )
            })
            .collect()
    })
}

#[cfg(feature = "gpu-training")]
impl AuditorBackend for GpuFeatureTransformer {
    fn id(&self) -> &'static str {
        "gpu_tabular_feature_transformer"
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<Vec<f64>> {
        Ok(self.predict_many(train, &[test], task, seed)?.remove(0))
    }

    fn predict_many(
        &self,
        train: &Table,
        tests: &[&Table],
        task: Task,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        for test in tests {
            validate_tables(train, test)?;
        }
        feature_transformer_predictions(train, tests, task, seed)
    }

    fn predict_two(
        &self,
        train: &Table,
        first: &Table,
        second: &Table,
        task: Task,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        validate_tables(train, first)?;
        validate_tables(train, second)?;
        let mut predictions = self.predict_many(train, &[first, second], task, seed)?;
        let second = predictions.pop().expect("two feature-transformer outputs");
        let first = predictions.pop().expect("two feature-transformer outputs");
        Ok((first, second))
    }
}

fn validate_tables(train: &Table, test: &Table) -> Result<()> {
    if train.rows == 0 || test.rows == 0 || train.features != test.features {
        return Err(DopeError::Data(
            "auditor train/test tables must be nonempty and schema-aligned".into(),
        ));
    }
    Ok(())
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}

fn loss(task: Task, target: &[f32], predictions: &[f64]) -> Result<f64> {
    if target.len() != predictions.len() || target.is_empty() {
        return Err(DopeError::Data("auditor prediction shape mismatch".into()));
    }
    Ok(match task {
        Task::Regression => {
            target
                .iter()
                .zip(predictions)
                .map(|(&target, &prediction)| (f64::from(target) - prediction).powi(2))
                .sum::<f64>()
                / target.len() as f64
        }
        Task::Binary => {
            target
                .iter()
                .zip(predictions)
                .map(|(&target, &prediction)| {
                    let probability = prediction.clamp(1e-7, 1.0 - 1e-7);
                    -f64::from(target) * probability.ln()
                        - (1.0 - f64::from(target)) * (1.0 - probability).ln()
                })
                .sum::<f64>()
                / target.len() as f64
        }
    })
}

fn column_means(table: &Table) -> Vec<f64> {
    table
        .columns
        .iter()
        .map(|column| {
            let observed = column.iter().copied().filter(|value| value.is_finite());
            let (sum, count) = observed.fold((0.0, 0usize), |(sum, count), value| {
                (sum + f64::from(value), count + 1)
            });
            sum / count.max(1) as f64
        })
        .collect()
}

fn value(table: &Table, feature: usize, row: usize, means: &[f64]) -> f64 {
    let value = table.columns[feature][row];
    if value.is_finite() {
        f64::from(value)
    } else {
        means[feature]
    }
}

struct ElasticNet;

impl AuditorBackend for ElasticNet {
    fn id(&self) -> &'static str {
        "elastic_net_glm"
    }

    #[allow(clippy::needless_range_loop)]
    fn predict(&self, train: &Table, test: &Table, task: Task, _seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let target_mean = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / train.rows as f64;
        let mut intercept = if task == Task::Binary {
            (target_mean.clamp(1e-5, 1.0 - 1e-5) / (1.0 - target_mean.clamp(1e-5, 1.0 - 1e-5))).ln()
        } else {
            target_mean
        };
        let mut weights = vec![0.0f64; train.features];
        let train_values = (0..train.rows)
            .map(|row| {
                (0..train.features)
                    .map(|feature| value(train, feature, row, &means))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let learning_rate = 0.15 / (train.features.max(1) as f64).sqrt();
        for _ in 0..240 {
            let mut intercept_gradient = 0.0;
            let mut gradients = vec![0.0; train.features];
            for row in 0..train.rows {
                let linear = intercept
                    + weights
                        .iter()
                        .zip(&train_values[row])
                        .map(|(weight, value)| weight * value)
                        .sum::<f64>();
                let prediction = if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear
                };
                let residual = prediction - f64::from(train.target[row]);
                intercept_gradient += residual;
                for feature in 0..train.features {
                    gradients[feature] += residual * train_values[row][feature];
                }
            }
            intercept -= learning_rate * intercept_gradient / train.rows as f64;
            for feature in 0..train.features {
                let gradient = gradients[feature] / train.rows as f64 + 1e-3 * weights[feature];
                weights[feature] -= learning_rate * gradient;
                let shrinkage = learning_rate * 1e-4;
                weights[feature] =
                    weights[feature].signum() * (weights[feature].abs() - shrinkage).max(0.0);
            }
        }
        Ok((0..test.rows)
            .map(|row| {
                let linear = intercept
                    + (0..test.features)
                        .map(|feature| weights[feature] * value(test, feature, row, &means))
                        .sum::<f64>();
                if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

fn bin(value: f64) -> usize {
    (value.clamp(0.0, 1.0) * 8.0).floor().min(7.0) as usize
}

struct Ga2m;