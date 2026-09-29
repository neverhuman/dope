

impl AuditorBackend for BoundedAuditor {
    fn id(&self) -> &'static str {
        self.inner.id()
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
        let (bounded_train, bounded_tests, _) = bounded_auditor_tables(
            train,
            tests,
            seed,
            AUDITOR_TRAIN_ROW_LIMIT,
            AUDITOR_FEATURE_LIMIT,
        );
        let bounded_refs = bounded_tests.iter().collect::<Vec<_>>();
        self.inner
            .predict_many(&bounded_train, &bounded_refs, task, seed)
    }

    fn permutation_importance(
        &self,
        train: &Table,
        holdout: &Table,
        task: Task,
        seed: u64,
    ) -> Result<PermutationImportance> {
        let (bounded_train, mut bounded_tests, selected) = bounded_auditor_tables(
            train,
            &[holdout],
            seed,
            AUDITOR_TRAIN_ROW_LIMIT,
            AUDITOR_FEATURE_LIMIT,
        );
        let mut importance = permutation_importance_for(
            self.inner.as_ref(),
            &bounded_train,
            &bounded_tests.remove(0),
            task,
            seed,
        )?;
        importance.feature_indices = importance
            .feature_indices
            .into_iter()
            .map(|feature| selected[feature])
            .collect();
        Ok(importance)
    }
}

pub fn auditor_backend(id: &str) -> Result<Box<dyn AuditorBackend>> {
    let inner: Box<dyn AuditorBackend> = match id {
        "elastic_net_glm" => Box::new(ElasticNet),
        "ga2m" => Box::new(Ga2m),
        "extremely_randomized_trees" => Box::new(ExtraTrees),
        "histogram_gbdt" => Box::new(HistogramGbdt),
        #[cfg(feature = "gpu-training")]
        "gpu_residual_mlp_ensemble" => Box::new(GpuResidualMlpEnsemble),
        #[cfg(feature = "gpu-training")]
        "gpu_tabular_feature_transformer" => Box::new(GpuFeatureTransformer),
        #[cfg(not(feature = "gpu-training"))]
        "gpu_residual_mlp_ensemble" | "gpu_tabular_feature_transformer" => {
            return Err(DopeError::Unsupported(format!(
                "frozen GPU auditor {id} requires the tch-rs/libtorch evaluation binary"
            )));
        }
        _ => return Err(DopeError::Data(format!("unknown frozen auditor {id}"))),
    };
    Ok(Box::new(BoundedAuditor { inner }))
}

#[cfg(feature = "gpu-training")]
fn gpu_error(context: &str, error: tch::TchError) -> DopeError {
    DopeError::Data(format!("{context}: {error}"))
}

#[cfg(feature = "gpu-training")]
fn gpu_device() -> Result<tch::Device> {
    if !tch::Cuda::is_available() {
        return Err(DopeError::Unsupported(
            "frozen GPU auditor requires a CUDA device".into(),
        ));
    }
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":4096:8") {
        return Err(DopeError::Unsupported(
            "frozen GPU auditor requires CUBLAS_WORKSPACE_CONFIG=:4096:8".into(),
        ));
    }
    Ok(tch::Device::Cuda(0))
}

#[cfg(feature = "gpu-training")]
fn deterministic_split(rows: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut fit = Vec::new();
    let mut validation = Vec::new();
    let mut rng = SplitMix64(seed ^ 0x6a09_e667_f3bc_c909);
    for row in 0..rows {
        if rng.next().is_multiple_of(5) {
            validation.push(row);
        } else {
            fit.push(row);
        }
    }
    if fit.is_empty() {
        fit.push(validation.pop().unwrap_or(0));
    }
    if validation.is_empty() {
        validation.push(*fit.last().unwrap_or(&0));
    }
    (fit, validation)
}

#[cfg(feature = "gpu-training")]
fn normalized_rows(
    table: &Table,
    indices: &[usize],
    means: &[f64],
    deviations: &[f64],
) -> Vec<f32> {
    let mut output = Vec::with_capacity(indices.len() * table.features);
    for &row in indices {
        for feature in 0..table.features {
            output.push(
                ((value(table, feature, row, means) - means[feature]) / deviations[feature]) as f32,
            );
        }
    }
    output
}

#[cfg(feature = "gpu-training")]
fn gpu_tensors(
    train: &Table,
    tests: &[&Table],
    seed: u64,
    device: tch::Device,
) -> Result<(
    tch::Tensor,
    tch::Tensor,
    tch::Tensor,
    tch::Tensor,
    Vec<tch::Tensor>,
)> {
    let means = column_means(train);
    let deviations = (0..train.features)
        .map(|feature| {
            let variance = (0..train.rows)
                .map(|row| {
                    let delta = value(train, feature, row, &means) - means[feature];
                    delta * delta
                })
                .sum::<f64>()
                / train.rows.max(1) as f64;
            variance.sqrt().max(1e-5)
        })
        .collect::<Vec<_>>();
    let (fit, validation) = deterministic_split(train.rows, seed);
    let x_fit = tch::Tensor::from_slice(&normalized_rows(train, &fit, &means, &deviations))
        .view([fit.len() as i64, train.features as i64])
        .to_device(device);
    let y_fit =
        tch::Tensor::from_slice(&fit.iter().map(|&row| train.target[row]).collect::<Vec<_>>())
            .view([fit.len() as i64, 1])
            .to_device(device);
    let x_validation =
        tch::Tensor::from_slice(&normalized_rows(train, &validation, &means, &deviations))
            .view([validation.len() as i64, train.features as i64])
            .to_device(device);
    let y_validation = tch::Tensor::from_slice(
        &validation
            .iter()
            .map(|&row| train.target[row])
            .collect::<Vec<_>>(),
    )
    .view([validation.len() as i64, 1])
    .to_device(device);
    let x_tests = tests
        .iter()
        .map(|test| {
            if test.features != train.features {
                return Err(DopeError::Data(
                    "GPU auditor train/test feature widths differ".into(),
                ));
            }
            let indices = (0..test.rows).collect::<Vec<_>>();
            Ok(
                tch::Tensor::from_slice(&normalized_rows(test, &indices, &means, &deviations))
                    .view([test.rows as i64, test.features as i64])
                    .to_device(device),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((x_fit, y_fit, x_validation, y_validation, x_tests))
}

#[cfg(feature = "gpu-training")]
fn tensor_predictions(output: tch::Tensor, task: Task, rows: usize) -> Result<Vec<f64>> {
    let output = if task == Task::Binary {
        output.sigmoid()
    } else {
        output.clamp(0.0, 1.0)
    }
    .to_device(tch::Device::Cpu)
    .view([rows as i64]);
    let values = Vec::<f32>::try_from(output).map_err(|error| gpu_error("GPU output", error))?;
    Ok(values.into_iter().map(f64::from).collect())
}

#[cfg(feature = "gpu-training")]
fn ridge_head(hidden: &tch::Tensor, target: &tch::Tensor, task: Task) -> Result<tch::Tensor> {
    let rows = hidden.size()[0];
    let width = hidden.size()[1];
    let ones = tch::Tensor::ones([rows, 1], (tch::Kind::Float, hidden.device()));
    let design = tch::Tensor::cat(&[&ones, hidden], 1);
    let fitted_target = if task == Task::Binary {
        target * 4.0 - 2.0
    } else {
        target.shallow_clone()
    };
    let transpose = design.transpose(0, 1);
    let gram = transpose.matmul(&design)
        + tch::Tensor::eye(width + 1, (tch::Kind::Float, hidden.device())) * 1e-3;
    let right = transpose.matmul(&fitted_target);
    tch::Tensor::f_linalg_solve(&gram, &right, true)
        .map_err(|error| gpu_error("GPU ridge solve", error))
}

#[cfg(feature = "gpu-training")]
fn ridge_predict(hidden: &tch::Tensor, head: &tch::Tensor) -> tch::Tensor {
    let ones = tch::Tensor::ones([hidden.size()[0], 1], (tch::Kind::Float, hidden.device()));
    tch::Tensor::cat(&[&ones, hidden], 1).matmul(head)
}

#[cfg(feature = "gpu-training")]
fn transform_row_batches(
    rows: &tch::Tensor,
    batch_rows: i64,
    transform: impl Fn(&tch::Tensor) -> tch::Tensor,
) -> tch::Tensor {
    let transformed = rows
        .split(batch_rows, 0)
        .into_iter()
        .map(|batch| transform(&batch))
        .collect::<Vec<_>>();
    tch::Tensor::cat(&transformed, 0)
}

#[cfg(feature = "gpu-training")]
fn residual_mlp_member(
    train: &Table,
    tests: &[&Table],
    task: Task,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    use tch::nn::Module;

    let device = gpu_device()?;
    tch::manual_seed(seed as i64);
    tch::Cuda::manual_seed_all(seed);
    let (x_fit, y_fit, _x_validation, _y_validation, x_tests) =
        gpu_tensors(train, tests, seed, device)?;
    let store = tch::nn::VarStore::new(device);
    let root = store.root();
    let width = 128i64;
    let input = tch::nn::linear(
        &root / "input",
        train.features as i64,
        width,
        Default::default(),
    );
    let hidden1 = tch::nn::linear(&root / "hidden1", width, width, Default::default());
    let hidden2 = tch::nn::linear(&root / "hidden2", width, width, Default::default());
    let representation = |xs: &tch::Tensor| {
        let base = input.forward(xs).relu();
        let residual = hidden2.forward(&hidden1.forward(&base).relu()).relu();
        base + residual
    };
    let head = ridge_head(&representation(&x_fit), &y_fit, task)?;
    x_tests
        .into_iter()
        .zip(tests)
        .map(|(test_tensor, test)| {
            tensor_predictions(
                ridge_predict(&representation(&test_tensor), &head),
                task,
                test.rows,
            )
        })
        .collect()
}

#[cfg(feature = "gpu-training")]
struct GpuResidualMlpEnsemble;

#[cfg(feature = "gpu-training")]
impl AuditorBackend for GpuResidualMlpEnsemble {
    fn id(&self) -> &'static str {
        "gpu_residual_mlp_ensemble"
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
        crate::libtorch::with_seeded_libtorch(seed, || {
            let members = (0..3)
                .map(|member| {
                    residual_mlp_member(train, tests, task, seed.wrapping_add(member * 1_000_003))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(tests
                .iter()
                .enumerate()
                .map(|(test, table)| {
                    (0..table.rows)
                        .map(|row| {
                            members.iter().map(|values| values[test][row]).sum::<f64>() / 3.0
                        })
                        .collect()
                })
                .collect())
        })
    }

    fn predict_two(
        &self,
        train: &Table,
        first: &Table,
        second: &Table,
        task: Task,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut predictions = self.predict_many(train, &[first, second], task, seed)?;
        let second = predictions.pop().expect("two residual-MLP outputs");
        let first = predictions.pop().expect("two residual-MLP outputs");
        Ok((first, second))
    }
}

include!("auditor_backends.rs");
