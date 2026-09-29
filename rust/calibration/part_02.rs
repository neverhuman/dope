

fn row_major(table: &Table) -> Vec<f32> {
    let mut values = vec![0.0; table.rows * table.features];
    for row in 0..table.rows {
        for column in 0..table.features {
            values[row * table.features + column] = table.columns[column][row];
        }
    }
    values
}

fn candidate_descriptor(candidate: &CandidateReport) -> CandidateDescriptor {
    CandidateDescriptor {
        quantization_bits: candidate.quantization_bits,
        marginal_knots: candidate.marginal_knots,
        dependence: candidate.dependence.clone(),
        target: candidate.target.clone(),
        effective_bytes: candidate.artifact_bytes,
    }
}

fn normalized_shortfall(candidate: &CandidateReport) -> f64 {
    [
        ((0.99 - candidate.utility_retention) / 0.99).max(0.0),
        ((0.95 - candidate.driver_agreement) / 0.95).max(0.0),
        ((0.90 - candidate.proxy_joint_fidelity) / 0.90).max(0.0),
        ((candidate.proxy_membership_auc - 0.60) / 0.40).max(0.0),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn candidate_regrets(candidates: &[CandidateReport]) -> BTreeMap<String, f64> {
    let compliant_oracle = candidates
        .iter()
        .filter(|candidate| candidate.compliant)
        .min_by(|left, right| {
            left.artifact_bytes
                .cmp(&right.artifact_bytes)
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        });
    let fallback_oracle = candidates.iter().min_by(|left, right| {
        normalized_shortfall(left)
            .total_cmp(&normalized_shortfall(right))
            .then_with(|| left.artifact_bytes.cmp(&right.artifact_bytes))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let oracle = compliant_oracle.or(fallback_oracle);
    let Some(oracle) = oracle else {
        return BTreeMap::new();
    };
    candidates
        .iter()
        .map(|candidate| {
            let regret = if compliant_oracle.is_some() {
                if !candidate.compliant {
                    1.0
                } else {
                    ((candidate.artifact_bytes as f64 / oracle.artifact_bytes.max(1) as f64)
                        .log2()
                        .max(0.0)
                        / 2.0_f64)
                        .min(1.0)
                }
            } else {
                let gate = (normalized_shortfall(candidate) - normalized_shortfall(oracle))
                    .clamp(0.0, 0.95);
                let bytes = 0.05
                    * (candidate.artifact_bytes as f64 / oracle.artifact_bytes.max(1) as f64)
                        .log2()
                        .clamp(0.0, 1.0);
                (gate + bytes).min(1.0)
            };
            (candidate.candidate_id.clone(), regret)
        })
        .collect()
}

type LabelCompilation = (
    Vec<LabelSample>,
    BTreeMap<String, usize>,
    HashMap<Vec<u16>, Vec<String>>,
);

fn compile_labels(packed: &PackedCorpus, manifest: &SplitManifest) -> Result<LabelCompilation> {
    let records: BTreeMap<&str, &DatasetRecord> = manifest
        .datasets
        .iter()
        .filter(|record| record.duplicate_of.is_none())
        .map(|record| (record.dataset_id.as_str(), record))
        .collect();
    let mut labels = Vec::new();
    let mut operators = BTreeMap::<String, usize>::new();
    let mut shapes = HashMap::<Vec<u16>, Vec<String>>::new();
    for dataset_id in packed
        .dataset_ids(None)
        .into_iter()
        .filter(|dataset_id| records[*dataset_id].split != "test")
    {
        let record = records[dataset_id];
        let table = packed.read(dataset_id)?;
        let descriptor = describe_table(&table, record.task);
        let compiled = compile_kernel_from_arrays(
            &row_major(&table),
            &table.target,
            table.rows,
            table.features,
            record.task,
            &CompileOptions {
                seed: Some(RELEASE_SEED),
                beam_width: Some(usize::MAX),
                ..Default::default()
            },
        )?;
        let kernel = decode_kernel(&compiled.artifact)?;
        let (kernel_dependence, kernel_target, kernel_noise) =
            kernel.symbolic().ok_or_else(|| {
                DopeError::Data("language calibration requires a symbolic kernel".into())
            })?;
        let dependence = match kernel_dependence {
            crate::model::Dependence::Independent => "independence",
            crate::model::Dependence::ChowLiu { .. } => "chow_liu",
            crate::model::Dependence::Triangular { .. } => "triangular_autoregressive",
            crate::model::Dependence::GaussianCopula { .. } => "gaussian_copula",
            crate::model::Dependence::SparseGraph { .. } => "sparse_graph",
            crate::model::Dependence::Poet { .. } => "poet",
            crate::model::Dependence::Vine { .. } => "vine",
            crate::model::Dependence::Mixture { .. } => "mixture",
        };
        *operators.entry(dependence.into()).or_default() += 1;
        *operators
            .entry(
                match kernel_target {
                    Target::SparseLinear { .. } => "sparse_linear",
                    Target::SparseLogistic { .. } => "sparse_logistic",
                    Target::SparseGam { .. } => "sparse_gam",
                    Target::Ga2m { .. } => "ga2m",
                    Target::Mars { .. } => "mars",
                    Target::ObliviousTree { .. } => "oblivious_tree",
                    Target::CompactNeuralResidual { .. } => "compact_neural_residual",
                }
                .into(),
            )
            .or_default() += 1;
        *operators
            .entry(
                match kernel_noise {
                    Noise::Homoscedastic { .. } => "homoscedastic",
                    Noise::BernoulliCalibration { .. } => "bernoulli_calibration",
                    Noise::Heteroscedastic { .. } => "heteroscedastic",
                    Noise::IsotonicCalibration { .. } => "isotonic_calibration",
                }
                .into(),
            )
            .or_default() += 1;
        for marginal in kernel.marginals {
            let name = match &marginal {
                Marginal::Constant { .. } => "constant",
                Marginal::Bernoulli { .. } => "bernoulli",
                Marginal::Grid { .. } => "grid",
                Marginal::QuantileSpline { .. } => "quantile_spline",
                Marginal::Beta { .. } => "beta",
                Marginal::Gaussian { .. } => "gaussian",
                Marginal::Histogram { .. } => "histogram",
                Marginal::ZeroInflated { .. } => "zero_inflated",
            };
            *operators.entry(name.into()).or_default() += 1;
            if let Marginal::QuantileSpline { values } = marginal {
                let low = values[0];
                let high = *values.last().expect("validated spline is non-empty");
                let scale = (high - low).max(1e-9);
                let shape: Vec<u16> = values
                    .iter()
                    .map(|value| (((value - low) / scale).clamp(0.0, 1.0) * 1000.0).round() as u16)
                    .collect();
                shapes
                    .entry(shape)
                    .or_default()
                    .push(record.content_fingerprint.clone());
            }
        }
        let regrets = candidate_regrets(&compiled.candidates);
        for candidate in &compiled.candidates {
            labels.push(LabelSample {
                dataset_id: dataset_id.into(),
                lineage: record.group_id.clone(),
                split: record.split.clone(),
                task: record.task,
                candidate_id: candidate.candidate_id.clone(),
                features: descriptor_features(&descriptor, &candidate_descriptor(candidate)),
                regret: regrets[&candidate.candidate_id],
                compliant: candidate.compliant,
            });
        }
    }
    Ok((labels, operators, shapes))
}

fn dictionary_entries(shapes: HashMap<Vec<u16>, Vec<String>>) -> Vec<DictionaryEntry> {
    let mut ordered: Vec<_> = shapes.into_iter().collect();
    ordered.sort_by(|(left, left_refs), (right, right_refs)| {
        right_refs
            .len()
            .cmp(&left_refs.len())
            .then_with(|| left.cmp(right))
    });
    let mut entries = Vec::new();
    for (shape, mut provenance) in ordered {
        provenance.sort();
        provenance.dedup();
        let reference_count = provenance.len();
        let piecewise: Vec<f32> = shape.iter().map(|value| *value as f32 / 1000.0).collect();
        let byte_cost = shape.len() * 2 + 8;
        let savings = reference_count as isize * byte_cost as isize
            - (byte_cost as isize + reference_count as isize * 2);
        if reference_count < 2 || savings <= 0 {
            continue;
        }
        let visualization = piecewise
            .iter()
            .enumerate()
            .map(|(index, value)| [index as f32 / (piecewise.len() - 1).max(1) as f32, *value])
            .collect();
        entries.push(DictionaryEntry {
            id: entries.len(),
            kind: "quantile_shape".into(),
            equation: "Q(u)=lo+(hi-lo)*linear_interp(u,points)".into(),
            piecewise,
            visualization,
            provenance,
            byte_cost,
            reference_count,
            corpus_byte_savings: savings,
        });
    }
    entries
}

fn quantile_knots(samples: &[&LabelSample], feature: usize) -> Vec<f64> {
    let mut values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.features[feature])
        .collect();
    values.sort_by(f64::total_cmp);
    (0..SPLINE_KNOTS)
        .map(|index| values[index * values.len().saturating_sub(1) / (SPLINE_KNOTS - 1)])
        .collect()
}

fn bin(knots: &[f64], value: f64) -> usize {
    knots
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - value).abs().total_cmp(&(*right - value).abs()))
        .map_or(0, |(index, _)| index)
}

fn group_shrink(coefficients: &mut [f64], regularization: f64) {
    let norm = coefficients
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let scale = if norm <= regularization {
        0.0
    } else {
        1.0 - regularization / norm
    };
    coefficients.iter_mut().for_each(|value| *value *= scale);
}

fn dataset_selection_losses(
    samples: &[&LabelSample],
    model: &Ga2mModel,
) -> Vec<(String, String, Task, f64, bool, f64, f64)> {
    let mut grouped = BTreeMap::<&str, Vec<&&LabelSample>>::new();
    for sample in samples {
        grouped.entry(&sample.dataset_id).or_default().push(sample);
    }
    grouped
        .into_values()
        .map(|candidates| {
            let selected = candidates
                .iter()
                .min_by(|left, right| {
                    model
                        .predict(&left.features)
                        .total_cmp(&model.predict(&right.features))
                        .then_with(|| left.candidate_id.cmp(&right.candidate_id))
                })
                .expect("dataset has candidate labels");
            let random = candidates.iter().map(|sample| sample.regret).sum::<f64>()
                / candidates.len() as f64;
            (
                selected.dataset_id.clone(),
                selected.lineage.clone(),
                selected.task,
                selected.regret,
                !selected.compliant && candidates.iter().any(|candidate| candidate.compliant),
                random,
                0.0,
            )
        })
        .collect()
}

fn macro_loss(samples: &[&LabelSample], model: &Ga2mModel) -> f64 {
    let losses = dataset_selection_losses(samples, model);
    let mut lineages = BTreeMap::<&str, Vec<f64>>::new();
    for (_, lineage, _, loss, _, _, _) in &losses {
        lineages.entry(lineage).or_default().push(*loss);
    }
    lineages
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .sum::<f64>()
        / lineages.len().max(1) as f64
}

include!("training_metrics.rs");
