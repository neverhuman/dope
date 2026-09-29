
fn quantized_kernel(kernel: &Kernel, bits: u8) -> Kernel {
    let mut output = kernel.clone();
    output.quantization_bits = bits;
    for schema in &mut output.schema {
        schema.missing_probability = quantize(schema.missing_probability, bits, true);
        schema.impute = quantize(schema.impute, bits, true);
        match &mut schema.transform {
            crate::model::Transform::Identity => {}
            crate::model::Transform::Log1p { scale } => {
                *scale = quantize(*scale, bits, false).max(1e-4)
            }
            crate::model::Transform::Power { exponent } => {
                *exponent = quantize(*exponent, bits, false).max(1e-4)
            }
            crate::model::Transform::Logit { epsilon } => {
                *epsilon = quantize(*epsilon, bits, true).clamp(1e-4, 0.499)
            }
            crate::model::Transform::Winsorize { lower, upper } => {
                *lower = quantize(*lower, bits, true);
                *upper = quantize(*upper, bits, true).max(*lower + 1e-4);
            }
        }
    }
    for marginal in &mut output.marginals {
        quantize_marginal(marginal, bits);
    }
    let (dependence, target, noise) = output
        .symbolic_mut()
        .expect("symbolic tournament only quantizes symbolic kernels");
    quantize_dependence(dependence, bits);
    match target {
        Target::SparseLinear { intercept, terms } | Target::SparseLogistic { intercept, terms } => {
            *intercept = quantize(*intercept, bits, false);
            terms
                .iter_mut()
                .for_each(|term| term.coefficient = quantize(term.coefficient, bits, false));
            terms.sort_by_key(|term| term.feature);
        }
        Target::SparseGam {
            intercept, terms, ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            quantize_additive(terms, bits);
        }
        Target::Ga2m {
            intercept,
            main_terms,
            interactions,
            ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            quantize_additive(main_terms, bits);
            for term in interactions {
                quantize_strict_unit(&mut term.left_knots, bits);
                quantize_strict_unit(&mut term.right_knots, bits);
                term.values
                    .iter_mut()
                    .for_each(|value| *value = quantize(*value, bits, false));
            }
        }
        Target::Mars {
            intercept, terms, ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            for term in terms {
                term.coefficient = quantize(term.coefficient, bits, false);
                term.factors
                    .iter_mut()
                    .for_each(|factor| factor.knot = quantize(factor.knot, bits, true));
            }
        }
        Target::ObliviousTree {
            thresholds, leaves, ..
        } => {
            thresholds
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, true));
            leaves
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
        }
        Target::CompactNeuralResidual {
            intercept,
            linear_terms,
            input_weights,
            hidden_biases,
            output_weights,
            ..
        } => {
            *intercept = quantize(*intercept, bits, false);
            linear_terms
                .iter_mut()
                .for_each(|term| term.coefficient = quantize(term.coefficient, bits, false));
            input_weights
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
            hidden_biases
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
            output_weights
                .iter_mut()
                .for_each(|value| *value = quantize(*value, bits, false));
        }
    }
    match noise {
        Noise::Homoscedastic { sigma } => *sigma = quantize(*sigma, bits, true),
        Noise::BernoulliCalibration { base_rate } => *base_rate = quantize(*base_rate, bits, true),
        Noise::Heteroscedastic {
            intercept,
            terms,
            minimum_sigma,
            maximum_sigma,
        } => {
            *intercept = quantize(*intercept, bits, false);
            terms
                .iter_mut()
                .for_each(|term| term.coefficient = quantize(term.coefficient, bits, false));
            *minimum_sigma = quantize(*minimum_sigma, bits, true);
            *maximum_sigma = quantize(*maximum_sigma, bits, true).max(*minimum_sigma);
        }
        Noise::IsotonicCalibration {
            knots,
            probabilities,
        } => {
            quantize_strict_unit(knots, bits);
            quantize_strict_unit(probabilities, bits);
        }
    }
    output
}

fn quantized_target_utility(kernel: &Kernel, table: &Table, completed: &[Vec<f32>]) -> f64 {
    let (_, target, _) = kernel
        .symbolic()
        .expect("symbolic utility only evaluates symbolic kernels");
    let mean = table.target.iter().sum::<f32>() / table.rows.max(1) as f32;
    match table
        .target
        .iter()
        .enumerate()
        .fold((0.0f64, 0.0f64), |(loss, null), (row, actual)| {
            let row_values: Vec<f32> = completed.iter().map(|column| column[row]).collect();
            let (linear, logistic) = evaluate_target(target, &row_values);
            if logistic {
                let probability = f64::from(1.0 / (1.0 + (-linear.clamp(-30.0, 30.0)).exp()))
                    .clamp(1e-9, 1.0 - 1e-9);
                let base = f64::from(mean).clamp(1e-9, 1.0 - 1e-9);
                let y = f64::from(*actual);
                (
                    loss - (y * probability.ln() + (1.0 - y) * (1.0 - probability).ln()),
                    null - (y * base.ln() + (1.0 - y) * (1.0 - base).ln()),
                )
            } else {
                (
                    loss + f64::from(*actual - linear.clamp(0.0, 1.0)).powi(2),
                    null + f64::from(*actual - mean).powi(2),
                )
            }
        }) {
        (loss, null_loss) if null_loss <= 1e-12 => {
            (1.0 - loss / table.rows.max(1) as f64).clamp(0.0, 1.0)
        }
        (loss, null_loss) => (1.0 - loss / null_loss).clamp(0.0, 1.0),
    }
}

fn quantized_driver_agreement(kernel: &Kernel, screen: &[(usize, f32)]) -> f64 {
    let (_, target, _) = kernel
        .symbolic()
        .expect("symbolic driver metric only evaluates symbolic kernels");
    let mut encoded: Vec<LinearTerm> = match target {
        Target::SparseLinear { terms, .. } | Target::SparseLogistic { terms, .. } => terms.clone(),
        Target::SparseGam { terms, .. } => terms
            .iter()
            .map(|term| LinearTerm {
                feature: term.feature,
                coefficient: term
                    .values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f32::max),
            })
            .collect(),
        Target::Ga2m {
            main_terms,
            interactions,
            ..
        } => main_terms
            .iter()
            .map(|term| LinearTerm {
                feature: term.feature,
                coefficient: term
                    .values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f32::max),
            })
            .chain(interactions.iter().flat_map(|term| {
                let coefficient = term
                    .values
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0, f32::max);
                [
                    LinearTerm {
                        feature: term.left,
                        coefficient,
                    },
                    LinearTerm {
                        feature: term.right,
                        coefficient,
                    },
                ]
            }))
            .collect(),
        Target::Mars { terms, .. } => terms
            .iter()
            .flat_map(|term| {
                term.factors.iter().map(|factor| LinearTerm {
                    feature: factor.feature,
                    coefficient: term.coefficient.abs(),
                })
            })
            .collect(),
        Target::ObliviousTree { features, .. } => features
            .iter()
            .map(|feature| LinearTerm {
                feature: *feature,
                coefficient: 1.0,
            })
            .collect(),
        Target::CompactNeuralResidual {
            linear_terms,
            hidden_features,
            hidden_width,
            input_weights,
            output_weights,
            ..
        } => {
            let mut influence: BTreeMap<u32, f32> = linear_terms
                .iter()
                .map(|term| (term.feature, term.coefficient.abs()))
                .collect();
            for (input, feature) in hidden_features.iter().enumerate() {
                let neural = (0..usize::from(*hidden_width))
                    .map(|unit| {
                        (output_weights[unit] * input_weights[unit * hidden_features.len() + input])
                            .abs()
                    })
                    .sum::<f32>();
                *influence.entry(*feature).or_default() += neural;
            }
            influence
                .into_iter()
                .map(|(feature, coefficient)| LinearTerm {
                    feature,
                    coefficient,
                })
                .collect()
        }
    };
    if screen.is_empty() {
        return 1.0;
    }
    let k = 20
        .min(5.max((screen.len() as f64 * 0.1).ceil() as usize))
        .min(screen.len())
        .max(1);
    let actual: std::collections::HashSet<usize> =
        screen.iter().take(k).map(|(feature, _)| *feature).collect();
    encoded.sort_by(|left, right| {
        right
            .coefficient
            .abs()
            .total_cmp(&left.coefficient.abs())
            .then_with(|| left.feature.cmp(&right.feature))
    });
    let overlap = encoded
        .iter()
        .take(k)
        .filter(|term| actual.contains(&(term.feature as usize)))
        .count() as f64
        / k as f64;
    let gains: BTreeMap<usize, f64> = screen
        .iter()
        .map(|(feature, score)| (*feature, f64::from(*score)))
        .collect();
    let ideal = screen
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, (_, score))| f64::from(*score) / ((rank + 2) as f64).log2())
        .sum::<f64>();
    let dcg = encoded
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, term)| {
            gains.get(&(term.feature as usize)).copied().unwrap_or(0.0) / ((rank + 2) as f64).log2()
        })
        .sum::<f64>();
    0.5 * overlap
        + 0.5
            * if ideal <= 1e-12 {
                1.0
            } else {
                (dcg / ideal).clamp(0.0, 1.0)
            }
}

fn stable_seed(table: &Table) -> u64 {
    let mut signatures: Vec<[u8; 32]> = table
        .columns
        .iter()
        .map(|column| {
            let mut values = column.clone();
            values.sort_by(f32::total_cmp);
            let mut hasher = blake3::Hasher::new();
            for value in values {
                hasher.update(&value.to_bits().to_le_bytes());
            }
            *hasher.finalize().as_bytes()
        })
        .collect();
    signatures.sort_unstable();
    let mut hasher = blake3::Hasher::new();
    for signature in signatures {
        hasher.update(&signature);
    }
    let mut target = table.target.clone();
    target.sort_by(f32::total_cmp);
    for value in target {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap())
}

fn pareto(candidates: &[CandidateReport]) -> Vec<CandidateReport> {
    let mut frontier: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            !candidates.iter().any(|other| {
                other.candidate_id != candidate.candidate_id
                    && other.artifact_bytes <= candidate.artifact_bytes
                    && other.score >= candidate.score
                    && (other.artifact_bytes < candidate.artifact_bytes
                        || other.score > candidate.score)
            })
        })
        .cloned()
        .collect();
    frontier.sort_by(|left, right| {
        left.artifact_bytes
            .cmp(&right.artifact_bytes)
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    frontier
}

fn maximum_normalized_gate_shortfall(candidate: &CandidateReport) -> f64 {
    [
        ((0.99 - candidate.utility_retention) / 0.99).max(0.0),
        ((0.95 - candidate.driver_agreement) / 0.95).max(0.0),
        ((0.90 - candidate.proxy_joint_fidelity) / 0.90).max(0.0),
        ((candidate.proxy_membership_auc - 0.60) / 0.40).max(0.0),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn target_parameter_bytes(target: &Target) -> usize {
    match target {
        Target::SparseLinear { terms, .. } | Target::SparseLogistic { terms, .. } => {
            8 + terms.len() * 8
        }
        Target::SparseGam { terms, .. } => {
            9 + terms
                .iter()
                .map(|term| 8 + 8 * term.knots.len())
                .sum::<usize>()
        }
        Target::Ga2m {
            main_terms,
            interactions,
            ..
        } => {
            9 + main_terms
                .iter()
                .map(|term| 8 + 8 * term.knots.len())
                .sum::<usize>()
                + interactions
                    .iter()
                    .map(|term| {
                        8 + 4 * (term.left_knots.len() + term.right_knots.len() + term.values.len())
                    })
                    .sum::<usize>()
        }
        Target::Mars { terms, .. } => {
            9 + terms
                .iter()
                .map(|term| 8 + term.factors.len() * 9)
                .sum::<usize>()
        }
        Target::ObliviousTree {
            features, leaves, ..
        } => 5 + features.len() * 8 + leaves.len() * 4,
        Target::CompactNeuralResidual {
            linear_terms,
            hidden_features,
            input_weights,
            hidden_biases,
            output_weights,
            ..
        } => {
            12 + linear_terms.len() * 8
                + hidden_features.len() * 4
                + 4 * (input_weights.len() + hidden_biases.len() + output_weights.len())
        }
    }
}

pub fn compile_kernel_from_arrays(
    features: &[f32],
    target: &[f32],
    rows: usize,
    columns: usize,
    task: Task,
    options: &CompileOptions,
) -> Result<CompileResult> {
    let table = Table::from_arrays_with_policy(
        features,
        target,
        rows,
        columns,
        task,
        &options.release_policy,
    )?;
    compile_table(&table, task, options)
}

pub fn compile_kernel_from_dir(
    path: &Path,
    task: Task,
    options: &CompileOptions,
) -> Result<CompileResult> {
    let table = Table::read_dataset_dir_with_policy(path, task, &options.release_policy)?;
    compile_table(&table, task, options)
}

fn backend_profile(id: &str) -> Option<(usize, u8, usize, u8)> {
    Some(match id {
        "independent_quantile" => (2, 0, 0, 8),
        "chow_liu" => (2, 1, 0, 8),
        "triangular_autoregressive" => (2, 2, 0, 8),
        "sparse_gaussian_copula" => (3, 3, 1, 10),
        "poet_factor" => (3, 4, 1, 10),
        "truncated_c_vine" => (3, 5, 2, 10),
        "two_component_copula_mixture" => (3, 6, 2, 10),
        "compact_neural_residual_symbolic" => (2, 2, 5, 8),
        "tabsds_rank" => (4, 1, 2, 12),
        "adversarial_random_forest" => (3, 6, 4, 10),
        "conditional_density_forest" => (4, 3, 4, 12),
        "forest_diffusion_vp" => (4, 4, 4, 12),
        "forest_diffusion_flow_matching" => (4, 5, 4, 12),
        "tvae" => (2, 4, 5, 8),
        "ctgan" => (3, 6, 5, 10),
        "taegan" => (3, 5, 5, 10),
        "tabddpm" => (4, 3, 5, 12),
        "tabsyn" => (4, 1, 5, 12),
        "single_table_autoregressive_transformer" => (3, 2, 5, 12),
        "masked_diffusion_transformer" => (4, 5, 5, 16),
        "legacy_tvae" => (2, 4, 5, 8),
        "legacy_ctgan" => (3, 6, 5, 10),
        "legacy_taegan" => (3, 5, 5, 10),
        "legacy_tabddpm" => (4, 3, 5, 12),
        "legacy_tabsyn" => (4, 1, 5, 12),
        "legacy_single_table_autoregressive_transformer" => (3, 2, 5, 12),
        "legacy_masked_diffusion_transformer" => (4, 5, 5, 16),
        "synthetic_pretrained_cross_table" => (3, 1, 1, 10),
        "per_dataset_adapter" => (3, 4, 1, 10),
        "symbolic_latent_diffusion_residual" => (4, 3, 5, 12),
        "symbolic_autoregressive_residual" => (2, 2, 5, 10),
        _ => return None,
    })
}

fn dependence_name(family: u8) -> &'static str {
    match family {
        0 => "independence",
        1 => "chow_liu",
        2 => "triangular_autoregressive",
        3 => "sparse_gaussian_copula",
        4 => "poet_factor",
        5 => "truncated_c_vine",
        6 => "two_component_copula_mixture",
        _ => unreachable!("bounded dependence family"),
    }
}

fn strongest_sparse_edges(
    correlation: Option<&Vec<f32>>,
    baseline_edges: &[CopulaEdge],
    width: usize,
    limit: usize,
) -> Vec<CopulaEdge> {
    let Some(matrix) = correlation else {
        return baseline_edges.iter().take(limit).cloned().collect();
    };
    let mut pairs = Vec::with_capacity(width.saturating_mul(width.saturating_sub(1)) / 2);
    for left in 0..width {
        for right in left + 1..width {
            pairs.push(CopulaEdge {
                parent: left as u32,
                child: right as u32,
                correlation: matrix[left * width + right],
            });
        }
    }
    pairs.sort_by(|left, right| {
        right
            .correlation
            .abs()
            .total_cmp(&left.correlation.abs())
            .then_with(|| left.parent.cmp(&right.parent))
            .then_with(|| left.child.cmp(&right.child))
    });
    pairs.truncate(limit);
    pairs
}

fn poet_dependence(
    correlation: Option<&Vec<f32>>,
    baseline_edges: &[CopulaEdge],
    width: usize,
) -> Dependence {
    let rank = width.clamp(1, 4);
    let mut loadings = Vec::with_capacity(width * rank);
    for feature in 0..width {
        for factor in 0..rank {
            let loading = correlation.map_or_else(
                || f32::from(feature == factor),
                |matrix| matrix[feature * width + factor] / (rank as f32).sqrt(),
            );
            loadings.push(FactorLoading {
                feature: feature as u32,
                factor: factor as u8,
                loading: loading.clamp(-1.0, 1.0),
            });
        }
    }
    Dependence::Poet {
        rank: rank as u8,
        loadings,
        residual_edges: strongest_sparse_edges(
            correlation,
            baseline_edges,
            width,
            width.saturating_mul(2),
        ),
    }
}

fn vine_dependence(
    correlation: Option<&Vec<f32>>,
    baseline_edges: &[CopulaEdge],
    width: usize,
) -> Dependence {
    let sparse = strongest_sparse_edges(correlation, baseline_edges, width, width.saturating_mul(3));
    Dependence::Vine {
        edges: sparse
            .into_iter()
            .enumerate()
            .map(|(index, edge)| VineEdge {
                left: edge.parent,
                right: edge.child,
                conditioning_depth: (index % 4) as u8,
                parameter: edge.correlation,
            })
            .collect(),
    }
}

fn neural_architecture(id: &str) -> Option<NeuralArchitecture> {
    match id {
        "tvae" | "micro_tvae_4_16" | "micro_tvae_8_24" | "micro_tvae_12_32" => {
            Some(NeuralArchitecture::Tvae)
        }
        "single_table_autoregressive_transformer" | "tiny_mat_16_2_32" | "tiny_mat_24_3_48" => {
            Some(NeuralArchitecture::MaskedAutoregressiveTransformer)
        }
        "tabsyn" => Some(NeuralArchitecture::TabSyn),
        "tabddpm_direct_rank" => Some(NeuralArchitecture::TabDdpm),
        _ => None,
    }
}

fn neural_profile(id: &str) -> NeuralProfile {
    match id {
        "micro_tvae_4_16" => NeuralProfile::MicroTvae4,
        "micro_tvae_8_24" => NeuralProfile::MicroTvae8,
        "micro_tvae_12_32" => NeuralProfile::MicroTvae12,
        "tiny_mat_16_2_32" => NeuralProfile::TinyMat16,
        "tiny_mat_24_3_48" => NeuralProfile::TinyMat24,
        _ => NeuralProfile::Full,
    }
}

fn randomized_gaussian_ranks(values: &[f32], seed: u64, column: usize) -> (Vec<f32>, Vec<f32>) {
    let mut observed = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    observed.sort_by(f32::total_cmp);
    let normal = Normal::new(0.0, 1.0).expect("standard normal");
    let mut ranks = Vec::with_capacity(values.len());
    let mut missing = Vec::with_capacity(values.len());
    for (row, value) in values.iter().enumerate() {
        if !value.is_finite() || observed.is_empty() {
            ranks.push(0.0);
            missing.push(1.0);
            continue;
        }
        let lower = observed.partition_point(|candidate| *candidate < *value - 1e-7);
        let upper = observed.partition_point(|candidate| *candidate <= *value + 1e-7);
        let counter = seed
            ^ (row as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ (column as u64).wrapping_mul(0xa076_1d64_78bd_642f);
        let jitter = (splitmix64(counter) >> 11) as f64 / (1u64 << 53) as f64;
        let probability = (lower as f64 + jitter * (upper - lower).max(1) as f64 + 0.5)
            / (observed.len() + 1) as f64;
        ranks.push(normal.inverse_cdf(probability.clamp(1e-6, 1.0 - 1e-6)) as f32);
        missing.push(0.0);
    }
    (ranks, missing)
}

fn correlation_tree_permutation(feature_ranks: &[Vec<f32>], target_ranks: &[f32]) -> Vec<u32> {
    let width = feature_ranks.len();
    if width <= 1 {
        return (0..width as u32).collect();
    }
    let root = (0..width)
        .max_by(|&left, &right| {
            dot(&feature_ranks[left], target_ranks)
                .abs()
                .total_cmp(&dot(&feature_ranks[right], target_ranks).abs())
                .then_with(|| right.cmp(&left))
        })
        .unwrap_or(0);
    let mut selected = vec![false; width];
    let mut parent = vec![root; width];
    let mut best = vec![f32::NEG_INFINITY; width];
    selected[root] = true;
    for feature in 0..width {
        best[feature] = dot(&feature_ranks[root], &feature_ranks[feature]).abs();
    }
    for _ in 1..width {
        let child = (0..width)
            .filter(|&feature| !selected[feature])
            .max_by(|&left, &right| {
                best[left]
                    .total_cmp(&best[right])
                    .then_with(|| right.cmp(&left))
            })
            .expect("unselected feature");
        selected[child] = true;
        for feature in 0..width {
            let score = dot(&feature_ranks[child], &feature_ranks[feature]).abs();
            if !selected[feature] && score > best[feature] {
                best[feature] = score;
                parent[feature] = child;
            }
        }
    }
    let mut children = vec![Vec::new(); width];
    for child in 0..width {
        if child != root {
            children[parent[child]].push(child);
        }
    }
    for entries in &mut children {
        entries.sort_unstable();
    }
    let mut permutation = Vec::with_capacity(width);
    let mut stack = vec![root];
    while let Some(feature) = stack.pop() {
        permutation.push(feature as u32);
        stack.extend(children[feature].iter().rev());
    }
    permutation
}

fn neural_training_data(
    table: &Table,
    fits: &[ColumnFit],
    seed: u64,
) -> crate::neural_train::NeuralTrainingData {
    let tokens = table.features + 1;
    let mut rank_columns = Vec::with_capacity(tokens);
    let mut missing_columns = Vec::with_capacity(tokens);
    for (column, values) in table
        .columns
        .iter()
        .chain(std::iter::once(&table.target))
        .enumerate()
    {
        let (ranks, missing) = randomized_gaussian_ranks(values, seed, column);
        rank_columns.push(ranks);
        missing_columns.push(missing);
    }
    let feature_permutation = correlation_tree_permutation(
        &rank_columns[..table.features],
        &rank_columns[table.features],
    );
    let mut ranks = Vec::with_capacity(table.rows * tokens);
    let mut missing = Vec::with_capacity(table.rows * tokens);
    for row in 0..table.rows {
        for column in feature_permutation
            .iter()
            .map(|value| *value as usize)
            .chain(std::iter::once(table.features))
        {
            ranks.push(rank_columns[column][row]);
            missing.push(missing_columns[column][row]);
        }
    }
    let target_fit = fit_column(&table.target);
    let target_marginal = target_fit
        .discrete
        .unwrap_or_else(|| Marginal::QuantileSpline {
            values: target_fit.quantiles[4].clone(),
        });
    let mut sorted_target = table.target.clone();
    sorted_target.sort_by(f32::total_cmp);
    let lower_tail = quantile(&sorted_target, 0.1);
    let upper_tail = quantile(&sorted_target, 0.9);
    let binary = table.target.iter().all(|value| matches!(*value, 0.0 | 1.0));
    let ones = table.target.iter().filter(|value| **value >= 0.5).count();
    let zeroes = table.rows.saturating_sub(ones);
    let rare_class = if ones < zeroes {
        Some(true)
    } else if zeroes < ones {
        Some(false)
    } else {
        None
    };
    let row_weights = table
        .target
        .iter()
        .map(|target| {
            if (binary && rare_class == Some(*target >= 0.5))
                || (!binary && (*target <= lower_tail || *target >= upper_tail))
            {
                2.0
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let normalization = fits
        .iter()
        .map(|fit| RankNormalization {
            location: 0.0,
            scale: 1.0,
            missing_probability: fit.schema.missing_probability,
            randomized_discrete: !matches!(
                fit.schema.kind,
                SchemaKind::Continuous | SchemaKind::Inflated
            ),
        })
        .chain(std::iter::once(RankNormalization {
            location: 0.0,
            scale: 1.0,
            missing_probability: 0.0,
            randomized_discrete: binary,
        }))
        .collect::<Vec<_>>();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"joint-rank-training-v1");
    for value in &ranks {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    for value in &missing {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    crate::neural_train::NeuralTrainingData {
        rows: table.rows,
        features: table.features,
        ranks,
        missing,
        row_weights,
        feature_permutation,
        target_marginal,
        normalization,
        training_hash: hasher.finalize().to_hex().to_string(),
    }
}
