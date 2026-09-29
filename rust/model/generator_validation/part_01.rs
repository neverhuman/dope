
impl JointGenerator {
    pub fn validate(&self, features: usize, task: Task) -> std::result::Result<(), String> {
        let tokens = features + 1;
        if self.feature_permutation.len() != features {
            return Err("neural feature permutation width mismatch".into());
        }
        let mut permutation = self.feature_permutation.clone();
        permutation.sort_unstable();
        if permutation != (0..features as u32).collect::<Vec<_>>() {
            return Err("neural feature permutation is not bijective".into());
        }
        validate_marginal(&self.target_marginal, 0)?;
        if task == Task::Binary && !matches!(self.target_marginal, Marginal::Bernoulli { .. }) {
            return Err("binary neural target requires a Bernoulli marginal".into());
        }
        if self.normalization.len() != tokens
            || self.normalization.iter().any(|normalization| {
                !normalization.location.is_finite()
                    || !normalization.scale.is_finite()
                    || normalization.scale <= 0.0
                    || !valid_probability(normalization.missing_probability)
            })
            || !valid_hash(&self.training_hash)
            || !valid_hash(&self.implementation_hash)
        {
            return Err("invalid neural normalization or provenance".into());
        }
        match (&self.architecture, &self.network) {
            (NeuralArchitecture::Tvae, JointNetwork::Tvae { decoder }) => {
                let (latent, hidden) = self
                    .profile
                    .tvae_dimensions()
                    .ok_or("TVAE profile is incompatible with its architecture")?;
                decoder.validate(tokens, latent, hidden)
            }
            (
                NeuralArchitecture::MaskedAutoregressiveTransformer,
                JointNetwork::MaskedAutoregressiveTransformer { transformer },
            ) => {
                let (width, _, ff_width) = self
                    .profile
                    .transformer_dimensions()
                    .ok_or("transformer profile is incompatible with its architecture")?;
                transformer.validate(tokens, width, ff_width)
            }
            (
                NeuralArchitecture::TabSyn,
                JointNetwork::TabSyn {
                    decoder,
                    denoiser,
                    alpha_cumprod,
                    inference_timesteps,
                },
            ) => {
                if self.profile != NeuralProfile::Full {
                    return Err("TabSyn uses the full profile".into());
                }
                decoder.validate(tokens, NEURAL_LATENT_WIDTH, NEURAL_HIDDEN_WIDTH)?;
                denoiser.validate()?;
                if alpha_cumprod.len() != DIFFUSION_TRAIN_STEPS
                    || alpha_cumprod
                        .iter()
                        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                    || !alpha_cumprod.windows(2).all(|pair| pair[0] >= pair[1])
                    || inference_timesteps.len() != DIFFUSION_INFERENCE_STEPS
                    || !inference_timesteps.windows(2).all(|pair| pair[0] > pair[1])
                    || inference_timesteps
                        .first()
                        .is_none_or(|step| usize::from(*step) >= DIFFUSION_TRAIN_STEPS)
                {
                    return Err("invalid TabSyn diffusion schedule".into());
                }
                Ok(())
            }
            (
                NeuralArchitecture::TabDdpm,
                JointNetwork::TabDdpm {
                    denoiser,
                    alpha_cumprod,
                    inference_timesteps,
                },
            ) => {
                if self.profile != NeuralProfile::Full {
                    return Err("TabDDPM uses the full profile".into());
                }
                denoiser.validate_dimensions(tokens + NEURAL_LATENT_WIDTH, tokens)?;
                validate_diffusion_schedule(alpha_cumprod, inference_timesteps)
            }
            _ => Err("neural architecture tag and tensor payload disagree".into()),
        }
    }
}

fn validate_diffusion_schedule(
    alpha_cumprod: &[f32],
    inference_timesteps: &[u8],
) -> std::result::Result<(), String> {
    if alpha_cumprod.len() != DIFFUSION_TRAIN_STEPS
        || alpha_cumprod
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || !alpha_cumprod.windows(2).all(|pair| pair[0] >= pair[1])
        || inference_timesteps.len() != DIFFUSION_INFERENCE_STEPS
        || !inference_timesteps.windows(2).all(|pair| pair[0] > pair[1])
        || inference_timesteps
            .first()
            .is_none_or(|step| usize::from(*step) >= DIFFUSION_TRAIN_STEPS)
    {
        return Err("invalid diffusion schedule".into());
    }
    Ok(())
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KernelProgram {
    Symbolic {
        dependence: Dependence,
        target: Target,
        noise: Noise,
    },
    NeuralJoint(JointGenerator),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Kernel {
    pub task: Task,
    pub rows_fitted: u64,
    pub features: u32,
    pub seed: u64,
    pub seed_policy: u8,
    pub quantization_bits: u8,
    pub compliant: bool,
    pub schema: Vec<ColumnSchema>,
    pub marginals: Vec<Marginal>,
    pub program: KernelProgram,
}

impl Kernel {
    pub fn validate(&self) -> std::result::Result<(), String> {
        let width = self.features as usize;
        if width == 0 {
            return Err("a data kernel requires at least one positional feature".into());
        }
        if self.schema.len() != width || self.marginals.len() != width {
            return Err("schema and marginal counts must equal positional feature count".into());
        }
        if !matches!(self.quantization_bits, 4 | 6 | 8 | 10 | 12 | 16) {
            return Err("unsupported quantization profile".into());
        }
        for schema in &self.schema {
            if !(0.0..=1.0).contains(&schema.missing_probability) || !schema.impute.is_finite() {
                return Err("invalid schema parameter".into());
            }
            match schema.transform {
                Transform::Identity => {}
                Transform::Log1p { scale } if scale.is_finite() && scale > 0.0 => {}
                Transform::Power { exponent } if exponent.is_finite() && exponent > 0.0 => {}
                Transform::Logit { epsilon } if (0.0..0.5).contains(&epsilon) => {}
                Transform::Winsorize { lower, upper }
                    if lower.is_finite()
                        && upper.is_finite()
                        && 0.0 <= lower
                        && lower < upper
                        && upper <= 1.0 => {}
                _ => return Err("invalid bounded transform".into()),
            }
        }
        for marginal in &self.marginals {
            validate_marginal(marginal, 0)?;
        }
        match &self.program {
            KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } => {
                validate_dependence(dependence, width, 0)?;
                validate_target(target, width)?;
                validate_noise(noise, width)?;
            }
            KernelProgram::NeuralJoint(generator) => generator.validate(width, self.task)?,
        }
        Ok(())
    }

    pub fn symbolic(&self) -> Option<(&Dependence, &Target, &Noise)> {
        match &self.program {
            KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } => Some((dependence, target, noise)),
            KernelProgram::NeuralJoint(_) => None,
        }
    }

    pub fn symbolic_mut(&mut self) -> Option<(&mut Dependence, &mut Target, &mut Noise)> {
        match &mut self.program {
            KernelProgram::Symbolic {
                dependence,
                target,
                noise,
            } => Some((dependence, target, noise)),
            KernelProgram::NeuralJoint(_) => None,
        }
    }
}

fn valid_probability(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn increasing(values: &[f32]) -> bool {
    values
        .windows(2)
        .all(|pair| pair[0].is_finite() && pair[0] <= pair[1])
        && values.last().is_none_or(|value| value.is_finite())
}

fn validate_marginal(marginal: &Marginal, depth: usize) -> std::result::Result<(), String> {
    if depth > 2 {
        return Err("marginal nesting exceeds its bound".into());
    }
    match marginal {
        Marginal::Constant { value } if valid_probability(*value) => Ok(()),
        Marginal::Bernoulli { probability } if valid_probability(*probability) => Ok(()),
        Marginal::Grid {
            values,
            probabilities,
        } if !values.is_empty()
            && values.len() == probabilities.len()
            && values.len() <= 65_536
            && increasing(values)
            && probabilities.iter().all(|value| valid_probability(*value))
            && probabilities.iter().sum::<f32>() > 0.0 =>
        {
            Ok(())
        }
        Marginal::QuantileSpline { values }
            if matches!(values.len(), 5 | 9 | 17 | 33 | 65) && increasing(values) =>
        {
            Ok(())
        }
        Marginal::Beta { alpha, beta }
            if alpha.is_finite() && *alpha > 0.0 && beta.is_finite() && *beta > 0.0 =>
        {
            Ok(())
        }
        Marginal::Gaussian { mean, sigma }
            if valid_probability(*mean) && sigma.is_finite() && *sigma > 0.0 =>
        {
            Ok(())
        }
        Marginal::Histogram {
            edges,
            probabilities,
        } if (2..=257).contains(&edges.len())
            && probabilities.len() + 1 == edges.len()
            && increasing(edges)
            && probabilities.iter().all(|value| valid_probability(*value))
            && probabilities.iter().sum::<f32>() > 0.0 =>
        {
            Ok(())
        }
        Marginal::ZeroInflated {
            point,
            probability,
            base,
        } if valid_probability(*point) && valid_probability(*probability) => {
            validate_marginal(base, depth + 1)
        }
        _ => Err("invalid bounded marginal".into()),
    }
}

fn valid_edge(edge: &CopulaEdge, width: usize) -> bool {
    (edge.parent as usize) < width
        && (edge.child as usize) < width
        && edge.parent != edge.child
        && (-0.995..=0.995).contains(&edge.correlation)
}

fn validate_dependence(
    dependence: &Dependence,
    width: usize,
    depth: usize,
) -> std::result::Result<(), String> {
    if depth > 2 {
        return Err("dependence mixture nesting exceeds its bound".into());
    }
    match dependence {
        Dependence::Independent => Ok(()),
        Dependence::ChowLiu { root, edges } => {
            if *root as usize >= width || edges.len() != width.saturating_sub(1) {
                return Err("invalid Chow-Liu tree shape".into());
            }
            let mut seen = vec![false; width];
            seen[*root as usize] = true;
            for edge in edges {
                let parent = edge.parent as usize;
                let child = edge.child as usize;
                if !valid_edge(edge, width) || !seen[parent] || seen[child] {
                    return Err("Chow-Liu edges must be bounded and topologically ordered".into());
                }
                seen[child] = true;
            }
            Ok(())
        }
        Dependence::GaussianCopula { cholesky }
            if cholesky.len() == width * width
                && cholesky
                    .iter()
                    .all(|value| value.is_finite() && value.abs() <= 1.0) =>
        {
            Ok(())
        }
        Dependence::SparseGraph { edges }
            if edges.len() <= width.saturating_mul(8)
                && edges.iter().all(|edge| valid_edge(edge, width)) =>
        {
            Ok(())
        }
        Dependence::Poet {
            rank,
            loadings,
            residual_edges,
        } if (1..=16).contains(rank)
            && loadings.len() <= width * usize::from(*rank)
            && loadings.iter().all(|loading| {
                (loading.feature as usize) < width
                    && loading.factor < *rank
                    && loading.loading.is_finite()
                    && loading.loading.abs() <= 1.0
            })
            && residual_edges.len() <= width.saturating_mul(4)
            && residual_edges.iter().all(|edge| valid_edge(edge, width)) =>
        {
            Ok(())
        }
        Dependence::Vine { edges }
            if edges.len() <= width.saturating_mul(8)
                && edges.iter().all(|edge| {
                    (edge.left as usize) < width
                        && (edge.right as usize) < width
                        && edge.left != edge.right
                        && edge.conditioning_depth <= 8
                        && (-0.995..=0.995).contains(&edge.parameter)
                }) =>
        {
            Ok(())
        }
        Dependence::Mixture {
            weights,
            components,
        } if !components.is_empty()
            && components.len() <= 8
            && weights.len() == components.len()
            && weights.iter().all(|weight| valid_probability(*weight))
            && weights.iter().sum::<f32>() > 0.0 =>
        {
            components
                .iter()
                .try_for_each(|component| validate_dependence(component, width, depth + 1))
        }
        Dependence::Triangular { terms }
            if terms.len() <= width.saturating_mul(8)
                && terms.iter().all(|term| {
                    term.parent < term.child
                        && (term.child as usize) < width
                        && term.coefficient.is_finite()
                        && term.coefficient.abs() <= 0.995
                }) =>
        {
            Ok(())
        }
        _ => Err("invalid bounded dependence operator".into()),
    }
}

fn validate_additive(term: &AdditiveTerm, width: usize) -> bool {
    (term.feature as usize) < width
        && (3..=17).contains(&term.knots.len())
        && term.knots.len() == term.values.len()
        && increasing(&term.knots)
        && term.values.iter().all(|value| value.is_finite())
}