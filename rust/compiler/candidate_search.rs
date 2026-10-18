include!("candidate_search/part_01.rs");
include!("candidate_search/part_02.rs");

#[path = "candidate_search/staged_cache.rs"]
pub mod staged_cache;

use crate::typed_data::prepared::PreparedCompileInput;
use staged_cache::{CacheIdentity, DevelopmentCache, DevelopmentMetrics, SearchPartition};

fn search_error(code: &'static str) -> DopeError {
    DopeError::Data(code.into())
}

/// Development selection only. Partition tags are assertions by the trusted
/// split owner, not an OS custody boundary. Final query data has no entry here.
pub struct DevelopmentSelection<'a> {
    input: &'a PreparedCompileInput<'a>,
    digest: [u8; 32],
}

impl<'a> DevelopmentSelection<'a> {
    pub fn new(input: &'a PreparedCompileInput<'a>, partition: SearchPartition) -> Result<Self> {
        partition.require_selection()?;
        let bytes = input
            .view()
            .private_json()
            .map_err(|_| search_error("invalid_development_selection"))?;
        Ok(Self {
            input,
            digest: *blake3::hash(&bytes).as_bytes(),
        })
    }
}

/// Implement with a pinned development model panel: train on synthetic rows,
/// score TSTR and drivers on D_select. The owner retains D_select labels; a
/// worker adapter must send only query features across its worker boundary.
/// Synthetic rows use the decoded artifact's units; replay the retained D01
/// transform before comparing with PreparedView's original-unit targets.
pub trait DevelopmentEvaluator {
    fn evaluate(
        &mut self,
        synthetic: &[f32],
        features: usize,
        selection: &PreparedCompileInput<'_>,
        seed: u64,
    ) -> Result<DevelopmentMetrics>;
}

pub struct DevelopmentSearchOptions {
    pub identity: CacheIdentity,
    pub policy: ReleasePolicy,
    pub sample_rows: usize,
    pub max_candidates: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DevelopmentObservation {
    pub candidate_digest: [u8; 32],
    pub wire_bytes: usize,
    pub metrics: DevelopmentMetrics,
    pub cache_hit: bool,
}

/// Immutable stage-3 handoff. No query features, labels, evaluator, or mutable
/// cache are accepted by confirmation. This is replay evidence, not a seal or
/// scientific certification; protected evaluation remains evaluator-owned.
pub struct FrozenDevelopmentRecipe {
    artifact: Vec<u8>,
    recipe_digest: [u8; 32],
    sample_rows: usize,
    seed_schedule: Vec<u64>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct FrozenRecipeConfirmation {
    pub recipe_digest: [u8; 32],
    pub artifact_digest: [u8; 32],
    pub sample_digests: Vec<[u8; 32]>,
    pub wire_bytes: usize,
    pub rows_per_seed: usize,
    pub qualified: bool,
}

impl FrozenDevelopmentRecipe {
    pub fn artifact(&self) -> &[u8] {
        &self.artifact
    }

    pub fn digest(&self) -> [u8; 32] {
        self.recipe_digest
    }

    pub fn confirm(&self, expected_recipe_digest: [u8; 32]) -> Result<FrozenRecipeConfirmation> {
        if expected_recipe_digest != self.recipe_digest {
            return Err(search_error("frozen_recipe_mismatch"));
        }
        let decoded = crate::codec::decode_kernel(&self.artifact)
            .map_err(|_| search_error("candidate_reload_failed"))?;
        let mut sample_digests = Vec::new();
        for &seed in &self.seed_schedule {
            let sample = development_sample(&decoded, self.sample_rows, seed)?;
            let mut digest = blake3::Hasher::new();
            for value in sample {
                digest.update(&value.to_bits().to_le_bytes());
            }
            sample_digests.push(*digest.finalize().as_bytes());
        }
        Ok(FrozenRecipeConfirmation {
            recipe_digest: self.recipe_digest,
            artifact_digest: *blake3::hash(&self.artifact).as_bytes(),
            sample_digests,
            wire_bytes: self.artifact.len(),
            rows_per_seed: self.sample_rows,
            qualified: false,
        })
    }
}

pub struct DevelopmentSearchResult {
    pub recipe: FrozenDevelopmentRecipe,
    pub observations: Vec<DevelopmentObservation>,
    pub rejected_over_cap: usize,
}

/// Bridge from the existing fit-only compiler tournament. Its proxy-ranked
/// winner is ignored: every retained candidate is evaluated on D_select.
pub fn select_compiled_development(
    compiled: &CompileResult,
    selection: &DevelopmentSelection<'_>,
    options: &DevelopmentSearchOptions,
    cache: &mut DevelopmentCache,
    evaluator: &mut impl DevelopmentEvaluator,
) -> Result<DevelopmentSearchResult> {
    if compiled.candidate_artifacts.len() > options.max_candidates {
        return Err(search_error("development_search_budget_exceeded"));
    }
    let proposals = compiled
        .candidate_artifacts
        .values()
        .map(|bytes| {
            crate::codec::decode_kernel(bytes).map_err(|_| search_error("candidate_reload_failed"))
        })
        .collect::<Result<Vec<_>>>()?;
    search_development_candidates(&proposals, selection, options, cache, evaluator)
}

fn development_sample(kernel: &Kernel, rows: usize, seed: u64) -> Result<Vec<f32>> {
    let cells = rows.checked_mul(kernel.features as usize + 1);
    if rows == 0 || cells.is_none_or(|n| n > 1_000_000) {
        return Err(search_error("development_sample_budget_exceeded"));
    }
    crate::sample::sample_symbolic_support(
        kernel,
        crate::sample::SampleOptions {
            rows,
            seed: Some(seed),
        },
        true,
    )
    .map_err(|_| search_error("candidate_sample_failed"))
}

/// Stage 1 consumes bounded fit-only structural proposals (symbolic or neural)
/// from the existing compiler. Stage 2 reloads their exact wire representation,
/// samples without source access, and invokes only the D_select evaluator.
/// The returned winner is frozen before stage 3 can run. No training-fit proxy
/// is used as TSTR, and no outcome here establishes production eligibility.
pub fn search_development_candidates(
    proposals: &[Kernel],
    selection: &DevelopmentSelection<'_>,
    options: &DevelopmentSearchOptions,
    cache: &mut DevelopmentCache,
    evaluator: &mut impl DevelopmentEvaluator,
) -> Result<DevelopmentSearchResult> {
    options.identity.validate()?;
    options
        .policy
        .validate()
        .map_err(|_| search_error("invalid_development_policy"))?;
    if options.sample_rows == 0
        || options.max_candidates == 0
        || options.max_candidates > 4096
        || proposals.len() > options.max_candidates
    {
        return Err(search_error("development_search_budget_exceeded"));
    }
    let mut observations = Vec::new();
    let mut winner: Option<(DevelopmentObservation, Vec<u8>, [u8; 32])> = None;
    let mut rejected_over_cap = 0;
    for proposal in proposals {
        let mut proposal = proposal.clone();
        proposal.compliant = false;
        proposal
            .validate()
            .map_err(|_| search_error("invalid_candidate_proposal"))?;
        if proposal.features as usize + 1 != selection.input.view().metadata().columns.len()
            || proposal.task.as_str()
                != match selection.input.view().metadata().task {
                    crate::schema::TaskKind::Regression => "regression",
                    crate::schema::TaskKind::Binary => "binary",
                    crate::schema::TaskKind::Multiclass => "multiclass",
                }
        {
            return Err(search_error("candidate_selection_mismatch"));
        }
        let artifact =
            encode_kernel(&proposal).map_err(|_| search_error("candidate_encode_failed"))?;
        if !options.policy.accepts_artifact_bytes(artifact.len()) {
            rejected_over_cap += 1;
            continue;
        }
        let candidate_digest = *blake3::hash(&artifact).as_bytes();
        let key = options.identity.key(
            SearchPartition::DevelopmentSelection,
            selection.digest,
            candidate_digest,
            options.sample_rows,
        )?;
        let cached = cache.get(SearchPartition::DevelopmentSelection, key)?;
        let metrics = if let Some(metrics) = cached {
            metrics
        } else {
            let decoded = crate::codec::decode_kernel(&artifact)
                .map_err(|_| search_error("candidate_reload_failed"))?;
            let mut metrics = DevelopmentMetrics::default();
            for &seed in &options.identity.seed_schedule {
                let synthetic = development_sample(&decoded, options.sample_rows, seed)?;
                let measured = evaluator
                    .evaluate(&synthetic, decoded.features as usize, selection.input, seed)
                    .map_err(|_| search_error("development_evaluation_failed"))?;
                measured.validate()?;
                metrics.tstr_loss +=
                    measured.tstr_loss / options.identity.seed_schedule.len() as f64;
                metrics.driver_agreement +=
                    measured.driver_agreement / options.identity.seed_schedule.len() as f64;
            }
            cache.insert(SearchPartition::DevelopmentSelection, key, metrics)?;
            metrics
        };
        let observation = DevelopmentObservation {
            candidate_digest,
            wire_bytes: artifact.len(),
            metrics,
            cache_hit: cached.is_some(),
        };
        if winner
            .as_ref()
            .is_none_or(|(best, _, _)| development_candidate_order(&observation, best).is_lt())
        {
            winner = Some((observation.clone(), artifact, key));
        }
        observations.push(observation);
    }
    let (selected, artifact, key) =
        winner.ok_or_else(|| search_error("no_eligible_development_candidate"))?;
    let mut frozen = blake3::Hasher::new();
    frozen.update(b"dope-frozen-development-recipe-v1\0");
    frozen.update(&key);
    frozen.update(options.policy.hash().as_bytes());
    frozen.update(&selected.metrics.tstr_loss.to_bits().to_le_bytes());
    frozen.update(&selected.metrics.driver_agreement.to_bits().to_le_bytes());
    Ok(DevelopmentSearchResult {
        recipe: FrozenDevelopmentRecipe {
            artifact,
            recipe_digest: *frozen.finalize().as_bytes(),
            sample_rows: options.sample_rows,
            seed_schedule: options.identity.seed_schedule.clone(),
        },
        observations,
        rejected_over_cap,
    })
}

#[cfg(test)]
mod development_proposal_tests {
    use super::*;

    #[test]
    fn cheap_structural_profiles_preserve_fit_recipe_and_remain_unqualified() {
        let compiled = compile_kernel_from_arrays(
            &[0.0, 0.3, 0.6, 1.0],
            &[0.0, 0.3, 0.6, 1.0],
            4,
            1,
            Task::Regression,
            &CompileOptions {
                beam_width: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let fit = crate::codec::decode_kernel(&compiled.artifact).unwrap();
        let proposals = development_structural_proposals(&fit, &[4, 8, 16]).unwrap();
        assert_eq!(
            proposals
                .iter()
                .map(|p| p.quantization_bits)
                .collect::<Vec<_>>(),
            vec![4, 8, 16]
        );
        assert!(proposals.iter().all(|p| !p.compliant));
        assert_eq!(encode_kernel(&fit).unwrap(), compiled.artifact);
        assert!(development_structural_proposals(&fit, &[255]).is_err());
        assert!(development_structural_proposals(&fit, &[]).is_err());
    }
}
