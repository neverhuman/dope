use serde::{Deserialize, Serialize};
use std::str::FromStr;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/kpi_contract_generated.rs"));
}

/// Release tiers describe enforceable artifact and evidence requirements.
/// They make no formal differential-privacy or de-identification claim.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AnonymizationTier {
    L0,
    L1,
    L2,
    #[default]
    L3,
}

impl AnonymizationTier {
    pub const fn byte_limit(self) -> Option<usize> {
        match self {
            Self::L3 => Some(generated::L3_ARTIFACT_LIMIT),
            Self::L2 => Some(generated::L2_ARTIFACT_LIMIT),
            Self::L1 | Self::L0 => None,
        }
    }
}

impl FromStr for AnonymizationTier {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "l0" => Ok(Self::L0),
            "l1" => Ok(Self::L1),
            "l2" => Ok(Self::L2),
            "l3" => Ok(Self::L3),
            _ => Err("tier must be l0, l1, l2, or l3".into()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ReleasePolicy {
    pub version: u8,
    pub tier: AnonymizationTier,
    pub maximum_artifact_bytes: Option<usize>,
    pub require_formal_dp: bool,
}

impl Default for ReleasePolicy {
    fn default() -> Self {
        Self::new(AnonymizationTier::L3, None, false).expect("fixed L3 policy")
    }
}

impl ReleasePolicy {
    pub fn new(
        tier: AnonymizationTier,
        maximum_artifact_bytes: Option<usize>,
        require_formal_dp: bool,
    ) -> crate::error::Result<Self> {
        if require_formal_dp {
            return Err(DopeError::Unsupported(
                "formal DP is unavailable: no verified backend is installed".into(),
            ));
        }
        if maximum_artifact_bytes == Some(0)
            || tier
                .byte_limit()
                .zip(maximum_artifact_bytes)
                .is_some_and(|(ceiling, requested)| requested > ceiling)
        {
            return Err(DopeError::Data(
                "the artifact byte limit must be positive and cannot raise the tier ceiling".into(),
            ));
        }
        Ok(Self {
            version: 1,
            tier,
            maximum_artifact_bytes: maximum_artifact_bytes.or(tier.byte_limit()),
            require_formal_dp,
        })
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.version != 1 {
            return Err(DopeError::Data("unsupported release-policy version".into()));
        }
        Self::new(
            self.tier,
            self.maximum_artifact_bytes,
            self.require_formal_dp,
        )?;
        Ok(())
    }

    pub fn hash(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("release policy serializes");
        blake3::hash(&bytes).to_hex().to_string()
    }

    pub fn accepts_artifact_bytes(&self, bytes: usize) -> bool {
        self.maximum_artifact_bytes
            .is_none_or(|limit| bytes <= limit)
    }
}

#[cfg(test)]
mod tier_policy_tests {
    use super::*;
    use crate::production::KpiContract;

    #[test]
    fn l3_hard_boundary_and_formal_dp_failure() {
        let policy = ReleasePolicy::default();
        assert_eq!(policy.maximum_artifact_bytes, Some(10_240));
        assert!(policy.accepts_artifact_bytes(10_240));
        assert!(!policy.accepts_artifact_bytes(10_241));
        assert!(ReleasePolicy::new(AnonymizationTier::L3, Some(10_241), false).is_err());
        assert!(ReleasePolicy::new(AnonymizationTier::L3, Some(512), false).is_ok());
        assert!(ReleasePolicy::new(AnonymizationTier::L3, None, true).is_err());
    }

    #[test]
    fn generated_tier_limits_match_the_embedded_v2_contract() {
        let contract = KpiContract::embedded().unwrap();
        contract.validate().unwrap();
        assert_eq!(contract.version, 2);
        assert_eq!(
            contract.tier_byte_limits.get("l3").copied(),
            AnonymizationTier::L3.byte_limit()
        );
        assert_eq!(
            contract.tier_byte_limits.get("l2").copied(),
            AnonymizationTier::L2.byte_limit()
        );
    }
}

use crate::compiler::{CompileOptions, CompileResult, compile_kernel_from_arrays};
use crate::error::{DopeError, Result};
use crate::model::Task;

pub const RELEASE_SEED: u64 = 1729;
pub const GENERATION_REPEATS: usize = 3;
pub const AUDITOR_SEEDS: [u64; 3] = [57721, 161803, 271828];
pub const SYNTHETIC_SIZE_MULTIPLIERS: [usize; 4] = [1, 2, 4, 8];
pub const GOLD_SIZE_MULTIPLIERS: [usize; 2] = [1, 4];
pub const DP_EPSILONS: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 8.0];
pub const DP_PARAMETER_BUDGETS: [usize; 4] = [16 << 10, 64 << 10, 256 << 10, 1 << 20];
pub const DEEP_CANDIDATE_SET_VERSION: &str = "deep-joint-v3";
pub const COMPACT_CANDIDATE_SET_VERSION: &str = "compact-neural-v1";
pub const DIRECT_DIFFUSION_CANDIDATE_SET_VERSION: &str = "direct-diffusion-v1";

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct FrozenBackend {
    pub id: String,
    pub group: String,
    pub implementation_hash: String,
    pub canonical_kernel_v3: bool,
    pub available: bool,
    pub failure: Option<String>,
}

fn backend(id: &str, group: &str, available: bool, failure: Option<&str>) -> FrozenBackend {
    FrozenBackend {
        id: id.into(),
        group: group.into(),
        implementation_hash: candidate_implementation_hash(id),
        canonical_kernel_v3: true,
        available,
        failure: failure.map(str::to_string),
    }
}

pub fn candidate_implementation_hash(id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    if id.starts_with("legacy_") {
        // The old candidate family remains a stable comparator even as the
        // joint generator implementation evolves.
        hasher.update(b"dope-legacy-deep-candidate-set-v1");
    } else {
        let lineage = if id == "tabddpm_direct_rank" {
            DIRECT_DIFFUSION_CANDIDATE_SET_VERSION
        } else if id.starts_with("micro_tvae_") || id.starts_with("tiny_mat_") {
            COMPACT_CANDIDATE_SET_VERSION
        } else {
            DEEP_CANDIDATE_SET_VERSION
        };
        hasher.update(lineage.as_bytes());
        hasher.update(include_bytes!("../compiler.rs"));
        hasher.update(include_bytes!("../compiler/target_fitting.rs"));
        hasher.update(include_bytes!("../compiler/candidate_search.rs"));
        hasher.update(include_bytes!("../compiler/neural_candidates.rs"));
        hasher.update(include_bytes!("../codec.rs"));
        hasher.update(include_bytes!("../codec/section_encoding.rs"));
        hasher.update(include_bytes!("../codec/decoding.rs"));
        hasher.update(include_bytes!("../model.rs"));
        hasher.update(include_bytes!("../model/generator_validation.rs"));
        hasher.update(include_bytes!("../neural.rs"));
        hasher.update(include_bytes!("../neural_train.rs"));
        hasher.update(include_bytes!("../neural_train/gpu_diffusion.rs"));
        hasher.update(include_bytes!("../../build.rs"));
        hasher.update(include_bytes!("../libtorch.rs"));
        // Include every moved source section in candidate lineage.
        hasher.update(include_bytes!("../codec/decoding/part_01.rs"));
        hasher.update(include_bytes!("../codec/decoding/part_02.rs"));
        hasher.update(include_bytes!("../codec/part_01.rs"));
        hasher.update(include_bytes!("../codec/part_02.rs"));
        hasher.update(include_bytes!("../codec/section_encoding/part_01.rs"));
        hasher.update(include_bytes!("../codec/section_encoding/part_02.rs"));
        hasher.update(include_bytes!("../compiler/candidate_search/part_01.rs"));
        hasher.update(include_bytes!("../compiler/candidate_search/part_02.rs"));
        hasher.update(include_bytes!("../compiler/neural_candidates/part_01.rs"));
        hasher.update(include_bytes!("../compiler/neural_candidates/part_02.rs"));
        hasher.update(include_bytes!("../compiler/neural_candidates/part_03.rs"));
        hasher.update(include_bytes!("../compiler/part_01.rs"));
        hasher.update(include_bytes!("../compiler/part_02.rs"));
        hasher.update(include_bytes!("../compiler/target_fitting/part_01.rs"));
        hasher.update(include_bytes!("../compiler/target_fitting/part_02.rs"));
        hasher.update(include_bytes!("../contract/part_01.rs"));
        hasher.update(include_bytes!("../contract/part_02.rs"));
        hasher.update(include_bytes!("../model/generator_validation/part_01.rs"));
        hasher.update(include_bytes!("../model/generator_validation/part_02.rs"));
        hasher.update(include_bytes!("../model/part_01.rs"));
        hasher.update(include_bytes!("../model/part_02.rs"));
        hasher.update(include_bytes!("../neural/part_01.rs"));
        hasher.update(include_bytes!("../neural/part_02.rs"));
        hasher.update(include_bytes!("../neural/part_03.rs"));
        hasher.update(include_bytes!("../neural_train/gpu/part_01.rs"));
        hasher.update(include_bytes!("../neural_train/gpu/part_02.rs"));
        hasher.update(include_bytes!("../neural_train/gpu_diffusion/part_01.rs"));
        hasher.update(include_bytes!("../neural_train/gpu_diffusion/part_02.rs"));
    }
    hasher.update(id.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// New profiles use a separate candidate set so historical campaign job counts stay fixed.
pub fn compact_neural_backends() -> Vec<FrozenBackend> {
    [
        "micro_tvae_4_16",
        "micro_tvae_8_24",
        "micro_tvae_12_32",
        "tiny_mat_16_2_32",
        "tiny_mat_24_3_48",
    ]
    .into_iter()
    .map(|id| {
        backend(
            id,
            "compact_neural",
            cfg!(feature = "gpu-training"),
            (!cfg!(feature = "gpu-training"))
                .then_some("compact neural training requires the gpu-training build"),
        )
    })
    .collect()
}

pub fn direct_diffusion_backends() -> Vec<FrozenBackend> {
    vec![backend(
        "tabddpm_direct_rank",
        "direct_diffusion_research",
        cfg!(feature = "gpu-training"),
        (!cfg!(feature = "gpu-training"))
            .then_some("direct rank diffusion requires the gpu-training build"),
    )]
}

/// The release candidate language is deliberately data- and path-independent.
/// Availability is recorded separately from membership so failures cannot alter
/// the tournament after seeing a dataset.
pub fn empirical_backends() -> Vec<FrozenBackend> {
    [
        ("independent_quantile", "symbolic", true, None),
        ("chow_liu", "symbolic", true, None),
        ("triangular_autoregressive", "symbolic", true, None),
        ("sparse_gaussian_copula", "symbolic", true, None),
        ("poet_factor", "symbolic", true, None),
        ("truncated_c_vine", "symbolic", true, None),
        ("two_component_copula_mixture", "symbolic", true, None),
        ("compact_neural_residual_symbolic", "symbolic", true, None),
        ("tabsds_rank", "non_parametric_tree", true, None),
        (
            "adversarial_random_forest",
            "non_parametric_tree",
            true,
            None,
        ),
        (
            "conditional_density_forest",
            "non_parametric_tree",
            true,
            None,
        ),
        ("forest_diffusion_vp", "non_parametric_tree", true, None),
        (
            "forest_diffusion_flow_matching",
            "non_parametric_tree",
            true,
            None,
        ),
        (
            "tvae",
            "per_dataset_deep_joint",
            cfg!(feature = "gpu-training"),
            Some("TVAE training requires the gpu-training build"),
        ),
        (
            "ctgan",
            "per_dataset_deep",
            false,
            Some("CTGAN has no verified deep-joint-v2 backend"),
        ),
        (
            "taegan",
            "per_dataset_deep",
            false,
            Some("TAEGAN has no verified deep-joint-v2 backend"),
        ),
        (
            "tabddpm",
            "per_dataset_deep",
            false,
            Some("TabDDPM is reserved for the bounded expansion rule"),
        ),
        (
            "tabsyn",
            "per_dataset_deep_joint",
            cfg!(feature = "gpu-training"),
            Some("TabSyn training requires the gpu-training build"),
        ),
        (
            "single_table_autoregressive_transformer",
            "per_dataset_deep_joint",
            cfg!(feature = "gpu-training"),
            Some("autoregressive transformer training requires the gpu-training build"),
        ),
        (
            "masked_diffusion_transformer",
            "per_dataset_deep",
            false,
            Some("masked diffusion transformer has no verified deep-joint-v2 backend"),
        ),
        ("legacy_tvae", "legacy_per_dataset_deep", true, None),
        ("legacy_ctgan", "legacy_per_dataset_deep", true, None),
        ("legacy_taegan", "legacy_per_dataset_deep", true, None),
        ("legacy_tabddpm", "legacy_per_dataset_deep", true, None),
        ("legacy_tabsyn", "legacy_per_dataset_deep", true, None),
        (
            "legacy_single_table_autoregressive_transformer",
            "legacy_per_dataset_deep",
            true,
            None,
        ),
        (
            "legacy_masked_diffusion_transformer",
            "legacy_per_dataset_deep",
            true,
            None,
        ),
        (
            "synthetic_pretrained_cross_table",
            "foundation_hybrid",
            true,
            None,
        ),
        ("per_dataset_adapter", "foundation_hybrid", true, None),
        (
            "symbolic_latent_diffusion_residual",
            "foundation_hybrid",
            true,
            None,
        ),
        (
            "symbolic_autoregressive_residual",
            "foundation_hybrid",
            true,
            None,
        ),
    ]
    .into_iter()
    .map(|(id, group, available, failure)| {
        backend(
            id,
            group,
            available,
            (!available).then_some(failure).flatten(),
        )
    })
    .collect()
}

pub fn deep_confirmation_backends() -> Vec<FrozenBackend> {
    const IDS: [&str; 14] = [
        "tvae",
        "single_table_autoregressive_transformer",
        "tabsyn",
        "legacy_tvae",
        "legacy_ctgan",
        "legacy_taegan",
        "legacy_tabddpm",
        "legacy_tabsyn",
        "legacy_single_table_autoregressive_transformer",
        "legacy_masked_diffusion_transformer",
        "tabsds_rank",
        "sparse_gaussian_copula",
        "forest_diffusion_vp",
        "synthetic_pretrained_cross_table",
    ];
    let registry = empirical_backends();
    IDS.into_iter()
        .map(|id| {
            registry
                .iter()
                .find(|backend| backend.id == id)
                .expect("frozen deep campaign candidate")
                .clone()
        })
        .collect()
}

/// One interface owns frozen candidate membership, availability, and fitting.
/// Unlinked candidates return failed evidence; callers must never replace that
/// failure with another candidate's score.
pub trait EmpiricalBackend: Send + Sync {
    fn spec(&self) -> &FrozenBackend;

    fn fit(
        &self,
        features: &[f32],
        target: &[f32],
        rows: usize,
        columns: usize,
        task: Task,
        options: &CompileOptions,
    ) -> Result<CompileResult>;
}

#[derive(Clone, Debug)]
struct FrozenBackendRunner {
    spec: FrozenBackend,
}