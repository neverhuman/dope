use serde::{Deserialize, Serialize};
use std::str::FromStr;

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
            Self::L3 => Some(10_240),
            Self::L2 => Some(32_768),
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
        hasher.update(DEEP_CANDIDATE_SET_VERSION.as_bytes());
        hasher.update(include_bytes!("compiler.rs"));
        hasher.update(include_bytes!("codec.rs"));
        hasher.update(include_bytes!("model.rs"));
        hasher.update(include_bytes!("neural.rs"));
        hasher.update(include_bytes!("neural_train.rs"));
        hasher.update(include_bytes!("libtorch.rs"));
        hasher.update(include_bytes!("../cpp/libtorch_determinism.cpp"));
    }
    hasher.update(id.as_bytes());
    hasher.finalize().to_hex().to_string()
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
            Some("CTGAN is not implemented in deep-joint-v2"),
        ),
        (
            "taegan",
            "per_dataset_deep",
            false,
            Some("TAEGAN is not implemented in deep-joint-v2"),
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
            Some("masked diffusion transformer is not implemented in deep-joint-v2"),
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

impl EmpiricalBackend for FrozenBackendRunner {
    fn spec(&self) -> &FrozenBackend {
        &self.spec
    }

    fn fit(
        &self,
        features: &[f32],
        target: &[f32],
        rows: usize,
        columns: usize,
        task: Task,
        options: &CompileOptions,
    ) -> Result<CompileResult> {
        if !self.spec.available {
            return Err(DopeError::Unsupported(format!(
                "frozen empirical backend {} is unavailable: {}",
                self.spec.id,
                self.spec.failure.as_deref().unwrap_or("not linked")
            )));
        }
        let mut backend_options = options.clone();
        backend_options.backend_id = Some(self.spec.id.clone());
        compile_kernel_from_arrays(features, target, rows, columns, task, &backend_options)
    }
}

pub fn empirical_backend(id: &str) -> Result<Box<dyn EmpiricalBackend>> {
    empirical_backends()
        .into_iter()
        .find(|backend| backend.id == id)
        .map(|spec| Box::new(FrozenBackendRunner { spec }) as Box<dyn EmpiricalBackend>)
        .ok_or_else(|| DopeError::Data(format!("unknown frozen empirical backend {id}")))
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct AuditorSpec {
    pub id: String,
    pub frozen_version: String,
    pub frozen_hash: String,
    pub telemetry_disabled: bool,
    pub available: bool,
    pub backend: String,
    pub reason: Option<String>,
}

pub fn auditor_specs() -> Vec<AuditorSpec> {
    [
        ("elastic_net_glm", true, "rust-native-elastic-net-v1"),
        ("ga2m", true, "rust-native-ga2m-v1"),
        (
            "extremely_randomized_trees",
            true,
            "rust-native-extra-trees-v1",
        ),
        ("histogram_gbdt", true, "rust-native-histogram-gbdt-v1"),
        (
            "gpu_residual_mlp_ensemble",
            cfg!(feature = "gpu-training"),
            "tch-rs-0.20-libtorch-2.7-residual-mlp-v1",
        ),
        (
            "gpu_tabular_feature_transformer",
            cfg!(feature = "gpu-training"),
            "tch-rs-0.20-libtorch-2.7-feature-transformer-v1",
        ),
    ]
    .into_iter()
    .map(|(id, available, frozen_hash)| AuditorSpec {
        id: id.into(),
        frozen_version: if available {
            env!("CARGO_PKG_VERSION").into()
        } else {
            "unavailable".into()
        },
        frozen_hash: if available {
            crate::auditor::implementation_hash(id)
        } else {
            frozen_hash.into()
        },
        telemetry_disabled: true,
        available,
        backend: if available && id.starts_with("gpu_") {
            "rust-tch-libtorch-training".into()
        } else if available {
            "rust-native".into()
        } else {
            "not-linked".into()
        },
        reason: (!available).then(|| {
            if id.starts_with("gpu_") {
                "required frozen GPU auditor needs the gpu-training build".into()
            } else {
                "required frozen native auditor is not implemented".into()
            }
        }),
    })
    .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DpAllocation {
    pub phase: String,
    pub epsilon: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DpCompositionReceipt {
    pub accountant: String,
    pub total_epsilon: f64,
    pub delta: f64,
    pub allocations: Vec<DpAllocation>,
    pub non_dp_shared_weights: bool,
    pub verified: bool,
}

impl DpCompositionReceipt {
    pub fn new(total_epsilon: f64, rows: usize) -> Self {
        let delta = (1.0 / (rows.max(1) as f64).powi(2)).min(1e-6);
        let allocations = [
            ("routing", 0.05),
            ("fitting", 0.90),
            ("selection_calibration", 0.05),
        ]
        .into_iter()
        .map(|(phase, fraction)| DpAllocation {
            phase: phase.into(),
            epsilon: total_epsilon * fraction,
        })
        .collect();
        let mut receipt = Self {
            accountant: "pure_epsilon_sequential_composition-v1".into(),
            total_epsilon,
            delta,
            allocations,
            non_dp_shared_weights: false,
            verified: false,
        };
        receipt.verified = receipt.verify(rows).is_ok();
        receipt
    }

    pub fn verify(&self, rows: usize) -> Result<()> {
        if !self.total_epsilon.is_finite() || self.total_epsilon <= 0.0 {
            return Err(DopeError::Data(
                "DP epsilon must be finite and positive".into(),
            ));
        }
        let expected_delta = (1.0 / (rows.max(1) as f64).powi(2)).min(1e-6);
        if (self.delta - expected_delta).abs() > 1e-15 {
            return Err(DopeError::Data(
                "DP delta does not match min(1e-6,1/n^2)".into(),
            ));
        }
        if self.non_dp_shared_weights {
            return Err(DopeError::Data(
                "formal-DP artifacts cannot reference non-DP shared weights".into(),
            ));
        }
        let expected = [
            ("routing", 0.05),
            ("fitting", 0.90),
            ("selection_calibration", 0.05),
        ];
        if self.allocations.len() != expected.len() {
            return Err(DopeError::Data(
                "DP receipt has the wrong phase count".into(),
            ));
        }
        for (allocation, (phase, fraction)) in self.allocations.iter().zip(expected) {
            if allocation.phase != phase
                || (allocation.epsilon - self.total_epsilon * fraction).abs() > 1e-12
            {
                return Err(DopeError::Data("DP receipt allocation mismatch".into()));
            }
        }
        let composed = self
            .allocations
            .iter()
            .map(|part| part.epsilon)
            .sum::<f64>();
        if (composed - self.total_epsilon).abs() > 1e-12 {
            return Err(DopeError::Data(
                "DP receipt does not compose to total epsilon".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FormalDpConfiguration {
    pub backend: String,
    pub parameter_budget_bytes: usize,
    pub epsilon: f64,
    pub delta: f64,
    pub receipt: DpCompositionReceipt,
    pub available: bool,
    pub failure: Option<String>,
}

pub fn formal_dp_configurations(rows: usize) -> Vec<FormalDpConfiguration> {
    let backends = [
        "dp_graphical_mst",
        "dp_kd_tree",
        "dp_tbart",
        "dp_tvae",
        "dp_tabsyn",
        "dp_conditional_transformer",
    ];
    let mut configurations = Vec::with_capacity(120);
    for epsilon in DP_EPSILONS {
        for backend in backends {
            for parameter_budget_bytes in DP_PARAMETER_BUDGETS {
                let receipt = DpCompositionReceipt::new(epsilon, rows);
                configurations.push(FormalDpConfiguration {
                    backend: backend.into(),
                    parameter_budget_bytes,
                    epsilon,
                    delta: receipt.delta,
                    receipt,
                    available: false,
                    failure: Some(
                        "formal-DP backend is frozen but is not linked in this build".into(),
                    ),
                });
            }
        }
    }
    configurations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::decode_kernel;

    #[test]
    fn release_languages_are_frozen() {
        let empirical = empirical_backends();
        assert_eq!(empirical.len(), 31);
        assert_eq!(
            empirical
                .iter()
                .filter(|item| item.group == "symbolic")
                .count(),
            8
        );
        assert_eq!(auditor_specs().len(), 6);
        assert_eq!(deep_confirmation_backends().len(), 14);
        assert!(
            ["ctgan", "taegan", "tabddpm", "masked_diffusion_transformer"]
                .iter()
                .all(|id| empirical
                    .iter()
                    .any(|item| item.id == *id && !item.available))
        );
        assert!(empirical.iter().all(|item| {
            empirical_backend(&item.id).is_ok_and(|backend| backend.spec().id == item.id)
        }));
    }

    #[test]
    fn every_empirical_backend_fits_binary_and_regression_fixtures() {
        let rows = 48usize;
        let features = (0..rows)
            .flat_map(|row| {
                let x = row as f32 / (rows - 1) as f32;
                [x, 1.0 - x, x * x, (x * 7.0).fract()]
            })
            .collect::<Vec<_>>();
        for task in [Task::Binary, Task::Regression] {
            let target = (0..rows)
                .map(|row| {
                    let x = row as f32 / (rows - 1) as f32;
                    if task == Task::Binary {
                        f32::from(x > 0.5)
                    } else {
                        (0.1 + 0.7 * x + 0.15 * (x * 9.0).sin()).clamp(0.0, 1.0)
                    }
                })
                .collect::<Vec<_>>();
            for spec in empirical_backends() {
                if !spec.available || spec.group == "per_dataset_deep_joint" {
                    continue;
                }
                let result = empirical_backend(&spec.id)
                    .unwrap()
                    .fit(
                        &features,
                        &target,
                        rows,
                        4,
                        task,
                        &CompileOptions {
                            seed: Some(1729),
                            ..Default::default()
                        },
                    )
                    .unwrap_or_else(|error| panic!("{} {task:?}: {error}", spec.id));
                assert_eq!(result.report.selected_candidate, spec.id);
                assert!(decode_kernel(&result.artifact).is_ok(), "{}", spec.id);
            }
        }
    }

    #[test]
    fn dp_frontier_has_verified_receipts() {
        let configurations = formal_dp_configurations(2_000);
        assert_eq!(configurations.len(), 120);
        assert!(
            configurations
                .iter()
                .all(|item| item.receipt.verify(2_000).is_ok())
        );
    }
}
