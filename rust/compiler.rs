include!("compiler/part_01.rs");
include!("compiler/part_02.rs");
include!("compiler/interaction_screen.rs");
include!("compiler/candidate_dependence.rs");
include!("compiler/capacity_refinement.rs");
include!("compiler/fair_allocation.rs");
include!("compiler/adaptive_allocation.rs");
include!("compiler/residual_oof.rs");
include!("compiler/residual_profiles.rs");
include!("compiler/target_preparation.rs");

#[cfg(all(test, feature = "gpu-training"))]
mod gpu_readout_tests;
