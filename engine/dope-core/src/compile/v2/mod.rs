//! Closed-loop exact-artifact compiler v2 (`BUILD/SELECT/CAL`, `P+` projection,
//! transitive bucketed lexicographic selection, and carry-forward tier monotonicity).

pub mod backoff;
pub mod loop_driver;
pub mod project;
pub mod roles;
pub mod select;
pub mod tiers;

pub use backoff::{
    run_privacy_backoff, CapacityBackoffStep, PrivacyBackoffOutcome, PRIVACY_BACKOFF_LADDER,
};

pub use loop_driver::{
    lineage_hash, run_exact_artifact_loop, CompileLedger, CompilerMove, DecodedCandidateArtifact,
};
pub use project::{
    compile_projected_sampler, project_moments_damped_newton, ProjectionFiveGapLadder,
    ProjectionResult, MIN_PROJECTION_ESS,
};
pub use roles::CompilerRoles;
pub use select::{select_best_decoded, CandidateScore};
pub use tiers::{resolve_carry_forward_tiers, TierAssignment, TIER_IMPROVEMENT_MARGIN};
