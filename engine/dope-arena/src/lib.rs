//! Unified Rust Arena for `dope-fidelity/v3`.
//!
//! Owns panel manifests, Rule A-1 duplicate-grouped `A/B/C/T/P` splits,
//! complete failure-accounting schedules, swap-ladder diagnostics (`--given`),
//! and frozen prediction-bundle reduction via `dope_core::dvs::v3`.

pub mod dopemat;
pub mod ledger;
pub mod panel;
pub mod report;
pub mod schedule;
pub mod split;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArenaError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("dope core error: {0}")]
    Core(#[from] dope_core::DopeError),
    #[error("dvs error: {0}")]
    Dvs(#[from] dope_core::dvs::DvsError),
    #[error("arena contract error: {0}")]
    Contract(String),
}

pub type ArenaResult<T> = std::result::Result<T, ArenaError>;
