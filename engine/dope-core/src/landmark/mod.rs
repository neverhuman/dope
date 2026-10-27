//! Native, bounded dataset descriptors. Real signatures and transfer reports
//! are private measurements; only synthetic signatures are artifact-derived.
//! No landmark score is an external transfer certification.

mod basis;
pub mod bundle;
mod cost;
mod folds;
mod frame;
mod learners;
mod panel;
mod progressive;
mod report;
mod ridge;
mod signature;
mod transfer;

/// Frozen coordinate layout; consumers must check the exact panel identity.
pub const LAYOUT_JSON: &str = include_str!("layout-native-v1.json");

pub use bundle::{
    parse_landmark_ids, squash_raw_gain, stratified_subsample, BundleScope, LandmarkBundle,
    LandmarkGroup, LandmarkMode, LandmarkScope, LandmarkStatus, RowSelection, SubsampleSpec,
    LANDMARK_SLOTS, PROGRESSIVE_ORDER,
};
pub use cost::WorkReport;
pub use folds::assign_folds;
pub use frame::LandmarkFrame;
pub use panel::{Family, LandmarkId, Recipe, PANEL, PANEL_V1};
pub use progressive::ProgressiveLandmarks;
pub use report::{
    CellState, FeatureSpectrum, GlobalCoordinates, LandmarkCell, Signature, SignatureKind,
};
pub use signature::{signature, signature_requested};
pub use transfer::{PairedComparison, TransferCell, TransferReference, TransferSignature};

use crate::{DopeError, Result};
use serde::{Deserialize, Serialize};

/// Options are part of the coordinate identity: consumers must bind the panel
/// AND this recipe, rather than comparing differently capped runs as identical.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LandmarkOptions {
    pub panel: String,
    pub folds: usize,
    pub ridge_rows: usize,
    pub small_rows: usize,
    pub knn_reference: usize,
    pub spectrum_features: usize,
    pub seed: u64,
    pub work_budget: u64,
}

impl Default for LandmarkOptions {
    fn default() -> Self {
        Self {
            panel: PANEL_V1.into(),
            folds: 5,
            ridge_rows: 8192,
            small_rows: 4096,
            knn_reference: 2048,
            spectrum_features: 64,
            seed: 0,
            work_budget: 2_000_000_000,
        }
    }
}

impl LandmarkOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.panel != PANEL_V1 {
            return Err(DopeError::Input("unsupported landmark panel".into()));
        }
        if !(2..=10).contains(&self.folds)
            || !(20..=50_000).contains(&self.ridge_rows)
            || !(20..=4096).contains(&self.small_rows)
            || !(1..=2048).contains(&self.knn_reference)
            || self.spectrum_features > 64
        {
            return Err(DopeError::Input("invalid landmark options".into()));
        }
        Ok(())
    }
}

pub(crate) fn job_seed(seed: u64, id: usize, fold: usize) -> u64 {
    crate::rng::hash3(seed ^ 0x4c41_4e44_4d41_524b, id as u64, fold as u64)
}
