//! Checked inference boundary for caller-supplied prepared conditioning.
//!
//! This validates the caller-declared recipe and tier before forwarding the
//! conditioning to the unchanged DOPE-MLC-v3 encoder. It does not authenticate
//! checkpoint provenance, preparation history, or data custody.

use dope_core::{
    mlc_v3::MlcV3Model,
    prepare::{DopeConditioning, FULL_REPORT_RECIPE, OBSERVED_RECIPE},
    DopeError, Result,
};

fn known_recipe(recipe: &str) -> bool {
    recipe == FULL_REPORT_RECIPE || recipe == OBSERVED_RECIPE
}

fn known_level(level: &str) -> bool {
    matches!(level, "low" | "small" | "medium" | "high" | "ultra")
}

/// Validate the declared conditioning recipe and tier, then forward through
/// the existing 629D encoder. Only the existing High encoder tier is accepted
/// by that encoder; other known tiers return its fixed safe tier error.
pub fn encode_prepared(
    model: &MlcV3Model,
    conditioning: &DopeConditioning,
    expected_recipe: &str,
) -> Result<Vec<f32>> {
    if !known_recipe(expected_recipe) {
        return Err(DopeError::Input(
            "unknown expected conditioning recipe".into(),
        ));
    }
    if !known_recipe(&conditioning.recipe) {
        return Err(DopeError::Input("unknown conditioning recipe".into()));
    }
    if conditioning.recipe != expected_recipe {
        return Err(DopeError::Input("conditioning recipe mismatch".into()));
    }
    if !known_level(&conditioning.level) {
        return Err(DopeError::Input("unsupported conditioning level".into()));
    }

    let columns: Vec<Vec<f32>> = conditioning
        .col_landmarks
        .iter()
        .map(|row| row.to_vec())
        .collect();
    model.encode_features_629(&conditioning.vec256, &columns, &conditioning.level)
}
