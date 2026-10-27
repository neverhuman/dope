//! Static, landmark-free table preparation and MLC-v3 conditioning
//! descriptors: a Rust port of the Python research path
//! (`research/levels/{scaling,descriptors,building_block_landmarks,multilevel_model}.py`)
//! that builds the inputs of `MlcV3Model::encode_features_629` without Python.
//!
//! * [`scaling`] — train-only `FitOnlyTableTransform` → [`PreparedTable`].
//! * [`descriptors`] — `vector116` / column spectrum from a native report
//!   ([`NativeReportView`]) and `table_vec24` / `column_desc` from the table.
//! * [`building_blocks`] — 68-d building-block landmark probes (`bb_vec68`, `col_bb7`).
//! * [`inter_landmark`] — 48-d inter-landmark summary stats (legacy full + observed-only).
//! * [`conditioning`] — `vec256` and `col_landmarks` (`64×17`) assembly + [`DopeConditioning::digest`].
//! * [`np`] — numpy-1.26-exact numerical helpers shared by the port.

pub mod building_blocks;
pub mod building_blocks_np;
mod building_blocks_simplex;
pub mod conditioning;
mod conditioning_builder;
mod conditioning_observed;
pub mod descriptors;
mod descriptors_table;
pub mod inter_landmark;
pub mod np;
mod np_linalg;
pub mod report_view;
pub mod scaling;
mod scaling_fit;
mod scaling_prepare;
mod scaling_table;

pub use building_blocks::{compute_building_block_landmarks, BuildingBlockDescriptors};
pub use conditioning::{
    assemble_observed_conditioning, assemble_observed_conditioning_cached,
    assemble_observed_conditioning_with_target_stats, assemble_v3_conditioning,
    build_conditioning_with_cached_table_stats, build_conditioning_with_descriptors,
    build_conditioning_with_full_table_stats, conditioning_digest, observed_report_descriptors,
    pad_col_landmarks, report_complete_landmarks, report_observed_landmarks, same_landmarks,
    subsample_scaled_rows, unsquash, BundleConditioningInput, DopeConditioning,
    LegacyReportConditioningInput, ObservedReportDescriptors, V3ConditioningBundle,
    COL_LANDMARK_DIM, CONDITIONING_DOMAIN, FULL_REPORT_RECIPE, MAX_COLS, OBSERVED_RECIPE,
};
pub use descriptors::{
    compute_level_contrasts, compute_table_descriptors, extract_report_descriptors,
    level_landmark_mask, Level, ReportDescriptors, ShapeOverride, TableDescriptors,
};
pub use inter_landmark::{
    compute_inter_landmark_summary_stats, compute_observed_inter_landmark_stats, ReportLandmarkView,
};
pub use report_view::{NativeReportView, ReportCellView, SpectrumView};
pub use scaling::{
    compute_native_target_stats, compute_observed_target_stats, prepare_table, prepare_table_cells,
    prepare_table_cells_with_ids, prepare_table_mixed, prepare_table_with_ids,
    CategoricalColumnEncoder, ColumnKind, FitOnlyTableTransform, NumericColumnScaler, PrepTask,
    PreparedTable, TargetScaler, TransformKind,
};

use crate::error::Result;
use crate::landmark::{
    LandmarkBundle, LandmarkGroup, LandmarkMode, LandmarkOptions, ProgressiveLandmarks, Signature,
    SubsampleSpec,
};

/// Build MLC-v3 / `DopeEncoder629` conditioning at `Level::High` from a
/// [`PreparedTable`], a [`LandmarkBundle`], an optional [`NativeReportView`],
/// and a [`LandmarkMode`].
///
/// Reuses the table's cached static `high_table_descriptors`, `classes`, and
/// full-table `target_stats` so progressive updates never rescan or sort
/// training columns.
pub fn build_conditioning(
    table: &PreparedTable,
    bundle: &LandmarkBundle,
    raw_report: Option<&NativeReportView>,
    mode: LandmarkMode,
) -> Result<DopeConditioning> {
    build_conditioning_sampled(table, bundle, raw_report, mode, None)
}

/// Build MLC-v3 / `DopeEncoder629` conditioning at `Level::High` with optional
/// `sample_positions` (0-based indices into `table`, e.g. from
/// `ProgressiveLandmarks::selection().positions`).
///
/// When `mode == LandmarkMode::Full` and `sample_positions` is `Some`,
/// building-block probes are evaluated on the selected sample rows while
/// `table_vec24`, `target_stats[85..87]`, and `shape[87..91]` use the complete
/// `PreparedTable`. Unsampled `Full` (`bundle.subsample_n.is_none()` and
/// `sample_positions.is_none()`) is bit-identical to the legacy full builder.
pub fn build_conditioning_sampled(
    table: &PreparedTable,
    bundle: &LandmarkBundle,
    raw_report: Option<&NativeReportView>,
    mode: LandmarkMode,
    sample_positions: Option<&[usize]>,
) -> Result<DopeConditioning> {
    build_conditioning_at_level(
        table,
        bundle,
        raw_report,
        mode,
        Level::High,
        sample_positions,
    )
}

/// Build MLC-v3 conditioning at an explicit [`Level`] with optional
/// `sample_positions`.
pub fn build_conditioning_at_level(
    table: &PreparedTable,
    bundle: &LandmarkBundle,
    raw_report: Option<&NativeReportView>,
    mode: LandmarkMode,
    level: Level,
    sample_positions: Option<&[usize]>,
) -> Result<DopeConditioning> {
    let tab_desc = table.table_descriptors(level);
    let bundle_input = BundleConditioningInput {
        values: &bundle.values,
        mask: &bundle.mask,
        subsample_n: bundle.subsample_n,
    };

    let sampled_buf = match sample_positions {
        Some(pos) => Some(subsample_scaled_rows(
            &table.x_scaled,
            table.rows,
            table.features,
            &table.y_scaled,
            pos,
        )?),
        None => None,
    };

    let rep_desc =
        raw_report.map(|r| extract_report_descriptors(r, level, ShapeOverride::default()));
    let legacy_input = match (raw_report, rep_desc.as_ref()) {
        (Some(r), Some(rd)) => {
            let (complete_values23, complete_mask23) =
                report_complete_landmarks(r.cells.iter().map(|c| (c.id, c.state.as_str(), c.gain)));
            let (observed_values23, observed_mask23) =
                report_observed_landmarks(r.cells.iter().map(|c| (c.id, c.state.as_str(), c.gain)));
            let evidence_sample = sampled_buf
                .as_ref()
                .map(|(sx, srows, sy)| (sx.as_slice(), *srows, sy.as_slice()));
            Some(LegacyReportConditioningInput {
                vector116: &rd.vector116,
                rep: ReportLandmarkView {
                    gains23: &rd.gains23,
                    mask23: &rd.mask23,
                    ses23: &rd.ses23,
                    lambdas23: &rd.lambdas23,
                    column_spectrum: &rd.column_spectrum,
                },
                column_prior_importance: &rd.column_prior_importance,
                complete_values23,
                complete_mask23,
                observed_values23,
                observed_mask23,
                full_table_target_stats: Some([
                    Some(table.target_stats[0]),
                    Some(table.target_stats[1]),
                    None,
                ]),
                evidence_sample,
            })
        }
        _ => None,
    };

    let mut cond = build_conditioning_with_cached_table_stats(
        &table.x_scaled,
        table.rows,
        table.features,
        &table.y_scaled,
        table.task.as_str(),
        Some(table.classes),
        Some(table.target_stats),
        &tab_desc.table_vec24,
        &tab_desc.column_desc,
        &tab_desc.empirical_driver_weights,
        Some(bundle_input),
        legacy_input,
        mode.as_str(),
        level.as_str(),
    )?;
    cond.bundle = bundle.sanitized();
    Ok(cond)
}

/// Construct a [`ProgressiveLandmarks`] session bound to `table`'s full
/// source-row and vocabulary digest (`table.source_rows_digest()`).
pub fn progressive_landmarks_for_table(
    table: &PreparedTable,
    opts: &LandmarkOptions,
    subsample: Option<SubsampleSpec>,
) -> Result<ProgressiveLandmarks> {
    let frame = table.to_landmark_frame()?;
    ProgressiveLandmarks::new_with_source_digest(
        &frame,
        opts,
        subsample,
        table.source_rows_digest(),
    )
}

/// Compute a single progressive landmark group (`Macro`, `Larger`, or `Fine`)
/// on a [`PreparedTable`], returning the group's [`LandmarkBundle`] (bound to
/// `table.source_rows_digest()`) and native [`Signature`].
///
/// For multi-step progressive execution that reuses fold preparation and
/// work-budget state across groups, use [`progressive_landmarks_for_table`] or
/// [`ProgressiveLandmarks::new_with_source_digest`].
pub fn compute_landmark_group(
    table: &PreparedTable,
    group: LandmarkGroup,
    subsample: Option<SubsampleSpec>,
    opts: &LandmarkOptions,
) -> Result<(LandmarkBundle, Signature)> {
    let mut driver = progressive_landmarks_for_table(table, opts, subsample)?;
    let (bundle, sig) = driver.next_group(group)?;
    Ok((bundle, sig.clone()))
}
