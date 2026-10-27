//! Static, landmark-free table preparation and MLC-v3 conditioning
//! descriptors: a Rust port of the Python research path
//! (`research/levels/{scaling,descriptors}.py`) that builds the inputs of
//! `MlcV3Model::encode_features_629` without Python.
//!
//! * [`scaling`] — train-only `FitOnlyTableTransform` → [`PreparedTable`].
//! * [`descriptors`] — `vector116` / column spectrum from a native report
//!   ([`NativeReportView`]) and `table_vec24` / `column_desc` from the table.
//! * [`np`] — numpy-1.26-exact numerical helpers shared by the port.

pub mod descriptors;
pub mod np;
pub mod report_view;
pub mod scaling;

pub use descriptors::{
    compute_level_contrasts, compute_table_descriptors, extract_report_descriptors,
    level_landmark_mask, Level, ReportDescriptors, ShapeOverride, TableDescriptors,
};
pub use report_view::{NativeReportView, ReportCellView, SpectrumView};
pub use scaling::{
    compute_native_target_stats, compute_observed_target_stats, prepare_table, prepare_table_cells,
    prepare_table_cells_with_ids, prepare_table_mixed, prepare_table_with_ids,
    CategoricalColumnEncoder, ColumnKind, FitOnlyTableTransform, NumericColumnScaler, PrepTask,
    PreparedTable, TargetScaler, TransformKind,
};
