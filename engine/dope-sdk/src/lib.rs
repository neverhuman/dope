//! Versioned DOPE integration facade. No model arithmetic or preparation is duplicated.
//! New input semantics require a new recipe and qualification, not a facade version bump.

mod nc1_view;

pub const SDK_CONTRACT: &str = "dope-sdk/v0";

/// Lane-A integration surface. Types are the original engine types, preserving numerical
/// behavior, source scopes, cancellation and private/public report boundaries.
pub mod v0 {
    pub use dope_core::{DopeError, Result, Task};

    pub mod target {
        pub use dope_core::target::Head;
    }

    pub mod landmark {
        pub use dope_core::landmark::{
            assign_folds, signature, signature_requested, CellState, ExecutionPlan, ExecutionWork,
            FeatureSpectrum, LandmarkBundle, LandmarkFrame, LandmarkGroup, LandmarkMode,
            LandmarkOptions, LandmarkScope, LandmarkStatus, ProgressiveLandmarks, RowSelection,
            Signature, SignatureKind, SubsampleSpec, WorkReport, LANDMARK_SLOTS, PANEL_V1,
        };
        pub mod bundle {
            pub use dope_core::landmark::bundle::{
                parse_landmark_ids, source_rows_digest, BundleScope, LandmarkBundle, LandmarkGroup,
                LandmarkMode, LandmarkScope, LandmarkStatus, RowSelection, SubsampleSpec,
                LANDMARK_SLOTS, PROGRESSIVE_ORDER,
            };
        }
    }

    pub mod mlc_v3 {
        pub use dope_core::mlc_v3::{
            MlcV3Model, MlcV3RoutedOutput, MlcV3RoutedTensors, MlcV3WeightInventory,
            MLC_V3_FEATURES_512_DIM, MLC_V3_FEATURES_629_DIM, MLC_V3_GATE_KEYS_14,
        };
    }

    pub mod prepare {
        pub use dope_core::prepare::{
            build_conditioning, build_conditioning_at_level, build_conditioning_sampled,
            compute_landmark_group, prepare_table, prepare_table_cells, prepare_table_cells_timed,
            prepare_table_cells_with_ids, prepare_table_cells_with_ids_timed,
            prepare_table_cells_with_policy, prepare_table_lossless, prepare_table_mixed,
            prepare_table_mixed_with_policy, prepare_table_timed, prepare_table_with_ids,
            prepare_table_with_ids_timed, prepare_table_with_policy,
            progressive_landmarks_for_table, ClassVocabPolicy, ColumnKind, DopeConditioning,
            FitOnlyTableTransform, NativeReportView, PrepTask, PrepareStageTimings, PreparedTable,
            COL_LANDMARK_DIM, FULL_REPORT_RECIPE, MAX_COLS, OBSERVED_RECIPE,
        };
        pub mod scaling {
            pub use dope_core::prepare::scaling::{ClassVocabPolicy, ColumnKind, PrepTask};
        }
    }
}

/// NC1 successor normalization; never reinterpret Lane-A model preparation.
pub mod nc1 {
    pub use crate::nc1_view::{TargetLastView, TargetLastViewExt};
    pub use dope_norm::*;
}

/// Explicit successor graph and evidence APIs; Lane-A remains unchanged.
pub mod mlc_v4 {
    pub use dope_core::mlc_v4::*;
}
pub mod dvs {
    pub use dope_core::dvs::*;
}
