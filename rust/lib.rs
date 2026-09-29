#![deny(unsafe_op_in_unsafe_fn)]

pub mod access;
pub mod auditor;
pub mod calibration;
pub mod campaign;
pub mod certification;
pub mod codec;
pub mod compiler;
pub mod contract;
pub mod corpus;
pub mod data;
pub mod deep_campaign;
pub mod embedding;
pub mod error;
pub mod fitness;
pub mod fitness_metrics;
pub mod inspect;
pub mod ledger;
#[cfg(feature = "gpu-training")]
mod libtorch;
pub mod model;
pub mod neural;
pub mod neural_train;
pub mod packed;
pub mod production;
pub mod regression_embeddings;
pub mod release;
pub mod router;
pub mod sample;

pub use auditor::{AuditorBackend, auditor_backend};
pub use calibration::{
    CalibrationResult, LanguageCalibration, TrainingOptions, load_language, train_language,
};
pub use campaign::{CampaignPhase, CampaignState, MetricsBundle, RouterEvidence};
pub use codec::{decode_kernel, encode_kernel, load_kernel};
pub use compiler::{
    CompileOptions, CompileResult, compile_kernel_from_arrays, compile_kernel_from_dir,
};
pub use contract::{
    EmpiricalBackend, auditor_specs, deep_confirmation_backends, empirical_backend,
    empirical_backends, formal_dp_configurations,
};
pub use corpus::{
    CorpusInventory, SplitManifest, ValidationSubmanifest, inventory_corpus, load_inventory,
    refresh_inventory_lineages, split_inventory, split_validation_manifest,
};
pub use embedding::{
    ACTION_EMBEDDING_DIMENSION, ACTION_EMBEDDING_VERSION, DatasetActionEmbedding, DatasetEmbedder,
    EmbeddingRouterQualification, TargetSelector, embed_dataset, load_numeric_csv,
    qualify_embedding_router, write_embedding_csv,
};
pub use error::{DopeError, Result};
pub use inspect::{Inspection, inspect_kernel};
pub use ledger::{CampaignLedger, LedgerStatus};
pub use model::{JointGenerator, Kernel, KernelProgram, NeuralArchitecture, Task};
pub use production::{
    BuildRcOptions, KpiAggregate, KpiCell, KpiContract, KpiSummary, RunContract, build_rc_manifest,
};
pub use regression_embeddings::{
    RegressionCorpusInventory, RegressionEmbeddingBatchManifest, RegressionEmbeddingBatchOptions,
    RegressionEmbeddingRecord, RegressionEmbeddingVerification, embed_regression_corpus,
    inspect_regression_corpus, verify_embedding_vector, verify_regression_embeddings,
};
pub use release::{ConversionOptions, ConversionResult, convert_release};
pub use router::{
    CandidateActionEmbedding, DatasetSketch, QuantizedRouter, RouterActionEmbedding, RouterBundle,
    RouterPrediction, SelectionPolicy,
};
pub use sample::{SampleOptions, sample_kernel, sample_kernel_to_csv};
