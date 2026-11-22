//! Owner-side functional witnesses (`LearningSignatureV1`) and Ridge Certificate.

pub mod certificate;
pub mod witnesses;

pub use certificate::{ridge_certificate, RidgeCertificate};
pub use witnesses::{LearningSignatureV1, WitnessGapReport, MAX_WITNESSES};
