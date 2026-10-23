//! # DOPE — Downstream Objective-Preserving Encoding
//!
//! Compile a table into a tiny generative *program* (a few kilobytes,
//! stored as safetensors) whose samples preserve what downstream models
//! learn from the table: models trained only on the synthetic rows recover
//! most of the predictive gain of models trained on the real rows.
//!
//! ```no_run
//! use dope_core::{fit, sample, FitOptions, SampleOptions, RawTable};
//! let table = RawTable::read("train.csv".as_ref())?;
//! let target = table.column_index("price")?;
//! let fitted = fit(&table, target, &FitOptions { budget: Some(4096), ..Default::default() })?;
//! std::fs::write("model.dope", &fitted.bytes)?;
//! let program = dope_core::decode(&fitted.bytes)?;
//! let rows = sample(&program, &SampleOptions { rows: 10_000, ..Default::default() })?;
//! # Ok::<(), dope_core::DopeError>(())
//! ```
//!
//! The pipeline is: intake ([`schema`]) → marginals ([`marginal`]) →
//! Gaussian dependence ([`copula`]) → conditional model ([`target`],
//! [`train`]) → noise ([`noise`]) → rate–distortion quantization
//! ([`quant`]) → safetensors ([`artifact`]). Sampling ([`sample`]) is one
//! tensor program that runs on CPU, CUDA or Metal through candle.

// Index loops mirror the linear algebra they implement, and the published
// AS241/Eigen coefficients are kept digit for digit.
#![allow(clippy::needless_range_loop, clippy::excessive_precision)]

pub mod artifact;
pub mod budget;
pub mod compile;
pub mod copula;
pub mod device;
pub mod encode;
pub mod error;
pub mod eval;
pub mod forest;
pub mod landmark;
pub mod learned;
pub mod levels;
pub mod marginal;
pub mod noise;
pub mod output;
pub mod program;
pub mod quant;
pub mod regimes;
pub mod rng;
pub mod sample;
pub mod sample_cpu;
pub mod schema;
pub mod screen;
pub mod stats;
pub mod table;
pub mod target;
pub mod train;
pub mod train_cpu;

pub use artifact::{byte_report, decode, ByteReport};
pub use budget::{ModelSize, ParameterCount};
pub use compile::{fit, FitOptions, FitReport, Fitted, DEFAULT_BUDGET};
pub use encode::{MlEncoder, Scaling};
pub use error::{DopeError, Result};
pub use program::Program;
pub use sample::{sample, SampleOptions, Samples};
pub use schema::Task;
pub use table::RawTable;
