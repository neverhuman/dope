include!("model/part_01.rs");
include!("model/part_02.rs");
mod residual;
pub use residual::{ResidualProfile, ResidualV1};

#[cfg(test)]
pub(crate) mod repro_tests;
