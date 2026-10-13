include!("compiler/part_01.rs");
include!("compiler/part_02.rs");

pub mod interactions;

#[cfg(all(test, feature = "gpu-training"))]
mod gpu_readout_tests;
