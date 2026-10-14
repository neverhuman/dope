include!("compiler/part_01.rs");
include!("compiler/part_02.rs");
include!("compiler/interaction_screen.rs");
include!("compiler/fair_allocation.rs");
include!("compiler/residual_oof.rs");

#[cfg(all(test, feature = "gpu-training"))]
mod gpu_readout_tests;
