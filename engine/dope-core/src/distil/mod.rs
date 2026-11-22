//! Teacher-distilled conditional student (`main + pairs + soft-Newton oblivious forest`).

pub mod main_effects;
pub mod pairs;
pub mod student;

pub use main_effects::{MainEffect, MainEffectSet};
pub use pairs::{add_pair_logits, mine_pairs, PairEffect};
pub use student::{probe_support_weights, DistilConfig, DistilledStudent};
