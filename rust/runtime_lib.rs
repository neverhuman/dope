//! Inference-only crate entry over the canonical modules, without compiling the
//! compiler, campaign ledger, certification workers, training or service routes.
//! Unit tests remain in the full root crate; deploy tests the shared sampler.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod codec;
pub mod error;
pub mod model;
pub mod neural;
pub mod sample;

pub use codec::{decode_kernel, encode_kernel};
