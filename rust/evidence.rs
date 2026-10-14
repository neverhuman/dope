//! New vNext evidence boundary. Historical production/ledger receipts are unchanged.
//! A verified receipt binds bytes to an independently trusted executor policy;
//! it does not grant scientific truth, isolation, privacy or release authority.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use rustix::fs::{Mode, OFlags, open, openat};
use serde::Deserializer;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};

use crate::error::{DopeError, Result};

const MAX_INTEGER: u64 = (1 << 53) - 1;
const MAX_MANIFEST: usize = 16 * 1024 * 1024;
const RUN_SCHEMA: &str = include_str!("../schemas/transfer-run-v1.schema.json");
const BEHAVIOR_SCHEMA: &str = include_str!("../schemas/behavior-contract-v1.schema.json");
const FREEZE_SCHEMA: &str = include_str!("../schemas/freeze-v1.schema.json");
const FIT_FIELDS: [&str; 8] = [
    "data_hash",
    "split_hash",
    "generator_hash",
    "configuration_hash",
    "feature_transform_hash",
    "target_transform_hash",
    "generator_fit_seed",
    "global_seed",
];

fn error(code: &'static str) -> DopeError {
    DopeError::Data(code.into())
}
fn require(condition: bool, code: &'static str) -> Result<()> {
    if condition { Ok(()) } else { Err(error(code)) }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

include!("evidence/json.rs");
include!("evidence/schema.rs");
include!("evidence/identity.rs");
include!("evidence/dependencies.rs");
include!("evidence/verify.rs");
include!("evidence/cache.rs");
include!("evidence/successors.rs");

#[cfg(test)]
mod tests {
    include!("evidence/tests.rs");
}
