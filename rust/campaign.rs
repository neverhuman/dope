use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufWriter, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::access::{AccessRole, ValidationAccessBroker};
use crate::certification::{
    GoldCellOptions, evaluate_gold_cell, evaluate_prepared_gold_cell, prepare_gold_cell,
};
use crate::contract::{auditor_specs, empirical_backends};
use crate::corpus::{
    SplitManifest, ValidationSubmanifest, validate_split_manifest, validate_validation_submanifest,
};
use crate::data::Table;
use crate::error::{DopeError, Result, io_error};
use crate::ledger::{
    CampaignLedger, FailureClass, HEARTBEAT_SECONDS, JOB_EVIDENCE_VERSION, JobEvidence, JobReceipt,
    JobState, LEASE_SECONDS, LeasedJob,
};
use crate::model::Task;
use crate::production::{
    ContentHashes, CoverageEntry, CoverageReport, GateEvidence, KPI_CONTRACT_SHA256, KpiCell,
    KpiSummary, StructuralProfile, aggregate_kpis, canonical_json, file_hashes, hashes, read_json,
    write_canonical,
};
use crate::router::{
    DATASET_SKETCH_VERSION, DatasetSketch, MAX_ROUTER_BUNDLE_BYTES, QuantizedRouter, RouterBundle,
    RouterLabel, RouterTrainingReport, SelectionPolicy, train_distilled_router,
};

pub const MIN_DURABLE_FREE_GIB: u64 = 150;
const GIB: u64 = 1024 * 1024 * 1024;

include!("campaign/cohort.rs");
include!("campaign/gold_evidence.rs");
include!("campaign/gold_baseline.rs");
include!("campaign/gold_blocks.rs");
include!("campaign/state.rs");
include!("campaign/router_training.rs");
include!("campaign/worker.rs");
include!("campaign/metrics.rs");
include!("campaign/validation_metrics.rs");
include!("campaign/release_decision.rs");
include!("campaign/tests.rs");
