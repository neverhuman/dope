//! Development probes for portable levels and landmarks.
//!
//! * [`nprng`] reproduces NumPy's seeded permutation, so [`inputs`] writes the probes' exact
//!   train/test rows from the pinned PMLB files.
//! * [`levels`] runs the levels probe against a `dope` binary and replays its receipts.
//! * [`landmarks`] replays the landmark-probe receipts.
//!
//! Everything here is selection-exposed development evidence on nine public tables. It is not
//! part of the shipped engine and supports no benchmark, certification or privacy claim.

use std::fmt;
use std::path::Path;

pub mod inputs;
pub mod landmarks;
pub mod levels;
pub mod nprng;
pub mod stats;

/// Every fallible probe operation returns this error. Messages name the file or the step that
/// failed and never echo table cells.
#[derive(Debug)]
pub enum Error {
    /// A pinned input, a receipt or an argument is missing or does not match the manifest.
    Input(String),
    /// Reading or writing a file failed.
    Io(String),
    /// The `dope` binary failed or wrote a report this tool cannot read.
    Engine(String),
}

impl Error {
    pub fn msg(m: impl Into<String>) -> Self {
        Error::Input(m.into())
    }

    pub fn engine(m: impl Into<String>) -> Self {
        Error::Engine(m.into())
    }

    pub fn io(path: &Path, e: std::io::Error) -> Self {
        Error::Io(format!("{}: {e}", path.display()))
    }

    pub fn csv(e: csv::Error) -> Self {
        Error::Input(e.to_string())
    }

    /// Agent-readable repair guidance: what the step was for, why it failed, the usual fixes,
    /// where it is documented, and what to do before rerunning.
    pub fn repair(&self) -> RepairHint {
        match self {
            Error::Input(_) => RepairHint {
                purpose: "read pinned probe inputs, receipts or arguments",
                reason: "a file is missing, malformed, or differs from the pinned digests",
                common_fixes: "re-download the pinned PMLB files, run `dope-probes split`, or point at a receipt written by dope-probes",
                docs_url: "bench/real-world/probes/README.md#what-reproduces-and-what-does-not",
                repair_hint: "check the reported path against bench/real-world/probes/inputs.json, then rerun the same command",
            },
            Error::Io(_) => RepairHint {
                purpose: "read or write a probe file",
                reason: "the filesystem operation failed",
                common_fixes: "check the path, permissions and free space",
                docs_url: "bench/real-world/probes/README.md#replay-no-data-needed",
                repair_hint: "rerun the same command after checking the path",
            },
            Error::Engine(_) => RepairHint {
                purpose: "fit, sample or evaluate through a dope binary",
                reason: "the binary failed or wrote a report this tool cannot read",
                common_fixes: "pass --dope a release build of the engine; the committed receipts used commit 65b1d2a",
                docs_url: "bench/real-world/probes/README.md#rebuild-the-inputs-and-rerun-the-levels-probe",
                repair_hint: "run the reported dope command by hand to see its own error, then rerun",
            },
        }
    }

    /// What usually fixes this error, for the person or agent who reruns the command.
    pub fn repair_hint(&self) -> &'static str {
        self.repair().repair_hint
    }

    /// Where the relevant behaviour is documented (repository-relative).
    pub fn docs_url(&self) -> &'static str {
        self.repair().docs_url
    }
}

/// Repair guidance attached to every [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepairHint {
    pub purpose: &'static str,
    pub reason: &'static str,
    pub common_fixes: &'static str,
    pub docs_url: &'static str,
    pub repair_hint: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Input(m) => write!(f, "input: {m}"),
            Error::Io(m) => write!(f, "io: {m}"),
            Error::Engine(m) => write!(f, "engine: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Read one JSON value per non-empty line.
pub fn load_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l)
                .map_err(|e| Error::msg(format!("{}:{}: {e}", path.display(), i + 1)))
        })
        .collect()
}

/// An integer with comma thousands separators.
pub fn fmt_int(v: i64) -> String {
    let digits = v.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if v < 0 {
        format!("-{out}")
    } else {
        out
    }
}
