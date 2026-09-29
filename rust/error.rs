use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum DopeError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SQLite error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("invalid data: {0}")]
    Data(String),
    #[error("invalid or unsupported artifact: {0}")]
    Codec(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, DopeError>;

/// Stable, source-data-free guidance that accompanies a command failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RepairHint {
    pub purpose: &'static str,
    pub reason: &'static str,
    pub common_fixes: &'static str,
    pub docs_url: &'static str,
    pub repair_hint: &'static str,
}

impl DopeError {
    pub fn repair_hint(&self) -> RepairHint {
        match self {
            Self::Io { .. } => RepairHint {
                purpose: "read or write a requested artifact",
                reason: "the filesystem operation failed",
                common_fixes: "check path, permissions, and available space",
                docs_url: "docs/testing.md#local-repair",
                repair_hint: "rerun the same command after checking the path and free space",
            },
            Self::Data(_) | Self::Csv(_) => RepairHint {
                purpose: "validate a numeric input table",
                reason: "the input does not meet the selected tier contract",
                common_fixes: "use headerless numeric features; normalize L3 values to [0,1] and use a binary 0/1 target",
                docs_url: "docs/testing.md#local-repair",
                repair_hint: "correct the reported row and column, then rerun compile or certify",
            },
            Self::Codec(_) | Self::Json(_) => RepairHint {
                purpose: "read a versioned artifact or evidence file",
                reason: "the bytes or schema did not validate",
                common_fixes: "check the artifact digest, producer version, and file completeness",
                docs_url: "docs/testing.md#local-repair",
                repair_hint: "regenerate the artifact from its recorded source and rerun inspection",
            },
            Self::Sql(_) => RepairHint {
                purpose: "open the local campaign ledger",
                reason: "the SQLite operation failed",
                common_fixes: "check ledger path, permissions, and schema version",
                docs_url: "docs/testing.md#local-repair",
                repair_hint: "repair the local ledger and rerun the same campaign step",
            },
            Self::Unsupported(_) => RepairHint {
                purpose: "check a requested backend or release capability",
                reason: "the current build cannot supply that capability",
                common_fixes: "use an available backend or the documented GPU build",
                docs_url: "docs/testing.md#local-repair",
                repair_hint: "inspect build features and rerun only the affected step",
            },
        }
    }
}

pub fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> DopeError {
    DopeError::Io {
        path: path.into(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_receipt_is_typed_and_omits_source_data() {
        let error = DopeError::Data("private-field-value".into());
        let receipt = serde_json::to_string(&error.repair_hint()).unwrap();
        assert!(receipt.contains("docs_url"));
        assert!(receipt.contains("common_fixes"));
        assert!(!receipt.contains("private-field-value"));
    }
}
