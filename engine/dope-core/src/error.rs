use thiserror::Error;

/// Every fallible DOPE operation returns this error.
///
/// Messages describe the failed operation and never echo cell values, so an
/// error can be logged next to private data without leaking it.
#[derive(Debug, Error)]
pub enum DopeError {
    #[error("input: {0}")]
    Input(String),
    #[error("artifact: {0}")]
    Artifact(String),
    #[error("budget: {0}")]
    Budget(String),
    #[error("device: {0}")]
    Device(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, DopeError>;

impl DopeError {
    /// What usually fixes this error, for the person or agent who reruns the command.
    pub fn repair_hint(&self) -> &'static str {
        match self {
            DopeError::Input(_) => {
                "check the CSV header, the --target column name and that the file has at least twenty labelled rows"
            }
            DopeError::Artifact(_) => {
                "the file is not a DOPE artifact from this engine version; refit it with `dope fit`"
            }
            DopeError::Budget(_) => {
                "raise --budget or --max-parameters, or pass --anonymize to reduce schema bytes"
            }
            DopeError::Device(_) => {
                "run `dope doctor` to list compiled backends; build with --features cuda or metal, or pass --device cpu"
            }
            DopeError::Io(_) => "check that the path exists and is readable or writable",
        }
    }

    /// Where the relevant behaviour is documented (repository-relative).
    pub fn docs_url(&self) -> &'static str {
        match self {
            DopeError::Device(_) => "engine/README.md#sampling-on-any-device",
            DopeError::Budget(_) | DopeError::Artifact(_) => "engine/README.md#the-program",
            _ => "engine/README.md#commands",
        }
    }
}

impl From<candle_core::Error> for DopeError {
    fn from(error: candle_core::Error) -> Self {
        DopeError::Device(error.to_string())
    }
}

impl From<csv::Error> for DopeError {
    fn from(error: csv::Error) -> Self {
        // csv errors can quote a record; keep only the position.
        let position = error
            .position()
            .map(|p| format!(" near line {}", p.line()))
            .unwrap_or_default();
        DopeError::Input(format!("malformed CSV{position}"))
    }
}

impl From<serde_json::Error> for DopeError {
    fn from(error: serde_json::Error) -> Self {
        DopeError::Artifact(format!("metadata JSON: {}", error.classify_name()))
    }
}

trait ClassifyName {
    fn classify_name(&self) -> &'static str;
}

impl ClassifyName for serde_json::Error {
    fn classify_name(&self) -> &'static str {
        match self.classify() {
            serde_json::error::Category::Io => "io",
            serde_json::error::Category::Syntax => "syntax",
            serde_json::error::Category::Data => "data",
            serde_json::error::Category::Eof => "eof",
        }
    }
}
