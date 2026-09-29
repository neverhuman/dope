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

pub fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> DopeError {
    DopeError::Io {
        path: path.into(),
        source,
    }
}
