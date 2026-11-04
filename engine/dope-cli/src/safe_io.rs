//! Bounded local-only I/O and explicit private, no-overwrite exports.
use serde::de::DeserializeOwned;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

pub(crate) fn read_bytes(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let before = fs::symlink_metadata(path).map_err(|_| "input_open_failed")?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err("input_not_regular".into());
    }
    if before.len() > limit as u64 {
        return Err("input_size_limit".into());
    }
    let file = File::open(path).map_err(|_| "input_open_failed")?;
    let opened = file.metadata().map_err(|_| "input_open_failed")?;
    if !opened.is_file() {
        return Err("input_not_regular".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != opened.dev() || before.ino() != opened.ino() {
            return Err("input_changed".into());
        }
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "input_read_failed")?;
    if bytes.len() > limit {
        return Err("input_size_limit".into());
    }
    Ok(bytes)
}
pub(crate) fn read_json<T: DeserializeOwned>(path: &Path, limit: usize) -> Result<T, String> {
    serde_json::from_slice(&read_bytes(path, limit)?)
        .map_err(|_| "invalid_next_version_json".into())
}
pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            "output_exists"
        } else {
            "output_open_failed"
        }
    })?;
    file.write_all(bytes).map_err(|_| "output_write_failed")?;
    file.sync_all().map_err(|_| "output_sync_failed")?;
    Ok(())
}
pub(crate) fn write_directory(path: &Path, files: &[(&str, Vec<u8>)]) -> Result<(), String> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            "output_exists"
        } else {
            "output_directory_failed"
        }
    })?;
    for (name, bytes) in files {
        write_new(&path.join(name), bytes)?;
    }
    Ok(())
}
pub(crate) fn json<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec_pretty(value).map_err(|_| "output_serialization_failed".into())
}
