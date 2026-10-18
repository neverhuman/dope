use super::exit_codes::{Code, Failure, Result};
use super::registry::Limits;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn local(path: &Path) -> Result<()> {
    let text = path.to_string_lossy();
    if text.len() > 4096 || text.is_empty() || text.contains("://") || text.starts_with("//") {
        return Err(Failure::usage("local_path_required"));
    }
    Ok(())
}

pub fn read(path: &Path, cap: u64) -> Result<Vec<u8>> {
    local(path)?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|_| Failure::new(Code::Incomplete, "input_unavailable_or_not_regular"))?;
    let meta = file
        .metadata()
        .map_err(|_| Failure::new(Code::Internal, "local_operation_failed"))?;
    if !meta.is_file() {
        return Err(Failure::usage("regular_file_required"));
    }
    if meta.len() > cap {
        return Err(Failure::budget());
    }
    let mut bytes = Vec::new();
    file.take(cap + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Failure::new(Code::Internal, "local_operation_failed"))?;
    if bytes.len() as u64 > cap {
        return Err(Failure::budget());
    }
    Ok(bytes)
}

pub fn outputs(paths: &[&Path]) -> Result<()> {
    let mut destinations = std::collections::BTreeSet::new();
    for path in paths {
        local(path)?;
        if fs::symlink_metadata(path).is_ok() {
            return Err(Failure::usage("output_already_exists"));
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent = fs::canonicalize(parent)
            .map_err(|_| Failure::new(Code::Internal, "output_parent_unavailable"))?;
        let name = path
            .file_name()
            .ok_or_else(|| Failure::usage("output_name_required"))?;
        if !destinations.insert(parent.join(name)) {
            return Err(Failure::usage("output_paths_must_differ"));
        }
    }
    Ok(())
}

pub fn write(path: &Path, bytes: &[u8], limits: &Limits) -> Result<()> {
    if bytes.len() as u64 > limits.max_output_bytes {
        return Err(Failure::budget());
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|_| Failure::new(Code::Internal, "output_create_failed"))?;
    file.write_all(bytes)
        .map_err(|_| Failure::new(Code::Internal, "output_write_failed"))
}

pub fn emit(value: &impl serde::Serialize, out: Option<&Path>, limits: &Limits) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|_| Failure::new(Code::Internal, "serialization_failed"))?;
    if let Some(path) = out {
        write(path, &bytes, limits)
    } else if bytes.len() as u64 > limits.max_output_bytes {
        Err(Failure::budget())
    } else {
        stdout(&bytes, limits)
    }
}

pub fn stdout(bytes: &[u8], limits: &Limits) -> Result<()> {
    if bytes.len() as u64 + 1 > limits.max_output_bytes {
        return Err(Failure::budget());
    }
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(bytes)
        .and_then(|()| stdout.write_all(b"\n"))
        .map_err(|_| Failure::new(Code::Internal, "output_write_failed"))
}

pub fn json(bytes: &[u8]) -> Result<serde_json::Value> {
    let text = std::str::from_utf8(bytes).map_err(|_| Failure::usage("invalid_json"))?;
    dope_kernel::metrics::strict_parse_json(text).map_err(|error| {
        if error.reason == "duplicate_json_key" {
            Failure::usage("duplicate_json_key")
        } else {
            Failure::usage("invalid_json")
        }
    })
}

/// Pin bounded captured bytes for older library APIs that accept paths only.
/// This is an in-memory regular file, never a user-controlled temporary path.
#[cfg(target_os = "linux")]
pub fn sink() -> Result<(File, PathBuf)> {
    use std::os::fd::{AsRawFd, FromRawFd};
    let fd = unsafe {
        libc::memfd_create(
            c"dope-cli".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    if fd < 0 {
        return Err(Failure::new(Code::Internal, "snapshot_failed"));
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let path = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
    Ok((file, path))
}

#[cfg(target_os = "linux")]
pub fn snapshot(bytes: &[u8]) -> Result<(File, PathBuf)> {
    use std::os::fd::AsRawFd;
    let (mut file, path) = sink()?;
    file.write_all(bytes)
        .map_err(|_| Failure::new(Code::Internal, "snapshot_failed"))?;
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
        return Err(Failure::new(Code::Internal, "snapshot_failed"));
    }
    Ok((file, path))
}

#[cfg(not(target_os = "linux"))]
pub fn sink() -> Result<(File, PathBuf)> {
    Err(Failure::unsupported())
}

#[cfg(not(target_os = "linux"))]
pub fn snapshot(_: &[u8]) -> Result<(File, PathBuf)> {
    Err(Failure::unsupported())
}
