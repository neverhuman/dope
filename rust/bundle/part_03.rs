fn errno() -> i32 {
    // SAFETY: `__errno_location` returns this thread's errno cell. It is read
    // only after `open` or `openat` has returned -1 on this same thread.
    unsafe { *__errno_location() }
}

fn c_string(bytes: &[u8]) -> Result<CString, BundleError> {
    CString::new(bytes).map_err(|_| err("unsafe_path"))
}

fn owned_fd(fd: i32) -> Result<OwnedFd, BundleError> {
    if fd < 0 {
        return Err(if errno() == ELOOP {
            err("symlink_component")
        } else {
            err("dependency_io_failed")
        });
    }
    // SAFETY: `fd` is a non-negative descriptor just returned by `open` or
    // `openat` and not stored elsewhere. `OwnedFd` takes exclusive ownership.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn read_relative(root: &Path, path: &str, limit: u64) -> Result<Vec<u8>, BundleError> {
    component_path_allow_manifest(path)?;
    let root_name = c_string(root.as_os_str().as_bytes())?;
    let directory_flags = O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
    // SAFETY: `root_name` is NUL-terminated. `open` uses O_DIRECTORY, O_NOFOLLOW,
    // and O_CLOEXEC, and returns a new descriptor or -1. `owned_fd` keeps it.
    let root_fd = owned_fd(unsafe { open(root_name.as_ptr(), directory_flags, 0) })?;
    let mut directory = root_fd;
    let mut pieces = path.split('/').peekable();
    let file_fd = loop {
        let part = pieces.next().ok_or_else(|| err("unsafe_path"))?;
        let name = c_string(part.as_bytes())?;
        if pieces.peek().is_none() {
            let flags = O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK;
            // SAFETY: `name` is one NUL-terminated path segment. `openat` is
            // relative to an owned directory descriptor and uses O_NOFOLLOW.
            break owned_fd(unsafe { openat(directory.as_raw_fd(), name.as_ptr(), flags, 0) })?;
        }
        directory = owned_fd({
            // SAFETY: `name` is one NUL-terminated path segment. `openat` is relative
            // to an owned directory descriptor and uses O_DIRECTORY plus O_NOFOLLOW.
            unsafe { openat(directory.as_raw_fd(), name.as_ptr(), directory_flags, 0) }
        })?;
    };
    let mut file = File::from(file_fd);
    let metadata = file.metadata().map_err(|_| err("dependency_io_failed"))?;
    if !metadata.is_file() {
        return Err(err("dependency_not_regular_file"));
    }
    if metadata.len() > limit {
        return Err(err("dependency_too_large"));
    }
    let mut data = Vec::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(size) => {
                let next = u64::try_from(data.len())
                    .ok()
                    .and_then(|len| len.checked_add(size as u64))
                    .ok_or_else(|| err("integer_overflow"))?;
                if next > limit {
                    return Err(err("dependency_too_large"));
                }
                data.extend_from_slice(&buffer[..size]);
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(_) => return Err(err("dependency_io_failed")),
        }
    }
    Ok(data)
}

fn component_path_allow_manifest(path: &str) -> Result<(), BundleError> {
    if path == MANIFEST_NAME {
        return Ok(());
    }
    component_path(path)
}

fn ensure_real_directory(path: &Path) -> Result<(), BundleError> {
    let name = c_string(path.as_os_str().as_bytes())?;
    // SAFETY: `name` is NUL-terminated. `open` uses O_DIRECTORY, O_NOFOLLOW, and
    // O_CLOEXEC. The descriptor is dropped after the directory check.
    let fd = owned_fd(unsafe { open(name.as_ptr(), O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC, 0) })?;
    drop(fd);
    Ok(())
}

fn inventory(root: &Path, declared: &BTreeSet<String>) -> Result<(), BundleError> {
    ensure_real_directory(root)?;
    let mut prefixes = BTreeSet::new();
    for path in declared {
        let pieces: Vec<&str> = path.split('/').collect();
        for index in 1..pieces.len() {
            prefixes.insert(pieces[..index].join("/"));
        }
    }
    let mut found = BTreeSet::new();
    walk(root, "", declared, &prefixes, &mut found)?;
    if found != *declared {
        return Err(err("unresolved_dependency"));
    }
    Ok(())
}

fn walk(
    root: &Path,
    relative: &str,
    declared: &BTreeSet<String>,
    prefixes: &BTreeSet<String>,
    found: &mut BTreeSet<String>,
) -> Result<(), BundleError> {
    let location = if relative.is_empty() {
        root.to_path_buf()
    } else {
        root.join(relative)
    };
    ensure_real_directory(&location)?;
    let entries = fs::read_dir(&location).map_err(|_| err("dependency_io_failed"))?;
    for entry in entries {
        let entry = entry.map_err(|_| err("dependency_io_failed"))?;
        let name = entry.file_name();
        let name = name.to_str().ok_or_else(|| err("unsafe_path"))?;
        if name.is_empty() || name.contains(['/', '\\', '\0']) {
            return Err(err("unsafe_path"));
        }
        let child = if relative.is_empty() {
            name.to_string()
        } else {
            format!("{relative}/{name}")
        };
        let kind = entry.file_type().map_err(|_| err("dependency_io_failed"))?;
        if kind.is_symlink() {
            return Err(err("symlink_component"));
        }
        if kind.is_dir() {
            if !prefixes.contains(&child) {
                return Err(err("undeclared_component"));
            }
            walk(root, &child, declared, prefixes, found)?;
        } else if kind.is_file() {
            if !declared.contains(&child) {
                return Err(err("undeclared_component"));
            }
            found.insert(child);
        } else {
            return Err(err("dependency_not_regular_file"));
        }
    }
    Ok(())
}

fn read_varint(data: &[u8], mut offset: usize) -> Result<(u64, usize), BundleError> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        if offset >= data.len() {
            return Err(err("truncated_artifact"));
        }
        if shift > 63 {
            return Err(err("noncanonical_varint"));
        }
        let byte = data[offset];
        offset += 1;
        let piece = u64::from(byte & 0x7f);
        if shift == 63 && piece > 1 {
            return Err(err("noncanonical_varint"));
        }
        value |= piece << shift;
        if byte & 0x80 == 0 {
            if shift > 0 && piece == 0 {
                return Err(err("noncanonical_varint"));
            }
            return Ok((value, offset));
        }
        shift += 7;
    }
}

pub fn preflight_dpk(artifact: &[u8], limits: &BundleLimits) -> Result<DpkSummary, BundleError> {
    debug_assert_eq!(HEADER_BYTES, crate::codec::HEADER_BYTES);
    if artifact.len() < HEADER_BYTES {
        return Err(err("truncated_artifact"));
    }
    if artifact.len() > limits.max_component_bytes {
        return Err(err("component_cap_exceeded"));
    }
    let v3 = if artifact.starts_with(crate::codec::MAGIC) {
        if artifact[4] != 3 || !matches!(artifact[5], 2 | 3) {
            return Err(err("unsupported_dpk_version"));
        }
        true
    } else if artifact.starts_with(crate::codec::V2_MAGIC) {
        if artifact[4] != 2 || artifact[5] > 1 {
            return Err(err("unsupported_dpk_version"));
        }
        false
    } else {
        return Err(err("unsupported_dpk_version"));
    };
    if artifact[6] != 1 {
        return Err(err("unsupported_dpk_version"));
    }
    if !matches!(artifact[7], 0 | 1) {
        return Err(err("unsupported_dpk_header"));
    }
    if !matches!(artifact[8], 4 | 6 | 8 | 10 | 12 | 16) {
        return Err(err("unsupported_quantization"));
    }
    let flags = u16::from_le_bytes([artifact[10], artifact[11]]);
    if flags & !1 != 0 {
        return Err(err("invalid_flags"));
    }
    let features = u32::from_le_bytes(artifact[12..16].try_into().unwrap_or([0; 4]));
    if features == 0 || usize::try_from(features).unwrap_or(usize::MAX) > limits.max_features {
        return Err(err("invalid_dimensions"));
    }
    let payload_length = u64::from_le_bytes(artifact[32..40].try_into().unwrap_or([0; 8]));
    if payload_length != (artifact.len() - HEADER_BYTES) as u64 {
        return Err(err("truncated_artifact"));
    }
    let payload = &artifact[HEADER_BYTES..];
    let maximum_tag = if v3 && artifact[5] >= 2 { 6 } else { 5 };
    let mut offset = 0usize;
    let mut prior = 0u8;
    let mut expanded = 0u64;
    while offset < payload.len() {
        let tag_byte = payload[offset];
        offset += 1;
        let tag = tag_byte & 0x7f;
        if tag == 0 || tag > maximum_tag || tag <= prior {
            return Err(err("unknown_section"));
        }
        prior = tag;
        if tag == 6 && artifact.len() > NEURAL_CAP {
            return Err(err("neural_cap_exceeded"));
        }
        let (encoded_length, next) = read_varint(payload, offset)?;
        offset = next;
        let raw_length = if tag_byte & 0x80 != 0 {
            let (raw_length, next) = read_varint(payload, offset)?;
            offset = next;
            if raw_length == 0 {
                return Err(err("invalid_dimensions"));
            }
            raw_length
        } else {
            encoded_length
        };
        let end = offset
            .checked_add(usize::try_from(encoded_length).map_err(|_| err("truncated_artifact"))?)
            .ok_or_else(|| err("integer_overflow"))?;
        if end > payload.len() {
            return Err(err("truncated_artifact"));
        }
        if tag == 6 && tag_byte & 0x80 == 0 {
            let body = &payload[offset..end];
            if body.windows(3).any(|window| window == b"NaN")
                || body.windows(8).any(|window| window == b"Infinity")
            {
                return Err(err("nonfinite_value"));
            }
        }
        if raw_length > limits.max_expanded_section_bytes as u64 {
            return Err(err("expansion_cap_exceeded"));
        }
        expanded = checked_add(expanded, raw_length)?;
        if expanded > limits.max_expanded_section_bytes as u64 {
            return Err(err("expansion_cap_exceeded"));
        }
        offset = end;
    }
    Ok(DpkSummary {
        features,
        minor: artifact[5],
        v3,
    })
}
