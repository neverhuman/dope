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

#[repr(C, align(8))]
struct DirBuf([u8; 8192]);

#[repr(C, align(8))]
struct StatxBuf([u8; 256]);

#[derive(Clone, Copy)]
enum EntryKind {
    File,
    Directory,
    Symlink,
    Other,
    Unknown,
}

#[derive(Clone, Copy)]
struct WalkBudget {
    max_entries: usize,
    max_name_bytes: usize,
}

struct WalkLists<'a> {
    declared: &'a BTreeSet<String>,
    prefixes: &'a BTreeSet<String>,
    found: &'a mut BTreeSet<String>,
}

fn inventory(
    root: &Path,
    declared: &BTreeSet<String>,
    budget: WalkBudget,
) -> Result<(), BundleError> {
    let root_name = c_string(root.as_os_str().as_bytes())?;
    let directory_flags = O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
    // SAFETY: `root_name` is NUL-terminated. `open` uses O_DIRECTORY, O_NOFOLLOW,
    // and O_CLOEXEC, and returns a new descriptor or -1. `owned_fd` keeps it.
    let root_fd = owned_fd(unsafe { open(root_name.as_ptr(), directory_flags, 0) })?;
    let mut prefixes = BTreeSet::new();
    for path in declared {
        let pieces: Vec<&str> = path.split('/').collect();
        for index in 1..pieces.len() {
            prefixes.insert(pieces[..index].join("/"));
        }
    }
    let mut found = BTreeSet::new();
    walk(
        root_fd,
        "",
        &mut WalkLists {
            declared,
            prefixes: &prefixes,
            found: &mut found,
        },
        budget,
    )?;
    if found != *declared {
        return Err(err("unresolved_dependency"));
    }
    Ok(())
}

fn walk(
    directory: OwnedFd,
    relative: &str,
    lists: &mut WalkLists<'_>,
    budget: WalkBudget,
) -> Result<(), BundleError> {
    let mut buffer = DirBuf([0u8; 8192]);
    let mut entry_count = 0usize;
    let mut name_bytes = 0usize;
    loop {
        let fd = directory.as_raw_fd();
        let ptr = buffer.0.as_mut_ptr();
        let len = buffer.0.len();
        // SAFETY: `fd` is an owned directory descriptor. `ptr` points at `len`
        // writable bytes. `getdents64` fills only that buffer and returns the
        // byte count or -1.
        let read = unsafe { getdents64(fd, ptr, len) };
        if read < 0 {
            if errno() == EINTR {
                continue;
            }
            return Err(err("dependency_io_failed"));
        }
        if read == 0 {
            break;
        }
        let mut offset = 0usize;
        let total = read as usize;
        while offset < total {
            if offset + 19 > total {
                return Err(err("dependency_io_failed"));
            }
            let reclen =
                u16::from_ne_bytes([buffer.0[offset + 16], buffer.0[offset + 17]]) as usize;
            if reclen < 20 || offset + reclen > total {
                return Err(err("dependency_io_failed"));
            }
            let d_type = buffer.0[offset + 18];
            let name_end = buffer.0[offset + 19..offset + reclen]
                .iter()
                .position(|byte| *byte == 0)
                .map(|index| offset + 19 + index)
                .ok_or_else(|| err("dependency_io_failed"))?;
            let name = std::str::from_utf8(&buffer.0[offset + 19..name_end])
                .map_err(|_| err("unsafe_path"))?;
            if name != "." && name != ".." {
                entry_count += 1;
                name_bytes += name.len();
                if entry_count > budget.max_entries || name_bytes > budget.max_name_bytes {
                    return Err(err("too_many_components"));
                }
                accept_entry(
                    &directory,
                    relative,
                    name,
                    kind_from_dtype(d_type),
                    lists,
                    budget,
                )?;
            }
            offset += reclen;
        }
    }
    Ok(())
}

fn accept_entry(
    directory: &OwnedFd,
    relative: &str,
    name: &str,
    observed: EntryKind,
    lists: &mut WalkLists<'_>,
    budget: WalkBudget,
) -> Result<(), BundleError> {
    if name.is_empty() || name.contains(['/', '\\', '\0']) {
        return Err(err("unsafe_path"));
    }
    let kind = if matches!(observed, EntryKind::Unknown) {
        entry_kind_at(directory.as_raw_fd(), name)?
    } else {
        observed
    };
    let child = if relative.is_empty() {
        name.to_string()
    } else {
        format!("{relative}/{name}")
    };
    match kind {
        EntryKind::Symlink => Err(err("symlink_component")),
        EntryKind::Directory => {
            if !lists.prefixes.contains(&child) {
                return Err(err("undeclared_component"));
            }
            let segment = c_string(name.as_bytes())?;
            let child_fd = owned_fd({
                // SAFETY: `segment` is one NUL-terminated path segment. `openat`
                // is relative to the owned directory descriptor and uses
                // O_DIRECTORY plus O_NOFOLLOW.
                unsafe {
                    openat(
                        directory.as_raw_fd(),
                        segment.as_ptr(),
                        O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC,
                        0,
                    )
                }
            })?;
            walk(child_fd, &child, lists, budget)
        }
        EntryKind::File => {
            if !lists.declared.contains(&child) {
                return Err(err("undeclared_component"));
            }
            lists.found.insert(child);
            Ok(())
        }
        EntryKind::Other | EntryKind::Unknown => Err(err("dependency_not_regular_file")),
    }
}

fn kind_from_dtype(d_type: u8) -> EntryKind {
    match d_type {
        4 => EntryKind::Directory,
        8 => EntryKind::File,
        10 => EntryKind::Symlink,
        0 => EntryKind::Unknown,
        _ => EntryKind::Other,
    }
}

fn entry_kind_at(dirfd: i32, name: &str) -> Result<EntryKind, BundleError> {
    let segment = c_string(name.as_bytes())?;
    let mut buffer = StatxBuf([0u8; 256]);
    // SAFETY: `segment` is one NUL-terminated path segment. `statx` uses
    // AT_SYMLINK_NOFOLLOW and writes the fixed 256-byte statx record.
    let status = unsafe {
        statx(
            dirfd,
            segment.as_ptr(),
            AT_SYMLINK_NOFOLLOW,
            STATX_TYPE,
            buffer.0.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(if errno() == ELOOP {
            err("symlink_component")
        } else {
            err("dependency_io_failed")
        });
    }
    let mode = u16::from_ne_bytes([buffer.0[28], buffer.0[29]]);
    Ok(match mode & 0o170000u16 {
        0o120000u16 => EntryKind::Symlink,
        0o040000u16 => EntryKind::Directory,
        0o100000u16 => EntryKind::File,
        _ => EntryKind::Other,
    })
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
        if artifact[4] != 3 || artifact[5] > 3 {
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
