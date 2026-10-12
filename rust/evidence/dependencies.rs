// Open relative to directory descriptors with NOFOLLOW for every component.
// The root directory is supplied by the trusted verifier, not by the receipt.
fn read_file(root: &Path, path: &str, manifest: bool) -> Result<(String, u64, Vec<u8>)> {
    require(
        !path.contains(['\\', '\0']) && path.split('/').all(|p| !matches!(p, "" | "." | "..")),
        "unsafe_dependency_path",
    )?;
    let directory_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut directory =
        open(root, directory_flags, Mode::empty()).map_err(|_| error("dependency_io_failed"))?;
    let mut pieces = path.split('/').peekable();
    let mut file = loop {
        let part = pieces
            .next()
            .ok_or_else(|| error("unsafe_dependency_path"))?;
        if pieces.peek().is_none() {
            let fd = openat(
                &directory,
                part,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
                Mode::empty(),
            )
            .map_err(|_| error("dependency_io_failed"))?;
            break File::from(fd);
        }
        directory = openat(&directory, part, directory_flags, Mode::empty())
            .map_err(|_| error("dependency_io_failed"))?;
    };
    require(
        file.metadata()
            .map_err(|_| error("dependency_io_failed"))?
            .is_file(),
        "dependency_not_regular_file",
    )?;
    let mut hasher = Sha256::new();
    let mut size = 0_u64;
    let mut data = Vec::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|_| error("dependency_io_failed"))?;
        if n == 0 {
            break;
        }
        size = size
            .checked_add(n as u64)
            .ok_or_else(|| error("dependency_too_large"))?;
        hasher.update(&buffer[..n]);
        if manifest {
            require(size <= MAX_MANIFEST as u64, "manifest_too_large")?;
            data.extend_from_slice(&buffer[..n]);
        }
    }
    Ok((format!("{:x}", hasher.finalize()), size, data))
}
fn visit<'a>(
    key: &'a str,
    nodes: &BTreeMap<&'a str, &'a Value>,
    visited: &mut BTreeSet<&'a str>,
    active: &mut BTreeSet<&'a str>,
) -> Result<()> {
    let node = nodes.get(key).ok_or_else(|| error("dependency_missing"))?;
    require(!active.contains(key), "dependency_cycle")?;
    if visited.contains(key) {
        return Ok(());
    }
    active.insert(key);
    for child in node["requires"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?
    {
        visit(text(child)?, nodes, visited, active)?;
    }
    active.remove(key);
    visited.insert(key);
    Ok(())
}
fn children(node: &Value) -> Result<BTreeSet<&str>> {
    node["requires"]
        .as_array()
        .ok_or_else(|| error("schema_mismatch"))?
        .iter()
        .map(text)
        .collect()
}
