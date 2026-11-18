#[derive(Clone, Debug)]
pub struct CampaignLedger {
    path: PathBuf,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Mitigate CVE-2026-119375 (RUSTSEC-2026-0459) before SQLite opens a database with a
/// sibling `-wal` file whose header page size disagrees with the main database header.
fn verify_wal_sibling(db_path: &Path) -> Result<()> {
    let mut wal_os = db_path.as_os_str().to_os_string();
    wal_os.push("-wal");
    let wal_path = PathBuf::from(wal_os);
    let wal_meta = match fs::symlink_metadata(&wal_path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_error(&wal_path, e)),
    };
    if !wal_meta.is_file() || wal_meta.file_type().is_symlink() {
        return Err(DopeError::Codec("campaign ledger WAL path must be a regular file".into()));
    }
    if wal_meta.len() == 0 {
        return Ok(());
    }
    if wal_meta.len() < 32 {
        return Err(DopeError::Codec("campaign ledger WAL header is truncated".into()));
    }
    let wal_hdr = {
        use std::io::Read;
        let mut f = fs::File::open(&wal_path).map_err(|e| io_error(&wal_path, e))?;
        let mut buf = [0u8; 32];
        f.read_exact(&mut buf).map_err(|e| io_error(&wal_path, e))?;
        buf
    };
    let magic = u32::from_be_bytes([wal_hdr[0], wal_hdr[1], wal_hdr[2], wal_hdr[3]]);
    if magic != 0x377f_0682 && magic != 0x377f_0683 {
        return Err(DopeError::Codec("campaign ledger WAL header magic mismatch".into()));
    }
    let wal_ps = u32::from_be_bytes([wal_hdr[8], wal_hdr[9], wal_hdr[10], wal_hdr[11]]);
    let db_meta = fs::symlink_metadata(db_path).map_err(|_| {
        DopeError::Codec("campaign ledger missing main file beside non-empty WAL".into())
    })?;
    if !db_meta.is_file() || db_meta.file_type().is_symlink() || db_meta.len() < 100 {
        return Err(DopeError::Codec("campaign ledger header missing beside non-empty WAL".into()));
    }
    let db_hdr = {
        use std::io::Read;
        let mut f = fs::File::open(db_path).map_err(|e| io_error(db_path, e))?;
        let mut buf = [0u8; 100];
        f.read_exact(&mut buf).map_err(|e| io_error(db_path, e))?;
        buf
    };
    if &db_hdr[..16] != b"SQLite format 3\0" {
        return Err(DopeError::Codec("campaign ledger SQLite header magic mismatch".into()));
    }
    let raw_ps = u16::from_be_bytes([db_hdr[16], db_hdr[17]]);
    let db_ps: u32 = if raw_ps == 1 { 65_536 } else { u32::from(raw_ps) };
    if !db_ps.is_power_of_two() || !(512..=65_536).contains(&db_ps) || wal_ps != db_ps {
        return Err(DopeError::Codec("campaign ledger WAL page size mismatch".into()));
    }
    Ok(())
}
