//! Scratch files of `dope_kpi emit-cli`.
//!
//! The fit rows of a cell are real rows. They are written to a file only the
//! current user can read, inside the run directory, and removed as soon as
//! the fit command returns: explicitly on the normal path (so a failed
//! removal stops the run) and by a drop guard on every early return. A
//! process that is killed, or that aborts on a panic as release builds do,
//! cannot clean up; `emit-cli` therefore empties its scratch directory
//! before it starts any command.

use super::jsonl::file_error;
use dope_arena::ArenaResult;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

/// Remove a file or a directory tree; a path that does not exist is fine.
pub fn remove_path(path: &Path) -> ArenaResult<()> {
    let removed = match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(path),
        Ok(_) => std::fs::remove_file(path),
        Err(err) => Err(err),
    };
    match removed {
        Err(err) if err.kind() != ErrorKind::NotFound => Err(file_error(path, &err)),
        _ => Ok(()),
    }
}

/// A path that may hold rows. `remove` reports a failed removal; dropping the
/// guard removes the path on every other exit that unwinds.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn remove(&self) -> ArenaResult<()> {
        remove_path(&self.0)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = remove_path(&self.0);
    }
}

/// Create a new file readable by the current user only.
pub fn write_private(path: &Path, bytes: &[u8]) -> ArenaResult<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path).map_err(|e| file_error(path, &e))?;
    file.write_all(bytes).map_err(|e| file_error(path, &e))
}

/// Length and SHA-256 of the packet bytes followed by the card bytes, when a card exists.
pub fn measure_packet(packet: &Path, card: &Path) -> ArenaResult<(u64, String)> {
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut block = vec![0u8; 1 << 16];
    for path in [packet, card] {
        if path == card && !card.is_file() {
            continue;
        }
        let mut file = File::open(path).map_err(|e| file_error(path, &e))?;
        loop {
            let got = file.read(&mut block).map_err(|e| file_error(path, &e))?;
            if got == 0 {
                break;
            }
            hasher.update(&block[..got]);
            total += got as u64;
        }
    }
    Ok((total, format!("{:x}", hasher.finalize())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kpi::jsonl::sha256_hex;

    #[test]
    fn scratch_files_are_private_and_always_removed() {
        let dir = crate::kpi::unit_test_dir("ext-cell-scratch");
        let path = dir.join("fit_rows.csv");
        {
            let scratch = Scratch(path.clone());
            write_private(&scratch.0, b"c0,target\n1,2\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = std::fs::metadata(&path).unwrap().permissions().mode();
                assert_eq!(mode & 0o777, 0o600);
            }
            // A second writer cannot replace the file behind the first one.
            assert!(write_private(&path, b"x").is_err());
            scratch.remove().unwrap();
            assert!(!path.exists());
            scratch.remove().unwrap();
            write_private(&path, b"again").unwrap();
        }
        assert!(!path.exists(), "dropping the guard removes the file");
        let unwound = std::panic::catch_unwind(|| {
            let scratch = Scratch(path.clone());
            write_private(&scratch.0, b"rows").unwrap();
            panic!("generator handling failed");
        });
        assert!(unwound.is_err() && !path.exists());

        std::fs::create_dir_all(dir.join("tree/inner")).unwrap();
        std::fs::write(dir.join("tree/inner/left.tmp"), b"x").unwrap();
        remove_path(&dir.join("tree")).unwrap();
        remove_path(&dir.join("tree")).unwrap();
        assert!(!dir.join("tree").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn packet_measure_covers_packet_then_card() {
        let dir = crate::kpi::unit_test_dir("ext-cell-packet");
        let (packet, card) = (dir.join("x.packet"), dir.join("x.card"));
        std::fs::write(&packet, b"packet-bytes").unwrap();
        assert_eq!(
            measure_packet(&packet, &card).unwrap(),
            (12, sha256_hex(b"packet-bytes"))
        );
        std::fs::write(&card, b"|card").unwrap();
        assert_eq!(
            measure_packet(&packet, &card).unwrap(),
            (17, sha256_hex(b"packet-bytes|card"))
        );
        std::fs::write(&packet, b"").unwrap();
        assert_eq!(
            measure_packet(&packet, &card).unwrap(),
            (5, sha256_hex(b"|card"))
        );
        std::fs::remove_file(&packet).unwrap();
        assert!(measure_packet(&packet, &card).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
