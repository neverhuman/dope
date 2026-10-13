//! Source-free sampling bundle envelope.
//!
//! The manifest names every byte a sampler must account for. Historical DPK
//! payloads stay intact: this module bounds expansion, then the existing
//! decoder and sampler run on the retained bytes. Verification is identity and
//! closure, not utility, privacy, or release authority.

use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fs;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use std::ffi::{CString, c_char};

const JSON_INT_MAX: u64 = (1 << 53) - 1;
const HEADER_BYTES: usize = 56;
const NEURAL_CAP: usize = 16 * 1024 * 1024;
const MANIFEST_NAME: &str = "manifest.json";
const ELOOP: i32 = 40;
const EINTR: i32 = 4;
const O_NOFOLLOW: i32 = libc::O_NOFOLLOW;
const O_DIRECTORY: i32 = libc::O_DIRECTORY;
const O_CLOEXEC: i32 = libc::O_CLOEXEC;
const O_NONBLOCK: i32 = libc::O_NONBLOCK;
const AT_SYMLINK_NOFOLLOW: i32 = libc::AT_SYMLINK_NOFOLLOW;
const STATX_TYPE: u32 = 1;

// Linux x86_64 values. Other architectures take the libc constants above.
#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(libc::O_NOFOLLOW == 0x20000);
    assert!(libc::O_DIRECTORY == 0x10000);
    assert!(libc::O_CLOEXEC == 0x80000);
    assert!(libc::O_NONBLOCK == 0x800);
    assert!(libc::AT_SYMLINK_NOFOLLOW == 0x100);
};

// SAFETY: these are the Linux libc entry points. Callers pass NUL-terminated
// paths and directory descriptors they own, and they check the returned fd.
unsafe extern "C" {
    fn open(path: *const c_char, flags: i32, mode: u32) -> i32;
    fn openat(dirfd: i32, path: *const c_char, flags: i32, mode: u32) -> i32;
    fn getdents64(fd: i32, dirp: *mut u8, count: usize) -> isize;
    fn statx(dirfd: i32, pathname: *const c_char, flags: i32, mask: u32, statxbuf: *mut u8) -> i32;
    fn __errno_location() -> *mut i32;
}

include!("bundle/part_01.rs");
include!("bundle/part_02.rs");
include!("bundle/part_03.rs");
include!("bundle/part_04.rs");
include!("bundle/part_05.rs");
