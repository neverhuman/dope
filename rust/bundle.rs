//! Source-free sampling bundle envelope.
//!
//! The manifest names every byte a sampler must account for. Historical DPK
//! payloads stay intact: this module bounds expansion, then the existing
//! decoder and sampler run on the retained bytes. Verification is identity and
//! closure, not utility, privacy, or release authority.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
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
const O_NOFOLLOW: i32 = 0x20000;
const O_DIRECTORY: i32 = 0x10000;
const O_CLOEXEC: i32 = 0x80000;
const O_NONBLOCK: i32 = 0x800;

// SAFETY: these are the Linux libc entry points. Callers pass NUL-terminated
// paths and directory descriptors they own, and they check the returned fd.
unsafe extern "C" {
    fn open(path: *const c_char, flags: i32, mode: u32) -> i32;
    fn openat(dirfd: i32, path: *const c_char, flags: i32, mode: u32) -> i32;
    fn __errno_location() -> *mut i32;
}

include!("bundle/part_01.rs");
include!("bundle/part_02.rs");
include!("bundle/part_03.rs");
include!("bundle/part_04.rs");
include!("bundle/part_05.rs");
