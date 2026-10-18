//! Dedicated fresh-worker restrictions. Pin the shared runtime first, confine
//! bundle reload to its directory, then seal sampling against all file access.
//! These irreversible boundaries cannot be installed in a compiler, test
//! supervisor, or long-lived service.

use crate::error::{DopeError, Result};
use sha2::Digest;

#[derive(serde::Serialize)]
pub struct RuntimeFile {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
}

/// Actual file-backed runtime mappings, counted once per file. This is a cold
/// file closure, not resident memory, a portable OS image, or learned state.
#[cfg(target_os = "linux")]
pub fn runtime_inventory() -> Result<Vec<RuntimeFile>> {
    let maps = std::fs::read_to_string("/proc/self/maps")
        .map_err(|_| DopeError::Unsupported("sampling_runtime_inventory_unavailable".into()))?;
    let mut paths = std::collections::BTreeSet::new();
    for line in maps.lines() {
        if let Some(at) = line.find('/') {
            let path = &line[at..];
            if path.ends_with(" (deleted)") {
                return Err(DopeError::Unsupported(
                    "sampling_runtime_mapping_unpinned".into(),
                ));
            }
            paths.insert(std::path::PathBuf::from(path));
        }
    }
    let mut files = Vec::new();
    for path in paths {
        if std::fs::metadata(&path)
            .map_err(|_| DopeError::Unsupported("sampling_runtime_mapping_unreadable".into()))?
            .len()
            > 512 * 1024 * 1024
        {
            return Err(DopeError::Unsupported(
                "sampling_runtime_mapping_too_large".into(),
            ));
        }
        let bytes = std::fs::read(&path)
            .map_err(|_| DopeError::Unsupported("sampling_runtime_mapping_unreadable".into()))?;
        files.push(RuntimeFile {
            name: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("runtime")
                .into(),
            bytes: bytes.len(),
            sha256: format!("{:x}", sha2::Sha256::digest(&bytes)),
        });
    }
    Ok(files)
}

#[cfg(not(target_os = "linux"))]
pub fn runtime_inventory() -> Result<Vec<RuntimeFile>> {
    Err(DopeError::Unsupported(
        "sampling_runtime_inventory_requires_linux".into(),
    ))
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub fn seal_sampling_process() -> Result<()> {
    // Initialize the fixed CPU pool before denying thread/process creation.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .map_err(|_| DopeError::Unsupported("sampling_pool_not_fresh".into()))?;
    close_source_descriptors()?;
    install_filter(false)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn close_source_descriptors() -> Result<()> {
    // Only write-only result streams survive. In particular stdin and all
    // inherited source FDs are closed, including aliases not marked CLOEXEC.
    for fd in [1, 2] {
        // SAFETY: F_GETFL reads flags without dereferencing userspace pointers.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 || flags & libc::O_ACCMODE != libc::O_WRONLY {
            return Err(DopeError::Unsupported(
                "sampling_result_stream_must_be_write_only".into(),
            ));
        }
    }
    // SAFETY: close_range takes only scalar arguments and cannot dereference
    // Rust memory. The fresh worker has already read and closed its artifact;
    // the range starts at 3, preserving the checked output descriptors 1 and 2.
    if unsafe { libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 0u32) } != 0 {
        return Err(DopeError::Unsupported(
            "sampling_fd_closure_unavailable".into(),
        ));
    }
    // SAFETY: close takes a scalar descriptor and cannot dereference Rust
    // memory. This dedicated worker never reads stdin; an already closed fd 0
    // returns EBADF, which does not affect the sealed sampling boundary.
    unsafe {
        libc::close(0);
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn install_filter(loading: bool) -> Result<()> {
    let instruction = |code, jt, jf, k| libc::sock_filter { code, jt, jf, k };
    let mut filter = vec![
        instruction(0x20, 0, 0, 4),
        instruction(0x15, 1, 0, 0xc000003e),
        instruction(0x06, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
        instruction(0x20, 0, 0, 0),
        instruction(0x35, 0, 1, 0x40000000),
        instruction(0x06, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
    ];
    if loading {
        // Landlock deliberately permits O_PATH handles. The loader needs only
        // ordinary read-only opens, so reject metadata and writable handles.
        for (syscall, offset) in [(libc::SYS_open, 24), (libc::SYS_openat, 32)] {
            filter.extend([
                instruction(0x15, 0, 3, syscall as u32),
                instruction(0x20, 0, 0, offset),
                instruction(0x45, 0, 1, (libc::O_PATH | libc::O_ACCMODE) as u32),
                instruction(0x06, 0, 0, libc::SECCOMP_RET_ERRNO | libc::EPERM as u32),
                instruction(0x20, 0, 0, 0),
            ]);
        }
    }
    // Memory, fixed-pool synchronization, signals, time and result I/O only.
    // New filesystem access, network, exec, process-memory/FD borrowing,
    // mount changes, io_uring and privileged alternatives are denied by default.
    let mut syscalls = vec![
        libc::SYS_read,
        libc::SYS_write,
        libc::SYS_writev,
        libc::SYS_close,
        libc::SYS_fstat,
        libc::SYS_lseek,
        libc::SYS_mmap,
        libc::SYS_mprotect,
        libc::SYS_munmap,
        libc::SYS_brk,
        libc::SYS_madvise,
        libc::SYS_futex,
        libc::SYS_sched_yield,
        libc::SYS_getpid,
        libc::SYS_gettid,
        libc::SYS_tgkill,
        libc::SYS_rt_sigaction,
        libc::SYS_rt_sigprocmask,
        libc::SYS_rt_sigreturn,
        libc::SYS_sigaltstack,
        libc::SYS_clock_gettime,
        libc::SYS_nanosleep,
        libc::SYS_clock_nanosleep,
        libc::SYS_restart_syscall,
        libc::SYS_exit,
        libc::SYS_exit_group,
    ];
    if loading {
        // Landlock confines these reads to the pinned bundle. New threads
        // inherit both boundaries, allowing the fixed sampling pool to start.
        syscalls.extend([
            libc::SYS_open,
            libc::SYS_openat,
            libc::SYS_newfstatat,
            libc::SYS_statx,
            libc::SYS_getdents64,
            libc::SYS_fcntl,
            libc::SYS_dup,
            libc::SYS_dup2,
            libc::SYS_dup3,
            libc::SYS_getrandom,
            libc::SYS_sched_getaffinity,
            libc::SYS_clone,
            libc::SYS_clone3,
            libc::SYS_rseq,
            libc::SYS_set_robust_list,
            libc::SYS_prlimit64,
            libc::SYS_close_range,
            libc::SYS_seccomp,
            libc::SYS_prctl,
        ]);
    }
    for syscall in syscalls {
        filter.push(instruction(0x15, 0, 1, syscall as u32));
        filter.push(instruction(0x06, 0, 0, libc::SECCOMP_RET_ALLOW));
    }
    filter.push(instruction(
        0x06,
        0,
        0,
        libc::SECCOMP_RET_ERRNO | libc::EPERM as u32,
    ));
    let program = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_mut_ptr(),
    };
    // SAFETY: the scalar prctl arguments set only NO_NEW_PRIVS. program points
    // to filter's contiguous initialized sock_filter entries; its u16 length
    // matches this bounded vector. Neither allocation moves or mutates during
    // the synchronous seccomp call, which copies the program before returning.
    // TSYNC applies the same restrictions to the already initialized CPU pool.
    // SAFETY: filter and program stay live and unchanged until the kernel copies them.
    let status = unsafe {
        if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
            -1
        } else {
            libc::syscall(
                libc::SYS_seccomp,
                libc::SECCOMP_SET_MODE_FILTER,
                libc::SECCOMP_FILTER_FLAG_TSYNC,
                &program,
            ) as i32
        }
    };
    if status != 0 {
        return Err(DopeError::Unsupported(
            "sampling_syscall_boundary_unavailable".into(),
        ));
    }
    Ok(())
}

/// Irreversible reload boundary for a dedicated, single-threaded worker.
/// Only the pinned directory may be read; no sockets, exec, or inherited
/// source descriptors survive. Call `seal_sampling_process` after reload.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub fn confine_bundle_directory(root: &std::path::Path) -> Result<()> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Component;
    let fail = || DopeError::Unsupported("bundle_directory_isolation_unavailable".into());
    if std::fs::read_dir("/proc/self/task")
        .map_err(|_| fail())?
        .count()
        != 1
    {
        return Err(DopeError::Unsupported("sampling_worker_not_fresh".into()));
    }
    if root
        .components()
        .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(DopeError::Unsupported("bundle_directory_unsafe".into()));
    }
    close_source_descriptors()?;
    // Walk every ancestor without following symlinks, including the root.
    let start = if root.is_absolute() { c"/" } else { c"." };
    // SAFETY: constant C string and scalar flags; returned fd is owned below.
    let fd = unsafe {
        libc::open(
            start.as_ptr(),
            libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(fail());
    }
    // SAFETY: fd was just created and has no other owner.
    let mut directory = unsafe { OwnedFd::from_raw_fd(fd) };
    for part in root.components() {
        if let Component::Normal(name) = part {
            let name = std::ffi::CString::new(name.as_bytes()).map_err(|_| fail())?;
            // SAFETY: owned directory and live C string; no symlinks followed.
            let fd = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_PATH | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            if fd < 0 {
                return Err(fail());
            }
            // SAFETY: fd was just created and has no other owner.
            directory = unsafe { OwnedFd::from_raw_fd(fd) };
        }
    }
    #[repr(C)]
    struct Ruleset {
        handled_access_fs: u64,
    }
    #[repr(C, packed)]
    struct PathRule {
        allowed_access: u64,
        parent_fd: i32,
    }
    // ABI 3 covers every filesystem right through TRUNCATE. Later rights
    // cannot grant source reads; the syscall filter denies their alternatives.
    let rules = Ruleset {
        handled_access_fs: (1 << 15) - 1,
    };
    let path = PathRule {
        allowed_access: (1 << 2) | (1 << 3),
        parent_fd: directory.as_raw_fd(),
    };
    // SAFETY: the kernel copies the bounded repr(C) structures synchronously.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            &rules,
            std::mem::size_of::<Ruleset>(),
            0,
        )
    };
    if fd < 0 {
        return Err(fail());
    }
    // SAFETY: fd was just created and has no other owner.
    let ruleset = unsafe { OwnedFd::from_raw_fd(fd as i32) };
    // SAFETY: owned FDs and live rule; NO_NEW_PRIVS only tightens privileges.
    let status = unsafe {
        libc::syscall(
            libc::SYS_landlock_add_rule,
            ruleset.as_raw_fd(),
            1,
            &path,
            0,
        ) != 0
            || libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0
            || libc::syscall(libc::SYS_landlock_restrict_self, ruleset.as_raw_fd(), 0) != 0
    };
    if status {
        return Err(fail());
    }
    drop(ruleset);
    drop(directory);
    // SAFETY: the dedicated worker has exactly one thread, checked above.
    for (key, _) in std::env::vars_os() {
        unsafe {
            std::env::remove_var(key);
        }
    }
    install_filter(true)
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub fn confine_bundle_directory(_: &std::path::Path) -> Result<()> {
    Err(DopeError::Unsupported(
        "sampling_isolation_requires_linux_x86_64".into(),
    ))
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub fn seal_sampling_process() -> Result<()> {
    Err(DopeError::Unsupported(
        "sampling_isolation_requires_linux_x86_64".into(),
    ))
}
