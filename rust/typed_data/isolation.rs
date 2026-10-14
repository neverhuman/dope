//! Dedicated fresh sampling process restriction. Load and verify the declared
//! artifact/runtime first; this irreversible boundary cannot be installed in
//! a compiler, test supervisor, or long-lived service.

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
    let instruction = |code, jt, jf, k| libc::sock_filter { code, jt, jf, k };
    let mut filter = vec![
        instruction(0x20, 0, 0, 4),
        instruction(0x15, 1, 0, 0xc000003e),
        instruction(0x06, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
        instruction(0x20, 0, 0, 0),
        instruction(0x35, 0, 1, 0x40000000),
        instruction(0x06, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
    ];
    // Memory, fixed-pool synchronization, signals, time and result I/O only.
    // New filesystem access, network, exec, process-memory/FD borrowing,
    // mount changes, io_uring and privileged alternatives are denied by default.
    for syscall in [
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
    ] {
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

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub fn seal_sampling_process() -> Result<()> {
    Err(DopeError::Unsupported(
        "sampling_isolation_requires_linux_x86_64".into(),
    ))
}
