use super::io::{CAPTURE_LIMIT, Captures, Request, Result};
use super::tree::{self, Identity};
use serde_json::json;
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
extern "C" fn interrupted(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

struct Owned {
    child: std::process::Child,
    leader: Identity,
    peak: u64,
    status: Option<i32>,
    finished: bool,
    group_signal_attempted: bool,
    cleanup_deadline: Option<Instant>,
    observer_fault: bool,
    found_descendant: bool,
}
fn wait_usage(pid: libc::pid_t) -> Result<Option<(libc::pid_t, i32, u64)>> {
    let mut status = 0;
    // SAFETY: rusage is a plain C output struct, initialized before passing a valid pointer.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: only this isolated supervisor's child is selected; pointers live for the call.
    let got = unsafe { libc::wait4(pid, &mut status, libc::WNOHANG, &mut usage) };
    if got == 0 {
        return Ok(None);
    }
    if got < 0 {
        return match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::EINTR) => Ok(None),
            Some(libc::ECHILD) => Ok(Some((0, 0, 0))),
            _ => Err("native_process_unavailable"),
        };
    }
    // This module is compiled ONLY on Linux, where ru_maxrss is KiB (not macOS bytes).
    let rss = u64::try_from(usage.ru_maxrss)
        .ok()
        .and_then(|n| n.checked_mul(1024))
        .ok_or("native_process_unavailable")?;
    Ok(Some((got, status, rss)))
}
impl Owned {
    fn observe(&self) -> Result<(Vec<Identity>, u64)> {
        if self.observer_fault {
            return Err("native_process_unavailable");
        }
        tree::snapshot(std::process::id() as libc::pid_t)
    }
    fn cleanup(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        // Retain the actual Child handle in the resource owner; custom wait4 below
        // performs reaping, and its anchored id must remain this exact leader.
        if self.child.id() as libc::pid_t != self.leader.pid {
            return Err("native_process_unavailable");
        }
        let mut failure = None;
        // Signal the anchored leader group only BEFORE any wait4 can release its PID.
        // Even a failed attempt cannot be repeated after reaping, because PGID may reuse.
        if !self.group_signal_attempted {
            if let Err(error) = tree::kill_group(self.leader) {
                failure = Some(error);
            }
            self.group_signal_attempted = true;
        }
        let deadline = *self
            .cleanup_deadline
            .get_or_insert_with(|| Instant::now() + Duration::from_secs(5));
        loop {
            match self.observe() {
                Ok((members, rss)) => {
                    self.peak = self.peak.max(rss);
                    for member in members {
                        if member.pid != self.leader.pid {
                            self.found_descendant = true;
                        }
                        if let Err(error) = tree::kill(member) {
                            failure = Some(error);
                        }
                    }
                }
                Err(error) => failure = Some(error),
            }
            // Reaping is independent of procfs observation: a persistent observer
            // failure cannot skip wait4 of the known killed leader/adopted children.
            loop {
                match wait_usage(-1) {
                    Ok(Some((0, _, _))) => {
                        self.finished = true;
                        return failure.map_or(Ok(()), Err);
                    }
                    Ok(Some((pid, status, peak))) => {
                        self.peak = self.peak.max(peak);
                        if pid == self.leader.pid {
                            self.status = Some(status);
                        } else {
                            self.found_descendant = true;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            if Instant::now() >= deadline {
                return Err("native_process_unavailable");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.cleanup();
        }
    }
}
fn leader_exited(pid: libc::pid_t) -> Result<bool> {
    // SAFETY: zeroed siginfo_t is an output buffer; waitid WNOWAIT retains PID ownership.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: pid is the unreaped child; info is valid and no other code waits for it.
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if rc < 0 {
        return if std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
            Ok(false)
        } else {
            Err("native_process_unavailable")
        };
    }
    // SAFETY: successful waitid initializes the Linux siginfo child-status union.
    Ok(unsafe { info.si_pid() } == pid)
}
fn limits(r: &Request, directory: i32) -> Result<impl FnMut() -> std::io::Result<()> + use<>> {
    // SAFETY: cpu_set_t is initialized and output size matches the passed buffer.
    let mut allowed: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: valid output buffer and the current process's affinity mask are requested.
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&allowed), &mut allowed) } != 0 {
        return Err("native_process_unavailable");
    }
    let cpu = (0..libc::CPU_SETSIZE as usize)
        // SAFETY: index is within CPU_SETSIZE and allowed is an initialized cpu_set_t.
        .find(|cpu| unsafe { libc::CPU_ISSET(*cpu, &allowed) })
        .ok_or("native_process_unavailable")?;
    let ram = r.ram_bytes as libc::rlim_t;
    let seconds = (r.wall_ms.div_ceil(1000) + 1).max(1) as libc::rlim_t;
    let owner = std::process::id() as libc::pid_t;
    Ok(move || {
        // SAFETY: pre_exec runs in the forked child; these bounded syscalls touch only
        // this child's session, parent-death behavior, cwd, affinity, priority and limits.
        unsafe {
            if libc::setsid() < 0
                || libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                || libc::getppid() != owner
                || libc::fchdir(directory) != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            let mut selected: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut selected);
            libc::CPU_SET(cpu, &mut selected);
            if libc::sched_setaffinity(0, std::mem::size_of_val(&selected), &selected) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            *libc::__errno_location() = 0;
            let nice = libc::getpriority(libc::PRIO_PROCESS, 0);
            if nice == -1 && *libc::__errno_location() != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if nice < 10 && libc::setpriority(libc::PRIO_PROCESS, 0, 10) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            for (kind, value) in [
                (libc::RLIMIT_AS, ram),
                (libc::RLIMIT_CPU, seconds),
                (libc::RLIMIT_FSIZE, CAPTURE_LIMIT as libc::rlim_t),
            ] {
                let bound = libc::rlimit {
                    rlim_cur: value,
                    rlim_max: value,
                };
                if libc::setrlimit(kind, &bound) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            // CLOEXEC preserves Rust's exec-error pipe until exec while closing every
            // nonstdio inherited fd at exec. Unsupported kernels refuse; no fallback.
            if libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 4u32) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    })
}
fn invoke_inner(
    r: Request,
    monitor_fault: bool,
    observer_fault: bool,
) -> Result<serde_json::Value> {
    let owner = std::process::id() as libc::pid_t;
    if !tree::snapshot(owner)?.0.is_empty() {
        return Err("native_process_unavailable");
    }
    // SAFETY: this dedicated single-command process adopts only its own orphaned
    // descendants; no preexisting child is present and this is never a shared library API.
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) } != 0 {
        return Err("native_process_unavailable");
    }
    INTERRUPTED.store(false, Ordering::Relaxed);
    for signal in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        // SAFETY: the handler only stores a lock-free atomic flag; no allocation or IO.
        if unsafe { libc::signal(signal, interrupted as *const () as libc::sighandler_t) }
            == libc::SIG_ERR
        {
            return Err("native_process_unavailable");
        }
    }
    let mut captures = Captures::create(&r)?;
    let setup = limits(&r, captures.directory.as_raw_fd())?;
    let start = Instant::now();
    let mut command = Command::new(&r.argv[0]);
    command
        .args(&r.argv[1..])
        .env_clear()
        .env("PATH", "/bin:/usr/bin")
        .env("LC_ALL", "C")
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::null())
        .stdout(
            captures
                .out
                .try_clone()
                .map_err(|_| "native_process_unavailable")?,
        )
        .stderr(
            captures
                .err
                .try_clone()
                .map_err(|_| "native_process_unavailable")?,
        );
    // SAFETY: setup contains only child-local syscall operations, with copied scalar data.
    unsafe {
        command.pre_exec(setup);
    }
    let child = command.spawn().map_err(|_| "native_process_unavailable")?;
    // This direct leader is unreaped and therefore cannot undergo PID reuse.
    let leader = Identity {
        pid: child.id() as libc::pid_t,
        start: 0,
    };
    let mut owned = Owned {
        child,
        leader,
        peak: 0,
        status: None,
        finished: false,
        group_signal_attempted: false,
        cleanup_deadline: None,
        observer_fault,
        found_descendant: false,
    };
    #[cfg(test)]
    LAST_SPAWNED_PID.store(leader.pid, Ordering::Relaxed);
    let mut outcome;
    loop {
        let sampled_at = Instant::now();
        let (members, rss) = owned.observe()?;
        owned.peak = owned.peak.max(rss);
        if monitor_fault {
            return Err("native_process_unavailable");
        }
        if owned.peak > r.ram_bytes {
            outcome = "over_budget";
            break;
        }
        if INTERRUPTED.load(Ordering::Relaxed) {
            outcome = "failed";
            break;
        }
        if start.elapsed().as_millis() >= u128::from(r.wall_ms) {
            outcome = "timeout";
            break;
        }
        if leader_exited(leader.pid)? {
            outcome = if members.iter().any(|id| id.pid != leader.pid) {
                "failed"
            } else {
                "completed"
            };
            break;
        }
        let remain = Duration::from_millis(r.wall_ms).saturating_sub(start.elapsed());
        std::thread::sleep(
            remain.min(Duration::from_millis(50).saturating_sub(sampled_at.elapsed())),
        );
    }
    owned.cleanup()?;
    let status = owned.status.ok_or("native_process_unavailable")?;
    let returncode = if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else if libc::WIFSIGNALED(status) {
        -libc::WTERMSIG(status)
    } else {
        return Err("native_process_unavailable");
    };
    if outcome == "completed" && (returncode != 0 || owned.found_descendant) {
        outcome = "failed";
    }
    let (stdout_sha256, stderr_sha256) = captures.hashes()?;
    let elapsed = u64::try_from(start.elapsed().as_nanos().div_ceil(1_000_000))
        .map_err(|_| "native_process_unavailable")?;
    if outcome == "completed" && (elapsed > r.wall_ms || owned.peak > r.ram_bytes) {
        outcome = "over_budget";
    }
    Ok(
        json!({"status":outcome,"wall_ms":elapsed,"peak_ram_bytes":owned.peak,
        "returncode":returncode,"rss_measurement":"per_command_polled_owned_tree_and_wait4_child_rusage",
        "stdout_sha256":stdout_sha256,"stderr_sha256":stderr_sha256}),
    )
}
pub(super) fn invoke(r: Request) -> Result<serde_json::Value> {
    invoke_inner(r, false, false)
}
#[cfg(test)]
pub(super) fn invoke_with_monitor_fault(r: Request) -> Result<serde_json::Value> {
    invoke_inner(r, true, false)
}

#[cfg(test)]
static LAST_SPAWNED_PID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
#[cfg(test)]
pub(super) fn last_spawned_pid() -> i32 {
    LAST_SPAWNED_PID.load(Ordering::Relaxed)
}
#[cfg(test)]
pub(super) fn invoke_with_persistent_observer_failure(r: Request) -> Result<serde_json::Value> {
    invoke_inner(r, false, true)
}
