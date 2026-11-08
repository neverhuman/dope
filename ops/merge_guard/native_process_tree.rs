//! Linux ownership is anchored to an unreaped leader and this isolated subreaper.
use super::io::Result;
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::os::fd::FromRawFd;

#[derive(Clone, Copy)]
pub(super) struct Identity {
    pub pid: libc::pid_t,
    pub start: u64,
}
struct Stat {
    parent: libc::pid_t,
    start: u64,
    rss: u64,
}
fn stat(pid: libc::pid_t) -> Result<Option<Stat>> {
    let path = format!("/proc/{pid}/stat");
    let text = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("native_process_unavailable"),
    };
    // comm may contain spaces and parentheses; fields after its FINAL ')' are fixed.
    let (_, fields) = text.rsplit_once(')').ok_or("native_process_unavailable")?;
    let parts: Vec<&str> = fields.split_whitespace().collect();
    let parent = parts
        .get(1)
        .and_then(|x| x.parse().ok())
        .ok_or("native_process_unavailable")?;
    let start = parts
        .get(19)
        .and_then(|x| x.parse().ok())
        .ok_or("native_process_unavailable")?;
    let pages: i64 = parts
        .get(21)
        .and_then(|x| x.parse().ok())
        .ok_or("native_process_unavailable")?;
    // SAFETY: sysconf reads a process-independent page-size constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 || pages < 0 {
        return Err("native_process_unavailable");
    }
    let rss = (pages as u64)
        .checked_mul(page as u64)
        .ok_or("native_process_unavailable")?;
    Ok(Some(Stat { parent, start, rss }))
}
fn direct_children(pid: libc::pid_t) -> Result<Vec<libc::pid_t>> {
    // Read every thread's children: a descendant may fork from a nonleader thread.
    let tasks = match fs::read_dir(format!("/proc/{pid}/task")) {
        Ok(x) => x,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(_) => return Err("native_process_unavailable"),
    };
    let mut out = Vec::new();
    for (index, entry) in tasks.enumerate() {
        if index >= 65536 {
            return Err("native_process_unavailable");
        }
        let entry = entry.map_err(|_| "native_process_unavailable")?;
        let read = || -> std::io::Result<String> {
            let mut text = String::new();
            fs::File::open(entry.path().join("children"))?
                .take((1 << 20) + 1)
                .read_to_string(&mut text)?;
            if text.len() > 1 << 20 {
                return Err(std::io::Error::other("proc_children_limit"));
            }
            Ok(text)
        };
        let text = match read() {
            Ok(x) => x,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => return Err("native_process_unavailable"),
        };
        for n in text.split_whitespace() {
            out.push(n.parse().map_err(|_| "native_process_unavailable")?);
            if out.len() > 65536 {
                return Err("native_process_unavailable");
            }
        }
    }
    Ok(out)
}
pub(super) fn snapshot(owner: libc::pid_t) -> Result<(Vec<Identity>, u64)> {
    let mut pending: Vec<_> = direct_children(owner)?
        .into_iter()
        .map(|pid| (pid, owner))
        .collect();
    let mut seen = BTreeMap::new();
    let mut total = 0u64;
    while let Some((pid, parent)) = pending.pop() {
        if seen.contains_key(&pid) {
            continue;
        }
        if seen.len() >= 65536 {
            return Err("native_process_unavailable");
        }
        if let Some(s) = stat(pid)? {
            if s.parent != parent {
                continue;
            } // Reject stale/reused child-list entries.
            total = total
                .checked_add(s.rss)
                .ok_or("native_process_unavailable")?;
            seen.insert(
                pid,
                Identity {
                    pid,
                    start: s.start,
                },
            );
            pending.extend(direct_children(pid)?.into_iter().map(|child| (child, pid)));
            if pending.len() > 65536 {
                return Err("native_process_unavailable");
            }
        }
    }
    Ok((seen.into_values().collect(), total))
}
pub(super) fn kill(id: Identity) -> Result<()> {
    // SAFETY: pidfd_open creates a descriptor for this currently named process;
    // start-time validation below ties it to the sampled descendant before signaling.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, id.pid, 0u32) };
    if fd < 0 {
        return if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err("native_process_unavailable")
        };
    }
    // SAFETY: successful pidfd_open returns a new owned fd, closed exactly once by File.
    let handle = unsafe { std::fs::File::from_raw_fd(fd as i32) };
    if stat(id.pid)?.is_some_and(|s| s.start == id.start) {
        use std::os::fd::AsRawFd;
        // SAFETY: pidfd pins the process identity atomically even if the numeric PID
        // subsequently exits/reuses; a null siginfo is permitted for this signal.
        if unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                handle.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0u32,
            )
        } != 0
            && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
        {
            return Err("native_process_unavailable");
        }
    }
    Ok(())
}
pub(super) fn kill_group(leader: Identity) -> Result<()> {
    // Caller guarantees this exact directly spawned leader is NOT yet reaped, so
    // its PID/PGID cannot be reused. Session setup ran successfully before spawn returned.
    // SAFETY: reading the unreaped child's PGID cannot mutate any process.
    let pgid = unsafe { libc::getpgid(leader.pid) };
    if pgid != leader.pid {
        return Err("native_process_unavailable");
    }
    // SAFETY: the anchored group belongs exclusively to this command; no other
    // session can join it. This call is made once, before any wait4 can reap the leader.
    if unsafe { libc::killpg(leader.pid, libc::SIGKILL) } != 0
        && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
    {
        return Err("native_process_unavailable");
    }
    Ok(())
}
