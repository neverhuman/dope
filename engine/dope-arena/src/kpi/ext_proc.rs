//! Child processes of `dope_kpi emit-cli`.
//!
//! A command is executed directly from its argument vector (no shell), with
//! stdin closed, stdout and stderr appended to a log file that is never read
//! back, a fixed environment that caps the child at one CPU thread, and a
//! wall-clock timeout after which the child and its process group are killed.

use super::jsonl::file_error;
use dope_arena::ArenaResult;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// Exit status 0.
    Success,
    /// Non-zero exit status, or killed by a signal.
    Failed,
    /// Still running at the timeout; killed.
    TimedOut,
    /// The program could not be started.
    SpawnFailed,
}

#[derive(Clone, Copy, Debug)]
pub struct Ran {
    pub exit: Exit,
    /// Wall-clock milliseconds from process start to exit (or to the timeout).
    pub wall_ms: f64,
}

/// The whole environment of a child: `PATH` and `HOME` of this process when
/// set, the C locale, and the caps that keep one cell on one CPU thread.
pub fn child_env() -> Vec<(&'static str, OsString)> {
    let mut env = Vec::new();
    for name in ["PATH", "HOME"] {
        if let Some(value) = std::env::var_os(name) {
            env.push((name, value));
        }
    }
    for (name, value) in [
        ("LANG", "C"),
        ("OMP_NUM_THREADS", "1"),
        ("RAYON_NUM_THREADS", "1"),
        ("CUDA_VISIBLE_DEVICES", ""),
    ] {
        env.push((name, OsString::from(value)));
    }
    env
}

fn millis(started: Instant) -> f64 {
    (started.elapsed().as_secs_f64() * 1e6).round() / 1e3
}

/// Poll often at first so short commands are timed to well under a millisecond.
fn poll_interval(elapsed: Duration) -> Duration {
    if elapsed < Duration::from_millis(100) {
        Duration::from_micros(500)
    } else if elapsed < Duration::from_secs(2) {
        Duration::from_millis(2)
    } else {
        Duration::from_millis(10)
    }
}

/// Kill the child and everything it started. The child has not been reaped
/// yet, so its process id (also its process-group id) cannot have been reused.
fn kill_tree(child: &mut Child) {
    // `kill` is a shell builtin, so signalling the group needs only `sh`.
    #[cfg(unix)]
    let _ = Command::new("sh")
        .arg("-c")
        .arg(format!("kill -s KILL -- -{}", child.id()))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

fn wait(child: &mut Child, started: Instant, timeout: Duration) -> (Exit, f64, String) {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let wall_ms = millis(started);
                let exit = if status.success() {
                    Exit::Success
                } else {
                    Exit::Failed
                };
                let note = match status.code() {
                    Some(code) => format!("exit code {code}"),
                    None => "killed by a signal".to_string(),
                };
                return (exit, wall_ms, note);
            }
            Ok(None) => {}
            Err(_) => {
                let wall_ms = millis(started);
                kill_tree(child);
                return (Exit::Failed, wall_ms, "wait failed".to_string());
            }
        }
        let elapsed = started.elapsed();
        if elapsed >= timeout {
            let wall_ms = millis(started);
            kill_tree(child);
            return (Exit::TimedOut, wall_ms, "timeout, killed".to_string());
        }
        std::thread::sleep(poll_interval(elapsed).min(timeout - elapsed));
    }
}

/// Run `argv` in `cwd`, appending its output to `log` between two marker
/// lines that carry `label` and the exit status only. An error is returned
/// only when the log cannot be written; every outcome of the command itself
/// is an `Exit`.
pub fn run_logged(
    argv: &[String],
    cwd: &Path,
    log: &Path,
    label: &str,
    timeout: Duration,
) -> ArenaResult<Ran> {
    let io_error = |e: std::io::Error| file_error(log, &e);
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(io_error)?;
    writeln!(log_file, "--- {label}: start").map_err(io_error)?;
    let stdout = log_file.try_clone().map_err(io_error)?;
    let stderr = log_file.try_clone().map_err(io_error)?;

    let (program, args) = argv.split_first().map_or(("", argv), |(p, a)| (p, a));
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .env_clear()
        .envs(child_env())
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);

    let started = Instant::now();
    let (exit, wall_ms, note) = match command.spawn() {
        Ok(mut child) => wait(&mut child, started, timeout),
        Err(_) => (Exit::SpawnFailed, 0.0, "not started".to_string()),
    };
    writeln!(log_file, "--- {label}: {note}").map_err(io_error)?;
    Ok(Ran { exit, wall_ms })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Vec<String> {
        vec!["sh".to_string(), "-c".to_string(), script.to_string()]
    }

    const LONG: Duration = Duration::from_secs(120);

    #[test]
    fn output_goes_to_the_log_and_the_exit_status_is_classified() {
        let dir = crate::kpi::unit_test_dir("ext-proc-exit");
        let log = dir.join("cell.log");
        let ok = run_logged(
            &sh("echo to-stdout; echo to-stderr >&2"),
            &dir,
            &log,
            "fit",
            LONG,
        );
        let ok = ok.unwrap();
        assert_eq!(ok.exit, Exit::Success);
        assert!(ok.wall_ms > 0.0 && ok.wall_ms < 120_000.0);
        let failed = run_logged(&sh("pwd; exit 3"), &dir, &log, "sample s0 m4", LONG).unwrap();
        assert_eq!(failed.exit, Exit::Failed);
        let signalled = run_logged(&sh("kill -s KILL $$"), &dir, &log, "sample s0 m1", LONG);
        assert_eq!(signalled.unwrap().exit, Exit::Failed);
        let absent = vec!["/nonexistent/kpi/generator".to_string()];
        let unstarted = run_logged(&absent, &dir, &log, "fit", LONG).unwrap();
        assert_eq!(
            (unstarted.exit, unstarted.wall_ms),
            (Exit::SpawnFailed, 0.0)
        );
        assert_eq!(
            run_logged(&[], &dir, &log, "fit", LONG).unwrap().exit,
            Exit::SpawnFailed
        );

        let text = std::fs::read_to_string(&log).unwrap();
        let cwd = std::fs::canonicalize(&dir).unwrap();
        let expected = format!(
            "--- fit: start\nto-stdout\nto-stderr\n--- fit: exit code 0\n\
             --- sample s0 m4: start\n{}\n--- sample s0 m4: exit code 3\n\
             --- sample s0 m1: start\n--- sample s0 m1: killed by a signal\n\
             --- fit: start\n--- fit: not started\n--- fit: start\n--- fit: not started\n",
            cwd.display()
        );
        assert_eq!(text, expected);
        // A log that cannot be opened is an error of the run, not of the command.
        let err = run_logged(
            &sh("true"),
            &dir,
            &dir.join("no/such/dir/x.log"),
            "fit",
            LONG,
        );
        assert!(err.unwrap_err().to_string().contains("x.log"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_environment_is_fixed_and_stdin_is_closed() {
        let dir = crate::kpi::unit_test_dir("ext-proc-env");
        let log = dir.join("env.log");
        // `read` sees end-of-file at once; with an inherited terminal it would block.
        let script = "if read -r line; then exit 7; fi; env";
        let ran = run_logged(&sh(script), &dir, &log, "fit", LONG).unwrap();
        assert_eq!(ran.exit, Exit::Success);
        let text = std::fs::read_to_string(&log).unwrap();
        let names: Vec<&str> = text
            .lines()
            .filter(|line| !line.starts_with("--- "))
            .filter_map(|line| line.split_once('=').map(|(name, _)| name))
            .collect();
        for line in [
            "LANG=C",
            "OMP_NUM_THREADS=1",
            "RAYON_NUM_THREADS=1",
            "CUDA_VISIBLE_DEVICES=",
        ] {
            assert!(text.lines().any(|l| l == line), "missing {line}");
        }
        // Whatever else this process has in its environment must not reach the child.
        let allowed = [
            "PATH",
            "HOME",
            "LANG",
            "OMP_NUM_THREADS",
            "RAYON_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
            // Set by the shell itself, not inherited.
            "PWD",
            "OLDPWD",
            "SHLVL",
            "_",
        ];
        for name in &names {
            assert!(allowed.contains(name), "unexpected variable {name}");
        }
        assert!(names.contains(&"PATH"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_timeout_kills_the_child_and_its_descendants() {
        let dir = crate::kpi::unit_test_dir("ext-proc-timeout");
        let log = dir.join("slow.log");
        let script = "sleep 60 & echo $! > grandchild.pid; echo started; wait";
        let started = Instant::now();
        let ran = run_logged(&sh(script), &dir, &log, "fit", Duration::from_millis(700)).unwrap();
        assert_eq!(ran.exit, Exit::TimedOut);
        assert!(
            ran.wall_ms >= 700.0 && ran.wall_ms < 30_000.0,
            "{}",
            ran.wall_ms
        );
        assert!(started.elapsed() < Duration::from_secs(30));
        let text = std::fs::read_to_string(&log).unwrap();
        assert_eq!(text, "--- fit: start\nstarted\n--- fit: timeout, killed\n");

        // The background `sleep` shared the child's process group and is gone too.
        let pid = std::fs::read_to_string(dir.join("grandchild.pid")).unwrap();
        let pid = pid.trim();
        // Field 3 of /proc/<pid>/stat is the state; `Z` is a dead process that
        // its new parent has not reaped yet. Elsewhere, probe with signal 0.
        let alive = || {
            if Path::new("/proc/self/stat").exists() {
                std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
                    let state = stat.rsplit(')').next().unwrap_or("").trim_start();
                    !state.starts_with('Z')
                })
            } else {
                let probe = format!("kill -s 0 {pid} 2>/dev/null");
                let status = Command::new("sh").arg("-c").arg(probe).status();
                status.unwrap().success()
            }
        };
        let deadline = Instant::now() + Duration::from_secs(20);
        while alive() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!alive(), "the grandchild survived the timeout");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
