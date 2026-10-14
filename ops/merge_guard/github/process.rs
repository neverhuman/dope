//! Termination is restricted to the process group created for this request.
use super::{Response, Transport};
use crate::guard::{Refused, Result};
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub struct ProcessTransport {
    pub(crate) limit: Duration,
}
impl Default for ProcessTransport {
    fn default() -> Self {
        Self {
            limit: Duration::from_secs(45),
        }
    }
}
fn terminate(child: &mut Child) {
    if let Ok(pid) = i32::try_from(child.id()) {
        if pid > 0 {
            // process_group(0) made this request a fresh group before exec.
            // No caller's screen, lock, or unrelated process belongs to it.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}
fn reader(
    mut stream: impl Read + Send + 'static,
    max: usize,
) -> mpsc::Receiver<std::io::Result<Vec<u8>>> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut data = Vec::new();
        let result = stream
            .by_ref()
            .take(max as u64 + 1)
            .read_to_end(&mut data)
            .and_then(|_| {
                if data.len() > max {
                    Err(std::io::Error::other("response exceeded bound"))
                } else {
                    Ok(data)
                }
            });
        let _ = sender.send(result);
    });
    receiver
}
impl Transport for ProcessTransport {
    fn run(&mut self, argv: &[String]) -> Result<Response> {
        let executable = argv.first().ok_or(Refused("api_unavailable"))?;
        let start = Instant::now();
        let mut child = Command::new(executable)
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(|_| Refused("api_unavailable"))?;
        let stdout = match child.stdout.take() {
            Some(v) => v,
            None => {
                terminate(&mut child);
                return Err(Refused("api_unavailable"));
            }
        };
        let stderr = match child.stderr.take() {
            Some(v) => v,
            None => {
                terminate(&mut child);
                return Err(Refused("api_unavailable"));
            }
        };
        let out = reader(stdout, 16 * 1024 * 1024);
        let err = reader(stderr, 64 * 1024);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {}
                Err(_) => {
                    terminate(&mut child);
                    return Err(Refused("api_unavailable"));
                }
            }
            if start.elapsed() >= self.limit {
                terminate(&mut child);
                return Err(Refused("api_unavailable"));
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        let capture = (|| {
            let bytes = out
                .recv_timeout(self.limit.saturating_sub(start.elapsed()))
                .map_err(|_| Refused("api_unavailable"))?
                .map_err(|_| Refused("api_unavailable"))?;
            let stderr = err
                .recv_timeout(self.limit.saturating_sub(start.elapsed()))
                .map_err(|_| Refused("api_unavailable"))?
                .map_err(|_| Refused("api_unavailable"))?;
            let stdout = String::from_utf8(bytes).map_err(|_| Refused("api_unavailable"))?;
            Ok(Response {
                code: status.code().ok_or(Refused("api_unavailable"))?,
                stdout,
                stderr: String::from_utf8_lossy(&stderr).into_owned(),
            })
        })();
        if capture.is_err() {
            terminate(&mut child);
        }
        capture
    }
}
