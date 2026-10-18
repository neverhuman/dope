use super::wire::{Request, Response, fail, json, read_request, respond};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::net::{SocketAddr, TcpListener};
use std::path::{Component, Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;
use std::time::Instant;

pub(super) const MAX_HEADERS: usize = 16 * 1024;
pub(super) const MAX_BODY: usize = 64 * 1024;
const MAX_FILE: u64 = 16 * 1024 * 1024;
const MAX_REPORT: usize = 1024 * 1024;
const MAX_CAPABILITIES: usize = 128;
const MAX_JOBS: usize = 512;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapabilityRequest {
    relative_path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobRequest {
    format: String,
    version: u8,
    kind: String,
    capability_id: String,
}

#[derive(Clone, Serialize)]
struct Receipt {
    format: &'static str,
    version: u8,
    job_id: String,
    job_state: &'static str,
    kind: String,
    input_sha256: String,
    input_bytes: u64,
    report_sha256: Option<String>,
    report: Option<Value>,
    utility_state: &'static str,
    empirical_privacy_state: &'static str,
    formal_dp_state: &'static str,
    release_authorization_state: &'static str,
    qualified: bool,
}

#[derive(Clone)]
struct Capability {
    relative_path: String,
    sha256: String,
    bytes: u64,
}

struct App {
    root: PathBuf,
    token: String,
    origin: String,
    capabilities: HashMap<String, Capability>,
    jobs: HashMap<String, Arc<JobEntry>>,
    active_jobs: Arc<AtomicUsize>,
}

struct JobEntry {
    receipt: Mutex<Receipt>,
    cancelled: Arc<AtomicBool>,
}

fn random_id() -> Result<String, &'static str> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .map_err(|_| "random_unavailable")?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn relative_file(root: &Path, relative: &str) -> Result<(PathBuf, Vec<u8>), &'static str> {
    if relative.is_empty() || relative.len() > 1024 || relative.contains(['\\', '%', '\0']) {
        return Err("path_refused");
    }
    let rel = Path::new(relative);
    if rel.is_absolute() || rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err("path_refused");
    }
    let mut current = root.to_path_buf();
    for part in rel.components() {
        current.push(part.as_os_str());
        let meta = fs::symlink_metadata(&current).map_err(|_| "path_unavailable")?;
        if meta.file_type().is_symlink() {
            return Err("path_refused");
        }
    }
    let canonical = fs::canonicalize(&current).map_err(|_| "path_unavailable")?;
    if !canonical.starts_with(root) {
        return Err("path_refused");
    }
    let meta = fs::metadata(&canonical).map_err(|_| "path_unavailable")?;
    if !meta.is_file() || meta.len() > MAX_FILE {
        return Err("file_limit");
    }
    let file = File::open(&canonical).map_err(|_| "path_unavailable")?;
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    file.take(MAX_FILE + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "file_unreadable")?;
    if bytes.len() as u64 > MAX_FILE {
        return Err("file_limit");
    }
    Ok((canonical, bytes))
}

impl App {
    fn authorized(&self, req: &Request, mutation: bool) -> bool {
        let host_ok = req
            .headers
            .get("host")
            .is_some_and(|h| h == self.origin.trim_start_matches("http://"));
        let origin_ok = req.headers.get("origin").is_some_and(|o| o == &self.origin);
        let auth_ok = req
            .headers
            .get("authorization")
            .is_some_and(|a| constant_eq(a, &format!("Bearer {}", self.token)));
        let csrf_ok = !mutation
            || req
                .headers
                .get("x-dope-csrf")
                .is_some_and(|v| constant_eq(v, &self.token));
        host_ok && origin_ok && auth_ok && csrf_ok
    }

    fn handle(&mut self, req: Request) -> Response {
        if !self.authorized(&req, req.method == "POST") {
            return fail(403, "request_refused");
        }
        if req.path.contains(['%', '\\']) || req.path.split('/').any(|p| p == "." || p == "..") {
            return fail(400, "path_refused");
        }
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/capabilities") => json(
                200,
                json!({
                    "format":"dope-local-capabilities","version":1,
                    "service_available":true,"supported_job_kinds":["inspect"],
                    "unsupported_job_kinds":["prepare","compile","evaluate"],
                    "utility_state":"unassessed","empirical_privacy_state":"unassessed",
                    "formal_dp_state":"not_claimed","release_authorization_state":"not_authorized",
                    "qualified":false,"limits":{"request_bytes":MAX_BODY,"file_bytes":MAX_FILE,
                    "capabilities":MAX_CAPABILITIES,"jobs":MAX_JOBS}
                }),
            ),
            ("POST", "/v1/capabilities") => self.add_capability(&req.body),
            ("POST", "/v1/jobs") => self.add_job(&req.body),
            ("GET", path) if path.starts_with("/v1/jobs/") => self.get_job(path, false),
            ("POST", path) if path.starts_with("/v1/jobs/") && path.ends_with("/cancel") => {
                self.cancel_job(path)
            }
            ("GET", path) if path.starts_with("/v1/receipts/") => self.get_job(path, true),
            ("GET", path) if path.starts_with("/v1/reports/") => self.get_report(path),
            _ => fail(404, "not_found"),
        }
    }

    fn add_capability(&mut self, body: &[u8]) -> Response {
        if self.capabilities.len() >= MAX_CAPABILITIES {
            return fail(429, "capability_limit");
        }
        let request: CapabilityRequest = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(_) => return fail(400, "request_invalid"),
        };
        let (_path, bytes) = match relative_file(&self.root, &request.relative_path) {
            Ok(v) => v,
            Err(e) => return fail(400, e),
        };
        let id = match random_id() {
            Ok(v) => v,
            Err(e) => return fail(500, e),
        };
        let sha = digest(&bytes);
        self.capabilities.insert(
            id.clone(),
            Capability {
                relative_path: request.relative_path,
                sha256: sha.clone(),
                bytes: bytes.len() as u64,
            },
        );
        json(
            201,
            json!({"format":"dope-local-capability","version":1,"capability_id":id,"sha256":sha,"bytes":bytes.len()}),
        )
    }

    fn add_job(&mut self, body: &[u8]) -> Response {
        if self.jobs.len() >= MAX_JOBS {
            return fail(429, "job_limit");
        }
        let request: JobRequest = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(_) => return fail(400, "request_invalid"),
        };
        if request.format != "dope-local-job"
            || request.version != 1
            || !matches!(
                request.kind.as_str(),
                "inspect" | "prepare" | "compile" | "evaluate"
            )
        {
            return fail(400, "protocol_unsupported");
        }
        if request.capability_id.len() != 32
            || !request.capability_id.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return fail(400, "capability_invalid");
        }
        let capability = match self.capabilities.get(&request.capability_id).cloned() {
            Some(v) => v,
            None => return fail(404, "capability_unknown"),
        };
        let (path, bytes) = match relative_file(&self.root, &capability.relative_path) {
            Ok(v) => v,
            Err(e) => return fail(400, e),
        };
        if digest(&bytes) != capability.sha256 || bytes.len() as u64 != capability.bytes {
            return fail(409, "capability_changed");
        }
        let id = match random_id() {
            Ok(v) => v,
            Err(e) => return fail(500, e),
        };
        let inspect = request.kind == "inspect";
        let receipt = Receipt {
            format: "dope-local-job-receipt",
            version: 1,
            job_id: id.clone(),
            job_state: if inspect { "queued" } else { "unsupported" },
            kind: request.kind,
            input_sha256: capability.sha256,
            input_bytes: capability.bytes,
            report_sha256: None,
            report: if inspect {
                None
            } else {
                Some(json!({"error":"executor_unavailable"}))
            },
            utility_state: "unassessed",
            empirical_privacy_state: "unassessed",
            formal_dp_state: "not_claimed",
            release_authorization_state: "not_authorized",
            qualified: false,
        };
        let entry = Arc::new(JobEntry {
            receipt: Mutex::new(receipt),
            cancelled: Arc::new(AtomicBool::new(false)),
        });
        self.jobs.insert(id.clone(), entry.clone());
        let mut state = "unsupported";
        if inspect {
            let active = self.active_jobs.fetch_add(1, Ordering::AcqRel);
            if active >= 2 {
                self.active_jobs.fetch_sub(1, Ordering::AcqRel);
                if let Ok(mut r) = entry.receipt.lock() {
                    r.job_state = "over_budget";
                    r.report = Some(json!({"error":"worker_limit"}));
                }
                state = "over_budget";
            } else {
                state = "queued";
                let path = path.clone();
                let cancel = entry.cancelled.clone();
                let slot = self.active_jobs.clone();
                std::thread::spawn(move || {
                    if let Ok(mut r) = entry.receipt.lock() {
                        if r.job_state != "cancelled" {
                            r.job_state = "running";
                        }
                    }
                    let start = Instant::now();
                    let result = dope_kernel::inspect_kernel(&path).map(|value| {
                        let hash = digest(&serde_json::to_vec_pretty(&value).unwrap_or_default());
                        let full = serde_json::to_value(value).unwrap_or(Value::Null);
                        let mut safe = full;
                        if let Some(o) = safe.as_object_mut() {
                            for k in [
                                "task",
                                "schema",
                                "marginals",
                                "dependence",
                                "target",
                                "noise",
                                "sexpr",
                            ] {
                                o.remove(k);
                            }
                        }
                        (safe, hash)
                    });
                    let (mut final_state, report, report_hash) = match result {
                        Ok((v, _hash))
                            if serde_json::to_vec(&v).map_or(true, |b| b.len() > MAX_REPORT) =>
                        {
                            ("incomplete", Some(json!({"error":"report_limit"})), None)
                        }
                        Ok((v, hash)) => ("complete", Some(v), Some(hash)),
                        Err(_) => ("failed", Some(json!({"error":"inspection_failed"})), None),
                    };
                    if cancel.load(Ordering::Acquire) {
                        final_state = "cancelled";
                    } else if start.elapsed() > Duration::from_secs(30) {
                        final_state = "timed_out";
                    }
                    if let Ok(mut r) = entry.receipt.lock() {
                        r.job_state = final_state;
                        r.report_sha256 = if final_state == "complete" {
                            report_hash
                        } else {
                            None
                        };
                        r.report = report;
                    }
                    slot.fetch_sub(1, Ordering::AcqRel);
                });
            }
        }
        json(
            202,
            json!({"job_id":id,"job_state":state,"qualified":false}),
        )
    }

    fn get_job(&self, path: &str, receipt: bool) -> Response {
        let prefix = if receipt {
            "/v1/receipts/"
        } else {
            "/v1/jobs/"
        };
        let id = path
            .strip_prefix(prefix)
            .unwrap_or("")
            .trim_end_matches("/cancel");
        if !valid_id(id) {
            return fail(404, "not_found");
        }
        match self.jobs.get(id) {
            Some(job) => match job.receipt.lock() {
                Ok(r) if receipt => json(200, serde_json::to_value(&*r).unwrap_or(Value::Null)),
                Ok(r) => json(
                    200,
                    json!({"job_id":id,"job_state":r.job_state,"qualified":false}),
                ),
                Err(_) => fail(500, "job_unavailable"),
            },
            _ => fail(404, "not_found"),
        }
    }

    fn get_report(&self, path: &str) -> Response {
        let id = path.strip_prefix("/v1/reports/").unwrap_or("");
        if !valid_id(id) {
            return fail(404, "not_found");
        }
        match self.jobs.get(id) {
            Some(job) => match job.receipt.lock() {
                Ok(r) if r.job_state == "complete" => json(
                    200,
                    json!({"report_sha256":r.report_sha256,"report":r.report,"qualified":false}),
                ),
                Ok(_) => fail(409, "report_unavailable"),
                Err(_) => fail(500, "job_unavailable"),
            },
            None => fail(404, "not_found"),
        }
    }

    fn cancel_job(&mut self, path: &str) -> Response {
        let id = path
            .strip_prefix("/v1/jobs/")
            .unwrap_or("")
            .strip_suffix("/cancel")
            .unwrap_or("");
        if !valid_id(id) {
            return fail(404, "not_found");
        }
        match self.jobs.get(id) {
            Some(job) => match job.receipt.lock() {
                Ok(mut r) if r.job_state == "queued" || r.job_state == "running" => {
                    job.cancelled.store(true, Ordering::Release);
                    r.job_state = "cancelled";
                    json(
                        200,
                        json!({"job_id":id,"job_state":"cancelled","qualified":false}),
                    )
                }
                Ok(_) => fail(409, "job_not_cancellable"),
                Err(_) => fail(500, "job_unavailable"),
            },
            None => fail(404, "not_found"),
        }
    }
}

fn valid_id(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
}
fn constant_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

pub fn run(args: Vec<String>) -> Result<(), &'static str> {
    let mut bind = "127.0.0.1:8787".to_owned();
    let mut root = std::env::current_dir().map_err(|_| "root_unavailable")?;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bind" => {
                i += 1;
                bind = args.get(i).ok_or("bind_required")?.clone();
            }
            "--root" => {
                i += 1;
                root = PathBuf::from(args.get(i).ok_or("root_required")?);
            }
            _ => return Err("argument_invalid"),
        };
        i += 1;
    }
    let addr: SocketAddr = bind.parse().map_err(|_| "bind_invalid")?;
    if !addr.ip().is_loopback() {
        return Err("loopback_only");
    }
    let root = fs::canonicalize(root).map_err(|_| "root_unavailable")?;
    if !root.is_dir() {
        return Err("root_unavailable");
    }
    let listener = TcpListener::bind(addr).map_err(|_| "bind_failed")?;
    let actual = listener.local_addr().map_err(|_| "bind_failed")?;
    let token = random_id()?;
    let host = match actual.ip() {
        std::net::IpAddr::V4(ip) => ip.to_string(),
        std::net::IpAddr::V6(ip) => format!("[{ip}]"),
    };
    let origin = format!("http://{host}:{}", actual.port());
    println!(
        "{}",
        json!({"service":"dope-local-service","origin":origin,"token":token,"protocol_version":1})
    );
    let mut app = App {
        root,
        token,
        origin,
        capabilities: HashMap::new(),
        jobs: HashMap::new(),
        active_jobs: Arc::new(AtomicUsize::new(0)),
    };
    for mut stream in listener.incoming().flatten() {
        let response = match read_request(&mut stream) {
            Ok(req) => app.handle(req),
            Err(code) => fail(400, code),
        };
        respond(&mut stream, response);
    }
    Ok(())
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
