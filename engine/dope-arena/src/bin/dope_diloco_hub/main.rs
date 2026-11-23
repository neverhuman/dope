//! Pure-Rust SSH-tunneled DiLoCo Outer-Momentum Parameter Hub CLI & HTTP server.

mod state;

use clap::Parser;
use state::{HubConfig, HubState, MAX_BODY_BYTES};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Parser, Debug, Clone)]
#[command(name = "dope_diloco_hub", about = "Pure-Rust DiLoCo SafeTensors hub")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8771")]
    bind: String,
    #[arg(long)]
    state_dir: PathBuf,
    #[arg(long, default_value = "v73-diloco")]
    sync_group: String,
    #[arg(long, default_value_t = 0.4)]
    outer_lr: f32,
    #[arg(long, default_value_t = 0.85)]
    outer_momentum: f32,
    #[arg(long, default_value_t = 0.15)]
    staleness_alpha: f32,
    #[arg(long, default_value_t = 50)]
    save_every: u64,
    #[arg(long, default_value_t = 0.5)]
    max_delta_rms: f32,
}

struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut tmp).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            return Err("eof before headers".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 64 * 1024 {
            return Err("http headers too large".into());
        }
    };
    let head_str = std::str::from_utf8(&buf[..header_end]).map_err(|_| "invalid utf8 headers")?;
    let mut lines = head_str.split("\r\n");
    let req_line = lines.next().ok_or("empty request")?;
    let mut parts = req_line.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_string();
    let path = parts.next().ok_or("missing path")?.to_string();
    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let content_len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if content_len > MAX_BODY_BYTES {
        return Err(format!("body exceeds cap: {content_len}"));
    }
    let mut body = Vec::with_capacity(content_len);
    body.extend_from_slice(&buf[header_end..]);
    if body.len() > content_len {
        body.truncate(content_len);
    }
    while body.len() < content_len {
        let need = (content_len - body.len()).min(65536);
        let mut chunk = vec![0u8; need];
        let n = stream
            .read(&mut chunk)
            .map_err(|e| format!("read body: {e}"))?;
        if n == 0 {
            return Err("truncated body".into());
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    ctype: &str,
    extra: &[(&str, &str)],
    body: &[u8],
) {
    let mut hdr = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        hdr.push_str(&format!("{k}: {v}\r\n"));
    }
    hdr.push_str("\r\n");
    let _ = stream.write_all(hdr.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn handle_client(mut stream: TcpStream, state: Arc<Mutex<HubState>>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let Ok(req) = read_http_request(&mut stream) else {
        write_http_response(&mut stream, "400 Bad Request", "text/plain", &[], b"bad\n");
        return;
    };
    let clean_path = req.path.split('?').next().unwrap_or(&req.path);
    match (req.method.as_str(), clean_path) {
        ("GET", "/status") => {
            let js = state
                .lock()
                .map(|s| s.status_json())
                .unwrap_or_else(|_| "{}".into());
            write_http_response(
                &mut stream,
                "200 OK",
                "application/json",
                &[],
                js.as_bytes(),
            );
        }
        ("GET", "/weights") => {
            let res = state.lock().ok().and_then(|s| {
                s.weights
                    .as_ref()
                    .map(|_| (s.round, s.global_sha.clone(), s.cached_bytes.clone()))
            });
            match res {
                Some((r, sha, bytes)) => {
                    let rs = r.to_string();
                    let sh = &sha[..sha.len().min(16)];
                    write_http_response(
                        &mut stream,
                        "200 OK",
                        "application/octet-stream",
                        &[("X-Diloco-Round", &rs), ("X-Diloco-Sha", sh)],
                        &bytes,
                    );
                }
                None => write_http_response(&mut stream, "404 Not Found", "text/plain", &[], b""),
            }
        }
        ("POST", "/sync") => {
            let wid = req
                .headers
                .get("x-worker-id")
                .map(String::as_str)
                .unwrap_or("anon");
            let mode = req
                .headers
                .get("x-payload-mode")
                .map(String::as_str)
                .unwrap_or("delta");
            let base_round = req
                .headers
                .get("x-base-round")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            let res = state
                .lock()
                .map_err(|_| "poisoned".to_string())
                .and_then(|mut s| s.apply_sync(wid, mode, base_round, &req.body));
            match res {
                Ok((r, sha, bytes)) => {
                    let rs = r.to_string();
                    let sh = &sha[..sha.len().min(16)];
                    write_http_response(
                        &mut stream,
                        "200 OK",
                        "application/octet-stream",
                        &[("X-Diloco-Round", &rs), ("X-Diloco-Sha", sh)],
                        &bytes,
                    );
                }
                Err(msg) => {
                    write_http_response(
                        &mut stream,
                        "400 Bad Request",
                        "text/plain",
                        &[],
                        msg.as_bytes(),
                    );
                }
            }
        }
        _ => write_http_response(&mut stream, "404 Not Found", "text/plain", &[], b"404\n"),
    }
}

fn main() {
    let args = Args::parse();
    let bind_addr = args.bind.clone();
    let cfg = HubConfig {
        state_dir: args.state_dir,
        sync_group: args.sync_group,
        outer_lr: args.outer_lr,
        outer_momentum: args.outer_momentum,
        staleness_alpha: args.staleness_alpha,
        save_every: args.save_every,
        max_delta_rms: args.max_delta_rms,
    };
    let state = Arc::new(Mutex::new(HubState::load_or_new(cfg)));
    let listener =
        TcpListener::bind(&bind_addr).unwrap_or_else(|e| panic!("bind {bind_addr}: {e}"));
    eprintln!("[dope_diloco_hub] listening on http://{bind_addr}");
    for incoming in listener.incoming().flatten() {
        let st = Arc::clone(&state);
        std::thread::spawn(move || handle_client(incoming, st));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use state::{encode_f32_safetensors, parse_f32_safetensors, TensorMap};

    fn mk_st(vals: &[f32]) -> Vec<u8> {
        let mut m = TensorMap::new();
        m.insert("w".into(), (vec![vals.len()], vals.to_vec()));
        encode_f32_safetensors(&m).expect("an f32 tensor map always serialises")
    }

    #[test]
    fn roundtrip_and_nesterov_diloco_step() {
        let tmp = std::env::temp_dir().join(format!("diloco_hub_t_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let cfg = HubConfig {
            state_dir: tmp.clone(),
            sync_group: "test-group".into(),
            outer_lr: 0.5,
            outer_momentum: 0.8,
            staleness_alpha: 0.25,
            save_every: 2,
            max_delta_rms: 10.0,
        };
        let mut st = HubState::load_or_new(cfg);
        let (r1, _, b1) = st
            .apply_sync("w0", "init", 0, &mk_st(&[1.0, 2.0]))
            .expect("init");
        assert_eq!(r1, 1);
        let (r2, _, b2) = st
            .apply_sync("w1", "init", 0, &mk_st(&[9.0, 9.0]))
            .expect("init2");
        assert_eq!(r2, 1);
        assert_eq!(b1, b2);
        let (r3, _, b3) = st
            .apply_sync("w0", "delta", 1, &mk_st(&[0.2, -0.4]))
            .expect("delta");
        assert_eq!(r3, 2);
        let parsed = parse_f32_safetensors(&b3).expect("parse");
        assert!((parsed["w"].1[0] - 0.82).abs() < 1e-5);
        assert!((parsed["w"].1[1] - 2.36).abs() < 1e-5);
        assert!(tmp.join("global_round_000002.safetensors").is_file());
        assert!(st
            .apply_sync("bad", "delta", 2, &mk_st(&[f32::NAN, 0.0]))
            .is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
