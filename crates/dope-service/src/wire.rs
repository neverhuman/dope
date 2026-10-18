use super::service::{MAX_BODY, MAX_HEADERS};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
const MAX_RESPONSE: usize = 1024 * 1024;

pub(super) struct Request {
    pub(super) method: String,
    pub(super) path: String,
    pub(super) headers: HashMap<String, String>,
    pub(super) body: Vec<u8>,
}
pub(super) struct Response {
    pub(super) status: u16,
    pub(super) body: Value,
}
pub(super) fn json(status: u16, body: Value) -> Response {
    Response { status, body }
}
pub(super) fn fail(status: u16, code: &str) -> Response {
    json(status, json!({"error":code}))
}

pub(super) fn read_request(stream: &mut TcpStream) -> Result<Request, &'static str> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|_| "socket_error")?;
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    while header.len() < MAX_HEADERS && !header.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).map_err(|_| "request_read_failed")? == 0 {
            return Err("request_incomplete");
        }
        header.push(byte[0]);
    }
    if !header.ends_with(b"\r\n\r\n") {
        return Err("header_limit");
    }
    let text = std::str::from_utf8(&header).map_err(|_| "request_invalid")?;
    let mut lines = text.split("\r\n");
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let method = first.next().ok_or("request_invalid")?.to_owned();
    let path = first.next().ok_or("request_invalid")?.to_owned();
    if first.next() != Some("HTTP/1.1") || !matches!(method.as_str(), "GET" | "POST") {
        return Err("request_invalid");
    }
    let mut headers = HashMap::new();
    for line in lines.filter(|x| !x.is_empty()) {
        let (name, value) = line.split_once(':').ok_or("request_invalid")?;
        let key = name.trim().to_ascii_lowercase();
        if headers.insert(key, value.trim().to_owned()).is_some() {
            return Err("request_invalid");
        }
    }
    if headers.contains_key("transfer-encoding") {
        return Err("request_invalid");
    }
    if method == "POST"
        && headers.get("content-type").map(String::as_str) != Some("application/json")
    {
        return Err("request_invalid");
    }
    let length = match headers.get("content-length") {
        Some(v) => v.parse::<usize>().map_err(|_| "request_invalid")?,
        None => 0,
    };
    if length > MAX_BODY || (method == "POST" && !headers.contains_key("content-length")) {
        return Err("request_limit");
    }
    let mut body = vec![0; length];
    stream
        .read_exact(&mut body)
        .map_err(|_| "request_incomplete")?;
    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

pub(super) fn respond(stream: &mut TcpStream, response: Response) {
    let mut status = response.status;
    let mut body = serde_json::to_vec(&response.body).unwrap_or_else(|_| b"{}".to_vec());
    if body.len() > MAX_RESPONSE {
        status = 413;
        body = b"{\"error\":\"response_limit\"}".to_vec();
    }
    let reason = match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        501 => "Not Implemented",
        _ => "Internal Server Error",
    };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'none'\r\n\r\n",
        status,
        reason,
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
    let _ = stream.flush();
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
