use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

fn svg(label: &str, color: &str) -> Vec<u8> {
    format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"230\" height=\"20\" role=\"img\" aria-label=\"Jankurai: {label}\">\n<title>Jankurai full audit: {label}</title>\n<rect width=\"110\" height=\"20\" fill=\"#555\"/>\n<rect x=\"110\" width=\"120\" height=\"20\" fill=\"{color}\"/>\n<g fill=\"#fff\" text-anchor=\"middle\" font-family=\"Verdana,sans-serif\" font-size=\"11\">\n<text x=\"55\" y=\"14\">Jankurai 1.7.1</text>\n<text x=\"170\" y=\"14\">{label}</text>\n</g>\n</svg>\n").into_bytes()
}

pub(super) fn render(report: &Value, head: &str, now: u64) -> Option<(Vec<u8>, &'static str)> {
    if head.len() != 40
        || !head
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || report["auditor_version"] != "1.7.1"
        || report["scope"] != json!({"mode":"full","paths":[]})
        || report["git"]["head"] != head
        || report["git"]["mode"] != "full"
    {
        return None;
    }
    let dirty = report["dirty_worktree"].as_bool()?;
    if report["git"]["dirty_worktree"].as_bool()? != dirty {
        return None;
    }
    let stamp = report["generated_at"].as_str()?.parse::<u64>().ok()?;
    if now.checked_sub(stamp)? > 7200 {
        return None;
    }
    let score = report["score"].as_u64()?;
    if score > 100 {
        return None;
    }
    let caps = report["caps_applied"].as_array()?;
    let findings = report["findings"].as_array()?;
    for finding in findings {
        if !matches!(
            finding["severity"].as_str()?,
            "critical" | "high" | "medium" | "low" | "info"
        ) {
            return None;
        }
    }
    let decision = &report["decision"];
    let status = decision["status"].as_str()?;
    if !matches!(status, "pass" | "fail") || decision["minimum_score"].as_u64()? != 85 {
        return None;
    }
    let passed = decision["passed"].as_bool()?
        && status == "pass"
        && score >= 85
        && caps.is_empty()
        && !findings
            .iter()
            .any(|f| matches!(f["severity"].as_str(), Some("high" | "critical")));
    let state = if !passed {
        "fail"
    } else if dirty {
        "local"
    } else {
        "pass"
    };
    let color = match state {
        "fail" => "#b31d28",
        "local" => "#777",
        _ => "#22863a",
    };
    Some((svg(&format!("{score}/100 {state}"), color), state))
}

pub(super) fn write(
    report: &Path,
    head: &str,
    output: &Path,
    check: Option<&Path>,
) -> io::Result<i32> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_secs();
    write_at(report, head, output, check, now)
}

pub(super) fn write_at(
    report: &Path,
    head: &str,
    output: &Path,
    check: Option<&Path>,
    now: u64,
) -> io::Result<i32> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = fs::read(report).ok();
    let parsed = raw
        .as_deref()
        .and_then(|bytes| serde_json::from_slice::<Value>(bytes).ok());
    let (body, status) = parsed
        .as_ref()
        .and_then(|value| render(value, head, now))
        .unwrap_or_else(|| (svg("unverified", "#777"), "invalid"));
    let mut receipt = json!({"expected_head":head,"status":status,
        "report_sha256":raw.map(|bytes| format!("{:x}",Sha256::digest(bytes))),
        "svg_sha256":format!("{:x}",Sha256::digest(&body))});
    let mut code = i32::from(status != "pass");
    if let Some(path) = check {
        let matches = fs::read(path).is_ok_and(|expected| expected == body);
        receipt["committed_badge_matches"] = json!(matches);
        if !matches {
            code = 1;
        }
    }
    fs::write(output, body)?;
    fs::write(
        output.with_extension("json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(code)
}
