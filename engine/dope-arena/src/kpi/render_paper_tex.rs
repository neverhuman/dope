//! Pure LaTeX token formatters for `dope_kpi render-paper-tables`.

use super::render::fixed3;
use super::render_ranked::{human_bytes, score1, signed_pts1};
use std::collections::BTreeMap;

pub(super) fn tex_escape(text: &str) -> String {
    text.replace('\\', r"\textbackslash{}")
        .replace('&', r"\&")
        .replace('%', r"\%")
        .replace('_', r"\_")
        .replace('#', r"\#")
}

pub(super) fn tex_minus(text: &str) -> String {
    if let Some(rest) = text.strip_prefix('-') {
        format!(r"\ensuremath{{-}}{rest}")
    } else {
        text.to_string()
    }
}

pub(super) fn tex_score(trg: f64) -> String {
    tex_minus(&score1(trg))
}

pub(super) fn tex_signed_pts(delta: f64) -> String {
    tex_minus(&signed_pts1(delta))
}

pub(super) fn tex_score_ci(trg: f64, lo: f64, hi: f64) -> String {
    format!("{} [{}, {}]", tex_score(trg), tex_score(lo), tex_score(hi))
}

pub(super) fn tex_delta_ci(delta: f64, lo: f64, hi: f64) -> String {
    format!(
        "{} [{}, {}]",
        tex_signed_pts(delta),
        tex_signed_pts(lo),
        tex_signed_pts(hi)
    )
}

pub(super) fn tex_p(p: f64) -> String {
    if p < 0.001 {
        r"\textless{}0.001".to_string()
    } else {
        fixed3(p)
    }
}

pub(super) fn tex_opt3(x: Option<f64>) -> String {
    x.map_or_else(|| "--".to_string(), |v| tex_minus(&fixed3(v)))
}

pub(super) fn tex_grouped_u64(n: u64) -> String {
    let raw = n.to_string();
    let mut out = String::new();
    for (i, ch) in raw.chars().enumerate() {
        if i > 0 && (raw.len() - i) % 3 == 0 {
            out.push_str("{,}");
        }
        out.push(ch);
    }
    out
}

pub(super) fn tex_bytes(bytes: Option<f64>) -> String {
    bytes.map_or_else(
        || "--".to_string(),
        |b| human_bytes(b).replace(',', "{,}").replace(' ', "~"),
    )
}

pub(super) fn short_guard(name: &str) -> &str {
    match name {
        "task_ci_low" => "tasks",
        "mlp_delta" => "mlp",
        "realism_coverage_delta" => "realism",
        "privacy_coverage_delta" => "privacy",
        "exact_copy_excess" => "copies",
        other => other,
    }
}

pub(super) fn display_name(arm_id: &str, labels: &BTreeMap<String, String>, bold: bool) -> String {
    let text = labels
        .get(arm_id)
        .map(|s| tex_escape(s))
        .unwrap_or_else(|| format!(r"\texttt{{{}}}", tex_escape(arm_id)));
    if bold {
        format!(r"\textbf{{{text}}}")
    } else {
        text
    }
}
