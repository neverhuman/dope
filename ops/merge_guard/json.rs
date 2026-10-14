//! Version-one receipt hashing, compatible with Python's ASCII JSON encoding.
use super::{Refused, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt::Write;

pub fn hash_bytes(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}
pub fn digest(value: &Value) -> Result<String> {
    Ok(hash_bytes(canonical(value)?.as_bytes()))
}
pub fn canonical(value: &Value) -> Result<String> {
    let mut out = String::new();
    encode(value, &mut out)?;
    Ok(out)
}
fn string(value: &str, out: &mut String) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 32 || (c as u32) >= 127 => {
                for unit in c.encode_utf16(&mut [0; 2]).iter() {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
fn number(value: &serde_json::Number) -> Result<String> {
    if value.is_i64() || value.is_u64() {
        return Ok(value.to_string());
    }
    let value = value
        .as_f64()
        .ok_or(Refused("invalid_or_incomplete_evidence"))?;
    if !value.is_finite() {
        return Err(Refused("invalid_or_incomplete_evidence"));
    }
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into());
    }
    let raw =
        serde_json::to_string(&value).map_err(|_| Refused("invalid_or_incomplete_evidence"))?;
    let sign = if raw.starts_with('-') { "-" } else { "" };
    let raw = raw.trim_start_matches('-');
    let (mantissa, exponent) = match raw.split_once('e') {
        Some((m, e)) => (
            m,
            e.parse::<i32>()
                .map_err(|_| Refused("invalid_or_incomplete_evidence"))?,
        ),
        None => (raw, 0),
    };
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let leading = digits.bytes().take_while(|c| *c == b'0').count();
    let digits = digits[leading..].trim_end_matches('0');
    let exp = exponent + decimal - leading as i32 - 1;
    if !(-4..16).contains(&exp) {
        let (first, rest) = digits.split_at(1);
        Ok(format!(
            "{sign}{first}{}e{}{abs:02}",
            if rest.is_empty() {
                String::new()
            } else {
                format!(".{rest}")
            },
            if exp < 0 { "-" } else { "+" },
            abs = exp.abs()
        ))
    } else if exp < 0 {
        Ok(format!(
            "{sign}0.{}{digits}",
            "0".repeat((-exp - 1) as usize)
        ))
    } else {
        let decimal = exp as usize + 1;
        if decimal >= digits.len() {
            Ok(format!(
                "{sign}{digits}{}.0",
                "0".repeat(decimal - digits.len())
            ))
        } else {
            Ok(format!(
                "{sign}{}.{}",
                &digits[..decimal],
                &digits[decimal..]
            ))
        }
    }
}
fn encode(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
        Value::Number(v) => out.push_str(&number(v)?),
        Value::String(v) => string(v, out),
        Value::Array(values) => {
            out.push('[');
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode(value, out)?;
            }
            out.push(']');
        }
        Value::Object(values) => {
            out.push('{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort();
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                string(key, out);
                out.push(':');
                encode(&values[*key], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}
