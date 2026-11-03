//! Deliberately narrow version-1 scalar TOML admission; unsupported syntax refuses.
use std::collections::BTreeMap;
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Scalar {
    Integer(u64),
    Boolean(bool),
    Text(String),
}
fn content(line: &str) -> Result<&str, &'static str> {
    let mut quoted = false;
    for (i, c) in line.char_indices() {
        if c == '\\' {
            return Err("budget_toml_unsupported");
        }
        if c == '"' {
            quoted = !quoted;
        }
        if c == '#' && !quoted {
            return Ok(line[..i].trim());
        }
    }
    if quoted {
        return Err("budget_toml_invalid");
    }
    Ok(line.trim())
}
fn scalar(value: &str) -> Result<Scalar, &'static str> {
    if value == "true" {
        return Ok(Scalar::Boolean(true));
    }
    if value == "false" {
        return Ok(Scalar::Boolean(false));
    }
    if let Some(text) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        if text.chars().any(|c| c == '"' || c.is_control()) {
            return Err("budget_toml_invalid");
        }
        return Ok(Scalar::Text(text.to_owned()));
    }
    if value.is_empty()
        || !value.bytes().all(|b| b.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err("budget_toml_unsupported");
    }
    value
        .parse()
        .map(Scalar::Integer)
        .map_err(|_| "budget_toml_invalid")
}
pub(super) fn parse(source: &str) -> Result<BTreeMap<String, Scalar>, &'static str> {
    let mut fields = BTreeMap::new();
    let mut admission = false;
    for line in source.lines() {
        let line = content(line)?;
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if line != "[admission]" || admission {
                return Err("budget_toml_unsupported");
            }
            admission = true;
            continue;
        }
        let (key, value) = line.split_once('=').ok_or("budget_toml_invalid")?;
        let key = key.trim();
        if key.is_empty()
            || !key
                .bytes()
                .enumerate()
                .all(|(i, b)| b == b'_' || b.is_ascii_lowercase() || (i > 0 && b.is_ascii_digit()))
        {
            return Err("budget_toml_unsupported");
        }
        let name = if admission {
            format!("admission.{key}")
        } else {
            key.to_owned()
        };
        if fields.insert(name, scalar(value.trim())?).is_some() {
            return Err("budget_toml_duplicate");
        }
    }
    if !admission {
        return Err("budget_toml_invalid");
    }
    Ok(fields)
}
