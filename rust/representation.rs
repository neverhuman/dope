//! Diagnostic representation normalization and clean-encoder distance (demoted
//! from headline ranking in TDL; retained for diagnostics and historical
//! receipt compatibility, T-038).
//!
//! The quantile map is fit on real fit rows only. Column names and category
//! levels become HMAC digests. The shipped manifest stores those digests, not
//! the source strings.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const REPRESENTATION_ROLE: &str = "diagnostic_only";
pub const HEADLINE_COMPONENT: bool = false;
pub const NORMALIZER: &str = "empirical_midrank_quantile_v1";
pub const DIAGNOSTIC_ENCODER: &str = "kumo_tabular_l";
/// Historical alias retained for legacy MFS compatibility tests.
pub const HEADLINE_ENCODER: &str = DIAGNOSTIC_ENCODER;
pub const TABULAR_PROTOCOL: &str = "train_on_synthetic_score_on_real";
pub const TABULAR_AUDITORS: &[&str] = &["kumo_tabular_l", "mitra_v2", "tabicl2"];
pub const CATEGORICAL_BUCKETS: u32 = 65_536;
pub const SENSITIVITY_ENCODERS: &[&str] = &["kumo_tabular_s", "mitra_v2", "tabicl2", "a0"];
pub const REFUSED_ENCODERS: &[&str] = &[
    "context2048",
    "foundation",
    "hyperion_v6all",
    "hyperion_v7",
    "tabdpt13",
    "tabpfn35",
    "ta_dope_v2",
    "ta_hyperion_v8",
    "ta_hyperion_v8braw",
    "target",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RepresentationDiagnostics {
    pub role: &'static str,
    pub headline_component: bool,
    pub encoder: String,
    pub normalizer: String,
    pub unit_half_distance: Option<f64>,
    pub closeness: Option<f64>,
}

impl RepresentationDiagnostics {
    pub fn from_scores(
        encoder: impl Into<String>,
        distance: Option<f64>,
        closeness: Option<f64>,
    ) -> Self {
        Self {
            role: REPRESENTATION_ROLE,
            headline_component: HEADLINE_COMPONENT,
            encoder: encoder.into(),
            normalizer: NORMALIZER.to_owned(),
            unit_half_distance: distance,
            closeness,
        }
    }
}

pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut keyed = [0u8; BLOCK];
    if key.len() > BLOCK {
        keyed[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        keyed[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0u8; BLOCK];
    let mut opad = [0u8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] = keyed[i] ^ 0x36;
        opad[i] = keyed[i] ^ 0x5c;
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(message);
    let inner_digest = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_digest);
    let mut out = [0u8; 32];
    out.copy_from_slice(&outer.finalize());
    out
}

pub fn hex16(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub fn salt_id(salt: &[u8]) -> String {
    hex16(&Sha256::digest(salt)[..8])
}

pub fn column_hash(salt: &[u8], text: &str) -> String {
    hex16(&hmac_sha256(salt, text.as_bytes())[..8])
}

pub fn category_code(salt: &[u8], level: &str) -> f64 {
    let digest = hmac_sha256(salt, level.as_bytes());
    let bucket = u32::from(digest[0]) << 8 | u32::from(digest[1]);
    (f64::from(bucket) + 0.5) / f64::from(CATEGORICAL_BUCKETS)
}

/// Mid-rank quantile map onto (0, 1), fit on `fit` and applied to `values`.
/// Values outside the fit range clamp to the extreme mid-ranks. They do not
/// become 0 or 1, and the map does not publish a min or a max.
pub fn midrank_quantile(fit: &[f64], values: &[f64]) -> Option<Vec<f64>> {
    if fit.is_empty()
        || !fit
            .iter()
            .chain(values.iter())
            .all(|value| value.is_finite())
    {
        return None;
    }
    let mut ordered = fit.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    let n = ordered.len() as f64;
    values
        .iter()
        .map(|value| midrank_one(&ordered, *value, n))
        .collect()
}

fn midrank_one(ordered: &[f64], value: f64, n: f64) -> Option<f64> {
    let less = ordered.partition_point(|item| *item < value);
    let right = ordered.partition_point(|item| *item <= value);
    if right == 0 {
        return Some(0.5 / n);
    }
    if less == ordered.len() {
        return Some((n - 0.5) / n);
    }
    if less == right {
        let left_image = (less as f64 - 0.5) / n;
        let right_image = (less as f64 + 0.5) / n;
        let x0 = ordered[less - 1];
        let x1 = ordered[less];
        if x1 == x0 {
            return Some(left_image);
        }
        let span = x1 - x0;
        let num = value - x0;
        if !span.is_finite() || !num.is_finite() {
            return None;
        }
        let t = num / span;
        let out = left_image + t * (right_image - left_image);
        return (out.is_finite() && out > 0.0 && out < 1.0).then_some(out);
    }
    let mut total = 0.0;
    for index in less..right {
        total += (index as f64 + 0.5) / n;
    }
    let out = total / (right - less) as f64;
    (out.is_finite() && out > 0.0 && out < 1.0).then_some(out)
}

pub fn unit_half_distance(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.is_empty() || left.len() != right.len() {
        return None;
    }
    if !left
        .iter()
        .chain(right.iter())
        .all(|value| value.is_finite())
    {
        return None;
    }
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !left_norm.is_finite() || !right_norm.is_finite() || left_norm == 0.0 || right_norm == 0.0 {
        return None;
    }
    let mut total = 0.0;
    for (a, b) in left.iter().zip(right) {
        let delta = a / left_norm - b / right_norm;
        total += delta * delta;
    }
    let distance = total.sqrt() / 2.0;
    (distance.is_finite() && (0.0..=1.0).contains(&distance)).then_some(distance)
}

/// Closeness is `1 - d / d_match` only when `d` beats the independent-marginal
/// null and Mitra and TabICL have the same sign on `(d - d_null)`.
pub fn representation_closeness(
    distance: f64,
    d_null: f64,
    d_match: f64,
    gap_mitra: f64,
    gap_tabicl: f64,
) -> Option<f64> {
    if ![distance, d_null, d_match, gap_mitra, gap_tabicl]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }
    if !(0.0..=1.0).contains(&distance)
        || !(0.0..=1.0).contains(&d_null)
        || d_match <= 0.0
        || d_match > 1.0
    {
        return None;
    }
    let gap = distance - d_null;
    if gap >= 0.0 || gap_mitra == 0.0 || gap_tabicl == 0.0 {
        return None;
    }
    if gap.signum() != gap_mitra.signum() || gap.signum() != gap_tabicl.signum() {
        return None;
    }
    Some((1.0 - distance / d_match).clamp(0.0, 1.0))
}

pub fn artifact_has_cleartext(artifact: &[u8], forbidden: &[&[u8]]) -> bool {
    forbidden.iter().any(|needle| {
        !needle.is_empty()
            && artifact
                .windows(needle.len())
                .any(|window| window == *needle)
    })
}

pub fn headline_encoder(name: &str) -> bool {
    name == HEADLINE_ENCODER
}

/// True only when utility was measured by conditioning the clean tabular
/// models on synthetic rows and scoring them on real rows, under the frozen map.
pub fn tabular_transfer_ok(normalizer: &str, protocol: &str, auditors: &[String]) -> bool {
    normalizer == NORMALIZER
        && protocol == TABULAR_PROTOCOL
        && auditors.len() == TABULAR_AUDITORS.len()
        && TABULAR_AUDITORS
            .iter()
            .all(|name| auditors.iter().any(|got| got == name))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnManifest {
    pub dtype: u8,
    pub hash: String,
    pub role: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashManifest {
    pub byte_length: usize,
    pub columns: Vec<ColumnManifest>,
    pub encoder: String,
    pub normalizer: String,
    pub salt_id: String,
}

fn nearest_existing(path: &Path) -> io::Result<PathBuf> {
    let mut cursor = path.to_path_buf();
    loop {
        if cursor.exists() {
            return cursor.canonicalize();
        }
        if !cursor.pop() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "lookup path has no existing ancestor",
            ));
        }
    }
}

fn lookup_inside_repo(directory: &Path, repo_root: &Path) -> io::Result<bool> {
    let root = repo_root.canonicalize()?;
    if directory.starts_with(&root) {
        return Ok(true);
    }
    Ok(nearest_existing(directory)?.starts_with(&root))
}

/// Writes the salt and the hash-to-label map for the end user.
///
/// `directory` must sit outside `repo_root`. The file mode is `0o600`.
/// The public hash manifest does not receive this path or these labels.
pub fn write_local_lookup(
    directory: &Path,
    repo_root: &Path,
    salt: &[u8],
    columns: &[(&str, u8, u8)],
    categories: &[&str],
) -> io::Result<PathBuf> {
    if lookup_inside_repo(directory, repo_root)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "column lookup must stay outside the repository",
        ));
    }
    fs::create_dir_all(directory)?;
    let path = directory.join("mfs-v3-column-lookup.json");
    let mut column_rows = Vec::with_capacity(columns.len());
    for (name, dtype, role) in columns {
        column_rows.push(serde_json::json!({
            "dtype": dtype,
            "hash": column_hash(salt, name),
            "label": name,
            "role": role,
        }));
    }
    let mut category_rows = Vec::with_capacity(categories.len());
    for label in categories {
        category_rows.push(serde_json::json!({
            "hash": column_hash(salt, label),
            "label": label,
        }));
    }
    let body = serde_json::json!({
        "categories": category_rows,
        "columns": column_rows,
        "format": "dope-mfs-v3-column-lookup",
        "normalizer": NORMALIZER,
        "salt_hex": hex16(salt),
        "salt_id": salt_id(salt),
    });
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;
    file.write_all(&serde_json::to_vec(&body).expect("lookup serializes"))?;
    file.write_all(b"\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = file.metadata()?.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(&path, permissions)?;
    }
    Ok(path)
}

pub fn hash_manifest(
    salt: &[u8],
    columns: &[(&str, u8, u8)],
    encoder: &str,
    byte_length: usize,
) -> HashManifest {
    HashManifest {
        byte_length,
        columns: columns
            .iter()
            .map(|(name, dtype, role)| ColumnManifest {
                dtype: *dtype,
                hash: column_hash(salt, name),
                role: *role,
            })
            .collect(),
        encoder: encoder.to_string(),
        normalizer: NORMALIZER.to_string(),
        salt_id: salt_id(salt),
    }
}

include!("representation/tests.rs");
