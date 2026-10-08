//! Frozen real-train normalization and the clean-encoder distance for MFS-v3.
//!
//! The quantile map is fit on real fit rows only. Column names and category
//! levels become HMAC digests. The shipped manifest stores those digests, not
//! the source strings.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub const NORMALIZER: &str = "empirical_midrank_quantile_v1";
pub const HEADLINE_ENCODER: &str = "kumo_tabular_l";
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
    Some(
        values
            .iter()
            .map(|value| midrank_one(&ordered, *value, n))
            .collect(),
    )
}

fn midrank_one(ordered: &[f64], value: f64, n: f64) -> f64 {
    let less = ordered.partition_point(|item| *item < value);
    let right = ordered.partition_point(|item| *item <= value);
    if right == 0 {
        return 0.5 / n;
    }
    if less == ordered.len() {
        return (n - 0.5) / n;
    }
    if less == right {
        let left_image = (less as f64 - 0.5) / n;
        let right_image = (less as f64 + 0.5) / n;
        let x0 = ordered[less - 1];
        let x1 = ordered[less];
        if x1 == x0 {
            return left_image;
        }
        let t = (value - x0) / (x1 - x0);
        return left_image + t * (right_image - left_image);
    }
    let mut total = 0.0;
    for index in less..right {
        total += (index as f64 + 0.5) / n;
    }
    total / (right - less) as f64
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
    if left_norm == 0.0 || right_norm == 0.0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231_case_1() {
        let key = [0x0bu8; 20];
        let digest = hmac_sha256(&key, b"Hi There");
        assert_eq!(
            hex16(&digest),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn midrank_clamps_inside_the_open_unit_interval() {
        let fit = [1.0, 2.0, 4.0];
        let got = midrank_quantile(&fit, &[0.0, 1.0, 3.0, 4.0, 9.0]).unwrap();
        let expect = [0.5 / 3.0, 0.5 / 3.0, 2.0 / 3.0, 2.5 / 3.0, 2.5 / 3.0];
        for (actual, expected) in got.iter().zip(expect) {
            assert!((actual - expected).abs() < 1e-12);
            assert!(*actual > 0.0 && *actual < 1.0);
        }
    }

    #[test]
    fn synthetic_column_uses_the_real_fit() {
        let real = [0.0, 1.0, 2.0];
        let synthetic = [10.0, 11.0, 12.0];
        let frozen = midrank_quantile(&real, &synthetic).unwrap();
        let refit = midrank_quantile(&synthetic, &synthetic).unwrap();
        let upper = 2.5 / 3.0;
        assert_ne!(frozen, refit);
        assert!(frozen.iter().all(|value| (*value - upper).abs() < 1e-12));
        assert!(frozen.iter().all(|value| *value < 1.0));
    }

    #[test]
    fn manifest_keeps_hashes_and_drops_source_text() {
        let manifest = hash_manifest(
            b"salt",
            &[("age", 1, 1), ("red", 2, 1)],
            HEADLINE_ENCODER,
            100,
        );
        let encoded = format!("{manifest:?}");
        assert!(!encoded.contains("age"));
        assert!(!encoded.contains("red"));
        assert_eq!(manifest.columns[0].hash.len(), 16);
        assert_eq!(manifest.salt_id.len(), 16);
        assert!(!artifact_has_cleartext(
            encoded.as_bytes(),
            &[b"age", b"red"]
        ));
    }

    #[test]
    fn cleartext_scan_catches_a_header_byte_string() {
        assert!(artifact_has_cleartext(b"col,age\n1", &[b"age"]));
        assert!(!artifact_has_cleartext(b"col,a1e\n1", &[b"age"]));
    }

    #[test]
    fn closeness_rejects_a_zero_match_and_a_sign_flip() {
        assert!(representation_closeness(0.2, 0.5, 0.0, -0.1, -0.1).is_none());
        assert!(representation_closeness(0.2, 0.5, 0.4, 0.1, -0.1).is_none());
        let value = representation_closeness(0.2, 0.5, 0.4, -0.1, -0.2).unwrap();
        assert!((value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn lookup_stays_outside_the_repo_and_holds_the_label() {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let refused = write_local_lookup(
            &repo.join("production"),
            &repo,
            b"salt",
            &[("age", 1, 1)],
            &["red"],
        );
        assert!(refused.is_err());
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        // Self-hosted runners set the process temp directory inside the checkout.
        // The lookup has to be created somewhere the guard accepts.
        let tmp = ["/dev/shm", "/var/tmp", "/tmp"]
            .into_iter()
            .map(|base| PathBuf::from(base).join(format!("dope-mfs-v3-lookup-{stamp}")))
            .chain(std::iter::once(
                std::env::temp_dir().join(format!("dope-mfs-v3-lookup-{stamp}")),
            ))
            .find(|dir| lookup_inside_repo(dir, &repo).ok() == Some(false))
            .expect("a directory outside the repository");
        fs::create_dir_all(&tmp).unwrap();
        let path = write_local_lookup(&tmp, &repo, b"salt", &[("age", 1, 1)], &["red"]).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        let lookup = fs::read(&path).unwrap();
        assert!(artifact_has_cleartext(&lookup, &[b"age", b"red"]));
        let fixture = tmp.join("artifact.csv");
        fs::write(&fixture, b"header,age\n1").unwrap();
        assert!(artifact_has_cleartext(
            &fs::read(&fixture).unwrap(),
            &[b"age"]
        ));
        let manifest = hash_manifest(b"salt", &[("age", 1, 1)], HEADLINE_ENCODER, 100);
        let encoded = format!("{manifest:?}");
        assert!(!artifact_has_cleartext(
            encoded.as_bytes(),
            &[b"age", b"red"]
        ));
        assert!(!encoded.contains(&path.display().to_string()));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn tabular_transfer_requires_the_clean_models_and_the_shared_map() {
        let auditors: Vec<String> = TABULAR_AUDITORS
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert!(tabular_transfer_ok(NORMALIZER, TABULAR_PROTOCOL, &auditors));
        assert!(!tabular_transfer_ok(
            NORMALIZER,
            "auditor_retention",
            &auditors
        ));
        assert!(!tabular_transfer_ok(
            NORMALIZER,
            TABULAR_PROTOCOL,
            &["catboost".into()]
        ));
    }

    #[test]
    fn refused_encoder_is_not_the_headline() {
        assert!(headline_encoder(HEADLINE_ENCODER));
        assert!(!headline_encoder("foundation"));
        assert!(!headline_encoder("tabpfn35"));
        assert!(REFUSED_ENCODERS.contains(&"foundation"));
        assert!(SENSITIVITY_ENCODERS.contains(&"mitra_v2"));
    }
}
