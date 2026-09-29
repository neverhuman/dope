fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn current_hostname() -> Result<String> {
    let path = Path::new("/etc/hostname");
    let hostname = fs::read_to_string(path).map_err(|error| io_error(path, error))?;
    hostname
        .trim()
        .split('.')
        .next()
        .filter(|hostname| !hostname.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| DopeError::Data("host identity is empty".into()))
}

fn require_named_host(actual: &str, expected: &str, operation: &str) -> Result<()> {
    if actual != expected {
        return Err(DopeError::Data(format!(
            "{operation} is restricted to {expected}; current host is {actual}"
        )));
    }
    Ok(())
}

fn require_host(expected: &str, operation: &str) -> Result<()> {
    require_named_host(&current_hostname()?, expected, operation)
}

pub fn available_bytes(path: &Path) -> Result<u64> {
    let stats = rustix::fs::statvfs(path).map_err(|error| {
        io_error(path, std::io::Error::from_raw_os_error(error.raw_os_error()))
    })?;
    Ok(stats.f_bavail.saturating_mul(stats.f_frsize))
}

fn require_durable_space(path: &Path) -> Result<u64> {
    let bytes = available_bytes(path)?;
    if bytes < MIN_DURABLE_FREE_GIB * GIB {
        return Err(DopeError::Data(format!(
            "campaign admission stopped: durable free space is {:.2} GiB, below {MIN_DURABLE_FREE_GIB} GiB",
            bytes as f64 / GIB as f64
        )));
    }
    Ok(bytes)
}

fn is_lower_hex(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn load_receipt_key(path: &Path) -> Result<[u8; 32]> {
    let text = fs::read_to_string(path).map_err(|error| io_error(path, error))?;
    let text = text.trim();
    if !is_lower_hex(text, &[64]) {
        return Err(DopeError::Data(
            "receipt key file must contain exactly 32 lowercase hexadecimal bytes".into(),
        ));
    }
    let mut key = [0u8; 32];
    for (index, slot) in key.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| DopeError::Data("invalid receipt key".into()))?;
    }
    Ok(key)
}

pub fn create_receipt_key(path: &Path) -> Result<ContentHashes> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
    }
    let mut random = File::open("/dev/urandom")
        .map_err(|error| DopeError::Data(format!("cannot open OS randomness: {error}")))?;
    let mut key = [0u8; 32];
    random
        .read_exact(&mut key)
        .map_err(|error| DopeError::Data(format!("cannot read OS randomness: {error}")))?;
    let encoded = key
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| io_error(path, error))?;
    output
        .write_all(encoded.as_bytes())
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all())
        .map_err(|error| io_error(path, error))?;
    file_hashes(path)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CorpusSampleVerification {
    pub format: String,
    pub version: u8,
    pub manifest: ContentHashes,
    pub selection_numerator: usize,
    pub selection_denominator: usize,
    pub sampled_datasets: usize,
    pub sample_sha256: String,
    pub sample_blake3: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CorpusSamplePlan {
    pub format: String,
    pub version: u8,
    pub paths: Vec<PathBuf>,
}

pub fn corpus_sample_plan(manifest_path: &Path) -> Result<CorpusSamplePlan> {
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let mut paths = manifest
        .datasets
        .iter()
        .filter(|record| {
            let digest = blake3::hash(record.dataset_id.as_bytes());
            let mut prefix = [0u8; 8];
            prefix.copy_from_slice(&digest.as_bytes()[..8]);
            u64::from_le_bytes(prefix) % 100 == 0
        })
        .map(|record| record.path.join("train.csv"))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(CorpusSamplePlan {
        format: "dope-corpus-sample-plan".into(),
        version: 1,
        paths,
    })
}

pub fn verify_corpus_sample(manifest_path: &Path) -> Result<CorpusSampleVerification> {
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let mut selected = manifest
        .datasets
        .iter()
        .filter(|record| {
            let digest = blake3::hash(record.dataset_id.as_bytes());
            let mut prefix = [0u8; 8];
            prefix.copy_from_slice(&digest.as_bytes()[..8]);
            u64::from_le_bytes(prefix) % 100 == 0
        })
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    if selected.is_empty() {
        return Err(DopeError::Data(
            "deterministic corpus sample unexpectedly selected no datasets".into(),
        ));
    }
    let mut aggregate_sha = Sha256::new();
    let mut aggregate_blake = blake3::Hasher::new();
    for record in &selected {
        let path = record.path.join("train.csv");
        let mut file = File::open(&path).map_err(|error| io_error(&path, error))?;
        let mut file_sha = Sha256::new();
        let mut file_blake = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| io_error(&path, error))?;
            if read == 0 {
                break;
            }
            file_sha.update(&buffer[..read]);
            file_blake.update(&buffer[..read]);
        }
        let file_sha = file_sha.finalize();
        let file_blake = file_blake.finalize();
        let id = record.dataset_id.as_bytes();
        let id_len = (id.len() as u64).to_le_bytes();
        aggregate_sha.update(id_len);
        aggregate_sha.update(id);
        aggregate_sha.update(file_sha);
        aggregate_blake.update(&id_len);
        aggregate_blake.update(id);
        aggregate_blake.update(file_blake.as_bytes());
    }
    Ok(CorpusSampleVerification {
        format: "dope-corpus-sample-verification".into(),
        version: 1,
        manifest: file_hashes(manifest_path)?,
        selection_numerator: 1,
        selection_denominator: 100,
        sampled_datasets: selected.len(),
        sample_sha256: format!("{:x}", aggregate_sha.finalize()),
        sample_blake3: aggregate_blake.finalize().to_hex().to_string(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CohortRecord {
    pub dataset_id: String,
    pub dataset_path: PathBuf,
    pub task: String,
    pub rows: usize,
    pub features: usize,
    pub lineage_group_id: String,
    pub structural_profile: String,
    pub partition: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CohortPlan {
    pub format: String,
    pub version: u8,
    pub kind: String,
    pub records: Vec<CohortRecord>,
    #[serde(default)]
    pub exclusions: Vec<CohortExclusion>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct CohortExclusion {
    pub dataset_id: String,
    pub reason: String,
}

fn cohort_order(seed: u64, profile: &str, lineage: &str, dataset: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    for value in [profile, lineage, dataset] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Selects campaign cohorts without opening validation-cert. Candidates are
/// admitted only after their train table agrees with the frozen inventory;
/// malformed or stale inventory entries are retained as explicit exclusions.
pub fn plan_cohort(manifest_path: &Path, validation_path: &Path, kind: &str) -> Result<CohortPlan> {
    if !matches!(kind, "baseline" | "training-gold" | "validation-select") {
        return Err(DopeError::Data(
            "cohort kind must be baseline, training-gold, or validation-select".into(),
        ));
    }
    let manifest: SplitManifest = read_json(manifest_path)?;
    validate_split_manifest(&manifest)?;
    let validation: ValidationSubmanifest = read_json(validation_path)?;
    validate_validation_submanifest(&validation)?;
    if validation.source_manifest_checksum != manifest.checksum {
        return Err(DopeError::Data(
            "cohort manifests do not share a source checksum".into(),
        ));
    }
    let select = validation
        .assignments
        .iter()
        .filter(|assignment| assignment.partition == "validation-select")
        .map(|assignment| {
            (
                assignment.dataset_id.as_str(),
                assignment.lineage_group_id.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut canonical = BTreeMap::<(String, String), ([u8; 32], CohortRecord)>::new();
    for record in &manifest.datasets {
        if kind == "training-gold"
            && (record.origin == "mirror"
                || record.duplicate_of.is_some()
                || record.near_duplicate_of.is_some())
        {
            continue;
        }
        let (eligible, lineage, partition) = if kind == "training-gold" {
            (
                record.split == "train",
                record.group_id.as_str(),
                "training_gold",
            )
        } else {
            (
                record.split == "validation" && select.contains_key(record.dataset_id.as_str()),
                select
                    .get(record.dataset_id.as_str())
                    .copied()
                    .unwrap_or(record.group_id.as_str()),
                "validation_select",
            )
        };
        if !eligible {
            continue;
        }
        let features = record.cols.saturating_sub(1);
        let profile = StructuralProfile::from_shape(record.task, record.rows, features)?.id();
        let order = cohort_order(manifest.seed, &profile, lineage, &record.dataset_id);
        let value = CohortRecord {
            dataset_id: record.dataset_id.clone(),
            dataset_path: record.path.clone(),
            task: record.task.as_str().into(),
            rows: record.rows,
            features,
            lineage_group_id: lineage.into(),
            structural_profile: profile.clone(),
            partition: partition.into(),
        };
        let cohort_profile = if kind == "validation-select" {
            String::new()
        } else {
            profile
        };
        let entry = canonical
            .entry((cohort_profile, lineage.into()))
            .or_insert_with(|| (order, value.clone()));
        if order < entry.0 {
            *entry = (order, value);
        }
    }
    let mut by_profile = BTreeMap::<String, Vec<([u8; 32], CohortRecord)>>::new();
    for ((profile, _), value) in canonical {
        by_profile.entry(profile).or_default().push(value);
    }
    let mut records = Vec::new();
    let mut exclusions = Vec::new();
    let mut profile_batches = by_profile.into_iter().collect::<Vec<_>>();
    profile_batches.sort_by(|left, right| {
        left.1
            .len()
            .cmp(&right.1.len())
            .then_with(|| left.0.cmp(&right.0))
    });
    for (_, mut profile_records) in profile_batches {
        profile_records.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.dataset_id.cmp(&right.1.dataset_id))
        });
        let take = match kind {
            "baseline" => 1,
            "training-gold" => 120,
            _ => usize::MAX,
        };
        let mut admitted = 0usize;
        for (_, record) in profile_records {
            if admitted == take {
                break;
            }
            if kind == "validation-select" {
                records.push(record);
                admitted += 1;
                continue;
            }
            match Table::read_dataset_dir(&record.dataset_path, table_task(&record.task)?) {
                Ok(table) if table.rows == record.rows && table.features == record.features => {
                    records.push(record);
                    admitted += 1;
                }
                Ok(table) => exclusions.push(CohortExclusion {
                    dataset_id: record.dataset_id,
                    reason: format!(
                        "inventory shape {}/{} differs from train shape {}/{}",
                        record.rows, record.features, table.rows, table.features
                    ),
                }),
                Err(error) => exclusions.push(CohortExclusion {
                    dataset_id: record.dataset_id,
                    reason: error.to_string(),
                }),
            }
        }
    }
    records.sort_by(|left, right| {
        left.structural_profile
            .cmp(&right.structural_profile)
            .then_with(|| left.lineage_group_id.cmp(&right.lineage_group_id))
    });
    if kind == "validation-select"
        && records
            .iter()
            .map(|record| &record.lineage_group_id)
            .collect::<BTreeSet<_>>()
            .len()
            != validation.validation_select_lineage_groups
    {
        return Err(DopeError::Data(
            "canonical validation-select cohort lost required lineages".into(),
        ));
    }
    Ok(CohortPlan {
        format: "dope-campaign-cohort".into(),
        version: 1,
        kind: kind.into(),
        records,
        exclusions,
    })
}