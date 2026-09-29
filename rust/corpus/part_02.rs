

fn metadata_group(path: &Path, default_group_key: &str) -> String {
    let metadata = fs::read(path.join("meta.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    if let Some(value) = metadata {
        let keys = [
            "source",
            "real_source",
            "source_path",
            "source_file",
            "source_real_data",
            "generator",
            "generator_name",
            "version",
            "generator_version",
            "recipe_family",
            "recipe",
            "generator_recipe",
            "lineage",
            "lineage_fingerprint",
        ];
        let selected: BTreeMap<_, _> = keys
            .into_iter()
            .filter_map(|key| value.get(key).map(|entry| (key, scrub_seed_fields(entry))))
            .collect();
        if !selected.is_empty() {
            return blake3::hash(&serde_json::to_vec(&selected).expect("JSON values serialize"))
                .to_hex()
                .to_string();
        }
    }
    blake3::hash(default_group_key.as_bytes())
        .to_hex()
        .to_string()
}

fn origin(root: &Path, path: &Path) -> String {
    let name = format!("{} {}", root.display(), path.display()).to_ascii_lowercase();
    if name.contains("remote_super") {
        "mirror".into()
    } else if name.contains("synthetic") {
        "synthetic".into()
    } else {
        "real".into()
    }
}

fn shape_stratum(rows: usize, features: usize) -> String {
    let row_bucket = rows.max(1).ilog2().min(31);
    let feature_bucket = features.max(1).ilog2().min(15);
    format!("r{row_bucket}-p{feature_bucket}")
}

fn difficulty_stratum(table: &Table) -> String {
    let signal = table
        .columns
        .iter()
        .map(|column| correlation(column, &table.target).abs())
        .fold(0.0f32, f32::max);
    let missing = table
        .columns
        .iter()
        .flatten()
        .filter(|value| value.is_nan())
        .count() as f64
        / (table.rows * table.features).max(1) as f64;
    let signal_bucket = if signal < 0.1 {
        "hard"
    } else if signal < 0.4 {
        "medium"
    } else {
        "easy"
    };
    let missing_bucket = if missing >= 0.1 { "missing" } else { "dense" };
    format!("{signal_bucket}-{missing_bucket}")
}

fn root_id(index: usize, root: &Path) -> String {
    let name = root
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("root");
    format!("{index}:{name}")
}

fn dataset_candidates(root: &Path) -> Result<Vec<PathBuf>> {
    if root.join("train.csv").is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut candidates = Vec::new();
    for entry in fs::read_dir(root).map_err(|error| io_error(root, error))? {
        let entry = entry.map_err(|error| io_error(root, error))?;
        candidates.push(entry.path());
    }
    candidates.sort();
    Ok(candidates)
}

include!("inventory_and_splits.rs");
