
pub fn inventory_corpus(roots: &[PathBuf]) -> Result<CorpusInventory> {
    if roots.is_empty() {
        return Err(DopeError::Data(
            "at least one corpus root is required".into(),
        ));
    }
    let mut canonical_roots = Vec::with_capacity(roots.len());
    let mut datasets = Vec::new();
    let mut exclusions = Vec::new();
    let mut exact_seen = BTreeMap::<String, (String, String)>::new();
    let mut near_seen = BTreeMap::<String, (String, String)>::new();

    for (root_index, requested_root) in roots.iter().enumerate() {
        let root = requested_root
            .canonicalize()
            .map_err(|error| io_error(requested_root, error))?;
        let root_id = root_id(root_index, &root);
        canonical_roots.push(root.clone());
        for path in dataset_candidates(&root)? {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            let exclude = |reason: &str, detail: String| ExclusionRecord {
                root_id: root_id.clone(),
                path: path.clone(),
                reason: reason.into(),
                detail,
            };
            if name.starts_with('.') {
                exclusions.push(exclude("hidden", "hidden corpus entry".into()));
                continue;
            }
            if !path.is_dir() {
                exclusions.push(exclude(
                    "unsupported",
                    "corpus entry is not a directory".into(),
                ));
                continue;
            }
            if let Some(reason) = exclusion_reason(name) {
                exclusions.push(exclude(
                    reason,
                    "directory name matched exclusion policy".into(),
                ));
                continue;
            }
            let train_path = path.join("train.csv");
            if !train_path.is_file() {
                exclusions.push(exclude("incomplete", "missing train.csv".into()));
                continue;
            }
            let test_path = path.join("test.csv");
            if !test_path.is_file() {
                exclusions.push(exclude("incomplete", "missing test.csv".into()));
                continue;
            }
            let raw_train = match Table::read_csv(&train_path, Task::Regression) {
                Ok(table) => table,
                Err(error) => {
                    exclusions.push(exclude("malformed", error.to_string()));
                    continue;
                }
            };
            let raw_test = if test_path.is_file() {
                match Table::read_csv(&test_path, Task::Regression) {
                    Ok(table) if table.features == raw_train.features => Some(table),
                    Ok(_) => {
                        exclusions.push(exclude(
                            "malformed",
                            "train.csv and test.csv feature widths differ".into(),
                        ));
                        continue;
                    }
                    Err(error) => {
                        exclusions.push(exclude("malformed", error.to_string()));
                        continue;
                    }
                }
            } else {
                None
            };
            let task = inferred_task(&raw_train, raw_test.as_ref());
            let train = if task == Task::Binary {
                Table::read_csv(&train_path, task)?
            } else {
                raw_train
            };
            let test = if task == Task::Binary {
                raw_test
                    .as_ref()
                    .map(|_| Table::read_csv(&test_path, task))
                    .transpose()?
            } else {
                raw_test
            };
            let mut content_hasher = blake3::Hasher::new();
            content_hasher.update(b"train\0");
            content_hasher.update(&canonical_table_bytes(&train));
            if let Some(test) = &test {
                content_hasher.update(b"test\0");
                content_hasher.update(&canonical_table_bytes(test));
            }
            let content_fingerprint = content_hasher.finalize().to_hex().to_string();
            let (statistical_fingerprint, near_fingerprint) =
                statistical_fingerprints(&train, test.as_ref());
            let relative = path.strip_prefix(&root).unwrap_or(&path);
            let relative = if relative.as_os_str().is_empty() {
                ".".into()
            } else {
                relative.to_string_lossy().into_owned()
            };
            let dataset_id = if roots.len() == 1 {
                relative
            } else {
                format!("{root_id}/{relative}")
            };
            let mut group_id = metadata_group(&path, &content_fingerprint);
            let duplicate_of =
                exact_seen
                    .get(&content_fingerprint)
                    .map(|(dataset_id, prior_group)| {
                        group_id.clone_from(prior_group);
                        dataset_id.clone()
                    });
            let near_duplicate_of = if duplicate_of.is_none() {
                near_seen
                    .get(&near_fingerprint)
                    .map(|(dataset_id, prior_group)| {
                        group_id.clone_from(prior_group);
                        dataset_id.clone()
                    })
            } else {
                None
            };
            exact_seen
                .entry(content_fingerprint.clone())
                .or_insert_with(|| (dataset_id.clone(), group_id.clone()));
            near_seen
                .entry(near_fingerprint.clone())
                .or_insert_with(|| (dataset_id.clone(), group_id.clone()));
            datasets.push(DatasetRecord {
                dataset_id,
                root_id: root_id.clone(),
                path: path.clone(),
                task,
                origin: origin(&root, &path),
                shape_stratum: shape_stratum(train.rows, train.features),
                difficulty_stratum: difficulty_stratum(&train),
                rows: train.rows,
                test_rows: test.as_ref().map_or(0, |table| table.rows),
                cols: train.features + 1,
                content_fingerprint,
                statistical_fingerprint,
                near_fingerprint,
                group_id,
                duplicate_of,
                near_duplicate_of,
                split: String::new(),
            });
        }
    }
    datasets.sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    exclusions.sort_by(|left, right| left.path.cmp(&right.path));
    if datasets.is_empty() {
        return Err(DopeError::Data(
            "no eligible train.csv datasets found".into(),
        ));
    }
    let mut inventory = CorpusInventory {
        format: "dope-corpus-inventory".into(),
        version: 2,
        roots: canonical_roots,
        discovered_paths: datasets.len() + exclusions.len(),
        eligible_paths: datasets.len(),
        datasets,
        exclusions,
        checksum: String::new(),
    };
    inventory.checksum = inventory_checksum(&inventory)?;
    Ok(inventory)
}

pub fn load_inventory(path: &Path) -> Result<CorpusInventory> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    let inventory: CorpusInventory = serde_json::from_slice(&bytes)?;
    if inventory.format != "dope-corpus-inventory"
        || inventory.version != 2
        || inventory.checksum != inventory_checksum(&inventory)?
    {
        return Err(DopeError::Data(
            "inventory checksum or version invalid".into(),
        ));
    }
    Ok(inventory)
}

pub fn refresh_inventory_lineages(inventory: &CorpusInventory) -> Result<CorpusInventory> {
    if inventory.format != "dope-corpus-inventory"
        || inventory.version != 2
        || inventory.checksum != inventory_checksum(inventory)?
    {
        return Err(DopeError::Data(
            "inventory checksum or version invalid".into(),
        ));
    }
    let mut refreshed = inventory.clone();
    refreshed
        .datasets
        .sort_by(|left, right| left.dataset_id.cmp(&right.dataset_id));
    let mut exact_seen = BTreeMap::<String, (String, String)>::new();
    let mut near_seen = BTreeMap::<String, (String, String)>::new();
    for record in &mut refreshed.datasets {
        let mut group_id = metadata_group(&record.path, &record.content_fingerprint);
        record.duplicate_of =
            exact_seen
                .get(&record.content_fingerprint)
                .map(|(dataset_id, prior_group)| {
                    group_id.clone_from(prior_group);
                    dataset_id.clone()
                });
        record.near_duplicate_of = if record.duplicate_of.is_none() {
            near_seen
                .get(&record.near_fingerprint)
                .map(|(dataset_id, prior_group)| {
                    group_id.clone_from(prior_group);
                    dataset_id.clone()
                })
        } else {
            None
        };
        exact_seen
            .entry(record.content_fingerprint.clone())
            .or_insert_with(|| (record.dataset_id.clone(), group_id.clone()));
        near_seen
            .entry(record.near_fingerprint.clone())
            .or_insert_with(|| (record.dataset_id.clone(), group_id.clone()));
        record.group_id = group_id;
        record.split.clear();
    }
    refreshed.checksum.clear();
    refreshed.checksum = inventory_checksum(&refreshed)?;
    Ok(refreshed)
}

fn split_hash(seed: u64, stratum: &str, group: &str) -> u64 {
    u64::from_le_bytes(
        blake3::hash(format!("{seed}:{stratum}:{group}").as_bytes()).as_bytes()[..8]
            .try_into()
            .expect("eight-byte hash prefix"),
    )
}

pub fn split_inventory(inventory: &CorpusInventory, seed: u64) -> Result<SplitManifest> {
    if inventory.format != "dope-corpus-inventory"
        || inventory.version != 2
        || inventory.checksum != inventory_checksum(inventory)?
    {
        return Err(DopeError::Data(
            "inventory checksum or version invalid".into(),
        ));
    }
    let mut group_strata = BTreeMap::<String, String>::new();
    let mut group_weights = BTreeMap::<String, usize>::new();
    for record in &inventory.datasets {
        *group_weights.entry(record.group_id.clone()).or_default() += 1;
        group_strata
            .entry(record.group_id.clone())
            .or_insert_with(|| {
                format!(
                    "{}:{}:{}:{}",
                    record.task.as_str(),
                    record.origin,
                    record.shape_stratum,
                    record.difficulty_stratum
                )
            });
    }
    let mut assignments = BTreeMap::<String, String>::new();
    let mut strata = BTreeMap::<String, Vec<String>>::new();
    for (group, stratum) in &group_strata {
        strata
            .entry(stratum.clone())
            .or_default()
            .push(group.clone());
    }
    const SPLITS: [&str; 3] = ["train", "validation", "test"];
    const FRACTIONS: [f64; 3] = [0.70, 0.15, 0.15];
    for (stratum, mut groups) in strata {
        groups.sort_by(|left, right| {
            group_weights[right]
                .cmp(&group_weights[left])
                .then_with(|| {
                    split_hash(seed, &stratum, left).cmp(&split_hash(seed, &stratum, right))
                })
                .then_with(|| left.cmp(right))
        });
        let total = groups
            .iter()
            .map(|group| group_weights[group])
            .sum::<usize>() as f64;
        let targets = FRACTIONS.map(|fraction| fraction * total);
        let mut assigned = [0.0f64; 3];
        for group in groups {
            let weight = group_weights[&group] as f64;
            let rotation = (split_hash(seed ^ 0x9e37_79b9, &stratum, &group) % 3) as usize;
            let split_index = (0..3)
                .min_by(|left, right| {
                    let cost = |candidate: usize| {
                        (0..3)
                            .map(|index| {
                                let next =
                                    assigned[index] + if index == candidate { weight } else { 0.0 };
                                (next - targets[index]).powi(2)
                            })
                            .sum::<f64>()
                    };
                    cost(*left).total_cmp(&cost(*right)).then_with(|| {
                        ((*left + 3 - rotation) % 3).cmp(&((*right + 3 - rotation) % 3))
                    })
                })
                .expect("three split candidates");
            assigned[split_index] += weight;
            assignments.insert(group, SPLITS[split_index].into());
        }
    }
    if assignments.len() == 1 {
        let group = assignments.keys().next().expect("one assignment").clone();
        assignments.insert(group, "train".into());
    } else if assignments.len() == 2 {
        let mut groups: Vec<_> = group_strata.keys().cloned().collect();
        groups.sort_by_key(|group| split_hash(seed, &group_strata[group], group));
        assignments.insert(groups[0].clone(), "train".into());
        assignments.insert(groups[1].clone(), "validation".into());
    } else if assignments.len() >= 3 {
        for required in ["train", "validation", "test"] {
            if !assignments.values().any(|split| split == required) {
                let candidate = group_strata
                    .iter()
                    .filter(|(group, _)| {
                        assignments
                            .values()
                            .filter(|split| *split == &assignments[*group])
                            .count()
                            > 1
                    })
                    .min_by_key(|(group, stratum)| split_hash(seed ^ 0xa5a5, stratum, group))
                    .map(|(group, _)| group.clone())
                    .ok_or_else(|| DopeError::Data("unable to form non-empty splits".into()))?;
                assignments.insert(candidate, required.into());
            }
        }
    }
    let mut datasets = inventory.datasets.clone();
    for record in &mut datasets {
        record.split.clone_from(&assignments[&record.group_id]);
    }
    let mut manifest = SplitManifest {
        format: "dope-corpus-split".into(),
        version: 2,
        seed,
        policy: "exact-and-quantile-rank-lsh-deduplicated-lineage-weighted-stratified-70-15-15"
            .into(),
        inventory_checksum: inventory.checksum.clone(),
        sealed_test: true,
        datasets,
        checksum: String::new(),
    };
    manifest.checksum = manifest_checksum(&manifest)?;
    validate_split_manifest(&manifest)?;
    Ok(manifest)
}

pub fn build_split_manifest_multi(
    roots: &[PathBuf],
    seed: u64,
) -> Result<(CorpusInventory, SplitManifest)> {
    let inventory = inventory_corpus(roots)?;
    let manifest = split_inventory(&inventory, seed)?;
    Ok((inventory, manifest))
}

pub fn build_split_manifest(root: &Path, seed: u64) -> Result<SplitManifest> {
    let (_, manifest) = build_split_manifest_multi(&[root.to_path_buf()], seed)?;
    Ok(manifest)
}

pub fn validate_split_manifest(manifest: &SplitManifest) -> Result<()> {
    if manifest.format != "dope-corpus-split"
        || manifest.version != 2
        || manifest.checksum != manifest_checksum(manifest)?
    {
        return Err(DopeError::Data(
            "split manifest checksum or version invalid".into(),
        ));
    }
    let mut groups = BTreeMap::new();
    let mut content = BTreeMap::new();
    let mut near = BTreeMap::new();
    let valid_splits: BTreeSet<&str> = ["train", "validation", "test"].into_iter().collect();
    for record in &manifest.datasets {
        if !valid_splits.contains(record.split.as_str()) {
            return Err(DopeError::Data("unknown split assignment".into()));
        }
        if groups
            .insert(record.group_id.clone(), record.split.clone())
            .is_some_and(|prior| prior != record.split)
        {
            return Err(DopeError::Data("lineage leakage across splits".into()));
        }
        if content
            .insert(record.content_fingerprint.clone(), record.split.clone())
            .is_some_and(|prior| prior != record.split)
        {
            return Err(DopeError::Data("duplicate leakage across splits".into()));
        }
        if near
            .insert(record.near_fingerprint.clone(), record.split.clone())
            .is_some_and(|prior| prior != record.split)
        {
            return Err(DopeError::Data(
                "near-duplicate leakage across splits".into(),
            ));
        }
    }
    Ok(())
}

pub fn write_inventory(inventory: &CorpusInventory, path: &Path) -> Result<()> {
    if inventory.checksum != inventory_checksum(inventory)? {
        return Err(DopeError::Data("inventory checksum invalid".into()));
    }
    fs::write(path, serde_json::to_vec_pretty(inventory)?).map_err(|error| io_error(path, error))
}

pub fn write_exclusions(inventory: &CorpusInventory, path: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    for exclusion in &inventory.exclusions {
        serde_json::to_writer(&mut bytes, exclusion)?;
        bytes.push(b'\n');
    }
    fs::write(path, bytes).map_err(|error| io_error(path, error))
}

pub fn write_split_manifest(manifest: &SplitManifest, path: &Path) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(manifest)?).map_err(|error| io_error(path, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn multi_root_inventory_accounts_for_every_immediate_child() {
        let root = std::env::temp_dir().join(format!("dope-inventory-{}", std::process::id()));
        let first = root.join("first/good");
        let duplicate = root.join("second/mirror");
        let excluded = root.join("first/cache-copy");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&duplicate).unwrap();
        fs::create_dir_all(&excluded).unwrap();
        for path in [&first, &duplicate] {
            let mut train = fs::File::create(path.join("train.csv")).unwrap();
            writeln!(train, "0,1,0").unwrap();
            writeln!(train, "1,0,1").unwrap();
            let mut test = fs::File::create(path.join("test.csv")).unwrap();
            writeln!(test, "1,1,1").unwrap();
        }
        let roots = vec![root.join("first"), root.join("second")];
        let inventory = inventory_corpus(&roots).unwrap();
        assert_eq!(inventory.discovered_paths, 3);
        assert_eq!(inventory.datasets.len(), 2);
        assert_eq!(inventory.exclusions.len(), 1);
        assert!(inventory.datasets[1].duplicate_of.is_some());
        let manifest = split_inventory(&inventory, 1729).unwrap();
        validate_split_manifest(&manifest).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn weighted_split_hits_requested_ratio_for_independent_groups() {
        let template = DatasetRecord {
            dataset_id: String::new(),
            root_id: "0:test".into(),
            path: PathBuf::from("/tmp/test"),
            task: Task::Regression,
            origin: "synthetic".into(),
            shape_stratum: "r9-p4".into(),
            difficulty_stratum: "medium-dense".into(),
            rows: 512,
            test_rows: 256,
            cols: 17,
            content_fingerprint: String::new(),
            statistical_fingerprint: String::new(),
            near_fingerprint: String::new(),
            group_id: String::new(),
            duplicate_of: None,
            near_duplicate_of: None,
            split: String::new(),
        };
        let datasets = (0..100)
            .map(|index| DatasetRecord {
                dataset_id: format!("dataset-{index:03}"),
                content_fingerprint: format!("content-{index:03}"),
                statistical_fingerprint: format!("statistics-{index:03}"),
                near_fingerprint: format!("near-{index:03}"),
                group_id: format!("group-{index:03}"),
                ..template.clone()
            })
            .collect();
        let mut inventory = CorpusInventory {
            format: "dope-corpus-inventory".into(),
            version: 2,
            roots: vec![PathBuf::from("/tmp")],
            datasets,
            exclusions: Vec::new(),
            discovered_paths: 100,
            eligible_paths: 100,
            checksum: String::new(),
        };
        inventory.checksum = inventory_checksum(&inventory).unwrap();
        let manifest = split_inventory(&inventory, 1729).unwrap();
        let mut counts = BTreeMap::<String, usize>::new();
        for record in manifest.datasets {
            *counts.entry(record.split).or_default() += 1;
        }
        assert_eq!(counts["train"], 70);
        assert_eq!(counts["validation"], 15);
        assert_eq!(counts["test"], 15);
    }
}
