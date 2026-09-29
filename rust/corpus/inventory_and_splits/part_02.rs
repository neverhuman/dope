

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
