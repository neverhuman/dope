#[test]
fn completed_receipt_reverifies_atomic_cache_and_corruption() {
    let f = Fixture::new();
    let v = f.verify().unwrap();
    let cache = f.root.join("cache");
    commit_completed_receipt(&cache, &v).unwrap();
    assert_eq!(
        fs::metadata(&cache).unwrap().permissions().mode() & 0o077,
        0
    );
    assert_eq!(
        fs::metadata(cache.join(format!(
            "{}.json",
            scientific_cache_key(v.identity()).unwrap()
        )))
        .unwrap()
        .permissions()
        .mode()
            & 0o077,
        0
    );
    assert_eq!(
        load_completed_receipt(&cache, &f.root, &f.policy())
            .unwrap()
            .receipt_hash(),
        v.receipt_hash()
    );
    fs::write(
        f.root
            .join(text(&f.receipt["dependencies"][0]["path"]).unwrap()),
        b"corrupt",
    )
    .unwrap();
    assert!(load_completed_receipt(&cache, &f.root, &f.policy()).is_err());
}

#[test]
fn full_identity_changes_cache_attempts_and_aliases_do_not_change_fit() {
    let mut f = Fixture::new();
    let first = f.verify().unwrap();
    let key = scientific_cache_key(first.identity()).unwrap();
    assert_eq!(
        key,
        "4bfc81f998d7d6a907b60e44948a4f211c5aa84f2ed865f731fb7fb253e8952e"
    );
    assert_eq!(
        fit_replicate_key(first.identity()).unwrap(),
        "6774caa4afc0fcad64da7139d288b94a849ecc786c7a0cfa45112e36175da5a1"
    );
    for (field, value) in first.identity().value().as_object().unwrap() {
        let mut changed = first.identity().value().clone();
        changed[field] = if field.ends_with("_hash") {
            "f".repeat(64).into()
        } else if let Some(n) = value.as_u64() {
            (n + 1).into()
        } else {
            match field.as_str() {
                "partition_role" => "exploratory",
                "precision" => "float32",
                _ => "seeded_best_effort",
            }
            .into()
        };
        assert_ne!(
            key,
            scientific_cache_key(&EvaluationIdentity::new(changed).unwrap()).unwrap(),
            "{field}"
        );
    }
    f.receipt["attempt"]["attempt_id"] = "b".repeat(64).into();
    for n in f.receipt["dependencies"].as_array_mut().unwrap() {
        let old = text(&n["path"]).unwrap();
        let alias = format!("alias-{old}");
        fs::hard_link(f.root.join(old), f.root.join(&alias)).unwrap();
        n["path"] = alias.into();
    }
    let after = f.verify().unwrap();
    assert_eq!(key, scientific_cache_key(after.identity()).unwrap());
    assert_eq!(
        fit_replicate_key(first.identity()).unwrap(),
        fit_replicate_key(after.identity()).unwrap()
    );
    assert_ne!(first.receipt_hash(), after.receipt_hash());
}
