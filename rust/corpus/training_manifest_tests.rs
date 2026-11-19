use super::*;
use std::io::Write;

fn hex(fill: u8) -> String {
    format!("{fill:02x}").repeat(32)
}

fn train(id: &str, rows: Vec<String>, families: &[&str]) -> TrainingBase {
    TrainingBase {
        base_dataset_id: id.into(),
        physical_rows_sha256: rows,
        source_family_ids: families
            .iter()
            .map(|family| (*family).to_string())
            .collect(),
        ancestor_base_ids: Vec::new(),
        origin: Origin::Real,
        generator: None,
        role: SourceRole::Train,
        task: TaskKind::Regression,
        target_declaration_sha256: hex(0x11),
        view_of: None,
    }
}

#[test]
fn legacy_inventory_does_not_read_protected_bytes() {
    let root = Path::new("target/training-manifest").join(format!("{}", std::process::id()));
    let public = root.join("public");
    let sealed = root.join("sealed");
    fs::create_dir_all(&public).unwrap();
    fs::create_dir_all(&sealed).unwrap();
    for (path, body) in [
        (public.join("train.csv"), "0,1,0\n"),
        (public.join("test.csv"), "1,0,1\n"),
        (sealed.join("train.csv"), "SECRET_LABEL,1\n"),
        (sealed.join("test.csv"), "SECRET_LABEL,0\n"),
    ] {
        let mut file = fs::File::create(path).unwrap();
        write!(file, "{body}").unwrap();
    }
    fs::write(
        sealed.join(MARKER_FILE),
        r#"{"schema":"dope-training-base/v1","role":"CONFIRM"}"#,
    )
    .unwrap();
    let inventory = super::super::inventory_corpus(&[root.clone()]).unwrap();
    assert_eq!(inventory.datasets.len(), 1);
    assert_eq!(inventory.exclusions.len(), 1);
    assert_eq!(inventory.exclusions[0].reason, "protected");
    assert_eq!(
        inventory.exclusions[0].detail,
        "training_manifest_protected"
    );
    let rendered = format!("{inventory:?}");
    assert!(!rendered.contains("SECRET_LABEL"));
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn malformed_marker_refuses_before_table_bytes() {
    let root = Path::new("target/training-manifest-bad").join(format!("{}", std::process::id()));
    let sealed = root.join("sealed");
    fs::create_dir_all(&sealed).unwrap();
    fs::write(sealed.join("train.csv"), "SECRET_LABEL,1\n").unwrap();
    fs::write(sealed.join("test.csv"), "SECRET_LABEL,0\n").unwrap();
    fs::write(sealed.join(MARKER_FILE), "{").unwrap();
    let error = super::super::inventory_corpus(&[root.clone()]).unwrap_err();
    let rendered = error.to_string();
    assert!(rendered.contains("training_manifest_schema"));
    assert!(!rendered.contains("SECRET_LABEL"));
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn derived_family_alias_is_refused() {
    let mut protected = train("secret-source", vec![hex(1)], &["family-secret"]);
    protected.role = SourceRole::Confirm;
    let mut renamed = train("public-name", vec![hex(2)], &["family-secret"]);
    renamed.ancestor_base_ids = vec!["secret-source".into()];
    let mut derived = train("derived", vec![hex(3)], &["other-family"]);
    derived.ancestor_base_ids = vec!["secret-source".into()];
    let copied = train("copied-bytes", vec![hex(1)], &["unrelated"]);
    let admission = admit(&[protected, renamed, derived, copied]).unwrap();
    assert_eq!(admission.accepted_base_count, 0);
    assert_eq!(admission.rejected.len(), 3);
    assert!(
        admission
            .rejected
            .iter()
            .all(|rejection| rejection.code == "training_manifest_ancestry")
    );
}

#[test]
fn reordering_does_not_increment_base_count() {
    let first = train("base", vec![hex(1), hex(2)], &["family-a"]);
    let mut reordered = train("reordered", vec![hex(2), hex(1)], &["family-a"]);
    reordered.target_declaration_sha256 = hex(0x22);
    let mut view = train("view", vec![hex(2), hex(1)], &["family-a"]);
    view.view_of = Some("base".into());
    view.target_declaration_sha256 = hex(0x33);
    let admission = admit(&[first, reordered, view]).unwrap();
    assert_eq!(admission.accepted_base_count, 1);
    assert_eq!(admission.accepted_base_ids, vec!["base".to_string()]);
    assert_eq!(admission.rejected.len(), 1);
    assert_eq!(admission.rejected[0].code, "training_manifest_duplicate");
    assert_eq!(
        physical_base_identity(&[hex(1), hex(2)]).unwrap(),
        physical_base_identity(&[hex(2), hex(1)]).unwrap()
    );
}

#[test]
fn seed_collision_is_rejected_by_realized_content() {
    let mut first = train("draw-a", vec![hex(4), hex(5)], &["mech"]);
    first.origin = Origin::StructuralSynthetic;
    first.generator = Some(Generator {
        mechanism_id: "generator-a".into(),
        seed: 1,
    });
    let mut second = train("draw-b", vec![hex(5), hex(4)], &["mech"]);
    second.origin = Origin::StructuralSynthetic;
    second.generator = Some(Generator {
        mechanism_id: "generator-a".into(),
        seed: 2,
    });
    let mut distinct = train("draw-c", vec![hex(6)], &["mech"]);
    distinct.origin = Origin::StructuralSynthetic;
    distinct.generator = Some(Generator {
        mechanism_id: "generator-a".into(),
        seed: 2,
    });
    let admission = admit(&[first, second, distinct]).unwrap();
    assert_eq!(admission.accepted_base_count, 2);
    assert_eq!(
        admission.rejected,
        vec![Rejection {
            base_dataset_id: "draw-b".into(),
            code: "training_manifest_seed",
        }]
    );
}

#[test]
fn dev_examples_and_marker_symlinks_stay_out_of_train() {
    let mut dev = train("dev-source", vec![hex(7)], &["family-dev"]);
    dev.role = SourceRole::Dev;
    let family = train("family-alias", vec![hex(8)], &["family-dev"]);
    let mut child = train("child", vec![hex(9)], &["family-train"]);
    child.ancestor_base_ids = vec!["dev-source".into()];
    let copied = train("copied", vec![hex(7)], &["family-other"]);
    let parent = train("parent", vec![hex(10)], &["family-train"]);
    let mut child_ok = train("child-ok", vec![hex(11)], &["family-train"]);
    child_ok.ancestor_base_ids = vec!["parent".into()];
    let admission = admit(&[dev, family, child, copied, parent, child_ok]).unwrap();
    assert_eq!(
        admission.accepted_base_ids,
        vec!["parent".to_string(), "child-ok".to_string()]
    );
    assert_eq!(admission.rejected.len(), 3);
    assert!(
        admission
            .rejected
            .iter()
            .all(|rejection| rejection.code == "training_manifest_ancestry")
    );

    let root = Path::new("target/training-manifest-link").join(format!("{}", std::process::id()));
    let sealed = root.join("sealed");
    fs::create_dir_all(&sealed).unwrap();
    fs::write(sealed.join("train.csv"), "SECRET_LABEL,1\n").unwrap();
    fs::write(sealed.join("test.csv"), "SECRET_LABEL,0\n").unwrap();
    std::os::unix::fs::symlink("missing-marker", sealed.join(MARKER_FILE)).unwrap();
    let error = super::super::inventory_corpus(&[root.clone()]).unwrap_err();
    let rendered = error.to_string();
    assert!(rendered.contains("training_manifest_schema"));
    assert!(!rendered.contains("SECRET_LABEL"));

    let follow_root = root.join("follow-root");
    let linked = follow_root.join("linked");
    fs::create_dir_all(&linked).unwrap();
    fs::write(linked.join("train.csv"), "SECRET_LABEL,1\n").unwrap();
    fs::write(linked.join("test.csv"), "SECRET_LABEL,0\n").unwrap();
    let real = root.join("real-marker.json");
    fs::write(
        &real,
        r#"{"schema":"dope-training-base/v1","role":"CONFIRM"}"#,
    )
    .unwrap();
    std::os::unix::fs::symlink(&real, linked.join(MARKER_FILE)).unwrap();
    let followed = super::super::inventory_corpus(&[follow_root]).unwrap_err();
    let followed_text = followed.to_string();
    assert!(followed_text.contains("training_manifest_schema"));
    assert!(!followed_text.contains("SECRET_LABEL"));
    fs::remove_dir_all(&root).unwrap();
}
