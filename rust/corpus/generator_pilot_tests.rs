use super::*;

fn plan(bases: usize, arms: &[(&str, u32)], task: &str, views: usize, role: SourceRole) -> Plan {
    Plan {
        seed: 11,
        shard_id: 3,
        bases,
        arms: arms
            .iter()
            .map(|(mechanism_id, weight)| Arm {
                mechanism_id: (*mechanism_id).to_string(),
                weight: *weight,
            })
            .collect(),
        task: task.to_string(),
        views_per_base: views,
        ancestor_role: role,
    }
}

#[test]
fn pilot_counts_one_hundred_thousand_bases_exactly() {
    let report = run(&plan(
        PILOT_BASES,
        &[("independent", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap();
    assert_eq!(report.schema, SCHEMA);
    assert_eq!(report.requested_bases, PILOT_BASES);
    assert_eq!(report.accepted_base_count, PILOT_BASES);
    assert_eq!(report.conditioning_base_count, 0);
    assert_eq!(report.requested_views, 0);
    assert_eq!(report.view_count, 0);
    assert!(report.rejected.is_empty());
    assert_eq!(
        report.mechanism_counts.get("independent").copied(),
        Some(PILOT_BASES)
    );
    assert_eq!(
        report.accepted_base_count,
        report.mechanism_counts.values().sum::<usize>()
    );
    let again = run(&plan(
        PILOT_BASES,
        &[("independent", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap();
    assert_eq!(report.content_sha256, again.content_sha256);
    assert_eq!(report.sample_row_sha256, again.sample_row_sha256);
}

#[test]
fn views_do_not_inflate_the_base_count() {
    let report = run(&plan(
        4,
        &[("independent", 1)],
        "binary",
        2,
        SourceRole::Train,
    ))
    .unwrap();
    assert_eq!(report.requested_bases, 4);
    assert_eq!(report.accepted_base_count, 4);
    assert_eq!(report.requested_views, 8);
    assert_eq!(report.view_count, 8);
    assert!(report.rejected.is_empty());
    assert_eq!(report.conditioning_base_count, 0);
}

#[test]
fn non_train_ancestor_is_rejected_and_counted() {
    let report = run(&plan(
        5,
        &[("real-conditioned", 1), ("counterfactual", 1)],
        "multiclass",
        0,
        SourceRole::Dev,
    ))
    .unwrap();
    assert_eq!(report.requested_bases, 5);
    assert_eq!(report.accepted_base_count, 0);
    assert_eq!(report.conditioning_base_count, 0);
    assert_eq!(report.rejected.len(), 5);
    assert!(
        report
            .rejected
            .iter()
            .all(|rejection| rejection.code == "training_manifest_ancestry")
    );
    assert_eq!(
        report.requested_bases,
        report.accepted_base_count + report.rejected.len()
    );
    let rendered = format!("{report:?}");
    assert!(!rendered.contains("DEV"));
}

#[test]
fn train_ancestor_admits_conditioned_bases_without_counting_itself() {
    let report = run(&plan(
        6,
        &[("real-conditioned", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap();
    assert_eq!(report.accepted_base_count, 6);
    assert_eq!(report.conditioning_base_count, 1);
    assert!(report.rejected.is_empty());
    assert_eq!(
        report.mechanism_counts.get("real-conditioned").copied(),
        Some(6)
    );
}

#[test]
fn same_plan_reproduces_content_and_a_new_shard_does_not() {
    let first = run(&plan(
        8,
        &[("independent", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap();
    let second = run(&plan(
        8,
        &[("independent", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap();
    assert_eq!(first, second);
    let mut other = plan(8, &[("independent", 1)], "regression", 0, SourceRole::Train);
    other.shard_id = 9;
    let shifted = run(&other).unwrap();
    assert_ne!(first.content_sha256, shifted.content_sha256);
    let conditioned = run(&plan(
        8,
        &[("counterfactual", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap();
    assert_ne!(first.content_sha256, conditioned.content_sha256);
    assert_eq!(conditioned.accepted_base_count, 8);
    assert_eq!(conditioned.conditioning_base_count, 1);
}

#[test]
fn unsupported_task_is_ledgered_without_echo() {
    let report = run(&plan(
        3,
        &[("independent", 1)],
        "secret-task-value",
        1,
        SourceRole::Train,
    ))
    .unwrap();
    assert_eq!(report.requested_bases, 3);
    assert_eq!(report.accepted_base_count, 0);
    assert_eq!(report.rejected.len(), 3);
    assert_eq!(report.requested_views, 3);
    assert_eq!(report.view_count, 0);
    assert!(report.mechanism_counts.is_empty());
    assert!(report.sample_row_sha256.is_none());
    assert!(
        report
            .rejected
            .iter()
            .all(|rejection| rejection.code == "generator_pilot_task")
    );
    let rendered = format!("{report:?}");
    assert!(!rendered.contains("secret-task-value"));
}

#[test]
fn unknown_mechanism_and_over_limit_refuse_without_echo() {
    let mechanism = run(&plan(
        2,
        &[("secret-mechanism", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap_err();
    let rendered = mechanism.to_string();
    assert!(rendered.contains("generator_pilot_mechanism"));
    assert!(!rendered.contains("secret-mechanism"));
    let limit = run(&plan(
        PILOT_BASES + 1,
        &[("independent", 1)],
        "regression",
        0,
        SourceRole::Train,
    ))
    .unwrap_err();
    assert!(limit.to_string().contains("generator_pilot_limit"));
    assert!(!limit.to_string().contains("100001"));
}

#[test]
fn mixture_counts_cover_every_arm() {
    let report = run(&plan(
        96,
        &[
            ("independent", 1),
            ("real-conditioned", 1),
            ("counterfactual", 1),
        ],
        "binary",
        0,
        SourceRole::Train,
    ))
    .unwrap();
    assert_eq!(report.accepted_base_count, 96);
    assert_eq!(report.conditioning_base_count, 1);
    assert_eq!(report.mechanism_counts.len(), 3);
    assert_eq!(report.mechanism_counts.values().sum::<usize>(), 96);
    assert!(report.mechanism_counts.values().all(|count| *count > 0));
    assert!(report.rejected.is_empty());
}
