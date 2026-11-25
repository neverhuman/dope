use super::*;

#[test]
fn detects_root_scratch_and_duplicate_files() {
    let tracked: BTreeSet<String> = [
        "README.md",
        "H1/metrics.json",
        "H42/README.md",
        "PYTHON_EXCEPTIONS.md",
        "receipts/review.json",
        "catboost_info/learn_error.tsv",
        "docs/archive/waves/H1/metrics.json",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let failures = check_root_cleanliness(&tracked);
    assert_eq!(failures.len(), 5, "unexpected failures: {failures:?}");
    assert!(failures.iter().any(|f| f.contains("H1/metrics.json")));
    assert!(failures.iter().any(|f| f.contains("H42/README.md")));
    assert!(failures.iter().any(|f| f.contains("PYTHON_EXCEPTIONS.md")));
    assert!(failures.iter().any(|f| f.contains("receipts/review.json")));
    assert!(failures.iter().any(|f| f.contains("catboost_info/")));
}

#[test]
fn detects_internal_hostnames_buckets_and_paths() {
    let bad = concat!(
        "ssh xbabe3\n",
        "bucket = s3://veox-private/data\n",
        "profile = veox-jepson\n",
        "dir = /home/ubuntu/dope-data\n",
        "clean = localhost:8787\n",
    );
    let failures = scan_content_for_leaks("tools/example.py", bad);
    assert_eq!(failures.len(), 4, "unexpected failures: {failures:?}");
    assert!(failures.iter().any(|f| f.contains("xbabe<N>")));
    assert!(failures.iter().any(|f| f.contains("s3://veox-")));
    assert!(failures.iter().any(|f| f.contains("veox-jepson")));
    assert!(failures.iter().any(|f| f.contains("/home/ubuntu/")));
}

#[test]
fn ignores_code_fences_and_inline_code_for_markdown_links() {
    let md = "\
See [Valid](docs/README.md#section) and `not [a](missing.md) link`.
```bash
echo '[ignored](nope.md)'
```
And [External](https://example.com) and [Anchor](#top) and [Real](ROADMAP.md).
";
    let targets = extract_markdown_targets(md);
    assert_eq!(
        targets,
        vec![
            (1, "docs/README.md#section".to_owned()),
            (5, "https://example.com".to_owned()),
            (5, "#top".to_owned()),
            (5, "ROADMAP.md".to_owned()),
        ]
    );
}

#[test]
fn checks_live_repository_hygiene() {
    let root = repository_root();
    let failures = run(&root).expect("repo-hygiene run");
    assert!(
        failures.is_empty(),
        "live repository hygiene failed:\n{}",
        failures.join("\n")
    );
}
