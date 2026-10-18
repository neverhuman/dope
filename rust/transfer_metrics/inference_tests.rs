use super::*;

fn fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/golden/aggregate/c02_crossed_{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn design(name: &str) -> CrossedDesign {
    serde_json::from_value(fixture(name)["input"].clone()).unwrap()
}

fn scheduled(mut panel: CrossedDesign) -> CrossedDesign {
    panel.scheduled_keys = panel.runs.iter().map(EvaluationRun::key).collect();
    panel
}

#[test]
fn namespaces_have_stable_substreams() {
    assert_eq!(
        task_substream(11, "task", "fit").unwrap(),
        9525942615105419655
    );
    assert_eq!(
        task_substream(11, "tâche/水", "fit").unwrap(),
        9973167365048566836
    );
    assert_ne!(
        task_substream(11, "task", "fit").unwrap(),
        task_substream(11, "task", "sample").unwrap()
    );
    let case = fixture("uint64_substream");
    assert!(crate::metrics::values_close(
        &evaluate(&case["input"]),
        &case["expected"],
        1e-12
    ));
}

#[test]
fn all_five_seeded_stress_cases_match_python_without_qualification() {
    for name in [
        "near_threshold",
        "low_gain",
        "heavy_tailed",
        "correlated",
        "few_source",
    ] {
        let case = fixture(name);
        let actual = crate::metrics::aggregate::evaluate(&case["input"]);
        assert!(crate::metrics::values_close(
            &actual,
            &case["expected"],
            1e-12
        ));
        assert_eq!(actual["qualification"], false);
        assert_eq!(actual["calibration"], "unmeasured");
        if name == "few_source" {
            assert_eq!(actual["status"], "insufficient_families");
            assert!(actual["bounds"].is_null());
        } else if name == "correlated" {
            assert_eq!(
                actual["bounds"]["gap/1024/tree"],
                actual["bounds"]["gap/2048/neural"]
            );
        }
    }
    let panel = design("low_gain");
    let p = panel.runs[0].pairs.as_ref().unwrap();
    let pairs = crate::transfer_metrics::PairedLosses::new(
        p.real.clone(),
        p.synthetic.clone(),
        p.null.clone(),
    )
    .unwrap();
    assert!(
        crate::transfer_metrics::loss_summary(
            &pairs,
            crate::transfer_metrics::TrainingScale::Positive(1.)
        )
        .unwrap()
        .retention
        .is_none()
    );
    let boot = bootstrap_effects(
        &design("heavy_tailed"),
        "gap",
        "method_new_source",
        199,
        6304,
    )
    .unwrap();
    let robust = heavy_tail_sensitivity(&boot, 3).unwrap();
    assert!(crate::metrics::values_close(
        &robust,
        &fixture("heavy_tailed")["robust_expected"],
        1e-12
    ));
    assert_eq!(robust["status"], "sensitivity_only");
    assert_eq!(robust["qualification"], false);
    assert!(robust["bounds"].is_null());
}

#[test]
fn wrong_clusters_learner_sources_duplicate_seeds_and_holes_are_refused() {
    for unit in ["query_row", "learner_seed", "fit_seed"] {
        let mut panel = design("correlated");
        panel.cluster_unit = unit.to_owned();
        assert_eq!(panel.validate().unwrap_err().reason, "invalid_cluster_unit");
    }
    let mut panel = design("correlated");
    let mut alias = panel.runs[0].clone();
    alias.fit_id = "aliased-fit".into();
    panel.runs.push(alias);
    assert_eq!(
        scheduled(panel).validate().unwrap_err().reason,
        "duplicate_fit_seed"
    );
    let mut panel = design("correlated");
    panel
        .runs
        .retain(|r| !(r.family_id == "s0" && r.task_axis_id == "t0"));
    let boot = bootstrap_effects(&scheduled(panel), "gap", "method_new_source", 199, 11).unwrap();
    assert_eq!(boot.status, "incomplete");
    assert!(boot.estimates.is_empty() && boot.samples.is_empty());
    assert!(simultaneous_ucb(&boot, 0.05).unwrap()["bounds"].is_null());
    let mut payload = fixture("near_threshold")["input"].clone();
    payload["cluster_unit"] = json!("PRIVATE-source-header-extrema");
    assert_eq!(
        evaluate(&payload),
        json!({"status": "failed", "reason": "invalid_crossed_design"})
    );
}

#[test]
fn nested_sampling_remains_paired_across_all_coordinates() {
    let mut panel = design("correlated");
    let original = panel.runs.clone();
    panel.runs.clear();
    for run in original {
        for sample in 0..2 {
            for learner in 0..2 {
                let mut r = run.clone();
                r.sample_id = sample.to_string();
                r.learner_seed_id = learner.to_string();
                for value in &mut r.pairs.as_mut().unwrap().synthetic {
                    *value += (sample + 2 * learner) as f64;
                }
                panel.runs.push(r);
            }
        }
    }
    let boot = bootstrap_effects(&scheduled(panel), "gap", "method_new_source", 399, 31).unwrap();
    assert_eq!(
        boot.samples["gap/1024/tree"],
        boot.samples["gap/2048/neural"]
    );
}

#[test]
fn shared_fit_and_task_shocks_survive_source_replication() {
    for axis in ["task", "fit"] {
        let mut panel = design("correlated");
        let original = panel.runs.clone();
        panel.runs.clear();
        for copy in 0..10 {
            for item in &original {
                let mut r = item.clone();
                r.family_id = format!("{copy}/{}", r.family_id);
                r.task_id = format!("{copy}/{}", r.task_id);
                let positive = if axis == "task" {
                    r.task_axis_id == "t0"
                } else {
                    r.fit_id == "f0"
                };
                let p = r.pairs.as_mut().unwrap();
                p.real.fill(10.);
                p.synthetic.fill(if positive { 13. } else { 7. });
                panel.runs.push(r);
            }
        }
        let boot =
            bootstrap_effects(&scheduled(panel), "gap", "method_new_source", 399, 31).unwrap();
        let variance = boot.samples["gap/1024/tree"]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            / 399.;
        assert!(variance > 3.);
    }
}

#[test]
fn fixed_artifact_and_default_budget_are_distinct_from_new_sources() {
    let mut panel = design("near_threshold");
    panel
        .runs
        .retain(|r| r.family_id == "s0" && r.task_axis_id == "t0" && r.fit_id == "f0");
    let panel = scheduled(panel);
    let boot = bootstrap_effects(&panel, "gap", "fixed_artifact", DEFAULT_DRAWS, 7).unwrap();
    assert_eq!(boot.draws, 10000);
    assert_eq!(boot.samples["gap/1024/tree"].len(), 10000);
    assert_eq!(simultaneous_ucb(&boot, 0.05).unwrap()["status"], "complete");
    let refusal = bootstrap_effects(&panel, "gap", "method_new_source", 199, 7).unwrap();
    assert_eq!(refusal.status, "insufficient_fits");
}
