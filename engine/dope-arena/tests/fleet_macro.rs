//! End-to-end check of `dope_fleet_macro` on a two-host fixture with a fixed clock.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_root(tag: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("fleet_macro_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}

fn write_lines(path: &Path, rows: &[Value]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
    fs::write(path, text).unwrap();
}

fn build_fixture(root: &Path) -> PathBuf {
    let runs = root.join("runs-root");
    let a = runs.join("hostA/runs/cpufleet-v72-x-s01");
    let step = |t: f64, s: u64, cum: u64, loss: f64| {
        json!({"time": t, "step": s, "unique_splits_seen": cum, "tables_per_s": 10.0, "loss": loss,
               "n_synth_prior": 4, "n_reg": 6, "n_bin": 5, "n_mc": 5, "arch": "single_enc_v7"})
    };
    write_lines(
        &a.join("train.jsonl"),
        &[
            step(1000.0, 1, 16, 2.0),
            step(1100.0, 2, 32, 1.0),
            step(1400.0, 3, 48, 1.0),
        ],
    );
    fs::write(
        a.join("config.json"),
        r#"{"arch":"single_enc_v7","world_size":1,"sync_mode":"diloco_outer_momentum","sync_dir":"/x/sync-A"}"#,
    )
    .unwrap();
    fs::create_dir_all(a.join("checkpoints")).unwrap();
    fs::write(a.join("checkpoints/step_000002_ema.safetensors"), b"").unwrap();
    write_lines(
        &a.join("exposure.jsonl"),
        &[
            json!({"step": 1, "time": 1000.0, "datasets": ["d1", "d2"], "splits": ["p1", "p2"]}),
            json!({"step": 2, "time": 1100.0, "datasets": ["d1", "d1"], "splits": ["p1", "p3"]}),
        ],
    );
    let b = runs.join("hostB/runs/other-s02");
    write_lines(
        &b.join("train.jsonl"),
        &[
            json!({"time": 1000.0, "step": 1, "tables_per_s": 5.0, "loss": 3.0, "world_size": 2}),
            json!({"time": 1200.0, "step": 2, "tables_per_s": 5.0, "loss": 3.0, "world_size": 2}),
            json!({"step": 3}),
        ],
    );
    let peer = runs.join("hostB/runs/sync-peer-s03");
    write_lines(
        &peer.join("train.jsonl"),
        &[
            json!({"time": 1000.0, "step": 1, "episodes_seen": 10, "sync_mode": "diloco_outer_momentum", "sync_round": 1}),
            json!({"time": 1010.0, "step": 2, "episodes_seen": 20, "sync_mode": "diloco_outer_momentum", "sync_round": 2}),
        ],
    );
    fs::write(peer.join("config.json"), r#"{"sync_dir":"/x/sync-A"}"#).unwrap();
    write_lines(
        &root.join("ckpt.jsonl"),
        &[
            json!({"run": "cpufleet-v72-x-s01", "step": 2, "variant": "full", "metric": "dtrg", "value": 0.02, "panel": "p"}),
            json!({"run": "cpufleet-v72-x-s01", "step": 3, "variant": "full", "metric": "dtrg", "value": -0.01, "panel": "p"}),
            json!({"run": "missing", "step": 1, "variant": "full", "metric": "dtrg", "value": 0.5, "panel": "p"}),
        ],
    );
    let guard = root.join("guard");
    fs::create_dir_all(&guard).unwrap();
    fs::write(
        guard.join("guard_state.json"),
        r#"{"levels":{"A1":0},"paused":{}}"#,
    )
    .unwrap();
    write_lines(
        &guard.join("guard_events.jsonl"),
        &[
            json!({"t": 1300.0, "utc": "u", "event": "PAUSE", "job": "A1", "hosts": ["hostA"], "reason": "r"}),
        ],
    );
    write_lines(
        &root.join("index.jsonl"),
        &[
            json!({"split": "train"}),
            json!({"split": "train"}),
            json!({"split": "val"}),
        ],
    );
    runs
}

fn run_macro(root: &Path, runs: &Path) -> (Value, String) {
    let out = root.join("out/fleet_macro.json");
    let status = Command::new(env!("CARGO_BIN_EXE_dope_fleet_macro"))
        .arg("--root")
        .arg(runs)
        .arg("--checkpoint-kpi")
        .arg(root.join("ckpt.jsonl"))
        .arg("--corpus-index")
        .arg(root.join("index.jsonl"))
        .arg("--guard-dir")
        .arg(root.join("guard"))
        .args(["--bucket-s", "300", "--now", "1450", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let text = fs::read_to_string(&out).unwrap();
    (serde_json::from_str(&text).unwrap(), text)
}

#[test]
fn fleet_macro_aggregates_a_two_host_campaign() {
    let root = fixture_root("e2e");
    let runs = build_fixture(&root);
    let (d, text) = run_macro(&root, &runs);
    assert_eq!(d["schema"], "dope.fleet-macro/v1");
    for end in [",", "}", "]"] {
        assert!(
            !text.contains(&format!("-0.0{end}")),
            "negative zero in output"
        );
    }
    assert_eq!(d["inputs"]["runs"], 3);
    assert_eq!(d["inputs"]["rows"], 7);
    assert_eq!(d["inputs"]["bad_rows"], 1);
    assert_eq!(d["hosts"], json!(["hostA", "hostB"]));

    let h = &d["headline"];
    assert_eq!(h["tables_total"], 1068.0);
    assert_eq!(h["active_runs"], 1);
    assert_eq!(h["tables_per_s_now"], 3.39);
    assert_eq!(h["tables_per_s_reported"], 10.0);
    assert_eq!(h["peak"], json!({"tables_per_s": 3.39, "t": 1200}));
    assert_eq!(h["checkpoints_saved"], 1);
    assert_eq!(h["checkpoints_unplaced"], 1);

    let s = d["series"].as_array().unwrap();
    assert_eq!(s.len(), 2);
    assert_eq!(s[0]["t"], 900);
    assert_eq!(s[0]["tables_per_s"], 0.173);
    assert_eq!(s[0]["loss_rel_median"], 0.875);
    assert_eq!(s[1]["by_host"]["hostB"], 3.333);
    assert_eq!(s[1]["cum_tables"], 1068.0);

    let a = d["runs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["host"] == "hostA")
        .unwrap();
    assert_eq!(a["family"], "cpufleet-v72-x");
    assert_eq!(a["synth_basis"], "logged");
    assert_eq!(a["synth_est"], 12.0);
    assert_eq!(a["real_est"], 36.0);
    assert_eq!(a["loss_change_pct"], -50.0);

    let c = &d["checkpoint_kpi"];
    assert_eq!(c["points"][0]["at"], 1100.0);
    assert_eq!(c["points"][1]["at"], 1400.0);
    assert_eq!(c["points"][2]["at"], Value::Null);
    assert_eq!(c["summary"]["full"]["n"], 3);
    assert_eq!(c["summary"]["full"]["above_noise"], 2);
    assert_eq!(c["summary"]["full"]["below_zero"], 1);

    let t = &d["topology"];
    assert_eq!(t["independent_runs"], 0);
    assert_eq!(t["data_parallel_runs"][0]["world_size"], 2);
    assert_eq!(t["sync_groups"][0]["group"], "/x/sync-A");
    assert_eq!(t["sync_groups"][0]["runs"], 2);
    assert_eq!(t["sync_groups"][0]["hosts"], 2);
    assert_eq!(t["sync_groups"][0]["max_round"], 2);
    assert_eq!(t["cross_run_weight_sharing"], true);

    let u = &d["uniqueness"];
    assert_eq!(u["corpus_train_tables"], 2);
    assert_eq!(u["real_episodes_est"], 1056.0);
    assert_eq!(u["fleet_avg_uses_per_train_table_est"], 528.0);
    assert_eq!(u["exposure"]["runs"][0]["distinct_splits"], 3);
    assert_eq!(u["exposure"]["runs"][0]["split_repeats"], 1);
    assert_eq!(u["exposure"]["runs"][0]["within_step_dataset_dups"], 1);
    assert_eq!(u["exposure"]["fleet"]["distinct_datasets"], 2);

    assert_eq!(d["guard"]["levels"]["A1"], 0);
    assert_eq!(d["guard"]["last_event"]["event"], "PAUSE");
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fleet_macro_handles_an_empty_root() {
    let root = fixture_root("empty");
    let empty = root.join("nothing");
    fs::create_dir_all(&empty).unwrap();
    let out = root.join("m.json");
    let status = Command::new(env!("CARGO_BIN_EXE_dope_fleet_macro"))
        .arg("--root")
        .arg(&empty)
        .args(["--now", "100", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let d: Value = serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(d["inputs"]["runs"], 0);
    assert_eq!(d["headline"]["tables_total"], 0.0);
    assert_eq!(d["series"], json!([]));
    assert_eq!(d["checkpoint_kpi"], Value::Null);
    assert_eq!(d["guard"], Value::Null);
    fs::remove_dir_all(&root).unwrap();
}
