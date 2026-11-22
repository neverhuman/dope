//! Black-box tests of `dope_kpi emit-cli` with generators that work: the rows
//! handed to a generator and read back from it are exact, the log is complete
//! and the outputs are reproducible (see `kpi_ext_support` for the fixture).

mod kpi_ext_support;
mod kpi_support;

use kpi_ext_support::{
    command_seed, files_under, fit_row, fixture, read_mat, sha256_hex, stem, write_arm_file,
    write_script, ArmDef, Fixture, MAT_GENERATOR, TEXT_GENERATOR,
};
use kpi_support::{json_file, json_lines};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Arms over the two fake generators. A generator that copies its fit rows
/// keeps real rows in its state, so those arms are declared `rows_at_sampler`
/// or `control`; `stats` keeps a header and a count only.
fn arms(fx: &Fixture) -> PathBuf {
    let text = write_script(&fx.dir, "gen.sh", TEXT_GENERATOR);
    let mat = write_script(&fx.dir, "genmat.sh", MAT_GENERATOR);
    let arm = |arm_id, class, io, script, fit_mode, sample_mode| ArmDef {
        arm_id,
        class,
        io,
        script,
        fit_mode,
        sample_mode,
    };
    write_arm_file(
        &fx.dir,
        "arms.json",
        &[
            arm("copy-csv", "rows_at_sampler", "csv", &text, "copy", "copy"),
            arm("copy-mat", "rows_at_sampler", "mat", &mat, "copy", "copy"),
            arm("shuffled", "control", "csv", &text, "copy", "shuffle"),
            arm("stats", "source_free", "csv", &text, "stats", "stats"),
        ],
    )
}

fn text(value: &Value) -> &str {
    value.as_str().expect("string field")
}

fn count(value: &Value) -> usize {
    value.as_array().expect("array field").len()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn copied_rows_come_back_bit_exact_through_csv_and_mat() {
    let fx = fixture("emit-round-trip", "3,5");
    let arms = arms(&fx);
    let run = fx.dir.join("run");
    let mut covered = BTreeSet::new();
    let (mut missing, mut integers, mut negative_zeros, mut files) = (0, 0, 0, 0);
    // Arms share nothing, so they are emitted into one run directory at the same time.
    let names = ["copy-csv", "copy-mat", "shuffled"];
    let summaries: Vec<Value> = std::thread::scope(|scope| {
        let emit = |arm| fx.emit_ok(&arms, arm, "run", &["--threads", "2"]);
        let running = names.map(|arm| scope.spawn(move || emit(arm)));
        running.map(|run| run.join().unwrap()).into()
    });
    for (arm, summary) in names.into_iter().zip(summaries) {
        assert_eq!(summary["rows_ok"], 20, "{arm}: {summary}");
        assert_eq!(summary["rows_failed"], 0);
        assert_eq!(summary["rows_unsupported"], 4);
        for cell in fx.cells.iter().filter(|cell| cell["status"] == "ok") {
            let table = fx.table_of(cell);
            let (fit_rows, width) = (count(&cell["a_idx"]), count(&cell["cols"]) + 1);
            covered.insert((text(&cell["task"]), text(&cell["view"])));
            for (seed, multiplier) in [(3, 4), (3, 1), (5, 4), (5, 1)] {
                let name = format!("{}.s{seed}.m{multiplier}.mat", stem(cell));
                let (rows, cols, values) = read_mat(&run.join(arm).join(&name));
                assert_eq!((rows, cols), (multiplier * fit_rows, width - 1), "{name}");
                // The generator repeats the fit rows in order until it has enough.
                for (position, got) in values.chunks(width).enumerate() {
                    let expected = fit_row(table, cell, position % fit_rows);
                    assert_eq!(bits(got), bits(&expected), "{arm}/{name} row {position}");
                    let features = &got[..width - 1];
                    missing += features.iter().filter(|v| v.is_nan()).count();
                    negative_zeros += features.iter().filter(|v| v.to_bits() == 1 << 31).count();
                    integers += features
                        .iter()
                        .filter(|v| v.fract() == 0.0 && **v > 1.0)
                        .count();
                }
                files += 1;
            }
        }
    }
    assert_eq!(files, 3 * 20);
    let expected_cells = [
        ("binary", "full"),
        ("multiclass", "full"),
        ("regression", "full"),
        ("regression", "p48"),
    ];
    assert_eq!(covered.into_iter().collect::<Vec<_>>(), expected_cells);
    // The comparison above really saw the hard cases.
    assert!(missing > 1000 && integers > 1000 && negative_zeros > 1000);

    // Text, binary and reordered-header generators give the same files.
    for path in files_under(&run.join("copy-csv")) {
        if path.extension().is_some_and(|ext| ext == "mat") {
            let name = path.file_name().unwrap();
            assert_eq!(read(&path), read(&run.join("copy-mat").join(name)));
            assert_eq!(read(&path), read(&run.join("shuffled").join(name)));
        }
    }
    // formats.md section 6: only a source-free arm reports an encoding size.
    for arm in ["copy-csv", "copy-mat", "shuffled"] {
        let rows = json_lines(&run.join(arm).join("emit.jsonl"));
        assert_eq!(rows.len(), 24);
        for row in &rows {
            assert_eq!(row["packet_bytes"], 0);
            assert_eq!(row["packet_sha256"], "");
        }
        let class = if arm == "shuffled" {
            "control"
        } else {
            "rows_at_sampler"
        };
        assert!(rows.iter().all(|row| row["artifact_class"] == class));
    }
}

const FIELDS: [&str; 16] = [
    "arch_id",
    "arm_id",
    "artifact_class",
    "cell_id",
    "cold_bytes",
    "decode_ms",
    "encode_ms",
    "file",
    "multiplier",
    "packet_bytes",
    "packet_sha256",
    "reason",
    "rows",
    "sample_seed",
    "status",
    "weights_sha256",
];

#[test]
fn the_log_has_one_complete_row_per_scheduled_file() {
    let fx = fixture("emit-log", "3,5");
    let arms = arms(&fx);
    let summary = fx.emit_ok(&arms, "stats", "run", &[]);
    let arm_dir = std::fs::canonicalize(fx.dir.join("run/stats")).unwrap();
    let rows = json_lines(&arm_dir.join("emit.jsonl"));
    let script = read(&fx.dir.join("gen.sh"));

    // Exactly one row per (cell, sample seed, multiplier), sorted by that key.
    let mut expected = Vec::new();
    for cell in &fx.cells {
        for (seed, multiplier) in [(3, 1), (3, 4), (5, 1), (5, 4)] {
            expected.push((text(&cell["cell_id"]).to_string(), seed, multiplier));
        }
    }
    let key = |row: &Value| {
        let (seed, multiplier) = (row["sample_seed"].as_u64(), row["multiplier"].as_u64());
        (
            text(&row["cell_id"]).to_string(),
            seed.unwrap(),
            multiplier.unwrap(),
        )
    };
    let keys: Vec<(String, u64, u64)> = rows.iter().map(key).collect();
    assert_eq!(keys, expected);
    assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(rows.len(), 6 * 2 * 2);

    let mut packet_sizes = Vec::new();
    for row in &rows {
        let fields: Vec<&str> = row
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(fields, FIELDS);
        assert_eq!(row["arm_id"], "stats");
        assert_eq!(row["arch_id"], "ext-cli/v1");
        assert_eq!(row["artifact_class"], "source_free");
        assert_eq!(text(&row["weights_sha256"]), sha256_hex(&script));
        assert_eq!(row["cold_bytes"], script.len());
        let (cell_id, seed, multiplier) = key(row);
        let cell = fx
            .cells
            .iter()
            .find(|cell| cell["cell_id"] == cell_id.as_str());
        let cell = cell.unwrap();
        let stem = stem(cell);
        let (encode_ms, decode_ms) = (row["encode_ms"].as_f64(), row["decode_ms"].as_f64());
        if cell["status"] != "ok" {
            assert_eq!(text(&cell["dataset_hash"]), "00000000000000a5");
            assert_eq!(row["status"], "unsupported");
            assert_eq!(row["reason"], "schedule_unsupported");
            assert_eq!((&row["rows"], &row["file"]), (&json!(0), &json!("")));
            assert_eq!(
                (&row["packet_bytes"], &row["packet_sha256"]),
                (&json!(0), &json!(""))
            );
            assert_eq!((encode_ms, decode_ms), (Some(0.0), Some(0.0)));
            assert!(!arm_dir.join(format!("_logs/{stem}.log")).exists());
            continue;
        }
        assert_eq!((&row["status"], &row["reason"]), (&json!("ok"), &json!("")));
        // The packet is the header line, the card is the line count of the fit rows.
        let packet = read(&arm_dir.join(format!("packets/{stem}.packet")));
        let card = read(&arm_dir.join(format!("packets/{stem}.card")));
        let (fit_rows, features) = (count(&cell["a_idx"]), count(&cell["cols"]));
        let header: Vec<String> = (0..features).map(|j| format!("c{j}")).collect();
        assert_eq!(
            packet,
            format!("{},target\n", header.join(",")).into_bytes()
        );
        assert_eq!(card, format!("{}\n", fit_rows + 1).into_bytes());
        assert_eq!(row["packet_bytes"], packet.len() + card.len());
        assert_eq!(
            text(&row["packet_sha256"]),
            sha256_hex(&[packet, card].concat())
        );
        packet_sizes.push(row["packet_bytes"].as_u64().unwrap());

        let file = format!("stats/{stem}.s{seed}.m{multiplier}.mat");
        assert_eq!(row["file"], file.as_str());
        assert_eq!(row["rows"], multiplier as usize * fit_rows);
        let (mat_rows, mat_cols, values) = read_mat(&fx.dir.join("run").join(&file));
        assert_eq!(
            (mat_rows, mat_cols),
            (multiplier as usize * fit_rows, features)
        );
        assert!(values.iter().all(|v| v.to_bits() == 0));
        assert!(encode_ms.unwrap() > 0.0 && decode_ms.unwrap() > 0.0);
    }

    // The summary on stdout is the summary on disk and agrees with the rows.
    assert_eq!(summary, json_file(&arm_dir.join("emit-summary.json")));
    packet_sizes.sort_unstable();
    let expected_summary = [
        ("command", json!("emit-cli")),
        ("arm_id", json!("stats")),
        ("arch_id", json!("ext-cli/v1")),
        ("artifact_class", json!("source_free")),
        ("io", json!("csv")),
        ("weights_sha256", json!(sha256_hex(&script))),
        ("cold_bytes", json!(script.len())),
        ("budget_bytes", Value::Null),
        ("emit_file", json!("stats/emit.jsonl")),
        ("schedule_cells", json!(6)),
        ("limit_cells", Value::Null),
        ("cells", json!(6)),
        ("cells_ok", json!(5)),
        ("cells_failed", json!(0)),
        ("cells_unsupported", json!(1)),
        ("cells_kept", json!(0)),
        ("cells_run", json!(5)),
        ("rows", json!(24)),
        ("rows_ok", json!(20)),
        ("rows_failed", json!(0)),
        ("rows_unsupported", json!(4)),
        ("rows_failed_by_reason", json!({})),
        ("cells_failed_by_reason", json!({})),
        ("packet_bytes_max", json!(packet_sizes[19])),
        // Five cells, four files each: the median cell is the third.
        ("packet_bytes_median", json!(packet_sizes[10] as f64)),
    ];
    for (field, value) in &expected_summary {
        assert_eq!(&summary[*field], value, "{field}");
    }
    assert!(summary["encode_ms_median"].as_f64().unwrap() > 0.0);
    assert!(summary["decode_ms_median"].as_f64().unwrap() > 0.0);
    assert_eq!(
        summary.as_object().unwrap().len(),
        expected_summary.len() + 2
    );

    // Per-cell logs: commands in schedule order, the documented seeds, a fixed
    // environment, a closed stdin, and fit rows that live under the run directory.
    for cell in fx.cells.iter().filter(|cell| cell["status"] == "ok") {
        let stem = stem(cell);
        let log = String::from_utf8(read(&arm_dir.join(format!("_logs/{stem}.log")))).unwrap();
        let markers: Vec<&str> = log
            .lines()
            .filter(|line| line.starts_with("--- "))
            .collect();
        let mut expected = vec!["--- fit: start".to_string(), "--- fit: exit code 0".into()];
        for (seed, multiplier) in [(3, 4), (3, 1), (5, 4), (5, 1)] {
            expected.push(format!("--- sample s{seed} m{multiplier}: start"));
            expected.push(format!("--- sample s{seed} m{multiplier}: exit code 0"));
            let rows = multiplier * count(&cell["a_idx"]);
            let line = format!(
                "sample seed={} rows={rows} task={}",
                command_seed(cell, seed, "ext-sample"),
                text(&cell["task"])
            );
            assert!(log.lines().any(|l| l == line), "{stem}: missing {line}");
        }
        assert_eq!(markers, expected);
        let (task, classes) = (text(&cell["task"]), cell["n_classes"].as_u64().unwrap());
        let task2 = if task == "regression" {
            task
        } else {
            "classification"
        };
        let fit_line = format!(
            "fit rows={}/_work/{stem}/fit_rows.csv seed={} task={task}/{task2} classes={classes}",
            arm_dir.display(),
            command_seed(cell, 0, "ext-fit")
        );
        assert!(
            log.lines().any(|l| l == fit_line),
            "{stem}: missing {fit_line}"
        );
        for line in [
            "LANG=C",
            "OMP_NUM_THREADS=1",
            "RAYON_NUM_THREADS=1",
            "CUDA_VISIBLE_DEVICES=",
        ] {
            assert!(log.lines().any(|l| l == line), "{stem}: missing {line}");
        }
        assert!(!log.contains("CANARY") && !log.contains("must-not-leak"));
        assert!(!log.contains("stdin is open") && !log.contains("CARGO"));
    }

    // Nothing but the documented files is left: no scratch, no stray output.
    assert!(!arm_dir.join("_work").exists());
    let left = files_under(&fx.dir.join("run"));
    assert_eq!(left.len(), 2 + 5 * (1 + 2 + 4), "{left:?}");
}

/// A log line without its two wall-clock fields.
fn timeless(mut row: Value) -> Value {
    let fields = row.as_object_mut().unwrap();
    assert!(fields.remove("encode_ms").is_some() && fields.remove("decode_ms").is_some());
    row
}

#[test]
fn outputs_do_not_depend_on_the_run_or_the_thread_count() {
    let fx = fixture("emit-repeat", "7");
    let arms = arms(&fx);
    // Three independent run directories, filled at the same time.
    std::thread::scope(|scope| {
        for (out, threads) in [("a", "1"), ("b", "1"), ("c", "4")] {
            let (fx, arms) = (&fx, &arms);
            scope.spawn(move || {
                for arm in ["copy-csv", "stats"] {
                    fx.emit_ok(arms, arm, out, &["--threads", threads]);
                }
            });
        }
    });
    let relative = |out: &str| -> Vec<PathBuf> {
        let root = fx.dir.join(out);
        let files = files_under(&root).into_iter();
        files
            .map(|path| path.strip_prefix(&root).unwrap().to_path_buf())
            .collect()
    };
    let names = relative("a");
    // Per arm: the row log, the summary, and a log and two matrices per supported
    // cell; plus five copied packets and five packet and card pairs.
    assert_eq!(names.len(), 2 * (2 + 5 * (1 + 2)) + 5 * (1 + 2));
    assert_eq!(names, relative("b"));
    assert_eq!(names, relative("c"));
    let (mut matrices, mut packets, mut logs) = (0, 0, 0);
    for name in &names {
        let [a, b, c] = ["a", "b", "c"].map(|out| read(&fx.dir.join(out).join(name)));
        let file = name.file_name().unwrap().to_str().unwrap();
        if file.ends_with(".mat") || name.parent().unwrap().ends_with("packets") {
            // Byte-identical across runs and thread counts.
            assert!(a == b && a == c, "{name:?} differs");
            matrices += usize::from(file.ends_with(".mat"));
            packets += usize::from(!file.ends_with(".mat"));
        } else if file == "emit.jsonl" {
            let [a, b, c] = ["a", "b", "c"].map(|out| {
                let rows = json_lines(&fx.dir.join(out).join(name)).into_iter();
                rows.map(timeless).collect::<Vec<Value>>()
            });
            assert_eq!(a.len(), 12);
            assert!(
                a == b && a == c,
                "{name:?} differs beyond the wall-clock fields"
            );
            logs += 1;
        }
    }
    assert_eq!((matrices, packets, logs), (20, 5 + 5 + 5, 2));
}
