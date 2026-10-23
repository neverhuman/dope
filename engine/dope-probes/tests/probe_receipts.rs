//! The probe receipts regenerate every number the levels and landmark plans cite, the
//! replay output matches its committed copies, and the split reproduces NumPy's rows.

use std::fs;
use std::path::PathBuf;

use dope_probes::inputs::{split_indices, split_table, Inputs};
use dope_probes::landmarks::{self, LandmarkRow, PROGS};
use dope_probes::levels::{self, LevelsRow};
use dope_probes::nprng::{permutation, seed_sequence_u64, Pcg64};
use dope_probes::stats::{median, np_sum, ranks, spearman};

fn probes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/real-world/probes")
}

fn levels_rows() -> Vec<LevelsRow> {
    levels::load(&probes_dir().join("results/levels-probe.jsonl")).expect("levels receipts")
}

fn landmark_rows() -> Vec<LandmarkRow> {
    landmarks::load(&probes_dir().join("results/landmark-probe.jsonl")).expect("landmark receipts")
}

fn golden(name: &str) -> String {
    fs::read_to_string(probes_dir().join("results").join(name)).expect("golden replay")
}

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "{a} differs from {b} by more than {tol}"
    );
}

// ---------------------------------------------------------------- NumPy-exact permutation

#[test]
fn seed_sequence_and_pcg64_match_numpy() {
    // np.random.SeedSequence(20261022).generate_state(4, np.uint64)
    assert_eq!(
        seed_sequence_u64(20261022, 4),
        [
            0x3f11bd87c38101d6,
            0x6a36bbb60373b2d9,
            0xdb85bd4395718386,
            0xb5caff9a1ea53c3f
        ]
    );
    // np.random.PCG64(20261022).random_raw(3)
    let mut g = Pcg64::new(20261022);
    let draws = [g.next_u64(), g.next_u64(), g.next_u64()];
    assert_eq!(
        draws,
        [0x24886d5322f4d186, 0x008f18fc6d0315aa, 0x2a0f5be4b15c9d9d]
    );
}

#[test]
fn permutation_matches_numpy() {
    // np.random.default_rng(seed).permutation(n): prefix and sum(i * p[i]) mod 2^61 - 1.
    let cases: [(usize, [usize; 8], u128); 4] = [
        (10, [0, 7, 8, 1, 9, 5, 4, 2], 203),
        (
            1728,
            [1280, 1645, 533, 473, 873, 548, 1066, 572],
            1308229171,
        ),
        (
            48842,
            [850, 29997, 3052, 41998, 48782, 44657, 14274, 19684],
            29165772301755,
        ),
        (
            70000,
            [15204, 59209, 65197, 18972, 68000, 32610, 7962, 36168],
            85664097422016,
        ),
    ];
    for (n, prefix, check) in cases {
        let p = permutation(20261022, n);
        assert_eq!(&p[..8], &prefix, "n = {n}");
        let m = (1u128 << 61) - 1;
        let s = p
            .iter()
            .enumerate()
            .fold(0u128, |acc, (i, &v)| (acc + (i as u128) * (v as u128)) % m);
        assert_eq!(s, check, "n = {n}");
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &v)| i == v));
    }
    assert_eq!(permutation(0, 12), [9, 2, 7, 4, 5, 11, 0, 3, 6, 10, 8, 1]);
    assert_eq!(
        permutation(u32::MAX, 12),
        [1, 4, 0, 6, 8, 7, 10, 3, 11, 5, 2, 9]
    );
    assert!(permutation(7, 0).is_empty());
    assert_eq!(permutation(7, 1), [0]);
}

// ---------------------------------------------------------------- inputs and split

#[test]
fn manifest_pins_every_receipt_table() {
    let inp = Inputs::load(&probes_dir().join("inputs.json")).expect("manifest");
    assert_eq!(inp.split.seed, 20261022);
    assert_eq!(inp.split.max_test_rows, 5000);
    assert_eq!(inp.split.max_train_rows, 12000);
    assert!(inp
        .pmlb_base
        .contains("7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68"));
    let hex = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
    for d in inp.tables.values() {
        assert!(hex(&d.tsv_gz) && hex(&d.tsv) && hex(&d.train_csv) && hex(&d.test_csv));
    }
    let names: Vec<&String> = inp.tables.keys().collect();
    let mut lv: Vec<String> = levels_rows().into_iter().map(|r| r.table).collect();
    let mut lm: Vec<String> = landmark_rows().into_iter().map(|r| r.table).collect();
    lv.sort();
    lm.sort();
    assert_eq!(lv.iter().collect::<Vec<_>>(), names);
    assert_eq!(lm.iter().collect::<Vec<_>>(), names);
    for r in landmark_rows() {
        let cls = inp.classification.contains(&r.table);
        assert_eq!(r.task, if cls { "cls" } else { "reg" }, "{}", r.table);
    }
}

#[test]
fn split_sizes_follow_the_protocol() {
    let inp = Inputs::load(&probes_dir().join("inputs.json")).expect("manifest");
    // (rows in the source table, test rows, training rows) of the nine probe tables.
    for (n, test, train) in [
        (8192, 1638, 6554),
        (22784, 4556, 12000),
        (40768, 5000, 12000),
        (48842, 5000, 12000),
        (1728, 345, 1383),
        (5000, 1000, 4000),
        (8124, 1624, 6500),
    ] {
        let (te, tr) = split_indices(n, &inp.split);
        assert_eq!((te.len(), tr.len()), (test, train), "n = {n}");
        assert!(te.iter().all(|i| !tr.contains(i)));
    }
}

#[test]
fn split_carries_fields_verbatim_and_rejects_bad_tables() {
    let inp = Inputs::load(&probes_dir().join("inputs.json")).expect("manifest");
    let dir = std::env::temp_dir().join(format!("dope-probes-split-{}", std::process::id()));
    let mut tsv = String::from("a\tb\ttarget\n");
    for i in 0..10 {
        tsv += &format!("{i}.50\t1e-3\t{}\n", i % 2);
    }
    split_table(tsv.as_bytes(), &inp.split, &dir).expect("split");
    let test = fs::read_to_string(dir.join("test.csv")).expect("test.csv");
    let train = fs::read_to_string(dir.join("train.csv")).expect("train.csv");
    // permutation(20261022, 10) starts 0, 7: two test rows, the other eight train.
    assert_eq!(test, "a,b,target\n0.50,1e-3,0\n7.50,1e-3,1\n");
    assert_eq!(train.lines().count(), 9);
    assert!(train.starts_with("a,b,target\n8.50,1e-3,0\n"));
    assert!(split_table(b"a\tb\n1\t2\n", &inp.split, &dir).is_err());
    assert!(split_table(b"a\ttarget\n1\t2\t3\n", &inp.split, &dir).is_err());
    let _ = fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------- statistics

#[test]
fn statistics_helpers() {
    assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), [3.5, 1.0, 3.5, 2.0]);
    assert_eq!(spearman(&[1.0, 2.0], &[1.0, 2.0]), None);
    assert_eq!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None);
    close(
        spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]).unwrap(),
        -1.0,
        1e-12,
    );
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    let v: Vec<f64> = (0..37).map(|i| 0.1 * i as f64).collect();
    close(np_sum(&v), 66.6, 1e-9);
}

// ---------------------------------------------------------------- levels receipts

#[test]
fn levels_replay_regenerates_cited_numbers() {
    let rows = levels_rows();
    assert_eq!(rows.len(), 9);
    let s = levels::summary(&rows);
    for (got, want) in s
        .independent
        .median
        .iter()
        .zip([0.960, 0.943, 0.859, 0.874])
    {
        close(*got, want, 5e-4);
    }
    close(s.carry_005.mean[2], 0.925, 5e-4);
    close(s.carry_005.mean[3], 0.924, 5e-4);
    assert_eq!(
        (
            s.independent.violations,
            s.carry_0.violations,
            s.carry_005.violations
        ),
        (4, 3, 0)
    );
    close(s.chain_saving_median, 0.369, 5e-4);
    close(s.chain_saving_min, 0.138, 5e-4);
    close(s.chain_saving_max, 0.585, 5e-4);
    assert_eq!((s.pool_beats_ultra, s.pool_beats_best_single), (8, 5));
    assert_eq!(levels::render(&rows), golden("levels-replay.md"));
}

#[test]
fn compare_levels_reports_identical_runs() {
    let rows = levels_rows();
    let out = levels::render_compare(&rows, &rows);
    assert!(out.contains("identical programs: 36/36; largest per-cell change 0.000"));
    assert!(out.contains("over the 9 common tables"));
}

#[test]
fn carry_policy_moves_up_only_past_the_margin() {
    let mut row: LevelsRow = levels_rows().remove(0);
    let audits = [0.90, 0.903, 0.95, 0.94];
    let robust = [0.80, 0.81, 0.85, 0.70];
    for (k, l) in ["low", "med", "high", "ultra"].iter().enumerate() {
        let lv = match *l {
            "low" => &mut row.levels.low,
            "med" => &mut row.levels.med,
            "high" => &mut row.levels.high,
            _ => &mut row.levels.ultra,
        };
        lv.self_audit = Some(audits[k]);
        lv.robust = robust[k];
    }
    assert_eq!(levels::policy(&row, 0.005), [0.80, 0.80, 0.85, 0.85]);
    assert_eq!(levels::policy(&row, 0.0), [0.80, 0.81, 0.85, 0.85]);
    assert_eq!(levels::independent(&row), robust);
    assert_eq!(levels::violations(&robust, 0.02), 1);
}

// ---------------------------------------------------------------- landmark receipts

#[test]
fn landmark_selection_table_matches_the_plan() {
    let rows = landmark_rows();
    let s = landmarks::selection(&rows);
    // (predictor, regret, hits, within-table Spearman, pooled Spearman, bias)
    let want = [
        ("self_audit", 0.0288, 2, 0.33, 0.87, Some(0.038)),
        ("panel2_min", 0.0026, 5, 0.50, 0.77, Some(0.003)),
        ("family_min", 0.0243, 2, 0.16, 0.64, Some(-0.168)),
        ("family_min_x_knn1", 0.0452, 2, 0.11, 0.81, Some(-0.098)),
        ("strong_min", 0.0262, 2, 0.22, 0.45, Some(-0.299)),
        ("gain_weighted", 0.0450, 2, 0.27, 0.63, Some(0.033)),
        ("pooled_strong", 0.0399, 3, 0.50, 0.71, Some(0.028)),
        ("signature_gap", 0.0622, 2, 0.01, 0.62, None),
    ];
    for (name, regret, hits, within, pooled, bias) in want {
        let d = &s.predictors[name];
        close(d.regret, regret, 5e-5);
        assert_eq!(d.hits, hits, "{name}");
        close(d.within_spearman, within, 5e-3);
        close(d.pooled_spearman.unwrap(), pooled, 5e-3);
        match (d.bias, bias) {
            (Some(a), Some(b)) => close(a, b, 5e-4),
            (None, None) => {}
            other => panic!("{name}: bias {other:?}"),
        }
    }
    close(s.always_low, 0.0099, 5e-5);
    close(s.always_ultra, 0.0526, 5e-5);
    close(s.t2_seed_std, 0.0109, 5e-5);
}

#[test]
fn landmark_picks_match_the_plan() {
    let want = [
        ("197_cpu_act", "low", ["low", "low", "low", "ultra"]),
        (
            "218_house_8L",
            "ultra",
            ["ultra", "ultra", "ultra", "ultra"],
        ),
        ("225_puma8NH", "low", ["ultra", "low", "high", "low"]),
        ("344_mv", "low", ["ultra", "low", "sum", "ultra"]),
        ("574_house_16H", "med", ["high", "med", "high", "high"]),
        ("adult", "med", ["low", "sum", "sum", "low"]),
        ("car_evaluation", "med", ["sum", "sum", "sum", "sum"]),
        ("churn", "med", ["high", "sum", "low", "low"]),
        ("mushroom", "sum", ["med", "ultra", "med", "high"]),
    ];
    let rows = landmark_rows();
    for (table, best, picks) in want {
        let r = rows.iter().find(|r| r.table == table).expect(table);
        let (b, _, p) = landmarks::table_picks(r);
        assert_eq!(b, best, "{table}");
        assert_eq!(p, picks, "{table}");
    }
}

#[test]
fn landmark_profiles_and_diagnostics_match_the_plan() {
    let rows = landmark_rows();
    let row = |t: &str| rows.iter().find(|r| r.table == t).expect(t);
    let prof = |t: &str| landmarks::profile(row(t));
    close(prof("car_evaluation").interaction, 0.52, 5e-3);
    close(prof("churn").interaction, 0.51, 5e-3);
    close(prof("197_cpu_act").linearity, 0.76, 5e-3);
    close(prof("197_cpu_act").interaction, -0.02, 5e-3);
    close(prof("churn").repeat_max_diff, 0.144, 5e-4);
    close(row("344_mv").real_profile["knn1"], 0.90, 5e-3);
    // §2.2 D: the 16 KiB program of 574_house_16H lost interaction structure.
    let r = row("574_house_16H");
    let (med, high) = (r.program("med"), r.program("high"));
    close(med.t2_family[0]["mlp"], 0.86, 5e-3);
    close(high.t2_family[0]["mlp"], 0.71, 5e-3);
    close(med.tstr["tree3"].ret.unwrap(), 0.78, 5e-3);
    close(high.tstr["tree3"].ret.unwrap(), 0.47, 5e-3);
    // §5.5b: kNN-1 transfer retention is structurally low.
    let worst_knn1 = rows
        .iter()
        .flat_map(|r| {
            PROGS
                .iter()
                .filter_map(move |p| r.program(p).tstr["knn1"].ret)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(worst_knn1 < 0.0);
    // §5.5a: on 197_cpu_act the engine's linear auditor falls far more than linear landmarks.
    let raw = landmarks::load_rawlin(&probes_dir().join("results/landmark-rawlin.jsonl"))
        .expect("rawlin");
    let r197 = raw.iter().find(|r| r.table == "197_cpu_act").expect("197");
    close(r197.programs["low"].unwrap(), 0.82, 5e-3);
    close(r197.programs["med"].unwrap(), 0.78, 5e-3);
    let c = row("197_cpu_act");
    close(c.program("low").t2_family[0]["linear"], 0.96, 5e-3);
    close(c.program("med").t2_family[0]["linear"], 0.66, 5e-3);
}

#[test]
fn landmark_replay_matches_committed_output() {
    let rows = landmark_rows();
    assert_eq!(landmarks::render(&rows), golden("landmark-replay.md"));
    let raw = landmarks::load_rawlin(&probes_dir().join("results/landmark-rawlin.jsonl"))
        .expect("rawlin");
    let diag = format!(
        "{}\n{}\n{}",
        landmarks::render_diff(&rows, "574_house_16H", "med", "high").expect("574"),
        landmarks::render_diff(&rows, "225_puma8NH", "low", "ultra").expect("225"),
        landmarks::render_rawlin(&rows, &raw)
    );
    assert_eq!(diag, golden("landmark-diagnostics.md"));
}

#[test]
fn mlp_auditor_varies_on_a_fixed_program() {
    // Repeated `dope eval` of one program on fixed files: linear and GBDT auditors agree to
    // rounding, the CPU MLP auditor does not (its initialization is not seeded).
    let text =
        fs::read_to_string(probes_dir().join("results/auditor-repeat.jsonl")).expect("receipts");
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("json"))
        .collect();
    for (table, min_spread) in [("197_cpu_act", 0.03), ("574_house_16H", 0.02)] {
        let group: Vec<&serde_json::Value> = rows.iter().filter(|r| r["table"] == table).collect();
        assert!(group.len() >= 3, "{table}");
        let one = |r: &serde_json::Value| r["program_sha256"].as_str().map(str::to_owned);
        assert!(
            group.iter().all(|r| one(r) == one(group[0])),
            "{table}: one program"
        );
        let spread = |name: &str| {
            let v: Vec<f64> = group
                .iter()
                .map(|r| r["per_auditor"][name].as_f64().expect(name))
                .collect();
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().copied().fold(f64::INFINITY, f64::min)
        };
        assert!(spread("linear") < 1e-9 && spread("gbdt") < 1e-9, "{table}");
        assert!(
            spread("mlp") >= min_spread,
            "{table}: mlp spread {}",
            spread("mlp")
        );
    }
}

#[test]
fn rerun_comparison_matches_committed_output() {
    let rerun =
        levels::load(&probes_dir().join("results/levels-probe-rust-rerun.jsonl")).expect("rerun");
    let out = levels::render_compare(&levels_rows(), &rerun);
    assert!(out.contains("identical programs: 25/36; largest per-cell change 0.176"));
    assert_eq!(out, golden("levels-rerun-compare.md"));
}

#[test]
fn errors_carry_repair_hints_and_existing_docs() {
    use dope_probes::Error;
    let errors = [
        Error::msg("x"),
        Error::engine("x"),
        Error::io(std::path::Path::new("p"), std::io::Error::other("x")),
    ];
    for e in errors {
        assert!(!e.repair_hint().is_empty(), "{e}");
        let doc = e.docs_url().split('#').next().unwrap_or_default();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(doc);
        assert!(path.exists(), "{}", path.display());
    }
}
