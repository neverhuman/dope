//! Black-box tests of the `dope_statpacket` binary: fit, info, truncate and
//! decode through files only, plus the failure paths.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_dope_statpacket");

fn dir(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("statpacket_cli_{name}"));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

fn ok_json(args: &[&str]) -> serde_json::Value {
    let out = run(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// A small deterministic generator (splitmix64) so the test needs no library code.
fn unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

fn write_mat(path: &Path, rows: usize, x_cols: usize, y_cols: usize, data: &[f32]) {
    assert_eq!(data.len(), rows * (x_cols + y_cols));
    let mut bytes = b"DOPEMAT1".to_vec();
    for v in [rows, x_cols, y_cols, 0] {
        bytes.extend((v as u32).to_le_bytes());
    }
    for v in data {
        bytes.extend(v.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

/// `(rows, x_cols, y_cols, cells)` of a DOPEMAT1 file.
fn read_mat(path: &Path) -> (usize, usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"DOPEMAT1");
    let word = |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap());
    let cells = bytes[24..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (word(0) as usize, word(1) as usize, word(2) as usize, cells)
}

/// 400 rows, 10 features (smooth, skewed, integer, binary, constant) and one target column.
fn fixture(target: impl Fn(usize, f64) -> f32) -> Vec<f32> {
    let mut state = 20_261_122u64;
    let mut data = Vec::new();
    for row in 0..400 {
        let common = unit(&mut state) - 0.5;
        for col in 0..10 {
            let u = unit(&mut state);
            data.push(match col {
                0..=4 => (common + u + col as f64) as f32,
                5 => (-(1.0 - u).ln() * 250.0) as f32,
                6 => (u * 40.0).floor() as f32,
                7 => (u < 0.3) as u8 as f32,
                8 => 12.5,
                _ => (common * 4.0 + u) as f32,
            });
        }
        data.push(target(row, common));
    }
    data
}

#[test]
fn fit_info_truncate_and_decode_through_files() {
    let d = dir("roundtrip");
    let (input, packet, card) = (d.join("a.mat"), d.join("p.dse"), d.join("card.json"));
    write_mat(&input, 400, 10, 1, &fixture(|_, c| (3.0 * c) as f32));
    let fit_args = |packet: &Path, extra: &[&str]| -> Vec<String> {
        let mut args: Vec<String> = [
            "fit",
            "--input",
            text(&input),
            "--task",
            "regression",
            "--packet",
            text(packet),
            "--card",
            text(&card),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        args.extend(extra.iter().map(|s| s.to_string()));
        args
    };
    let call = |v: Vec<String>| ok_json(&v.iter().map(String::as_str).collect::<Vec<_>>());
    let report = call(fit_args(&packet, &[]));
    let bytes = std::fs::read(&packet).unwrap();
    assert_eq!(report["status"], "ok");
    assert_eq!(report["level"], "E");
    assert_eq!(report["packet_bytes"], bytes.len());
    assert_eq!(report["ledger"]["total"], bytes.len());
    assert_eq!(report["ledger"]["constant_columns"], 1);
    assert_eq!(report["ledger"]["discrete_columns"], 1);
    assert!(bytes.len() <= 2048 && &bytes[..4] == b"DSE1");
    let info = ok_json(&["info", "--packet", text(&packet)]);
    assert_eq!(info["ledger"], report["ledger"]);
    assert_eq!(info["packet_sha256"], report["packet_sha256"]);

    // Fitting twice gives the same bytes; a level cap equals truncation of the full packet.
    let again = d.join("again.dse");
    call(fit_args(&again, &[]));
    assert_eq!(std::fs::read(&again).unwrap(), bytes);
    for level in ["A", "B", "C", "D"] {
        let (capped, cut) = (d.join("capped.dse"), d.join("cut.dse"));
        let capped_report = call(fit_args(&capped, &["--level", level]));
        ok_json(&[
            "truncate",
            "--packet",
            text(&packet),
            "--level",
            level,
            "--output",
            text(&cut),
        ]);
        assert_eq!(capped_report["level"], level);
        assert_eq!(
            std::fs::read(&capped).unwrap(),
            std::fs::read(&cut).unwrap()
        );
    }
    // A tighter budget degrades the level instead of failing.
    let small = d.join("small.dse");
    let small_report = call(fit_args(&small, &["--max-bytes", "300"]));
    assert_eq!(small_report["status"], "ok");
    assert!(std::fs::read(&small).unwrap().len() <= 300);
    assert_ne!(small_report["level"], "E");

    // Decode: whole range equals two chunks; repeated runs are identical.
    let decode = |out: &Path, rows: &str, row0: &str, with_card: bool| {
        let mut args = vec![
            "decode",
            "--packet",
            text(&packet),
            "--rows",
            rows,
            "--row0",
            row0,
            "--seed",
            "7",
            "--output",
            text(out),
        ];
        if with_card {
            args.extend(["--card", text(&card)]);
        }
        ok_json(&args)
    };
    let (whole, head, tail, repeat, plain) = (
        d.join("w.mat"),
        d.join("h.mat"),
        d.join("t.mat"),
        d.join("r.mat"),
        d.join("u.mat"),
    );
    assert_eq!(decode(&whole, "1000", "0", true)["units"], "raw");
    decode(&head, "500", "0", true);
    decode(&tail, "500", "500", true);
    decode(&repeat, "1000", "0", true);
    assert_eq!(decode(&plain, "1000", "0", false)["units"], "unit");
    let (rows, x_cols, y_cols, cells) = read_mat(&whole);
    assert_eq!((rows, x_cols, y_cols), (1000, 10, 0));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut joined = read_mat(&head).3;
    joined.extend(read_mat(&tail).3);
    assert_eq!(bits(&cells), bits(&joined));
    assert_eq!(
        std::fs::read(&whole).unwrap(),
        std::fs::read(&repeat).unwrap()
    );
    // Raw units: the constant column is restored, the binary and integer columns stay on their levels.
    for row in cells.chunks(10) {
        assert_eq!(row[8], 12.5);
        assert!(row[7] == 0.0 || row[7] == 1.0);
        assert_eq!(row[6], row[6].round());
        assert!(row.iter().all(|v| v.is_finite()));
    }
    // Unit space: every cell within the clip, the constant column at 0.
    let unit_cells = read_mat(&plain).3;
    assert!(unit_cells.iter().all(|v| v.abs() <= 6.0));
    assert!(unit_cells.chunks(10).all(|row| row[8] == 0.0));
}

#[test]
fn classification_inputs_and_failures() {
    let d = dir("failures");
    let (input, packet, card) = (d.join("a.mat"), d.join("p.dse"), d.join("card.json"));
    write_mat(&input, 400, 10, 1, &fixture(|_, c| (c > 0.1) as u8 as f32));
    let fit = |task: &str, classes: &str| {
        run(&[
            "fit",
            "--input",
            text(&input),
            "--task",
            task,
            "--classes",
            classes,
            "--packet",
            text(&packet),
            "--card",
            text(&card),
        ])
    };
    let out = fit("binary", "0");
    assert!(out.status.success());
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["level"], "E");
    // Binary adds one prior byte and eight class-mean bytes per rank block.
    assert_eq!(report["ledger"]["sections"][0], 3);
    let card_text = std::fs::read_to_string(&card).unwrap();
    assert!(card_text.contains("\"kind\": \"classes\""));

    // A label outside the declared classes: exit code 2, static message, no value echoed.
    let mut poisoned = fixture(|_, c| (c > 0.1) as u8 as f32);
    poisoned[10] = 31_337.0;
    write_mat(&input, 400, 10, 1, &poisoned);
    let out = fit("binary", "0");
    assert_eq!(out.status.code(), Some(2));
    let message = String::from_utf8(out.stderr).unwrap();
    assert_eq!(
        message.trim(),
        "dope_statpacket: stat packet: a label is outside the declared classes"
    );
    assert!(out.stdout.is_empty() && !message.contains("31337") && !message.contains("a.mat"));

    // Unknown task and a damaged packet.
    assert_eq!(fit("ranking", "0").status.code(), Some(2));
    write_mat(&input, 400, 10, 1, &fixture(|_, c| c as f32));
    assert!(fit("regression", "0").status.success());
    let mut bytes = std::fs::read(&packet).unwrap();
    let at = bytes.len() / 2;
    bytes[at] ^= 0x10;
    let damaged = d.join("damaged.dse");
    std::fs::write(&damaged, &bytes).unwrap();
    let out = run(&[
        "decode",
        "--packet",
        text(&damaged),
        "--rows",
        "5",
        "--output",
        text(&d.join("x.mat")),
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        String::from_utf8(out.stderr).unwrap().trim(),
        "dope_statpacket: single-encoding: checksum mismatch"
    );
    assert!(!d.join("x.mat").exists());
    let out = run(&["info", "--packet", text(&d.join("missing.dse"))]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        String::from_utf8(out.stderr).unwrap().trim(),
        "dope_statpacket: stat packet: file error (NotFound)"
    );
}
