//! End-to-end CLI tests for `dope fit --format v8|v8-analytic|auto-v8` and `DPA1` tag `8`
//! decoding in `dope sample` (including `--v8-weights`, `--v8-zero-init`, `--anonymize`,
//! target-last CSV round-trip, determinism, and fail-closed fingerprint/truncation refusals).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_dope"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap()
}

fn ok(dir: &Path, args: &[&str]) -> Output {
    let out = run(dir, args);
    assert!(
        out.status.success(),
        "dope {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn refused(dir: &Path, args: &[&str]) -> String {
    let out = run(dir, args);
    assert_eq!(out.status.code(), Some(1), "dope {args:?} did not exit 1");
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        err.starts_with("error: ") && !err.contains("panicked"),
        "{err}"
    );
    err
}

fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dope-auto-v8-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn write_mixed(dir: &Path, n: usize) {
    let mut rng = Lcg(77);
    let mut csv = String::from("age,city,label,spend\n");
    for _ in 0..n {
        let age = 20 + (rng.next() * 50.0) as u32;
        let city = ["paris", "lima", "oslo"][(rng.next() * 3.0) as usize];
        let spend = rng.next() * 3.0 - 1.5;
        let z = 0.08 * (age as f64 - 45.0) + spend + if city == "oslo" { 1.0 } else { -0.4 };
        let label = if z + rng.next() - 0.5 > 0.0 {
            "yes"
        } else {
            "no"
        };
        csv += &format!("{age},{city},{label},{spend:.3}\n");
    }
    std::fs::write(dir.join("mixed.csv"), csv).unwrap();
}

/// Build a valid zero-initialized canonical F32 `rank <= 2` SingleEncV8 SafeTensors checkpoint.
fn build_zero_v8_safetensors(d_model: usize, seed_bias: f32) -> Vec<u8> {
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let add_lin = |m: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   name: &str,
                   in_d: usize,
                   out_d: usize,
                   val: f32| {
        m.insert(
            format!("{name}.weight"),
            (vec![out_d, in_d], vec![val; out_d * in_d]),
        );
        m.insert(format!("{name}.bias"), (vec![out_d], vec![0.0; out_d]));
    };
    let add_ln = |m: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>, name: &str, d: usize| {
        m.insert(format!("{name}.weight"), (vec![d], vec![1.0; d]));
        m.insert(format!("{name}.bias"), (vec![d], vec![0.0; d]));
    };

    // Owner encoder (1 block) + Receiver decoder (1 block)
    add_lin(&mut tensors, "enc_col_in", 128, d_model, 0.0);
    add_lin(&mut tensors, "enc_glob_in", 160, d_model, 0.0);
    for pfx in ["enc_blocks.0", "dec_blocks.0"] {
        let edge_d = if pfx.starts_with("enc") { 16 } else { 4 };
        add_ln(&mut tensors, &format!("{pfx}.norm1"), d_model);
        add_lin(
            &mut tensors,
            &format!("{pfx}.q_proj"),
            d_model,
            d_model,
            0.0,
        );
        add_lin(
            &mut tensors,
            &format!("{pfx}.k_proj"),
            d_model,
            d_model,
            0.0,
        );
        add_lin(
            &mut tensors,
            &format!("{pfx}.v_proj"),
            d_model,
            d_model,
            0.0,
        );
        add_lin(
            &mut tensors,
            &format!("{pfx}.o_proj"),
            d_model,
            d_model,
            0.0,
        );
        add_lin(&mut tensors, &format!("{pfx}.edge_proj"), edge_d, 4, 0.0);
        add_ln(&mut tensors, &format!("{pfx}.norm2"), d_model);
        add_lin(
            &mut tensors,
            &format!("{pfx}.ff1"),
            d_model,
            d_model * 2,
            0.0,
        );
        add_lin(
            &mut tensors,
            &format!("{pfx}.ff2"),
            d_model * 2,
            d_model,
            0.0,
        );
    }
    add_ln(&mut tensors, "enc_final_norm", d_model);
    add_lin(&mut tensors, "enc_code_head", d_model, 64, seed_bias);

    add_lin(&mut tensors, "dec_col_in", 48, d_model, 0.0);
    add_lin(&mut tensors, "dec_glob_in", 32, d_model, 0.0);
    add_lin(&mut tensors, "dec_code_in", 64, d_model, 0.0);
    add_ln(&mut tensors, "dec_final_norm", d_model);
    add_lin(&mut tensors, "delta_knots", d_model, 17, 0.0);
    add_lin(&mut tensors, "delta_L", d_model, 12, 0.0);
    add_lin(&mut tensors, "delta_W", d_model, 8, 0.0);
    add_lin(&mut tensors, "node_sel", d_model, 24, 0.0);
    add_lin(&mut tensors, "node_thr", d_model, 24, 0.0);
    add_lin(&mut tensors, "node_leaf", d_model, 512, 0.0);
    add_lin(&mut tensors, "bilin_u", d_model, 4, 0.0);
    add_lin(&mut tensors, "bilin_v", d_model, 4, 0.0);
    add_lin(&mut tensors, "bilin_w", d_model, 32, 0.0);
    add_lin(&mut tensors, "flow_scale_head", d_model, 2, 0.0);
    add_lin(&mut tensors, "flow_shift_head", d_model, 2, 0.0);
    add_lin(&mut tensors, "gate_head", d_model, 2, 0.0);

    let mut body = Vec::new();
    let mut header_map = serde_json::Map::new();
    for (name, (shape, data)) in tensors {
        let start = body.len();
        for v in data {
            body.extend_from_slice(&v.to_le_bytes());
        }
        let end = body.len();
        header_map.insert(
            name,
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end]
            }),
        );
    }
    let mut hdr = serde_json::to_vec(&header_map).unwrap();
    while hdr.len() % 8 != 0 {
        hdr.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr.len() + body.len());
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&body);
    out
}

#[test]
fn v8_analytic_and_auto_v8_round_trip_within_2kib_and_deterministic() {
    let dir = workdir("analytic");
    write_mixed(&dir, 300);

    // 1. Explicit weightless DSE8 analytic (`--format v8-analytic`), target in middle -> output target-last!
    let fit = ok(
        &dir,
        &[
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--format",
            "v8-analytic",
            "--budget",
            "2048",
            "--seed",
            "11",
            "--json",
            "-o",
            "v8a.dope",
        ],
    );
    let art = std::fs::read(dir.join("v8a.dope")).unwrap();
    assert_eq!(&art[..4], b"DPA1");
    assert_eq!(art[4], 8, "DPA1 tag must be 8 for V8");
    assert_eq!(
        &art[5..13],
        &[0u8; 8],
        "analytic V8 must have zero fingerprint"
    );
    assert!(art.len() <= 2048, "artifact exceeded 2048 B: {}", art.len());
    let rep: serde_json::Value = serde_json::from_slice(&fit.stdout).unwrap();
    assert_eq!(rep["encoder"], "v8");
    assert_eq!(rep["choice"], "Dse8Analytic");

    // Deterministic sample twice + verify target-last header & valid categorical labels.
    let s1 = ok(&dir, &["sample", "v8a.dope", "-n", "60", "--seed", "7"]);
    let s2 = ok(&dir, &["sample", "v8a.dope", "-n", "60", "--seed", "7"]);
    assert_eq!(s1.stdout, s2.stdout);
    let text = String::from_utf8(s1.stdout).unwrap();
    let rows: Vec<&str> = text.lines().collect();
    assert_eq!(rows.len(), 61);
    assert_eq!(rows[0], "age,city,spend,label");
    for row in &rows[1..] {
        let cols: Vec<&str> = row.split(',').collect();
        assert_eq!(cols.len(), 4);
        assert!(cols[1].is_empty() || ["paris", "lima", "oslo"].contains(&cols[1]));
        assert!(["yes", "no"].contains(&cols[3]));
    }

    // 2. 3-way `--format auto-v8` + `--anonymize` stays strictly <= 2048 B and hides column names!
    ok(
        &dir,
        &[
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--format",
            "auto-v8",
            "--anonymize",
            "--budget",
            "2048",
            "--seed",
            "13",
            "-q",
            "-o",
            "autov8.dope",
        ],
    );
    let auto_art = std::fs::read(dir.join("autov8.dope")).unwrap();
    assert!(auto_art.len() <= 2048);
    let s_anon = ok(&dir, &["sample", "autov8.dope", "-n", "25", "--seed", "3"]);
    let anon_text = String::from_utf8(s_anon.stdout).unwrap();
    assert!(!anon_text.contains("paris") && !anon_text.contains("spend"));
    assert_eq!(anon_text.lines().count(), 26);

    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn v8_neural_weights_fingerprint_round_trip_and_fail_closed_refusals() {
    let dir = workdir("neural");
    write_mixed(&dir, 260);
    let w1 = build_zero_v8_safetensors(16, 0.0);
    let w2 = build_zero_v8_safetensors(16, 0.01);
    std::fs::write(dir.join("w1.safetensors"), &w1).unwrap();
    std::fs::write(dir.join("w2.safetensors"), &w2).unwrap();

    // Fit neural V8 DPA1 tag-8 artifact with w1.
    ok(
        &dir,
        &[
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--format",
            "v8",
            "--v8-weights",
            "w1.safetensors",
            "--budget",
            "2048",
            "--seed",
            "5",
            "-q",
            "-o",
            "nn.dope",
        ],
    );
    let art = std::fs::read(dir.join("nn.dope")).unwrap();
    assert_eq!(&art[..4], b"DPA1");
    assert_eq!(art[4], 8);
    assert_ne!(
        &art[5..13],
        &[0u8; 8],
        "neural V8 must carry non-zero fingerprint"
    );
    assert!(art.len() <= 2048);

    // Sampling with matching w1 succeeds.
    let s_ok = ok(
        &dir,
        &[
            "sample",
            "nn.dope",
            "-n",
            "30",
            "--seed",
            "2",
            "--v8-weights",
            "w1.safetensors",
        ],
    );
    assert_eq!(String::from_utf8(s_ok.stdout).unwrap().lines().count(), 31);

    // Sampling with --v8-zero-init also succeeds (analytic fallback).
    let s_z = ok(
        &dir,
        &[
            "sample",
            "nn.dope",
            "-n",
            "30",
            "--seed",
            "2",
            "--v8-zero-init",
        ],
    );
    assert_eq!(String::from_utf8(s_z.stdout).unwrap().lines().count(), 31);

    // Sampling without weights OR with mismatched w2 MUST fail closed!
    let e_miss = refused(&dir, &["sample", "nn.dope", "-n", "10"]);
    assert!(e_miss.contains("requires SingleEncV8 weights"), "{e_miss}");
    let e_wrong = refused(
        &dir,
        &[
            "sample",
            "nn.dope",
            "-n",
            "10",
            "--v8-weights",
            "w2.safetensors",
        ],
    );
    assert!(e_wrong.contains("fingerprint mismatch"), "{e_wrong}");

    // Truncated DPA1 V8 artifact MUST fail closed!
    std::fs::write(dir.join("cut.dope"), &art[..art.len() / 2]).unwrap();
    let _ = refused(
        &dir,
        &[
            "sample",
            "cut.dope",
            "-n",
            "10",
            "--v8-weights",
            "w1.safetensors",
        ],
    );

    // Cross-version flag guards fail closed!
    let e_flag = refused(
        &dir,
        &[
            "fit",
            "mixed.csv",
            "-t",
            "label",
            "--format",
            "1",
            "--v8-weights",
            "w1.safetensors",
            "-o",
            "x.dope",
        ],
    );
    assert!(e_flag.contains("--v8-weights applies"), "{e_flag}");

    std::fs::remove_dir_all(dir).ok();
}
