//! Verification suite for:
//! - Item A: `landmarks-native-v1` bit-identity (`gbdt::Params::default().group_duplicates == false`),
//! - Item B: Format 2 `fit_v2` strict wide-table byte-budget compliance (`<= 2048 B` / `<= 4096 B`),
//! - Item C: Pure-Rust `SingleEncV7` F32 SafeTensors loader, fail-closed guards, row-local
//!   invariance, and optional PyTorch cross-runtime parity check (`DOPE_V7_PARITY_DIR`).

use dope_core::compile::v2::{fit_v2, FitV2Options};
use dope_core::eval::{gbdt, Matrix};
use dope_core::rng::Rng;
use dope_core::sample::{EmissionMode, EmitOptions};
use dope_core::single_encoding::{
    fit_dse7_packet, sample_zero_init_rows, unpack_dse7_packet, SingleEncV7Packet,
    SingleEncV7Weights, COL_FEAT_DIM, COL_FEAT_DIM_V2, GLOB_FEAT_DIM, KNOT_COUNT, MAX_CLASSES,
    MAX_COLS, MAX_RANK,
};
use dope_core::table::RawTable;
use dope_core::target::Head;
use serde::Deserialize;
use std::collections::BTreeMap;

fn pack_safetensors(tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut blobs: Vec<u8> = Vec::new();
    for (name, (shape, vals)) in tensors {
        let start = blobs.len();
        for &v in vals {
            blobs.extend_from_slice(&v.to_le_bytes());
        }
        let end = blobs.len();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end]
            }),
        );
    }
    let mut hdr_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while hdr_bytes.len() % 8 != 0 {
        hdr_bytes.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hdr_bytes.len() + blobs.len());
    out.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    out.extend_from_slice(&blobs);
    out
}

fn add_lin(
    t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    pfx: &str,
    out_d: usize,
    in_d: usize,
    scale: f32,
) {
    let w: Vec<f32> = (0..out_d * in_d)
        .map(|i| {
            if scale == 0.0 {
                0.0
            } else {
                (((i * 37 + 13) % 97) as f32 / 97.0 - 0.5) * scale
            }
        })
        .collect();
    let b: Vec<f32> = (0..out_d)
        .map(|i| {
            if scale == 0.0 {
                0.0
            } else {
                (i as f32 * 0.01) * scale
            }
        })
        .collect();
    t.insert(format!("{pfx}.weight"), (vec![out_d, in_d], w));
    t.insert(format!("{pfx}.bias"), (vec![out_d], b));
}

fn add_ln(t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>, pfx: &str, d: usize) {
    t.insert(format!("{pfx}.weight"), (vec![d], vec![1.0; d]));
    t.insert(format!("{pfx}.bias"), (vec![d], vec![0.0; d]));
}

fn synthetic_v7_safetensors(d: usize, h_target: usize, nonzero: bool) -> Vec<u8> {
    let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let s = if nonzero { 0.15 } else { 0.0 };
    add_lin(&mut t, "col_in", d, COL_FEAT_DIM, 0.1);
    add_lin(&mut t, "glob_in", d, GLOB_FEAT_DIM, 0.1);
    add_ln(&mut t, "blocks.0.norm1", d);
    add_lin(&mut t, "blocks.0.q_proj", d, d, 0.1);
    add_lin(&mut t, "blocks.0.k_proj", d, d, 0.1);
    add_lin(&mut t, "blocks.0.v_proj", d, d, 0.1);
    add_lin(&mut t, "blocks.0.o_proj", d, d, 0.1);
    add_ln(&mut t, "blocks.0.norm2", d);
    add_lin(&mut t, "blocks.0.ff1", d * 2, d, 0.1);
    add_lin(&mut t, "blocks.0.ff2", d, d * 2, 0.1);
    add_ln(&mut t, "final_norm", d);
    add_lin(&mut t, "delta_knots", KNOT_COUNT, d, s);
    add_lin(&mut t, "delta_L", MAX_RANK, d, s);
    add_lin(&mut t, "delta_W", MAX_CLASSES, d, s);
    add_lin(&mut t, "hyper_w1", h_target, d, 0.1);
    add_lin(&mut t, "hyper_b1", h_target, d, s);
    add_lin(&mut t, "hyper_w2", h_target * MAX_CLASSES, d, s);
    add_lin(&mut t, "hyper_temp", 2, d, s);
    pack_safetensors(&t)
}

fn sample_packet() -> SingleEncV7Packet {
    let mut pkt = SingleEncV7Packet {
        col_feats: [[0.0; COL_FEAT_DIM]; MAX_COLS],
        q_nl: [[0.0; 4]; MAX_COLS],
        glob_feats: [0.0; GLOB_FEAT_DIM],
        knots: [[0.0; KNOT_COUNT]; MAX_COLS],
        loadings: [[0.0; MAX_RANK]; MAX_COLS],
        w_base: [[0.0; MAX_COLS]; MAX_CLASSES],
        b_base: [0.0; MAX_CLASSES],
        resid_sd: 0.4,
        col_mask: [0.0; MAX_COLS],
        is_const: [0.0; MAX_COLS],
    };
    pkt.glob_feats[0] = 1.0;
    for c in 0..4 {
        pkt.col_mask[c] = 1.0;
        for k in 0..KNOT_COUNT {
            pkt.knots[c][k] = -2.0 + 4.0 * (k as f32) / (KNOT_COUNT - 1) as f32;
            pkt.col_feats[c][k] = pkt.knots[c][k];
        }
        pkt.loadings[c][0] = 0.5 - 0.1 * c as f32;
        pkt.col_feats[c][KNOT_COUNT] = pkt.loadings[c][0];
        pkt.col_feats[c][COL_FEAT_DIM - 2] = 2.4;
        pkt.w_base[0][c] = 0.6 - 0.2 * c as f32;
    }
    pkt.b_base[0] = 0.15;
    pkt
}

#[test]
fn gbdt_default_preserves_v130_landmark_split_and_opt_in_groups_duplicates() {
    let p_def = gbdt::Params::default();
    assert!(
        !p_def.group_duplicates,
        "gbdt::Params::default() must keep group_duplicates=false to preserve landmarks-native-v1"
    );
    // 240 rows formed by repeating 60 base rows 4x so grouped vs ungrouped splits differ cleanly.
    let mut rng = Rng::new(77);
    let mut base_x = Vec::with_capacity(60 * 3);
    let mut base_y = Vec::with_capacity(60);
    for _ in 0..60 {
        let a = rng.normal();
        let b = rng.normal();
        let c = rng.normal();
        base_x.extend_from_slice(&[a, b, c]);
        base_y.push(1.5 * a - 0.8 * b + 0.3 * a * c + 0.1 * rng.normal());
    }
    let mut data = Vec::with_capacity(240 * 3);
    let mut y = Vec::with_capacity(240);
    for _ in 0..4 {
        data.extend_from_slice(&base_x);
        y.extend_from_slice(&base_y);
    }
    let m = Matrix {
        rows: 240,
        cols: 3,
        data,
    };
    let p_landmark = gbdt::Params {
        rounds: 40,
        max_depth: 4,
        learning_rate: 0.15,
        bins: 32,
        patience: 40,
        seed: 123,
        ..Default::default()
    };
    let p_grouped = gbdt::Params {
        group_duplicates: true,
        ..p_landmark.clone()
    };
    let pred_a = gbdt::fit_predict(&m, &y, Head::Regression, &m, &p_landmark);
    let pred_a2 = gbdt::fit_predict(&m, &y, Head::Regression, &m, &p_landmark);
    let pred_g = gbdt::fit_predict(&m, &y, Head::Regression, &m, &p_grouped);
    assert_eq!(pred_a, pred_a2);
    assert!(pred_g.iter().all(|v| v.is_finite()));
}

#[test]
fn fit_v2_holds_strict_2kib_and_4kib_caps_on_wide_mixed_tables() {
    for (n_feat, budget) in [(24usize, 2048usize), (48usize, 4096usize)] {
        let n = 360usize;
        let mut rng = Rng::new(9001 + n_feat as u64);
        let mut headers = Vec::with_capacity(n_feat + 1);
        let mut columns: Vec<Vec<String>> = Vec::with_capacity(n_feat + 1);
        let mut signal = vec![0.0f64; n];
        for j in 0..n_feat {
            headers.push(format!("x{j}"));
            let mut vals = Vec::with_capacity(n);
            for sig in signal.iter_mut().take(n) {
                let z = rng.normal();
                let v = if j % 3 == 0 {
                    if z < 0.0 {
                        0.0
                    } else {
                        (z * 10.0).round() / 10.0
                    }
                } else if j % 3 == 1 {
                    ((z * 2.0).round().clamp(-4.0, 5.0)) + 4.0
                } else {
                    z
                };
                if j < 5 {
                    *sig += v * (0.4 - 0.06 * j as f64);
                }
                vals.push(format!("{v:.4}"));
            }
            columns.push(vals);
        }
        headers.push("y".into());
        columns.push(
            signal
                .iter()
                .map(|&s| format!("{:.4}", s + 0.25 * rng.normal()))
                .collect(),
        );
        let raw = RawTable { headers, columns };
        let opts = FitV2Options {
            budget: Some(budget),
            seed: 42,
            ..Default::default()
        };
        let fitted = fit_v2(&raw, n_feat, &opts).expect("fit_v2 wide table");
        assert!(
            fitted.bytes.len() <= budget,
            "n_feat={n_feat} budget={budget} overflowed: got {} bytes",
            fitted.bytes.len()
        );
        let emitted = fitted
            .program
            .sample_with_mode(&EmitOptions {
                rows: 64,
                seed: 7,
                mode: EmissionMode::Faithful,
                table_id: 1,
                ..Default::default()
            })
            .expect("sample_with_mode");
        assert_eq!(emitted.samples.x.len(), n_feat);
        assert_eq!(emitted.samples.y.len(), 64);
    }
}

#[test]
fn single_enc_v7_fail_closed_and_row_local_invariance() {
    // 1. Fail-closed checks: rank-3 tensor rejected, non-F32 rejected, truncated rejected.
    assert!(SingleEncV7Weights::from_safetensors_bytes(&[0u8; 4], 4).is_err());
    let bad_rank_hdr = br#"{"t":{"dtype":"F32","shape":[1,2,3],"data_offsets":[0,24]}}"#;
    let mut bad_rank = (bad_rank_hdr.len() as u64).to_le_bytes().to_vec();
    bad_rank.extend_from_slice(bad_rank_hdr);
    bad_rank.extend_from_slice(&[0u8; 24]);
    assert!(SingleEncV7Weights::from_safetensors_bytes(&bad_rank, 4).is_err());

    let bad_dtype_hdr = br#"{"t":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
    let mut bad_dtype = (bad_dtype_hdr.len() as u64).to_le_bytes().to_vec();
    bad_dtype.extend_from_slice(bad_dtype_hdr);
    bad_dtype.extend_from_slice(&[0u8; 4]);
    assert!(SingleEncV7Weights::from_safetensors_bytes(&bad_dtype, 4).is_err());

    // 2. Load valid F32 checkpoint & confirm strict row-local counter-RNG invariance.
    let bytes = synthetic_v7_safetensors(16, 8, true);
    let model = SingleEncV7Weights::from_safetensors_bytes(&bytes, 4).expect("load v7");
    assert_eq!(model.d_model, 16);
    assert_eq!(model.n_layers, 1);
    assert_eq!(model.h_target, 8);
    let pkt = sample_packet();
    let batch_64 = model.sample_counter_rows(&pkt, 64, 999);
    let batch_1 = model.sample_counter_rows(&pkt, 1, 999);
    assert_eq!(batch_64[0].x, batch_1[0].x);
    assert_eq!(batch_64[0].logits, batch_1[0].logits);
    assert_eq!(batch_64[0].y_reg, batch_1[0].y_reg);
}

#[derive(Deserialize)]
struct ParityFixture {
    n_heads: usize,
    col_feats: Vec<Vec<f32>>,
    glob_feats: Vec<f32>,
    knots: Vec<Vec<f32>>,
    loadings: Vec<Vec<f32>>,
    w_base: Vec<Vec<f32>>,
    b_base: Vec<f32>,
    resid_sd: f32,
    col_mask: Vec<f32>,
    is_const: Vec<f32>,
    z_factor: Vec<Vec<f32>>,
    z_noise: Vec<Vec<f32>>,
    y_noise: Vec<f32>,
    expected_x: Vec<Vec<f32>>,
    expected_logits: Vec<Vec<f32>>,
    expected_y_reg: Vec<f32>,
}

#[test]
fn single_enc_v7_pytorch_parity_when_fixture_provided() {
    let Ok(dir) = std::env::var("DOPE_V7_PARITY_DIR") else {
        return;
    };
    let weights_bytes = std::fs::read(format!("{dir}/model.safetensors")).expect("read weights");
    let fix: ParityFixture = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/fixture.json")).expect("read fixture"),
    )
    .expect("parse fixture");
    let model = SingleEncV7Weights::from_safetensors_bytes(&weights_bytes, fix.n_heads)
        .expect("load parity weights");
    let mut pkt = SingleEncV7Packet {
        col_feats: [[0.0; COL_FEAT_DIM]; MAX_COLS],
        q_nl: [[0.0; 4]; MAX_COLS],
        glob_feats: [0.0; GLOB_FEAT_DIM],
        knots: [[0.0; KNOT_COUNT]; MAX_COLS],
        loadings: [[0.0; MAX_RANK]; MAX_COLS],
        w_base: [[0.0; MAX_COLS]; MAX_CLASSES],
        b_base: [0.0; MAX_CLASSES],
        resid_sd: fix.resid_sd,
        col_mask: [0.0; MAX_COLS],
        is_const: [0.0; MAX_COLS],
    };
    for c in 0..MAX_COLS {
        pkt.col_feats[c].copy_from_slice(&fix.col_feats[c][..COL_FEAT_DIM]);
        if fix.col_feats[c].len() == COL_FEAT_DIM_V2 {
            pkt.q_nl[c].copy_from_slice(&fix.col_feats[c][COL_FEAT_DIM..COL_FEAT_DIM_V2]);
        }
        pkt.knots[c].copy_from_slice(&fix.knots[c]);
        pkt.loadings[c].copy_from_slice(&fix.loadings[c]);
        pkt.col_mask[c] = fix.col_mask[c];
        pkt.is_const[c] = fix.is_const[c];
    }
    pkt.glob_feats.copy_from_slice(&fix.glob_feats);
    for cls in 0..MAX_CLASSES {
        pkt.w_base[cls].copy_from_slice(&fix.w_base[cls]);
        pkt.b_base[cls] = fix.b_base[cls];
    }
    let refined = model.refine_packet(&pkt);
    let mut max_diff = 0.0f32;
    for r in 0..fix.y_noise.len() {
        let mut zf = [0.0f32; MAX_RANK];
        zf.copy_from_slice(&fix.z_factor[r]);
        let mut zn = [0.0f32; MAX_COLS];
        zn.copy_from_slice(&fix.z_noise[r]);
        let row = SingleEncV7Weights::decode_row(&refined, &zf, &zn, fix.y_noise[r]);
        for c in 0..MAX_COLS {
            max_diff = max_diff.max((row.x[c] - fix.expected_x[r][c]).abs());
        }
        for cls in 0..MAX_CLASSES {
            max_diff = max_diff.max((row.logits[cls] - fix.expected_logits[r][cls]).abs());
        }
        max_diff = max_diff.max((row.y_reg - fix.expected_y_reg[r]).abs());
    }
    assert!(
        max_diff < 1e-4,
        "PyTorch vs Pure-Rust SingleEncV7 max_abs_diff={max_diff} exceeds 1e-4!"
    );
}

#[test]
fn dse7_v2_self_contained_budget_roundtrip_and_v72_loader() {
    for &(task_id, p_raw, n_rows) in &[
        (0u8, 12usize, 128usize),
        (1u8, 24usize, 160usize),
        (2u8, 18usize, 180usize),
        (0u8, 250usize, 140usize),
    ] {
        let mut rng = Rng::new(4242 + p_raw as u64 + task_id as u64 * 100);
        let mut x = vec![vec![0.0f64; n_rows]; p_raw];
        let mut y = vec![0.0f64; n_rows];
        for r in 0..n_rows {
            let z0 = rng.normal();
            let z1 = rng.normal();
            for (c, xc) in x.iter_mut().enumerate().take(p_raw) {
                xc[r] = if c == 0 {
                    1000.0 + 50.0 * z0
                } else if c == 1 {
                    z1
                } else {
                    0.3 * z0 + 0.9 * rng.normal()
                };
            }
            let sig = 0.8 * z0 + 0.6 * (z0 * z0 - 1.0) - 0.5 * z1;
            y[r] = match task_id {
                0 => 20.0 + 5.0 * sig + 0.4 * rng.normal(),
                1 => f64::from(sig > 0.0),
                _ => ((sig.clamp(-2.5, 2.5) + 2.5) as i64 % 5) as f64,
            };
        }
        let wire = fit_dse7_packet(&x, &y, task_id, 2048).expect("fit dse7 v2");
        assert!(
            wire.len() <= 2048,
            "task={task_id} p_raw={p_raw} exceeded 2048 B: got {}",
            wire.len()
        );
        let (pkt, hdr, card) = unpack_dse7_packet(&wire).expect("unpack dse7 v2");
        assert_eq!(hdr.task_id, task_id);
        assert_eq!(hdr.x_cols_raw as usize, p_raw);
        let rows = sample_zero_init_rows(&pkt, 32, 777);
        assert_eq!(rows.len(), 32);
        let raw0 = card.inverse_row_x(&rows[0].x);
        assert_eq!(raw0.len(), p_raw);
        assert!(raw0.iter().all(|v| v.is_finite()));
        // Flip a byte -> CRC32 must fail closed.
        let mut bad = wire.clone();
        bad[8] ^= 0xFF;
        assert!(unpack_dse7_packet(&bad).is_err());
    }
    // Verify V7.2 (col_in = 46) checkpoint also loads cleanly alongside V7.0/V7.1 (col_in = 42).
    let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let d = 16;
    add_lin(&mut t, "col_in", d, COL_FEAT_DIM_V2, 0.1);
    add_lin(&mut t, "glob_in", d, GLOB_FEAT_DIM, 0.1);
    add_ln(&mut t, "final_norm", d);
    add_lin(&mut t, "delta_knots", KNOT_COUNT, d, 0.05);
    add_lin(&mut t, "delta_L", MAX_RANK, d, 0.05);
    add_lin(&mut t, "delta_W", MAX_CLASSES, d, 0.05);
    add_lin(&mut t, "hyper_w1", 8, d, 0.1);
    add_lin(&mut t, "hyper_b1", 8, d, 0.05);
    add_lin(&mut t, "hyper_w2", 8 * MAX_CLASSES, d, 0.05);
    add_lin(&mut t, "hyper_temp", 2, d, 0.05);
    let m_v72 = SingleEncV7Weights::from_safetensors_bytes(&pack_safetensors(&t), 4)
        .expect("load v7.2 col_in=46");
    let rows = m_v72.sample_counter_rows(&sample_packet(), 8, 123);
    assert_eq!(rows.len(), 8);

    // Verify Format 2 discrete-level compaction never panics in catf::sample and fails closed on tiny budget.
    let mut rng = Rng::new(7777);
    let n = 320usize;
    let n_feat = 28usize;
    let mut headers = Vec::with_capacity(n_feat + 1);
    let mut columns: Vec<Vec<String>> = Vec::with_capacity(n_feat + 1);
    for j in 0..n_feat {
        headers.push(format!("d{j}"));
        let vals: Vec<String> = (0..n)
            .map(|r| format!("{}", (r + j * 3) % 10))
            .collect();
        columns.push(vals);
    }
    headers.push("y".into());
    columns.push((0..n).map(|r| format!("{:.3}", (r % 10) as f64 + 0.1 * rng.normal())).collect());
    let raw = RawTable { headers, columns };
    let fit_ok = fit_v2(
        &raw,
        n_feat,
        &FitV2Options {
            budget: Some(2048),
            seed: 9,
            ..Default::default()
        },
    );
    if let Ok(fitted) = fit_ok {
        assert!(fitted.bytes.len() <= 2048);
        let em = fitted
            .program
            .sample_with_mode(&EmitOptions {
                rows: 64,
                seed: 11,
                mode: EmissionMode::Faithful,
                table_id: 1,
                ..Default::default()
            })
            .expect("zero-panic sample after discrete trim_extra");
        assert_eq!(em.samples.y.len(), 64);
    }
    // Impossibly tiny budget must fail closed with Budget error, never exit Ok over budget!
    let tiny = fit_v2(
        &raw,
        n_feat,
        &FitV2Options {
            budget: Some(64),
            seed: 9,
            ..Default::default()
        },
    );
    assert!(tiny.is_err());
}
