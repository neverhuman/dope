//! Shared FIT evidence, `ScaleCardV8::forward_y` f64, and `DPA1` wrapper-budget correspondence
//! between training reserved packs and serving `fit_dpa1_v8` packs across canonical and renamed
//! schemas, plus historical checkpoint preservation (`active_rank_v1 == false`).

use dope_core::auto::v8::AutoV8Options;
use dope_core::auto::v8_budget::{
    compare_reserved_and_serving_dpa1, dpa_v8_wrapper_bytes_from_json,
    estimate_dpa_v8_wrapper_bytes, DpaV8KindSpec, DpaV8SchemaSpec,
};
use dope_core::auto::v8_dpa::{fit_dpa1_v8, V8DpaMode, V8Model};
use dope_core::evidence::EvidenceTier;
use dope_core::rng::Rng;
use dope_core::schema::Task;
use dope_core::single_encoding::v8::{
    extract_fit_evidence_v8, fit_dse8_packet, fit_dse8_packet_with_encoder, fit_unit_meta,
    unpack_dse8_packet, CodeSectionV8, Dse8FitOptions, ScaleCardV8, SingleEncV8Weights,
    DEC_COL_DIM, DEC_EDGE_DIM, DEC_GLOB_DIM, V8_MAX_CLASSES, V8_MAX_RANK,
};
use dope_core::single_encoding::KNOT_COUNT;
use dope_core::table::RawTable;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn build_v8_ckpt(active_rank_v1: bool) -> Vec<u8> {
    let (d_model, n_heads, code_dim) = (32usize, 4usize, 64usize);
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let add_lin = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   name: &str,
                   out_d: usize,
                   in_d: usize| {
        let mut w = vec![0.0f32; out_d * in_d];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = ((i % 7) as f32 - 3.0) * 0.015625;
        }
        t.insert(format!("{name}.weight"), (vec![out_d, in_d], w));
        t.insert(format!("{name}.bias"), (vec![out_d], vec![0.0f32; out_d]));
    };
    let add_ln = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>, name: &str, d: usize| {
        t.insert(format!("{name}.weight"), (vec![d], vec![1.0f32; d]));
        t.insert(format!("{name}.bias"), (vec![d], vec![0.0f32; d]));
    };
    add_lin(&mut tensors, "enc_col_in", d_model, 128);
    add_lin(&mut tensors, "enc_glob_in", d_model, 160);
    add_ln(&mut tensors, "enc_blocks.0.norm1", d_model);
    add_lin(&mut tensors, "enc_blocks.0.q_proj", d_model, d_model);
    add_lin(&mut tensors, "enc_blocks.0.k_proj", d_model, d_model);
    add_lin(&mut tensors, "enc_blocks.0.v_proj", d_model, d_model);
    add_lin(&mut tensors, "enc_blocks.0.o_proj", d_model, d_model);
    add_lin(&mut tensors, "enc_blocks.0.edge_proj", n_heads, 16);
    add_ln(&mut tensors, "enc_blocks.0.norm2", d_model);
    add_lin(&mut tensors, "enc_blocks.0.ff1", d_model * 2, d_model);
    add_lin(&mut tensors, "enc_blocks.0.ff2", d_model, d_model * 2);
    add_ln(&mut tensors, "enc_final_norm", d_model);
    add_lin(&mut tensors, "enc_code_head", code_dim, d_model);

    add_lin(&mut tensors, "dec_col_in", d_model, DEC_COL_DIM);
    add_lin(&mut tensors, "dec_glob_in", d_model, DEC_GLOB_DIM);
    add_lin(&mut tensors, "dec_code_in", d_model, code_dim);
    add_ln(&mut tensors, "dec_blocks.0.norm1", d_model);
    add_lin(&mut tensors, "dec_blocks.0.q_proj", d_model, d_model);
    add_lin(&mut tensors, "dec_blocks.0.k_proj", d_model, d_model);
    add_lin(&mut tensors, "dec_blocks.0.v_proj", d_model, d_model);
    add_lin(&mut tensors, "dec_blocks.0.o_proj", d_model, d_model);
    add_lin(
        &mut tensors,
        "dec_blocks.0.edge_proj",
        n_heads,
        DEC_EDGE_DIM,
    );
    add_ln(&mut tensors, "dec_blocks.0.norm2", d_model);
    add_lin(&mut tensors, "dec_blocks.0.ff1", d_model * 2, d_model);
    add_lin(&mut tensors, "dec_blocks.0.ff2", d_model, d_model * 2);
    add_ln(&mut tensors, "dec_final_norm", d_model);

    add_lin(&mut tensors, "delta_knots", KNOT_COUNT, d_model);
    add_lin(&mut tensors, "delta_L", V8_MAX_RANK, d_model);
    add_lin(&mut tensors, "delta_W", V8_MAX_CLASSES, d_model);
    add_lin(&mut tensors, "node_sel", 24, d_model);
    add_lin(&mut tensors, "node_thr", 24, d_model);
    add_lin(&mut tensors, "node_leaf", 8 * 8 * V8_MAX_CLASSES, d_model);
    add_lin(&mut tensors, "bilin_u", 4, d_model);
    add_lin(&mut tensors, "bilin_v", 4, d_model);
    add_lin(&mut tensors, "bilin_w", 4 * V8_MAX_CLASSES, d_model);
    add_lin(&mut tensors, "flow_scale_head", 2, d_model);
    add_lin(&mut tensors, "flow_shift_head", 2, d_model);
    add_lin(&mut tensors, "gate_head", 2, d_model);

    let mut hdr = serde_json::Map::new();
    if active_rank_v1 {
        hdr.insert(
            "__metadata__".into(),
            serde_json::json!({ "dope_dse8_decoder": "active-rank-v1" }),
        );
    }
    let mut raw = Vec::new();
    if active_rank_v1 {
        let start = raw.len();
        raw.push(1u8);
        hdr.insert(
            "__dope_dse8_decoder_active_rank_v1__".into(),
            serde_json::json!({
                "dtype": "U8",
                "shape": [1],
                "data_offsets": [start, raw.len()],
            }),
        );
    }
    for (name, (shape, vals)) in &tensors {
        let start = raw.len();
        for &v in vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        hdr.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, raw.len()],
            }),
        );
    }
    let hb = serde_json::to_vec(&hdr).unwrap();
    let mut out = Vec::with_capacity(8 + hb.len() + raw.len());
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(&hb);
    out.extend_from_slice(&raw);
    out
}

fn make_raw_table(names: &[String], x: &[Vec<f64>], y: &[f64]) -> RawTable {
    let mut headers = names.to_vec();
    headers.push("target".to_string());
    let mut columns: Vec<Vec<String>> = x
        .iter()
        .map(|col| col.iter().map(|&v| format!("{v:.6}")).collect())
        .collect();
    columns.push(y.iter().map(|&v| format!("{v:.6}")).collect());
    RawTable { headers, columns }
}

#[test]
fn shared_dpa_budget_and_fit_evidence_match_serving_canonical_and_renamed() {
    let mut rng = Rng::new(174);
    let (n, p) = (128usize, 48usize);
    let mut x = vec![vec![0.0f64; n]; p];
    let mut y = vec![0.0f64; n];
    for r in 0..n {
        for col in x.iter_mut() {
            col[r] = (rng.normal() * 1e6).round() / 1e6;
        }
        y[r] = ((x[0][r] - x[1][r] + rng.normal()) * 1e6).round() / 1e6;
    }

    let model = V8Model::from_safetensors_auto(&build_v8_ckpt(true)).expect("active-rank-v1");
    let canonical_names: Vec<String> = (0..p).map(|j| format!("c{j}")).collect();
    let renamed_names: Vec<String> = (0..p)
        .map(|j| {
            let h = Sha256::digest(j.to_string().as_bytes());
            format!(
                "measurement_{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                h[0], h[1], h[2], h[3], h[4], h[5]
            )
        })
        .collect();

    let mut ranks = Vec::new();
    for (names, anonymize) in [
        (&canonical_names, false),
        (&renamed_names, false),
        (&renamed_names, true),
    ] {
        let spec = DpaV8SchemaSpec {
            feature_names: names.clone(),
            feature_kinds: Some(vec![DpaV8KindSpec::Real; p]),
            target_name: Some("target".into()),
            target_labels: None,
        };
        let spec_json = serde_json::to_string(&spec).unwrap();
        let wrap_bytes = dpa_v8_wrapper_bytes_from_json(&spec_json, anonymize).unwrap();
        if names == &canonical_names && !anonymize {
            assert_eq!(wrap_bytes, estimate_dpa_v8_wrapper_bytes(&x, &y, 0));
        }

        let raw_tbl = make_raw_table(names, &x, &y);
        let mut opts = AutoV8Options {
            budget: 2048,
            code_tier: 1,
            ..Default::default()
        };
        opts.format1.seed = 4242;
        opts.format1.task = Some(Task::Regression);
        opts.format1.anonymize = anonymize;
        let served = fit_dpa1_v8(&raw_tbl, p, &opts, Some(&model), V8DpaMode::V8).unwrap();

        let meta = fit_unit_meta(&x, &y, 0, false).unwrap();
        let reserved = fit_dse8_packet(
            &x,
            &y,
            &Dse8FitOptions {
                task_id: 0,
                budget_bytes: 2048 - wrap_bytes,
                code_tier: 1,
                reserve_neural_code: true,
                col_meta: Some(&meta),
                seed: 4242,
                ..Default::default()
            },
        )
        .unwrap();

        let cmp = compare_reserved_and_serving_dpa1(&reserved, &served.bytes).unwrap();
        assert_eq!(cmp.wrapper_bytes, wrap_bytes);
        assert_eq!(cmp.reserved_inner_bytes, cmp.serving_inner_bytes);
        assert_eq!(
            wrap_bytes + cmp.reserved_inner_bytes,
            cmp.serving_total_bytes
        );
        assert!(cmp.header_bytes_equal, "header bytes differed");
        assert!(cmp.section_tags_equal, "section tags differed");
        assert!(cmp.section_lengths_equal, "section lengths differed");
        assert!(
            cmp.non_neural_sections_byte_equal,
            "non-neural TLV sections differed: {:?}",
            cmp.sections
        );
        ranks.push((wrap_bytes, cmp.rank_use));

        // Verify shared FIT evidence + FSQ code matches serving's encoded code byte-for-byte!
        let srv_pkt = unpack_dse8_packet(&served.bytes[wrap_bytes..]).unwrap();
        let card =
            ScaleCardV8::fit(&x, &y, 0, srv_pkt.header.p_eff as usize, false, Some(&meta)).unwrap();
        let x_unit_all = card.forward_x(&x);
        let (y_sc, valid_rows) = card.forward_y(&y);
        let x_unit_valid: Vec<Vec<f32>> = x_unit_all
            .iter()
            .map(|col| valid_rows.iter().map(|&r| col[r]).collect())
            .collect();
        let active_meta: Vec<_> = card.col_indices.iter().map(|&j| meta[j]).collect();
        let row_ids: Vec<u64> = (0..n as u64).collect();
        let ev = extract_fit_evidence_v8(
            &x,
            &x_unit_valid,
            &y_sc,
            &valid_rows,
            &card,
            Some(&active_meta),
            EvidenceTier::T2,
            4242,
            Some(&row_ids),
            true,
        )
        .unwrap();
        let expected_code =
            CodeSectionV8::from_unit_vec(&model.weights().encode_evidence_code(&ev).unwrap(), 1);
        assert_eq!(srv_pkt.code.packed, expected_code.packed);
    }
    // Canonical (91 B) keeps higher rank than renamed unanonymized (~390 B); anonymize recovers it!
    assert!(ranks[1].0 > ranks[0].0);
    assert!(ranks[0].1 > ranks[1].1);
    assert!(ranks[2].0 < ranks[1].0);
}

#[test]
fn historical_checkpoint_preserves_prior_evidence_behavior() {
    let mut rng = Rng::new(99);
    let (n, p) = (64usize, 6usize);
    let mut x = vec![vec![0.0f64; n]; p];
    let mut y = vec![0.0f64; n];
    for (r, yr) in y.iter_mut().enumerate() {
        for col in x.iter_mut() {
            col[r] = if r % 4 == 0 { 1.0 } else { rng.normal() };
        }
        *yr = if x[0][r] + 0.5 * rng.normal() > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let w_hist = SingleEncV8Weights::from_safetensors_auto(&build_v8_ckpt(false)).unwrap();
    assert!(!w_hist.active_rank_v1);
    let meta = fit_unit_meta(&x, &y, 1, false).unwrap();
    let ids: Vec<u64> = (0..n as u64).rev().collect();
    let pkt_none = fit_dse8_packet_with_encoder(
        &x,
        &y,
        &Dse8FitOptions {
            task_id: 1,
            code_tier: 1,
            col_meta: Some(&meta),
            fit_row_ids: None,
            seed: 17,
            ..Default::default()
        },
        &w_hist,
    )
    .unwrap();
    let pkt_ids = fit_dse8_packet_with_encoder(
        &x,
        &y,
        &Dse8FitOptions {
            task_id: 1,
            code_tier: 1,
            col_meta: Some(&meta),
            fit_row_ids: Some(&ids),
            seed: 17,
            ..Default::default()
        },
        &w_hist,
    )
    .unwrap();
    assert_eq!(
        pkt_none, pkt_ids,
        "historical checkpoint must ignore fit_row_ids"
    );
}

#[test]
fn active_rank_v1_checkpoints_and_packets_fail_closed_under_v300_rules() {
    use dope_core::single_encoding::v8::{build_decode_plan_v8, FLAG_ACTIVE_RANK_V1};

    // 1. F32 active-rank-v1 checkpoint carries the U8 guard tensor refused by v3.0.0 F32-only loaders.
    let ckpt_active = build_v8_ckpt(true);
    let hlen = u64::from_le_bytes(ckpt_active[..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&ckpt_active[8..8 + hlen]).unwrap();
    let marker = hdr
        .get("__dope_dse8_decoder_active_rank_v1__")
        .expect("missing U8 guard tensor");
    assert_eq!(marker["dtype"], "U8");
    assert_eq!(marker["shape"], serde_json::json!([1]));
    let w_active = SingleEncV8Weights::from_safetensors_auto(&ckpt_active).unwrap();
    assert!(w_active.active_rank_v1);

    // Marker without matching __metadata__ MUST be refused.
    let mut bad_hdr = hdr.clone();
    bad_hdr.remove("__metadata__");
    let bad_hb = serde_json::to_vec(&bad_hdr).unwrap();
    let mut bad_ckpt = Vec::new();
    bad_ckpt.extend_from_slice(&(bad_hb.len() as u64).to_le_bytes());
    bad_ckpt.extend_from_slice(&bad_hb);
    bad_ckpt.extend_from_slice(&ckpt_active[8 + hlen..]);
    assert!(SingleEncV8Weights::from_safetensors_auto(&bad_ckpt).is_err());

    // Metadata-only active-rank-v1 WITHOUT U8 guard tensor MUST also be refused!
    let mut meta_only_hdr = hdr.clone();
    meta_only_hdr.remove("__dope_dse8_decoder_active_rank_v1__");
    let mo_hb = serde_json::to_vec(&meta_only_hdr).unwrap();
    let mut mo_ckpt = Vec::new();
    mo_ckpt.extend_from_slice(&(mo_hb.len() as u64).to_le_bytes());
    mo_ckpt.extend_from_slice(&mo_hb);
    mo_ckpt.extend_from_slice(&ckpt_active[8 + hlen..]);
    assert!(SingleEncV8Weights::from_safetensors_auto(&mo_ckpt).is_err());

    // 2. Packets fitted with active-rank-v1 set FLAG_ACTIVE_RANK_V1 (0x40), outside v3.0.0's 0x3F mask.
    let mut rng = Rng::new(300);
    let (n, p) = (64usize, 6usize);
    let mut x = vec![vec![0.0f64; n]; p];
    let mut y = vec![0.0f64; n];
    for (r, yr) in y.iter_mut().enumerate() {
        for col in x.iter_mut() {
            col[r] = rng.normal();
        }
        *yr = x[0][r] - 0.5 * x[1][r] + 0.2 * rng.normal();
    }
    let meta = fit_unit_meta(&x, &y, 0, false).unwrap();
    let pkt_active_b = fit_dse8_packet_with_encoder(
        &x,
        &y,
        &Dse8FitOptions {
            code_tier: 1,
            col_meta: Some(&meta),
            seed: 5,
            ..Default::default()
        },
        &w_active,
    )
    .unwrap();
    assert_ne!(
        pkt_active_b[10] & FLAG_ACTIVE_RANK_V1,
        0,
        "active-rank-v1 packet must set flag bit 0x40 refused by v3.0.0"
    );
    let pkt_hist_b = fit_dse8_packet(&x, &y, &Dse8FitOptions::default()).unwrap();
    assert_eq!(pkt_hist_b[10] & FLAG_ACTIVE_RANK_V1, 0);

    // Historical checkpoint MUST refuse to decode an active-rank-v1 packet!
    let w_hist = SingleEncV8Weights::from_safetensors_auto(&build_v8_ckpt(false)).unwrap();
    let pkt_active = unpack_dse8_packet(&pkt_active_b).unwrap();
    assert!(build_decode_plan_v8(&pkt_active, Some(&w_hist)).is_err());
    assert!(build_decode_plan_v8(&pkt_active, Some(&w_active)).is_ok());
}
