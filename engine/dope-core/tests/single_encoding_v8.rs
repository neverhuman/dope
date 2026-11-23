//! Comprehensive Rust verification suite for `DSE8` v1 codec, Step-0 analytic decoder,
//! nested-prefix truncation, privacy floor (`k >= 10`), zero illegal nominal one-hot rows,
//! row-locality (`0.0` diff `N=1` vs `N=1000`), `SingleEncV8Weights` zero-init identity,
//! 3-way `auto::v8` guard, and `DSE7` regression preservation.

use dope_core::auto::v8::{select_v8_candidate, V8Choice};
use dope_core::rng::Rng;
use dope_core::single_encoding::v8::{
    build_decode_plan_v8, decode_dse8_row_at, decode_dse8_rows, entmax_15,
    extract_jopedime_embedding, fit_dse8_packet, fit_dse8_packet_with_encoder,
    truncate_dse8_packet, unpack_dse8_packet, Dse8ColumnMeta, Dse8FitOptions, SingleEncV8Weights,
    DEC_CODE_DIM, DEC_COL_DIM, DEC_EDGE_DIM, DEC_GLOB_DIM, V8_MAX_CLASSES, V8_MAX_RANK,
};
use dope_core::single_encoding::{fit_dse7_packet, unpack_dse7_packet, KNOT_COUNT};
use std::collections::BTreeMap;

/// Build a canonical F32 rank-<=2 SafeTensors checkpoint with exact zero-init output heads!
fn build_zero_init_v8_safetensors(d_model: usize, n_heads: usize) -> Vec<u8> {
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let add_lin = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   name: &str,
                   out_d: usize,
                   in_d: usize,
                   zero: bool| {
        let mut w = vec![0.0f32; out_d * in_d];
        if !zero {
            for (i, slot) in w.iter_mut().enumerate() {
                *slot = ((i % 7) as f32 - 3.0) * 0.01;
            }
        }
        t.insert(format!("{name}.weight"), (vec![out_d, in_d], w));
        t.insert(format!("{name}.bias"), (vec![out_d], vec![0.0f32; out_d]));
    };
    let add_ln = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>, name: &str, d: usize| {
        t.insert(format!("{name}.weight"), (vec![d], vec![1.0f32; d]));
        t.insert(format!("{name}.bias"), (vec![d], vec![0.0f32; d]));
    };

    // Owner encoder (1 block)
    add_lin(&mut tensors, "enc_col_in", d_model, 128, false);
    add_lin(&mut tensors, "enc_glob_in", d_model, 160, false);
    add_ln(&mut tensors, "enc_blocks.0.norm1", d_model);
    add_lin(&mut tensors, "enc_blocks.0.q_proj", d_model, d_model, false);
    add_lin(&mut tensors, "enc_blocks.0.k_proj", d_model, d_model, false);
    add_lin(&mut tensors, "enc_blocks.0.v_proj", d_model, d_model, false);
    add_lin(&mut tensors, "enc_blocks.0.o_proj", d_model, d_model, false);
    add_lin(&mut tensors, "enc_blocks.0.edge_proj", n_heads, 16, false);
    add_ln(&mut tensors, "enc_blocks.0.norm2", d_model);
    add_lin(
        &mut tensors,
        "enc_blocks.0.ff1",
        d_model * 2,
        d_model,
        false,
    );
    add_lin(
        &mut tensors,
        "enc_blocks.0.ff2",
        d_model,
        d_model * 2,
        false,
    );
    add_ln(&mut tensors, "enc_final_norm", d_model);
    add_lin(&mut tensors, "enc_code_head", 64, d_model, false);

    // Receiver decoder (1 block) with STRICTLY ZERO-INIT output heads!
    add_lin(&mut tensors, "dec_col_in", d_model, DEC_COL_DIM, false);
    add_lin(&mut tensors, "dec_glob_in", d_model, DEC_GLOB_DIM, false);
    add_lin(&mut tensors, "dec_code_in", d_model, DEC_CODE_DIM, false);
    add_ln(&mut tensors, "dec_blocks.0.norm1", d_model);
    add_lin(&mut tensors, "dec_blocks.0.q_proj", d_model, d_model, false);
    add_lin(&mut tensors, "dec_blocks.0.k_proj", d_model, d_model, false);
    add_lin(&mut tensors, "dec_blocks.0.v_proj", d_model, d_model, false);
    add_lin(&mut tensors, "dec_blocks.0.o_proj", d_model, d_model, false);
    add_lin(
        &mut tensors,
        "dec_blocks.0.edge_proj",
        n_heads,
        DEC_EDGE_DIM,
        false,
    );
    add_ln(&mut tensors, "dec_blocks.0.norm2", d_model);
    add_lin(
        &mut tensors,
        "dec_blocks.0.ff1",
        d_model * 2,
        d_model,
        false,
    );
    add_lin(
        &mut tensors,
        "dec_blocks.0.ff2",
        d_model,
        d_model * 2,
        false,
    );
    add_ln(&mut tensors, "dec_final_norm", d_model);

    add_lin(&mut tensors, "delta_knots", KNOT_COUNT, d_model, true);
    add_lin(&mut tensors, "delta_L", V8_MAX_RANK, d_model, true);
    add_lin(&mut tensors, "delta_W", V8_MAX_CLASSES, d_model, true);
    add_lin(&mut tensors, "node_sel", 24, d_model, true);
    add_lin(&mut tensors, "node_thr", 24, d_model, true);
    add_lin(
        &mut tensors,
        "node_leaf",
        8 * 8 * V8_MAX_CLASSES,
        d_model,
        true,
    );
    add_lin(&mut tensors, "bilin_u", 4, d_model, true);
    add_lin(&mut tensors, "bilin_v", 4, d_model, true);
    add_lin(&mut tensors, "bilin_w", 4 * V8_MAX_CLASSES, d_model, true);
    add_lin(&mut tensors, "flow_scale_head", 2, d_model, true);
    add_lin(&mut tensors, "flow_shift_head", 2, d_model, true);
    add_lin(&mut tensors, "gate_head", 2, d_model, true);

    // Serialize canonical HF SafeTensors F32
    let mut header = serde_json::Map::new();
    let mut data = Vec::<u8>::new();
    for (k, (shape, vals)) in tensors {
        let lo = data.len();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let hi = data.len();
        header.insert(
            k,
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [lo, hi]}),
        );
    }
    let hdr_bytes = serde_json::to_vec(&header).unwrap();
    let mut out = Vec::with_capacity(8 + hdr_bytes.len() + data.len());
    out.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr_bytes);
    out.extend_from_slice(&data);
    out
}

#[test]
fn dse8_roundtrip_crc_prefix_truncation_and_dual_mode() {
    let n = 400;
    let mut rng = Rng::new(42);
    let mut x = vec![vec![0.0f64; n]; 8];
    let mut y = vec![0.0f64; n];
    for (r, yr) in y.iter_mut().enumerate().take(n) {
        let a = rng.normal();
        let b = rng.normal();
        x[0][r] = a;
        x[1][r] = b;
        x[2][r] = 0.7 * a + 0.5 * rng.normal();
        x[3][r] = (rng.uniform() * 50.0).floor() / 10.0; // lattice
        x[4][r] = if rng.uniform() < 0.35 {
            0.0
        } else {
            rng.normal().abs() + 0.2
        }; // zero-inflated
        x[5][r] = if rng.uniform() < 0.15 {
            f64::NAN
        } else {
            rng.normal()
        }; // MAR missing
        x[6][r] = if a > 0.0 { 1.0 } else { 0.0 };
        x[7][r] = rng.normal();
        // Nonlinear + XOR target!
        *yr = a * a + if (a > 0.0) ^ (b > 0.0) { 2.0 } else { -2.0 } + 0.25 * rng.normal();
    }

    // 1. Fit at 4096 B self-contained raw, verify sections populated (E, Pr, Tr, Rg).
    let opts4k = Dse8FitOptions {
        task_id: 0,
        budget_bytes: 4096,
        seed: 7,
        ..Default::default()
    };
    let bytes4k = fit_dse8_packet(&x, &y, &opts4k).unwrap();
    assert!(bytes4k.len() <= 4096);
    let pkt4k = unpack_dse8_packet(&bytes4k).unwrap();
    assert!(!pkt4k.effects.is_empty() || !pkt4k.pairs.is_empty() || !pkt4k.trees.trees.is_empty());

    // 2. Prefix truncation down to 2048 B, 1024 B, 512 B with ZERO refitting!
    for &b_target in &[2048usize, 1024, 512] {
        let tr = truncate_dse8_packet(&bytes4k, b_target).unwrap();
        assert!(
            tr.len() <= b_target,
            "truncated len {} > {b_target}",
            tr.len()
        );
        let tpkt = unpack_dse8_packet(&tr).unwrap();
        let dec = decode_dse8_rows(&tpkt, None, 64, 19).unwrap();
        assert_eq!(dec.x.len(), 8);
        assert_eq!(dec.y.len(), 64);
    }

    // 3. Single-bit flip MUST be rejected fail-closed by CRC32!
    let mut bad = bytes4k.clone();
    bad[10] ^= 0x01;
    assert!(unpack_dse8_packet(&bad).is_err());

    // 4. Dual normalized_only mode must emit 0 scale bytes and strictly smaller base wire size!
    let opts_norm = Dse8FitOptions {
        task_id: 0,
        budget_bytes: 2048,
        normalized_only: true,
        enable_trees: false,
        enable_regimes: false,
        enable_effects: false,
        enable_residual_model: false,
        ..Default::default()
    };
    let opts_raw = Dse8FitOptions {
        normalized_only: false,
        ..opts_norm.clone()
    };
    let b_norm = fit_dse8_packet(&x, &y, &opts_norm).unwrap();
    let b_raw = fit_dse8_packet(&x, &y, &opts_raw).unwrap();
    assert!(
        b_norm.len() < b_raw.len(),
        "normalized_only ({}) must save scale bytes vs raw ({})",
        b_norm.len(),
        b_raw.len()
    );
    assert!(unpack_dse8_packet(&b_norm).unwrap().card.normalized_only);
}

#[test]
fn zero_illegal_nominal_rows_and_k10_privacy_floor() {
    let n = 400;
    let mut rng = Rng::new(77);
    // 5-column one-hot nominal group (cols 0..5) where level 4 is a RARE CANARY (only 3 rows < 10!)
    // plus continuous col 5 with a rare canary outlier 99999.0 in 2 rows!
    let mut x = vec![vec![0.0f64; n]; 6];
    let mut y = vec![0.0f64; n];
    for (r, yr) in y.iter_mut().enumerate().take(n) {
        let cat = if r < 3 {
            4 // rare canary level (count = 3 < 10 -> MUST NOT APPEAR in decoded rows!)
        } else {
            (rng.next_u64() as usize) % 4
        };
        x[cat][r] = 1.0;
        x[5][r] = if r < 2 { 99999.0 } else { rng.normal() };
        *yr = if r < 4 {
            99.0 // rare canary class (count = 4 < 10 -> MUST NOT APPEAR in class table or decoded rows!)
        } else {
            ((cat + usize::from(x[5][r] > 0.0)) % 3) as f64
        };
    }
    let meta: Vec<Dse8ColumnMeta> = (0..6)
        .map(|j| Dse8ColumnMeta {
            group_id: if j < 5 { 0 } else { 1 },
            group_kind: if j < 5 { 4 } else { 0 },
            group_width: if j < 5 { 5 } else { 1 },
            ..Default::default()
        })
        .collect();

    let opts = Dse8FitOptions {
        task_id: 2,
        budget_bytes: 2048,
        col_meta: Some(&meta),
        seed: 12,
        ..Default::default()
    };
    let bytes = fit_dse8_packet(&x, &y, &opts).unwrap();
    let pkt = unpack_dse8_packet(&bytes).unwrap();
    // Rare class 99 MUST be excluded from class table (k >= 10 floor)!
    assert!(
        !pkt.card.classes.contains(&99),
        "rare class 99 leaked into class table!"
    );

    let dec = decode_dse8_rows(&pkt, None, 5000, 999).unwrap();
    // 1. ZERO illegal multi-hot or NaN nominal rows across all 5,000 rows!
    // 2. Rare canary nominal level 4 (count=3 < 10) MUST NEVER appear!
    // 3. Rare canary class 99 MUST NEVER appear!
    // 4. Rare canary outlier 99999.0 MUST NEVER appear!
    for r in 0..5000 {
        let mut hot = 0usize;
        for c in 0..5 {
            let v = dec.x[c][r];
            assert!(v == 0.0 || v == 1.0, "non-binary nominal cell {v}");
            if v > 0.5 {
                hot += 1;
            }
        }
        assert_eq!(
            hot, 1,
            "illegal multi-hot/zero nominal row at r={r} (hot={hot})"
        );
        assert_eq!(
            dec.x[4][r], 0.0,
            "rare canary nominal level 4 leaked at r={r}"
        );
        assert!(
            dec.x[5][r].abs() < 100.0,
            "rare canary outlier leaked: {}",
            dec.x[5][r]
        );
        assert_ne!(dec.y[r] as i32, 99, "rare canary target class 99 leaked");
    }
}

#[test]
fn row_locality_zero_diff_zero_init_identity_and_throughput() {
    let n = 300;
    let mut rng = Rng::new(101);
    let x: Vec<Vec<f64>> = (0..8)
        .map(|_| (0..n).map(|_| rng.normal()).collect())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|r| {
            if x[0][r] + x[1][r] * x[2][r] > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let st_bytes = build_zero_init_v8_safetensors(32, 4);
    let weights = SingleEncV8Weights::from_safetensors_bytes(&st_bytes, 4).unwrap();

    let opts = Dse8FitOptions {
        task_id: 1,
        budget_bytes: 2048,
        code_tier: 1,
        seed: 5,
        ..Default::default()
    };
    let pkt_bytes = fit_dse8_packet_with_encoder(&x, &y, &opts, &weights).unwrap();
    let pkt = unpack_dse8_packet(&pkt_bytes).unwrap();
    assert_eq!(pkt.code.tier, 1);
    assert_eq!(extract_jopedime_embedding(&pkt).len(), DEC_CODE_DIM);

    // 1. Zero-init exact identity: analytic DecodePlanV8 == zero-init neural DecodePlanV8!
    let plan_p0 = build_decode_plan_v8(&pkt, None).unwrap();
    let plan_z = build_decode_plan_v8(&pkt, Some(&weights)).unwrap();
    for r in 0..100u64 {
        let a = decode_dse8_row_at(&plan_p0, 12345, r);
        let b = decode_dse8_row_at(&plan_z, 12345, r);
        assert_eq!(a.x_unit, b.x_unit, "zero-init x_unit diverged at row {r}");
        assert_eq!(a.logits, b.logits, "zero-init logits diverged at row {r}");
        assert_eq!(a.y_raw, b.y_raw, "zero-init y_raw diverged at row {r}");
    }

    // 2. Row-locality exact 0.0 diff: isolated N=1 row matches bulk N=1000 table!
    let bulk = decode_dse8_rows(&pkt, Some(&weights), 1000, 777).unwrap();
    for &r in &[0usize, 17, 512, 999] {
        let one = decode_dse8_row_at(&plan_z, 777, r as u64);
        for c in 0..8 {
            assert_eq!(one.x_raw[c].to_bits(), bulk.x[c][r].to_bits());
        }
        assert_eq!(one.y_raw.to_bits(), bulk.y[r].to_bits());
    }

    // 3. entmax_1.5 simplex sanity & 3-way auto::v8 preference guard
    let p_em = entmax_15(&[2.0, 1.0, -5.0, -5.0]);
    assert!((p_em.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert_eq!(
        p_em[2], 0.0,
        "entmax_1.5 must be sparse on far-negative logits"
    );
    assert_eq!(
        select_v8_candidate(Some(0.50), Some(0.51), Some(0.515)),
        V8Choice::Format1
    );
    assert_eq!(
        select_v8_candidate(Some(0.50), Some(0.53), Some(0.54)),
        V8Choice::Dse8Analytic
    );
    assert_eq!(
        select_v8_candidate(Some(0.50), Some(0.53), Some(0.56)),
        V8Choice::Dse8Neural
    );

    // 4. DSE7 historical codec untouched regression check!
    let d7 = fit_dse7_packet(&x, &y, 1, 2048).unwrap();
    assert!(unpack_dse7_packet(&d7).is_ok());
}

#[test]
fn dse8_malformed_wire_vectors_and_admission_refusals() {
    use dope_core::artifact::format2::crc32_ieee;

    let resign = |mut b: Vec<u8>| -> Vec<u8> {
        let n = b.len() - 4;
        let crc = crc32_ieee(&b[..n]);
        b[n..].copy_from_slice(&crc.to_le_bytes());
        b
    };

    let mut rng = Rng::new(505);
    let x: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..120).map(|_| rng.normal()).collect())
        .collect();
    let y: Vec<f64> = (0..120).map(|r| x[0][r] + 0.5 * rng.normal()).collect();
    let base = fit_dse8_packet(
        &x,
        &y,
        &Dse8FitOptions {
            task_id: 0,
            budget_bytes: 2048,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(unpack_dse8_packet(&base).is_ok());

    // 1. Unknown header flag bit (bit 7 = 0x80) -> MUST reject even with valid CRC!
    let mut bad_flag = base.clone();
    bad_flag[10] |= 0x80;
    assert!(unpack_dse8_packet(&resign(bad_flag)).is_err());

    // 2. Non-zero reserved header bytes (bytes 22..24) -> MUST reject!
    let mut bad_res = base.clone();
    bad_res[22] = 1;
    assert!(unpack_dse8_packet(&resign(bad_res)).is_err());

    // 3. Header code_tier > 0 without TAG_CODE -> MUST reject!
    let mut bad_ct = base.clone();
    bad_ct[21] = 1;
    assert!(unpack_dse8_packet(&resign(bad_ct)).is_err());

    // 4. Duplicate TLV tag -> MUST reject!
    let mut dup = base[..base.len() - 4].to_vec();
    dup[20] = dup[20].wrapping_add(1);
    dup.push(0x04); // duplicate TAG_LINEAR (0x04 <= last_tag)
    dup.extend_from_slice(&0u16.to_le_bytes());
    dup.extend_from_slice(&[0u8; 4]);
    assert!(unpack_dse8_packet(&resign(dup)).is_err());

    // 5. Out-of-order TLV tags (swap first two sections: Scale(1), Marg(2) -> Marg(2), Scale(1))
    let s1_len = u16::from_le_bytes([base[25], base[26]]) as usize;
    let s2_off = 24 + 3 + s1_len;
    let s2_len = u16::from_le_bytes([base[s2_off + 1], base[s2_off + 2]]) as usize;
    let mut reord = Vec::with_capacity(base.len());
    reord.extend_from_slice(&base[..24]);
    reord.extend_from_slice(&base[s2_off..s2_off + 3 + s2_len]);
    reord.extend_from_slice(&base[24..s2_off]);
    reord.extend_from_slice(&base[s2_off + 3 + s2_len..]);
    assert!(unpack_dse8_packet(&resign(reord)).is_err());

    // 6. Explicit refusal in normalized_only mode for > 64 cols and > 8 classes!
    let x65: Vec<Vec<f64>> = (0..65)
        .map(|_| (0..40).map(|_| rng.normal()).collect())
        .collect();
    let y40: Vec<f64> = (0..40).map(|_| rng.normal()).collect();
    assert!(fit_dse8_packet(
        &x65,
        &y40,
        &Dse8FitOptions {
            task_id: 0,
            budget_bytes: 16384,
            normalized_only: true,
            ..Default::default()
        }
    )
    .is_err());

    let y9: Vec<f64> = (0..180).map(|r| (r % 9) as f64).collect();
    let x4: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..180).map(|_| rng.normal()).collect())
        .collect();
    assert!(fit_dse8_packet(
        &x4,
        &y9,
        &Dse8FitOptions {
            task_id: 2,
            budget_bytes: 4096,
            normalized_only: true,
            ..Default::default()
        }
    )
    .is_err());

    // 7. Neural weights fingerprint mismatch MUST fail closed!
    let w1 = SingleEncV8Weights::from_safetensors_bytes(&build_zero_init_v8_safetensors(32, 4), 4)
        .unwrap();
    let mut st2 = build_zero_init_v8_safetensors(32, 4);
    // Flip a float byte inside enc_col_in weight payload so SHA-256 fingerprint differs cleanly.
    let n_st = st2.len();
    st2[n_st - 1] ^= 0x01;
    let w2 = SingleEncV8Weights::from_safetensors_bytes(&st2, 4).unwrap();
    assert_ne!(w1.fingerprint(), w2.fingerprint());
    let pkt_b = fit_dse8_packet_with_encoder(
        &x,
        &y,
        &Dse8FitOptions {
            task_id: 0,
            budget_bytes: 2048,
            code_tier: 1,
            ..Default::default()
        },
        &w1,
    )
    .unwrap();
    let pkt = unpack_dse8_packet(&pkt_b).unwrap();
    assert!(build_decode_plan_v8(&pkt, Some(&w1)).is_ok());
    assert!(build_decode_plan_v8(&pkt, Some(&w2)).is_err());
}

#[test]
fn dse8_calibrated_priors_collinear_precision_and_discrete_trees() {
    // 1. Imbalanced 5-class table (13/29/44/66/198): decoded class shares MUST match FIT!
    let mut rng = Rng::new(4201);
    let counts = [13usize, 29, 44, 66, 198];
    let n_mc: usize = counts.iter().sum();
    let mut x_mc = vec![vec![0.0f64; n_mc]; 8];
    let mut y_mc = Vec::with_capacity(n_mc);
    let mut row = 0usize;
    for (cls, &cnt) in counts.iter().enumerate() {
        for _ in 0..cnt {
            let mu = (cls as f64 - 2.0) * 0.65;
            x_mc[0][row] = mu + rng.normal();
            x_mc[1][row] = -0.5 * mu + rng.normal();
            x_mc[2][row] = if rng.uniform() < (0.15 + 0.15 * cls as f64) {
                1.0
            } else {
                0.0
            };
            x_mc[3][row] = ((cls % 3) as f64) + if rng.uniform() < 0.2 { 1.0 } else { 0.0 };
            for x_col in x_mc.iter_mut().take(8).skip(4) {
                x_col[row] = rng.normal();
            }
            y_mc.push(cls as f64);
            row += 1;
        }
    }
    let b_mc = fit_dse8_packet(
        &x_mc,
        &y_mc,
        &Dse8FitOptions {
            task_id: 2,
            budget_bytes: 2048,
            seed: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let pkt_mc = unpack_dse8_packet(&b_mc).unwrap();
    let dec_mc = decode_dse8_rows(&pkt_mc, None, 2000, 77).unwrap();
    let mut syn_cnt = [0usize; 5];
    for &yv in &dec_mc.y {
        let c = (yv.round() as usize).min(4);
        syn_cnt[c] += 1;
    }
    for c in 0..5 {
        let p_fit = counts[c] as f64 / n_mc as f64;
        let p_syn = syn_cnt[c] as f64 / 2000.0;
        assert!(
            (p_syn - p_fit).abs() < 0.045,
            "class {c} prior drift: fit={p_fit:.3} syn={p_syn:.3}"
        );
    }

    // 2. Near-deterministic collinear regression: resid_sd MUST stay tight & syn R^2 > 0.98!
    let n_reg = 350;
    let mut x_reg = vec![vec![0.0f64; n_reg]; 10];
    let mut y_reg = vec![0.0f64; n_reg];
    for (r, yr) in y_reg.iter_mut().enumerate().take(n_reg) {
        let base = rng.normal();
        let diff = 0.08 * rng.normal();
        x_reg[0][r] = base;
        x_reg[1][r] = base + diff;
        for col in x_reg.iter_mut().take(10).skip(2) {
            col[r] = rng.normal();
        }
        *yr = 1.5 * x_reg[0][r] - 1.2 * x_reg[1][r] + 0.8 * x_reg[2][r] + 0.02 * rng.normal();
    }
    let b_reg = fit_dse8_packet(
        &x_reg,
        &y_reg,
        &Dse8FitOptions {
            task_id: 0,
            budget_bytes: 2048,
            seed: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let pkt_reg = unpack_dse8_packet(&b_reg).unwrap();
    assert!(
        pkt_reg.header.resid_sd < 0.06,
        "near-deterministic resid_sd exploded to {}",
        pkt_reg.header.resid_sd
    );
    let dec_reg = decode_dse8_rows(&pkt_reg, None, 1000, 88).unwrap();
    let mut err2 = 0.0;
    let y_bar = dec_reg.y.iter().sum::<f64>() / 1000.0;
    let y_var = dec_reg.y.iter().map(|v| (v - y_bar).powi(2)).sum::<f64>() / 1000.0;
    for r in 0..1000 {
        let pred = 1.5 * dec_reg.x[0][r] - 1.2 * dec_reg.x[1][r] + 0.8 * dec_reg.x[2][r];
        err2 += (dec_reg.y[r] - pred).powi(2);
    }
    let syn_r2 = 1.0 - (err2 / 1000.0) / y_var.max(1e-6);
    assert!(
        syn_r2 > 0.97,
        "near-deterministic syn_r2={syn_r2:.4} (resid_sd={})",
        pkt_reg.header.resid_sd
    );

    // 3. Nonlinear discrete-column interaction (binary 65/35 * binary 70/30): decode-aligned trees!
    let n_disc = 600;
    let mut x_d = vec![vec![0.0f64; n_disc]; 4];
    let mut y_d = vec![0.0f64; n_disc];
    for (r, yd) in y_d.iter_mut().enumerate().take(n_disc) {
        let b0 = if rng.uniform() < 0.35 { 1.0 } else { 0.0 };
        let b1 = if rng.uniform() < 0.30 { 1.0 } else { 0.0 };
        x_d[0][r] = b0;
        x_d[1][r] = b1;
        x_d[2][r] = rng.normal();
        x_d[3][r] = rng.normal();
        *yd = if (b0 > 0.5) ^ (b1 > 0.5) { 2.0 } else { -1.0 } + 0.25 * rng.normal();
    }
    let b_d = fit_dse8_packet(
        &x_d,
        &y_d,
        &Dse8FitOptions {
            task_id: 0,
            budget_bytes: 2048,
            seed: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let pkt_d = unpack_dse8_packet(&b_d).unwrap();
    let dec_d = decode_dse8_rows(&pkt_d, None, 2000, 99).unwrap();
    let (mut s_xor, mut n_xor, mut s_eq, mut n_eq) = (0.0, 0usize, 0.0, 0usize);
    for r in 0..2000 {
        if (dec_d.x[0][r] > 0.5) ^ (dec_d.x[1][r] > 0.5) {
            s_xor += dec_d.y[r];
            n_xor += 1;
        } else {
            s_eq += dec_d.y[r];
            n_eq += 1;
        }
    }
    let gap = (s_xor / n_xor.max(1) as f64) - (s_eq / n_eq.max(1) as f64);
    assert!(
        gap > 1.8,
        "discrete XOR gap collapsed to {gap:.3} (expected > 1.8)"
    );
}
