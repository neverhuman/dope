//! Pure-Rust verification of large-model `SingleEncV8` SafeTensors support:
//! - `BF16` + `F32` loading (`<= 1 GiB` cap),
//! - Auto-inferred attention heads (`4 / 8 / 12 / 16` heads),
//! - Dynamic nested FSQ code widths (`64 | 128 | 256` dims = Tiers 1/2/3 inside `<= 2,048 B`),
//! - Fail-closed rejection of non-finite `BF16` (`NaN`/`Inf`), unsupported dtypes, and invalid code dims.

use dope_core::auto::V8Model;
use dope_core::rng::Rng;
use dope_core::single_encoding::v8::{
    decode_dse8_rows, fit_dse8_packet, fit_dse8_packet_with_encoder, unpack_dse8_packet,
    Dse8FitOptions, SingleEncV8Weights, DEC_COL_DIM, DEC_EDGE_DIM, DEC_GLOB_DIM, V8_MAX_CLASSES,
    V8_MAX_RANK,
};
use dope_core::single_encoding::KNOT_COUNT;
use std::collections::BTreeMap;

fn f32_to_bf16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    // Round-to-nearest-even to BF16 upper 16 bits.
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    ((rounded >> 16) as u16).to_le_bytes()
}

fn build_v8_safetensors(
    d_model: usize,
    n_heads: usize,
    code_dim: usize,
    ff_mult: usize,
    dtype: &str,
    zero_heads: bool,
) -> Vec<u8> {
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let add_lin = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   name: &str,
                   out_d: usize,
                   in_d: usize,
                   zero: bool| {
        let mut w = vec![0.0f32; out_d * in_d];
        if !zero {
            for (i, slot) in w.iter_mut().enumerate() {
                *slot = ((i % 7) as f32 - 3.0) * 0.015625; // exact power-of-2 in BF16 and F32!
            }
        }
        t.insert(format!("{name}.weight"), (vec![out_d, in_d], w));
        t.insert(format!("{name}.bias"), (vec![out_d], vec![0.0f32; out_d]));
    };
    let add_ln = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>, name: &str, d: usize| {
        t.insert(format!("{name}.weight"), (vec![d], vec![1.0f32; d]));
        t.insert(format!("{name}.bias"), (vec![d], vec![0.0f32; d]));
    };

    // Owner encoder (1 block, arbitrary ff_mult and code_dim in 64|128|256)
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
        d_model * ff_mult,
        d_model,
        false,
    );
    add_lin(
        &mut tensors,
        "enc_blocks.0.ff2",
        d_model,
        d_model * ff_mult,
        false,
    );
    add_ln(&mut tensors, "enc_final_norm", d_model);
    add_lin(&mut tensors, "enc_code_head", code_dim, d_model, false);

    // Receiver decoder (1 block)
    add_lin(&mut tensors, "dec_col_in", d_model, DEC_COL_DIM, false);
    add_lin(&mut tensors, "dec_glob_in", d_model, DEC_GLOB_DIM, false);
    add_lin(&mut tensors, "dec_code_in", d_model, code_dim, false);
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
        d_model * ff_mult,
        d_model,
        false,
    );
    add_lin(
        &mut tensors,
        "dec_blocks.0.ff2",
        d_model,
        d_model * ff_mult,
        false,
    );
    add_ln(&mut tensors, "dec_final_norm", d_model);

    add_lin(&mut tensors, "delta_knots", KNOT_COUNT, d_model, zero_heads);
    add_lin(&mut tensors, "delta_L", V8_MAX_RANK, d_model, zero_heads);
    add_lin(&mut tensors, "delta_W", V8_MAX_CLASSES, d_model, zero_heads);
    add_lin(&mut tensors, "node_sel", 24, d_model, zero_heads);
    add_lin(&mut tensors, "node_thr", 24, d_model, zero_heads);
    add_lin(
        &mut tensors,
        "node_leaf",
        8 * 8 * V8_MAX_CLASSES,
        d_model,
        zero_heads,
    );
    add_lin(&mut tensors, "bilin_u", 4, d_model, zero_heads);
    add_lin(&mut tensors, "bilin_v", 4, d_model, zero_heads);
    add_lin(
        &mut tensors,
        "bilin_w",
        4 * V8_MAX_CLASSES,
        d_model,
        zero_heads,
    );
    add_lin(&mut tensors, "flow_scale_head", 2, d_model, zero_heads);
    add_lin(&mut tensors, "flow_shift_head", 2, d_model, zero_heads);
    add_lin(&mut tensors, "gate_head", 2, d_model, zero_heads);

    let mut header = serde_json::Map::new();
    let mut data = Vec::<u8>::new();
    for (k, (shape, vals)) in tensors {
        let lo = data.len();
        for v in vals {
            if dtype == "BF16" {
                data.extend_from_slice(&f32_to_bf16_le(v));
            } else {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        let hi = data.len();
        header.insert(
            k,
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [lo, hi]}),
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
fn v8_bf16_auto_heads_and_tier3_code_roundtrip() {
    // 12-head, ff_mult=4, Tier-3 (256-dim = 128 B code) BF16 checkpoint!
    let bf16_bytes = build_v8_safetensors(96, 12, 256, 4, "BF16", true);
    let f32_bytes = build_v8_safetensors(96, 12, 256, 4, "F32", true);
    assert!(
        bf16_bytes.len() * 2 < f32_bytes.len() + 8192,
        "BF16 should halve payload size vs F32"
    );

    let model = V8Model::from_safetensors_auto(&bf16_bytes).expect("load BF16 V8Model");
    let w = model.weights();
    assert_eq!(w.d_model, 96);
    assert_eq!(w.n_heads, 12);
    assert_eq!(w.code_dim, 256);
    assert_eq!(w.preferred_code_tier(), 3);

    // Fit a <= 2,048 B packet with Tier-3 (128 B) nested code and verify exact zero-init decode parity!
    let n = 160;
    let mut rng = Rng::new(1234);
    let mut x = vec![vec![0.0f64; n]; 6];
    let mut y = vec![0.0f64; n];
    for (r, yr) in y.iter_mut().enumerate().take(n) {
        for xc in &mut x {
            xc[r] = rng.normal();
        }
        *yr = 0.8 * x[0][r] - 0.5 * x[1][r] + 0.3 * rng.normal();
    }
    let opts = Dse8FitOptions {
        task_id: 0,
        budget_bytes: 2048,
        seed: 7,
        ..Default::default()
    };
    let pkt_bytes = fit_dse8_packet_with_encoder(&x, &y, &opts, w).expect("fit Tier-3 packet");
    assert!(pkt_bytes.len() <= 2048);
    let pkt = unpack_dse8_packet(&pkt_bytes).expect("unpack Tier-3 packet");
    assert_eq!(
        pkt.code.tier, 3,
        "expected Tier-3 (128 B) code in 2048 B packet"
    );
    assert_eq!(pkt.code.packed.len(), 128);

    // Non-zero BF16 vs F32 weights with exact dyadic values must yield identical NeuralPlanV8!
    let bf16_nz = build_v8_safetensors(48, 8, 128, 2, "BF16", false);
    let f32_nz = build_v8_safetensors(48, 8, 128, 2, "F32", false);
    let w_bf = SingleEncV8Weights::from_safetensors_auto(&bf16_nz).expect("load BF16 nz");
    let w_f3 = SingleEncV8Weights::from_safetensors_auto(&f32_nz).expect("load F32 nz");
    assert_eq!(w_bf.n_heads, 8);
    assert_eq!(w_bf.preferred_code_tier(), 2);

    let p0_bytes = fit_dse8_packet(&x, &y, &opts).unwrap();
    let p0 = unpack_dse8_packet(&p0_bytes).unwrap();
    let dec_bf = decode_dse8_rows(&p0, Some(&w_bf), 32, 99).unwrap();
    let dec_f3 = decode_dse8_rows(&p0, Some(&w_f3), 32, 99).unwrap();
    for r in 0..32 {
        assert!(
            (dec_bf.y[r] - dec_f3.y[r]).abs() < 1e-6,
            "BF16/F32 parity drift at row {r}: {} vs {}",
            dec_bf.y[r],
            dec_f3.y[r]
        );
    }
}

#[test]
fn v8_safetensors_rejects_bad_dtype_bad_code_dim_and_bf16_nan() {
    // 1. Unsupported dtype (F16) rejected fail-closed.
    let bad_dtype = build_v8_safetensors(32, 4, 64, 2, "F16", true);
    assert!(SingleEncV8Weights::from_safetensors_auto(&bad_dtype).is_err());

    // 2. Unsupported code_dim (32) rejected fail-closed.
    let bad_code = build_v8_safetensors(32, 4, 32, 2, "BF16", true);
    assert!(SingleEncV8Weights::from_safetensors_auto(&bad_code).is_err());

    // 3. Non-finite BF16 NaN (0x7FC0) rejected fail-closed.
    let mut bad_nan = build_v8_safetensors(32, 4, 64, 2, "BF16", true);
    let n = bad_nan.len();
    bad_nan[n - 2] = 0xC0;
    bad_nan[n - 1] = 0x7F;
    assert!(SingleEncV8Weights::from_safetensors_auto(&bad_nan).is_err());

    // 4. Overlapping data_offsets (SafeTensors zip-bomb amplification) rejected before allocation.
    let overlap_hdr = serde_json::json!({
        "t0": {"dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 128]},
        "t1": {"dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 128]}
    });
    let oh = serde_json::to_vec(&overlap_hdr).unwrap();
    let mut overlap_blob = Vec::with_capacity(8 + oh.len() + 128);
    overlap_blob.extend_from_slice(&(oh.len() as u64).to_le_bytes());
    overlap_blob.extend_from_slice(&oh);
    overlap_blob.resize(overlap_blob.len() + 128, 0u8);
    let err = SingleEncV8Weights::from_safetensors_auto(&overlap_blob).unwrap_err();
    assert!(
        err.to_string().contains("overlapping"),
        "expected overlapping offsets error, got: {err}"
    );

    // 5. Oversized header length (> MAX_HEADER_BYTES = 2 MiB) rejected before parsing.
    let mut huge_hdr = vec![0u8; 64];
    let bad_hdr_len: u64 = (4 * 1024 * 1024) as u64;
    huge_hdr[..8].copy_from_slice(&bad_hdr_len.to_le_bytes());
    assert!(SingleEncV8Weights::from_safetensors_auto(&huge_hdr).is_err());

    // 6. Declared data_offsets past payload end rejected before allocation.
    let oob_hdr = serde_json::json!({
        "t0": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [0, 2_097_152]}
    });
    let oob = serde_json::to_vec(&oob_hdr).unwrap();
    let mut oob_blob = Vec::with_capacity(8 + oob.len() + 64);
    oob_blob.extend_from_slice(&(oob.len() as u64).to_le_bytes());
    oob_blob.extend_from_slice(&oob);
    oob_blob.resize(oob_blob.len() + 64, 0u8);
    assert!(SingleEncV8Weights::from_safetensors_auto(&oob_blob).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn v8_safetensors_allocation_failure_returns_err_under_ulimit_v() {
    use std::process::Command;

    let exe = std::env::current_exe().expect("current test executable");
    // Step 1: measure child VmSize after holding the 32 MiB BF16 payload (so ulimit -v is portable).
    let probe = Command::new(&exe)
        .args([
            "--exact",
            "v8_safetensors_oom_child_probe",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("DOPE_OOM_PROBE_MODE", "measure")
        .output()
        .expect("run measure child");
    assert!(
        probe.status.success(),
        "measure child failed: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let stdout = String::from_utf8_lossy(&probe.stdout);
    let vmsize_kb: u64 = stdout
        .lines()
        .find_map(|l| l.split_once("DOPE_VMSIZE_KB=").map(|(_, rhs)| rhs))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("parse DOPE_VMSIZE_KB from stdout: {stdout}"));

    // Step 2: cap virtual memory at baseline + 32 MiB (strictly < 256 MiB required for decoded Vec<f32>).
    let cap_kb = vmsize_kb + 32 * 1024;
    let run = Command::new("bash")
        .arg("-c")
        .arg(format!("ulimit -v {cap_kb} && exec \"$@\""))
        .arg("--")
        .arg(&exe)
        .args([
            "--exact",
            "v8_safetensors_oom_child_probe",
            "--ignored",
            "--test-threads=1",
        ])
        .env("DOPE_OOM_PROBE_MODE", "expect_err")
        .output()
        .expect("run ulimit -v child");
    assert!(
        run.status.success(),
        "ulimit -v child did not return clean Err (cap_kb={cap_kb}): stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "subprocess helper invoked by v8_safetensors_allocation_failure_returns_err_under_ulimit_v"]
fn v8_safetensors_oom_child_probe() {
    let mode = std::env::var("DOPE_OOM_PROBE_MODE").unwrap_or_default();
    // Build a valid 128 MiB BF16 blob (67,108,864 elements -> requires 256 MiB decoded Vec<f32>).
    let n_elem: usize = 8192 * 8192;
    let payload_bytes = n_elem * 2;
    let hdr = serde_json::json!({
        "big": {"dtype": "BF16", "shape": [8192, 8192], "data_offsets": [0, payload_bytes]}
    });
    let hb = serde_json::to_vec(&hdr).unwrap();
    let mut blob = vec![0u8; 8 + hb.len() + payload_bytes];
    blob[..8].copy_from_slice(&(hb.len() as u64).to_le_bytes());
    blob[8..8 + hb.len()].copy_from_slice(&hb);

    if mode == "measure" {
        let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
        let kb = status
            .lines()
            .find_map(|l| l.strip_prefix("VmSize:"))
            .and_then(|rest| rest.split_whitespace().next())
            .expect("VmSize");
        println!("DOPE_VMSIZE_KB={kb}");
        return;
    }

    let err = SingleEncV8Weights::from_safetensors_auto(&blob)
        .expect_err("expected fallible allocation error under tight ulimit -v");
    assert!(
        err.to_string().contains("allocation failed"),
        "expected allocation failure error, got: {err}"
    );
}

#[test]
fn v8_active_rank_v1_capability_identity_and_placeholder_reservation() {
    use dope_core::auto::v8_dpa::estimate_dpa_v8_wrapper_bytes;
    use dope_core::single_encoding::v8::build_decode_plan_v8;

    let mut rng = Rng::new(42);
    let n = 96usize;
    let p = 6usize;
    let mut x = vec![vec![0.0f64; n]; p];
    let mut y = vec![0.0f64; n];
    for r in 0..n {
        for col in x.iter_mut() {
            col[r] = rng.normal();
        }
        y[r] = 0.8 * x[0][r] - 0.5 * x[1][r] + 0.3 * x[0][r] * x[1][r] + 0.2 * rng.normal();
    }
    // 1. Reserve exact TAG_CODE + TAG_WEIGHTS_FP placeholder at Tier 1 without weights.
    let opts_res = Dse8FitOptions {
        task_id: 0,
        budget_bytes: 2048,
        code_tier: 1,
        reserve_neural_code: true,
        seed: 7,
        ..Default::default()
    };
    let pkt_res_b = fit_dse8_packet(&x, &y, &opts_res).expect("reserved fit");
    assert!(pkt_res_b.len() <= 2048);
    let pkt_res = unpack_dse8_packet(&pkt_res_b).expect("unpack reserved");
    assert_eq!(pkt_res.header.code_tier, 1);
    assert_eq!(pkt_res.code.packed.len(), 32);
    assert_eq!(pkt_res.weights_fp, Some([0xFFu8; 8]));
    let wrap_b = estimate_dpa_v8_wrapper_bytes(&x, &y, 0);
    assert!(
        wrap_b > 13 && wrap_b < 256,
        "unexpected wrapper bytes: {wrap_b}"
    );

    // 2. Patch SafeTensors header with __metadata__ dope_dse8_decoder=active-rank-v1 vs historical vs unknown.
    let raw_hist = build_v8_safetensors(32, 4, 64, 2, "F32", true);
    let w_hist = SingleEncV8Weights::from_safetensors_auto(&raw_hist).expect("historical load");
    assert!(!w_hist.active_rank_v1);

    let add_meta = |base: &[u8], cap: &str, with_u8_marker: bool| -> Vec<u8> {
        let hlen = u64::from_le_bytes(base[..8].try_into().unwrap()) as usize;
        let mut map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&base[8..8 + hlen]).unwrap();
        map.insert(
            "__metadata__".into(),
            serde_json::json!({ "dope_dse8_decoder": cap }),
        );
        let body = &base[8 + hlen..];
        if with_u8_marker {
            let off = body.len();
            map.insert(
                "__dope_dse8_decoder_active_rank_v1__".into(),
                serde_json::json!({ "dtype": "U8", "shape": [1], "data_offsets": [off, off + 1] }),
            );
        }
        let hb = serde_json::to_vec(&map).unwrap();
        let mut out = Vec::with_capacity(8 + hb.len() + body.len() + usize::from(with_u8_marker));
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        out.extend_from_slice(body);
        if with_u8_marker {
            out.push(1u8);
        }
        out
    };

    let raw_bad = add_meta(&raw_hist, "future-unknown-v99", false);
    assert!(SingleEncV8Weights::from_safetensors_auto(&raw_bad).is_err());

    // Metadata-only active-rank-v1 without the U8 v3.0.0 guard tensor MUST be refused!
    let raw_meta_only = add_meta(&raw_hist, "active-rank-v1", false);
    assert!(SingleEncV8Weights::from_safetensors_auto(&raw_meta_only).is_err());

    // Build an active-rank-v1 checkpoint where ONLY delta_L bias on inactive rank 11 is non-zero!
    let mut raw_active = add_meta(&raw_hist, "active-rank-v1", true);
    {
        let hlen = u64::from_le_bytes(raw_active[..8].try_into().unwrap()) as usize;
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&raw_active[8..8 + hlen]).unwrap();
        let offs = map["delta_L.bias"]["data_offsets"].as_array().unwrap();
        let lo = offs[0].as_u64().unwrap() as usize;
        let slot = 8 + hlen + lo + 11 * 4;
        raw_active[slot..slot + 4].copy_from_slice(&5.0f32.to_le_bytes());
    }
    let w_active = SingleEncV8Weights::from_safetensors_auto(&raw_active).expect("active load");
    assert!(w_active.active_rank_v1);

    // Analytic packet without weights_fp MUST be refused by active-rank-v1 checkpoint (fail-closed)!
    let pkt_an_b = fit_dse8_packet(&x, &y, &Dse8FitOptions::default()).unwrap();
    let pkt_an = unpack_dse8_packet(&pkt_an_b).unwrap();
    assert!(build_decode_plan_v8(&pkt_an, Some(&w_active)).is_err());

    // Fitted neural packet with matching fingerprint: rank_use <= 6 < 11 -> inactive rank 11 MUST NOT flip is_identity!
    let pkt_nn_b = fit_dse8_packet_with_encoder(
        &x,
        &y,
        &Dse8FitOptions {
            code_tier: 1,
            seed: 9,
            ..Default::default()
        },
        &w_active,
    )
    .expect("fit with active-rank-v1");
    let pkt_nn = unpack_dse8_packet(&pkt_nn_b).unwrap();
    assert!((pkt_nn.header.rank_use as usize) < 11);
    let plan = build_decode_plan_v8(&pkt_nn, Some(&w_active)).expect("active decode plan");
    assert!(plan.neural.as_ref().unwrap().is_identity);
    for j in 0..pkt_nn.header.p_eff as usize {
        assert_eq!(plan.l_eff[j][11], 0.0);
    }
}
