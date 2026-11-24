//! Shared FIT evidence (`extract_fit_evidence_v8`) and exact `V7Schema` wrapper-byte FFI tests.

use dope_core::evidence::EvidenceTier;
use dope_core::single_encoding::v8::{
    extract_fit_evidence_v8, fit_unit_meta, unpack_dse8_packet, ScaleCardV8,
};
use dope_ffi::v8::{block, dope_ffi_v8_schema_wrapper_bytes, offsets, p_cap, v8_episode_prepared};

fn synth_table(n: usize, p: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f32 / u64::MAX as f32) * 2.0 - 1.0
    };
    let mut x = vec![0.0f32; n * p];
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..p {
            x[i * p + j] = next();
        }
        y[i] = 0.7 * x[i * p] - 0.4 * x[i * p + 1] + 0.2 * next();
    }
    (x, y)
}

#[test]
fn prepared_schema_wrapper_bytes_and_shared_fit_evidence_match_core() {
    let (n_fit, n_qry, p) = (96usize, 16usize, 12usize);
    let (xf, yf) = synth_table(n_fit, p, 11);
    let (xq, yq) = synth_table(n_qry, p, 29);
    let ids: Vec<u64> = (1000..1000 + n_fit as u64).collect();
    let schema_json = format!(
        "{{\"feature_names\":[{}],\"target_name\":\"target\"}}",
        (0..p)
            .map(|j| format!("\"measurement_long_col_{j:02}\""))
            .collect::<Vec<_>>()
            .join(",")
    );
    let w_bytes = unsafe {
        dope_ffi_v8_schema_wrapper_bytes(schema_json.as_ptr(), schema_json.len() as u32, 0)
    };
    assert!(w_bytes > 40, "expected schema wrapper bytes, got {w_bytes}");

    let ep = v8_episode_prepared(
        &xf,
        n_fit,
        p,
        &yf,
        &xq,
        n_qry,
        &yq,
        0,
        2048,
        false,
        2,
        77,
        Some(&ids),
        1,
        true,
        w_bytes as usize,
    )
    .unwrap();
    assert!(ep.packet.len() + w_bytes as usize <= 2048);
    let pkt = unpack_dse8_packet(&ep.packet).unwrap();
    let p_eff = pkt.header.p_eff as usize;

    let mut x_raw = vec![vec![0.0f64; n_fit]; p];
    for i in 0..n_fit {
        for j in 0..p {
            x_raw[j][i] = xf[i * p + j] as f64;
        }
    }
    let y_raw: Vec<f64> = yf.iter().map(|&v| v as f64).collect();
    let meta = fit_unit_meta(&x_raw, &y_raw, 0, false).unwrap();
    let card = ScaleCardV8::fit(&x_raw, &y_raw, 0, p_eff, false, Some(&meta)).unwrap();
    let x_unit_all = card.forward_x(&x_raw);
    let (y_sc, valid) = card.forward_y(&y_raw);
    let x_unit: Vec<Vec<f32>> = x_unit_all
        .iter()
        .map(|col| valid.iter().map(|&r| col[r]).collect())
        .collect();
    let active_meta: Vec<_> = card.col_indices.iter().map(|&j| meta[j]).collect();
    let ev = extract_fit_evidence_v8(
        &x_raw,
        &x_unit,
        &y_sc,
        &valid,
        &card,
        Some(&active_meta),
        EvidenceTier::T2,
        77,
        Some(&ids),
        true,
    )
    .unwrap();
    let off = offsets(p_cap(p));
    assert_eq!(
        &ep.floats[off[block::EV_GLOB]..off[block::EV_GLOB] + 160],
        &ev.glob
    );
    for c in 0..p_eff {
        assert_eq!(
            &ep.floats[off[block::EV_COL] + c * 128..off[block::EV_COL] + (c + 1) * 128],
            &ev.col[c]
        );
    }
}
