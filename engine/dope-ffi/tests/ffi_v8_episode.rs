//! The V8 native stage is a pure function of the FIT rows: query rows never reach the packet,
//! the decoder inputs or the evidence bank; rows land in the packet's own unit space; column
//! meta is in that same unit space; and the C ABI agrees with the Rust entry point.

use dope_core::marginal::column_meta::META_FIELDS;
use dope_core::single_encoding::v8::{
    build_decoder_inputs, unpack_dse8_packet, DEC_COL_DIM, V8_MAX_COLS,
};
use dope_ffi::v8::{
    block, dope_ffi_v8_episode, dope_ffi_v8_len, dope_ffi_v8_offset, offsets, p_cap, total,
    v8_episode, MISC_LEN,
};
use dope_ffi::Code;

/// Deterministic table: `n` rows, `p` columns (every 5th an integer 0..5, every 7th a 0.25
/// grid), row-major, with a planted target for `task`.
fn table(n: usize, p: usize, task: u8, salt: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = vec![0.0f32; n * p];
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut signal = 0.0;
        for j in 0..p {
            let v = if j % 5 == 4 {
                (next() * 6.0).floor()
            } else if j % 7 == 6 {
                10.0 + 0.25 * (next() * 40.0).floor()
            } else {
                next() * 2.0 - 1.0
            };
            x[i * p + j] = v as f32;
            signal += v * if j % 2 == 0 { 1.0 } else { -0.5 } / if j % 7 == 6 { 10.0 } else { 1.0 };
        }
        y[i] = match task {
            0 => (3.0 * signal + 5.0 + 0.3 * next()) as f32,
            1 => (signal > 0.0) as u8 as f32,
            _ => ((signal.clamp(-1.99, 1.99) + 2.0).floor() * 2.0) as f32, // class ids 0, 2, 4, 6
        };
    }
    (x, y)
}

const P: usize = 11;
const N_FIT: usize = 400;
const N_QRY: usize = 150;

fn episode(task: u8, qry_salt: u64, normalized_only: bool) -> dope_ffi::v8::V8Episode {
    let (xf, yf) = table(N_FIT, P, task, 1);
    let (xq, yq) = table(N_QRY, P, task, qry_salt);
    v8_episode(
        &xf,
        N_FIT,
        P,
        &yf,
        &xq,
        N_QRY,
        &yq,
        task,
        2048,
        normalized_only,
        2,
        7,
    )
    .expect("episode")
}

fn misc(ep: &dope_ffi::v8::V8Episode) -> [f32; MISC_LEN] {
    let off = offsets(p_cap(P));
    ep.floats[off[block::MISC]..][..MISC_LEN]
        .try_into()
        .unwrap()
}

#[test]
fn packet_decoder_inputs_and_evidence_ignore_query_rows() {
    for task in [0u8, 1, 2] {
        let a = episode(task, 2, false);
        let b = episode(task, 99, false);
        assert_eq!(
            a.packet, b.packet,
            "task {task}: packet depends on query rows"
        );
        let off = offsets(p_cap(P));
        let fit_end = off[block::HEAD] + N_FIT * (p_cap(P) + 1);
        assert_eq!(
            a.floats[..fit_end],
            b.floats[..fit_end],
            "task {task}: a FIT block depends on query rows"
        );
        assert_ne!(
            a.floats[fit_end..],
            b.floats[fit_end..],
            "query rows were not written"
        );
        assert!(a.packet.len() <= 2048);
        assert_eq!(misc(&a)[9] as usize, a.packet.len());
    }
}

#[test]
fn deterministic_and_decoder_inputs_come_from_the_packet_alone() {
    let a = episode(0, 2, false);
    assert_eq!(a, episode(0, 2, false));
    let pkt = unpack_dse8_packet(&a.packet).unwrap();
    let p_eff = pkt.header.p_eff as usize;
    assert_eq!(misc(&a)[2] as usize, p_eff);
    assert!((2..=P).contains(&p_eff));
    let (dec_col, _, dec_glob, _) = build_decoder_inputs(&pkt);
    let off = offsets(p_cap(P));
    for (c, want) in dec_col.iter().enumerate().take(p_eff) {
        assert_eq!(
            a.floats[off[block::DEC_COL] + c * DEC_COL_DIM..][..DEC_COL_DIM],
            want[..]
        );
    }
    assert_eq!(
        a.floats[off[block::DEC_GLOB]..][..dec_glob.len()],
        dec_glob[..]
    );
    // Unused columns are marked and zero.
    for c in p_eff..p_cap(P) {
        assert_eq!(a.floats[off[block::COL_INDEX] + c], -1.0);
    }
    // Evidence is finite everywhere.
    assert!(a.floats[off[block::EV_COL]..off[block::EV_GLOB] + 160]
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn rows_are_in_the_packet_unit_space_and_meta_shares_it() {
    let (xf, _) = table(N_FIT, P, 0, 1);
    let ep = episode(0, 2, false);
    let pkt = unpack_dse8_packet(&ep.packet).unwrap();
    let p_eff = pkt.header.p_eff as usize;
    let w = p_cap(P);
    let off = offsets(w);
    let card = &pkt.card;
    for c in 0..p_eff {
        let j = ep.floats[off[block::COL_INDEX] + c] as usize;
        assert_eq!(j, card.col_indices[c]);
        let (center, scale) = (card.center_active[c], card.scale_active[c]);
        // FIT rows: unit value * scale + center gives the raw value back (wire precision).
        for i in [0usize, 17, 399] {
            let unit = ep.floats[off[block::HEAD] + i * w + c];
            let raw = xf[i * P + j];
            let tol = 2e-2 * scale.abs().max(1.0) + 1e-3 * raw.abs();
            assert!(
                (unit * scale + center - raw).abs() <= tol,
                "col {j} row {i}"
            );
        }
        let meta = &ep.floats[off[block::COL_META] + c * META_FIELDS..][..META_FIELDS];
        let (support, q0) = (meta[4], meta[6]);
        if j % 5 == 4 {
            // Six integer levels with >= 10 rows each are atoms, which take precedence.
            assert_eq!(support, 2.0, "col {j}");
        } else if j % 7 == 6 {
            // The 0.25 grid (40 levels): step 0.25 raw = 0.25 / scale in unit space.
            assert_eq!(support, 1.0, "col {j}");
            assert!(
                (q0 * scale - 0.25).abs() < 5e-3,
                "col {j}: q0 {q0} scale {scale}"
            );
        }
    }
}

#[test]
fn class_codes_follow_the_packet_class_list() {
    let ep = episode(2, 5, false);
    let pkt = unpack_dse8_packet(&ep.packet).unwrap();
    let classes = &pkt.card.classes;
    let w = p_cap(P);
    let off = offsets(w);
    let (_, yf) = table(N_FIT, P, 2, 1);
    let y_at = off[block::HEAD] + N_FIT * w;
    for (i, &v) in yf.iter().enumerate() {
        let code = ep.floats[y_at + i];
        match classes.binary_search(&(v as i32)) {
            Ok(k) => assert_eq!(code, k as f32),
            Err(_) => assert_eq!(code, -1.0),
        }
    }
    // A query class the packet never saw is masked, never folded into class 0.
    let (xf, yf) = table(N_FIT, P, 2, 1);
    let (xq, mut yq) = table(N_QRY, P, 2, 5);
    yq[0] = 42.0;
    let ep = v8_episode(&xf, N_FIT, P, &yf, &xq, N_QRY, &yq, 2, 2048, false, 2, 7).unwrap();
    let q_at = off[block::HEAD] + N_FIT * (w + 1) + N_QRY * w;
    assert_eq!(ep.floats[q_at], -1.0);
}

#[test]
fn normalized_only_keeps_rows_as_given() {
    // Pre-normalized input: the card is the identity, so unit rows equal the input.
    let (mut xf, yf) = table(N_FIT, P, 1, 1);
    for v in xf.iter_mut() {
        *v = (*v - 0.5).clamp(-3.0, 3.0);
    }
    let (xq, yq) = table(N_QRY, P, 1, 3);
    let ep = v8_episode(&xf, N_FIT, P, &yf, &xq, N_QRY, &yq, 1, 2048, true, 2, 7).unwrap();
    let w = p_cap(P);
    let off = offsets(w);
    assert_eq!(misc(&ep)[8], 1.0);
    for c in 0..(misc(&ep)[2] as usize) {
        let j = ep.floats[off[block::COL_INDEX] + c] as usize;
        assert!((ep.floats[off[block::HEAD] + 3 * w + c] - xf[3 * P + j]).abs() < 1e-6);
    }
}

#[test]
fn bad_shapes_are_codes_and_the_c_abi_matches() {
    let (xf, yf) = table(N_FIT, P, 0, 1);
    let (xq, yq) = table(N_QRY, P, 0, 2);
    assert_eq!(
        v8_episode(
            &xf,
            N_FIT,
            P + 1,
            &yf,
            &xq,
            N_QRY,
            &yq,
            0,
            2048,
            false,
            2,
            7
        ),
        Err(Code::Shape)
    );
    assert_eq!(
        v8_episode(&xf, N_FIT, P, &yf, &xq, N_QRY, &yq, 3, 2048, false, 2, 7),
        Err(Code::Shape)
    );
    let mut bad_y = yf.clone();
    bad_y[3] = f32::NAN;
    assert_eq!(
        v8_episode(&xf, N_FIT, P, &bad_y, &xq, N_QRY, &yq, 0, 2048, false, 2, 7),
        Err(Code::Target)
    );

    let want = v8_episode(&xf, N_FIT, P, &yf, &xq, N_QRY, &yq, 0, 2048, false, 2, 7).unwrap();
    let len = dope_ffi_v8_len(P as u32, N_FIT as u32, N_QRY as u32) as usize;
    assert_eq!(len, total(P, N_FIT, N_QRY));
    assert_eq!(
        dope_ffi_v8_offset(block::HEAD as u32, P as u32) as usize,
        offsets(P)[block::HEAD]
    );
    assert_eq!(dope_ffi_v8_offset(block::COUNT as u32, P as u32), u64::MAX);
    assert_eq!(
        total(200, 10, 0),
        offsets(V8_MAX_COLS)[block::HEAD] + 10 * (V8_MAX_COLS + 1)
    );
    let mut out = vec![0.0f32; len];
    let mut pkt = vec![0u8; 65_536];
    let mut pkt_len = 0u32;
    let call = |out: &mut [f32], cap: u64, pkt: &mut [u8], pkt_len: &mut u32| unsafe {
        dope_ffi_v8_episode(
            xf.as_ptr(),
            N_FIT as u32,
            P as u32,
            yf.as_ptr(),
            xq.as_ptr(),
            N_QRY as u32,
            yq.as_ptr(),
            0,
            2048,
            0,
            2,
            7,
            out.as_mut_ptr(),
            cap,
            pkt.as_mut_ptr(),
            pkt.len() as u32,
            pkt_len,
        )
    };
    assert_eq!(
        call(&mut out, len as u64, &mut pkt, &mut pkt_len),
        Code::Ok as i32
    );
    assert_eq!(out, want.floats);
    assert_eq!(&pkt[..pkt_len as usize], &want.packet[..]);
    assert_eq!(
        call(&mut out, len as u64 - 1, &mut pkt, &mut pkt_len),
        Code::Capacity as i32
    );
}
