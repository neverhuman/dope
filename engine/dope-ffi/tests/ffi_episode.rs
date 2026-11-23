//! The per-episode native stage is a pure function of the FIT rows: query rows never reach the
//! packet, outputs are deterministic, and the C ABI agrees with the Rust entry point.

use dope_core::single_encoding::{unpack_dse7_packet, MAX_COLS};
use dope_ffi::{
    dope_ffi_dse7_episode, dope_ffi_dse7_len, dope_ffi_dse7_offset, dse7_episode, layout, Code,
};

/// Deterministic table: `n` rows, `p` columns, row-major, with a target for `task`.
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
            } else {
                next() * 2.0 - 1.0
            };
            x[i * p + j] = v as f32;
            signal += v * if j % 2 == 0 { 1.0 } else { -0.5 };
        }
        y[i] = match task {
            0 => (signal + 0.3 * next()) as f32,
            1 => (signal > 0.0) as u8 as f32,
            _ => ((signal.clamp(-1.99, 1.99) + 2.0).floor() * 2.0) as f32, // class ids 0, 2, 4, 6
        };
    }
    (x, y)
}

fn episode(task: u8, qry_salt: u64) -> dope_ffi::Dse7Episode {
    let (p, n_fit, n_qry) = (9, 300, 120);
    let (xf, yf) = table(n_fit, p, task, 1);
    let (xq, yq) = table(n_qry, p, task, qry_salt);
    dse7_episode(&xf, n_fit, p, &yf, &xq, n_qry, &yq, task, 2048).expect("episode")
}

#[test]
fn packet_and_head_ignore_query_rows() {
    for task in [0u8, 1, 2] {
        let a = episode(task, 2);
        let b = episode(task, 3);
        assert_eq!(
            a.packet, b.packet,
            "task {task}: query rows changed the packet"
        );
        let fit_end = layout::HEAD + 300 * (MAX_COLS + 1);
        assert_eq!(
            a.floats[..fit_end],
            b.floats[..fit_end],
            "task {task}: FIT block depends on query"
        );
        assert_ne!(
            a.floats[fit_end..],
            b.floats[fit_end..],
            "task {task}: query block should differ"
        );
    }
}

#[test]
fn deterministic_and_within_budget() {
    for task in [0u8, 1, 2] {
        let a = episode(task, 2);
        assert_eq!(a, episode(task, 2));
        assert!(a.packet.len() <= 2048);
        let (_, header, card) = unpack_dse7_packet(&a.packet).unwrap();
        let p_eff = header.p_eff as usize;
        assert_eq!(a.floats[layout::MISC + 1], p_eff as f32);
        for c in 0..MAX_COLS {
            let want = card.col_indices.get(c).map_or(-1.0, |&j| j as f32);
            assert_eq!(a.floats[layout::COL_INDEX + c], want);
        }
        // Unused packet columns are zero in every row.
        let rows = &a.floats[layout::HEAD..layout::HEAD + 300 * MAX_COLS];
        assert!(rows
            .chunks(MAX_COLS)
            .all(|r| r[p_eff..].iter().all(|&v| v == 0.0)));
    }
}

#[test]
fn class_codes_follow_the_fit_class_list() {
    let ep = episode(2, 2);
    let y_fit = &ep.floats[layout::HEAD + 300 * MAX_COLS..][..300];
    let k = ep.floats[layout::MISC + 2] as usize;
    assert!(y_fit
        .iter()
        .all(|&c| c >= 0.0 && (c as usize) < k && c.fract() == 0.0));
    // A query class unseen in FIT is masked (-1), never folded into another class.
    let (p, n_fit) = (9, 300);
    let (xf, yf) = table(n_fit, p, 2, 1);
    let (xq, mut yq) = table(4, p, 2, 9);
    yq[0] = 99.0;
    let out = dse7_episode(&xf, n_fit, p, &yf, &xq, 4, &yq, 2, 2048).unwrap();
    let at = layout::HEAD + (n_fit + 4) * MAX_COLS + n_fit;
    assert_eq!(out.floats[at], -1.0);
}

#[test]
fn bad_shapes_are_codes_not_panics() {
    let (xf, yf) = table(10, 3, 0, 1);
    assert_eq!(
        dse7_episode(&xf, 10, 4, &yf, &[], 0, &[], 0, 2048),
        Err(Code::Shape)
    );
    assert_eq!(
        dse7_episode(&xf, 10, 3, &yf, &[], 0, &[], 3, 2048),
        Err(Code::Shape)
    );
    assert_eq!(
        dse7_episode(&xf, 10, 3, &yf, &[], 0, &[], 0, 0),
        Err(Code::Shape)
    );
    let mut bad = yf.clone();
    bad[3] = f32::NAN;
    assert_eq!(
        dse7_episode(&xf, 10, 3, &bad, &[], 0, &[], 0, 2048),
        Err(Code::Target)
    );
    // One class only: the fit refuses it with a code.
    let ones = vec![1.0f32; 10];
    assert_eq!(
        dse7_episode(&xf, 10, 3, &ones, &[], 0, &[], 1, 2048),
        Err(Code::Fit)
    );
}

#[test]
fn c_abi_matches_rust_entry_point() {
    let (p, n_fit, n_qry) = (9usize, 300usize, 120usize);
    let (xf, yf) = table(n_fit, p, 1, 1);
    let (xq, yq) = table(n_qry, p, 1, 2);
    let want = dse7_episode(&xf, n_fit, p, &yf, &xq, n_qry, &yq, 1, 2048).unwrap();
    let len = dope_ffi_dse7_len(n_fit as u32, n_qry as u32) as usize;
    assert_eq!(len, want.floats.len());
    assert_eq!(dope_ffi_dse7_offset(11), layout::HEAD as u64);
    assert_eq!(dope_ffi_dse7_offset(12), u64::MAX);
    let mut out = vec![f32::NAN; len];
    let mut pkt = vec![0u8; 4096];
    let mut pkt_len = 0u32;
    let code = unsafe {
        dope_ffi_dse7_episode(
            xf.as_ptr(),
            n_fit as u32,
            p as u32,
            yf.as_ptr(),
            xq.as_ptr(),
            n_qry as u32,
            yq.as_ptr(),
            1,
            2048,
            out.as_mut_ptr(),
            len as u64,
            pkt.as_mut_ptr(),
            4096,
            &mut pkt_len,
        )
    };
    assert_eq!(code, Code::Ok as i32);
    assert_eq!(out, want.floats);
    assert_eq!(&pkt[..pkt_len as usize], &want.packet[..]);
    // Too-small output buffer is refused before any work.
    let code = unsafe {
        dope_ffi_dse7_episode(
            xf.as_ptr(),
            n_fit as u32,
            p as u32,
            yf.as_ptr(),
            xq.as_ptr(),
            n_qry as u32,
            yq.as_ptr(),
            1,
            2048,
            out.as_mut_ptr(),
            (len - 1) as u64,
            pkt.as_mut_ptr(),
            4096,
            &mut pkt_len,
        )
    };
    assert_eq!(code, Code::Capacity as i32);
    let code = unsafe {
        dope_ffi_dse7_episode(
            std::ptr::null(),
            1,
            1,
            yf.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            2048,
            out.as_mut_ptr(),
            len as u64,
            pkt.as_mut_ptr(),
            4096,
            &mut pkt_len,
        )
    };
    assert_eq!(code, Code::NullPointer as i32);
}
