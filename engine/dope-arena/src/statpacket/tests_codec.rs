//! Wire-format tests: round trip, documented byte formulas, the design byte
//! table, the prefix property, corruption rejection and budget fitting.

use super::codec::{
    decoder_sha8, fit_to_budget, ledger, pack, packed_len, truncate, unpack, BudgetStep,
};
use super::packet::{f16_to_f64, f64_to_f16, Level, Packet, Task, CRC_LEN, HEADER_LEN};
use super::testkit::{synthetic_packet, Kind};
use super::StatError;

fn mixed(j: usize) -> Kind {
    match j % 7 {
        2 => Kind::Explicit(2 + j % 5),
        4 => Kind::Lattice(5 + j % 12),
        6 => Kind::Constant,
        _ => Kind::Continuous,
    }
}

fn tasks() -> Vec<(Task, u8)> {
    vec![
        (Task::Regression, 0),
        (Task::Binary, 2),
        (Task::Multiclass, 5),
        (Task::Multiclass, 32),
        (Task::Multilabel, 8),
        (Task::Multilabel, 32),
    ]
}

#[test]
fn unpack_of_pack_is_the_quantised_packet() {
    for (t, (task, k)) in tasks().into_iter().enumerate() {
        for p in [1usize, 7, 24, 300] {
            let packet = synthetic_packet(task, k, p, mixed, 40 + t as u64);
            for level in Level::ALL {
                let bytes = pack(&packet, level).unwrap();
                let back = unpack(&bytes).unwrap();
                assert_eq!(back, packet.at_level(level).unwrap());
                assert_eq!(pack(&back, level).unwrap(), bytes);
                assert_eq!(bytes.len(), packed_len(&packet, level));
            }
        }
    }
}

/// Bytes at P = 8 / 24 / 48 / 64 for levels A..E, from the encoding design.
const DESIGN_TABLE: [(Task, u8, [[usize; 4]; 5]); 3] = [
    (
        Task::Regression,
        0,
        [
            [90, 202, 370, 482],
            [122, 298, 562, 738],
            [178, 466, 898, 1186],
            [259, 709, 1384, 1834],
            [339, 853, 1624, 2138],
        ],
    ),
    (
        Task::Binary,
        2,
        [
            [91, 203, 371, 483],
            [131, 307, 571, 747],
            [195, 483, 915, 1203],
            [276, 726, 1401, 1851],
            [364, 878, 1649, 2163],
        ],
    ),
    (
        Task::Multiclass,
        5,
        [
            [130, 290, 530, 690],
            [182, 406, 742, 966],
            [258, 594, 1098, 1434],
            [435, 1125, 2160, 2850],
            [583, 1337, 2468, 3222],
        ],
    ),
];

#[test]
fn all_continuous_lengths_equal_the_design_byte_table() {
    for (task, k, rows) in DESIGN_TABLE {
        for (ci, p) in [8usize, 24, 48, 64].into_iter().enumerate() {
            let packet = synthetic_packet(task, k, p, |_| Kind::Continuous, 7);
            for level in Level::ALL {
                let bytes = pack(&packet, level).unwrap();
                assert_eq!(
                    bytes.len(),
                    rows[level as usize][ci],
                    "{task:?} P={p} level {}",
                    level.name()
                );
            }
        }
    }
}

/// The byte formulas of the specification, written out independently of `sections`.
fn formula(packet: &Packet, level: Level) -> usize {
    let p = packet.cols();
    let (o, m) = (packet.outputs(), packet.regime_means());
    let pc = packet.continuous_cols();
    let atoms: usize = packet
        .columns
        .iter()
        .map(|c| match c {
            super::packet::ColumnMarginal::Discrete(t) => {
                let a = t.prob.len();
                let values = match t.values {
                    super::packet::AtomValues::Lattice { .. } => 4 + a,
                    super::packet::AtomValues::Explicit(_) => 2 * a,
                };
                values + (6 * a).div_ceil(8)
            }
            _ => 0,
        })
        .sum();
    let prior = match packet.task {
        Task::Regression => 0,
        Task::Binary | Task::Multiclass => packet.k as usize - 1,
        Task::Multilabel => packet.k as usize,
    };
    let s = [
        prior + 2,
        p + 5 * pc + atoms,
        p * o + 4 * o,
        4 * p + 4 * m,
        3 * pc,
        4 * p + 4 * m,
        6 * pc,
        p.div_ceil(8) + 4 * o * p,
        4 * p + 4 * m,
        16 * (2 * if p <= 256 { 1 } else { 2 } + o),
    ];
    HEADER_LEN + s[..level.sections()].iter().sum::<usize>() + CRC_LEN
}

#[test]
fn measured_lengths_equal_the_documented_formulas() {
    for (task, k) in tasks() {
        for p in [3usize, 24, 48, 131, 300] {
            let packet = synthetic_packet(task, k, p, mixed, 3);
            for level in Level::ALL {
                let bytes = pack(&packet, level).unwrap();
                assert_eq!(bytes.len(), formula(&packet, level));
                let led = ledger(&bytes).unwrap();
                let parts = led.header + led.sections.iter().sum::<usize>() + led.crc;
                assert_eq!((parts, led.total), (bytes.len(), bytes.len()));
                assert_eq!(
                    led.sections[1],
                    led.s2_flag_plane + led.s2_knot_payload + led.s2_atom_payload
                );
            }
        }
    }
    // Atom payload bytes per column: explicit 2A + ceil(6A/8), lattice 4 + A + ceil(6A/8).
    let two = synthetic_packet(Task::Regression, 0, 1, |_| Kind::Explicit(2), 1);
    let sixteen = synthetic_packet(Task::Regression, 0, 1, |_| Kind::Lattice(16), 1);
    assert_eq!(
        ledger(&pack(&two, Level::A).unwrap())
            .unwrap()
            .s2_atom_payload,
        6
    );
    assert_eq!(
        ledger(&pack(&sixteen, Level::A).unwrap())
            .unwrap()
            .s2_atom_payload,
        32
    );
    // Multilabel rows of the byte table (all continuous): level E at P = 24 / 48.
    for (labels, want) in [(8u8, [1841, 3452]), (32, [5225, 9716])] {
        for (p, bytes) in [24usize, 48].into_iter().zip(want) {
            let packet = synthetic_packet(Task::Multilabel, labels, p, |_| Kind::Continuous, 9);
            assert_eq!(pack(&packet, Level::E).unwrap().len(), bytes);
        }
    }
}

#[test]
fn truncation_is_a_prefix_of_every_level() {
    let mut variants = Vec::new();
    for (task, k) in tasks() {
        let dense = synthetic_packet(task, k, 37, mixed, 11);
        // The two budget fallbacks must keep the property too.
        variants.push(fit_to_budget(&dense, packed_len(&dense, Level::D) - 9).packet);
        variants.push(fit_to_budget(&dense, packed_len(&dense, Level::A) - 9).packet);
        variants.push(dense);
    }
    for packet in variants {
        let top = packet.level;
        let full = pack(&packet, top).unwrap();
        for level in Level::ALL.into_iter().filter(|&l| l <= top) {
            let cut = truncate(&full, level).unwrap();
            assert_eq!(cut, pack(&packet, level).unwrap());
            assert_eq!(
                full[HEADER_LEN..cut.len() - CRC_LEN],
                cut[HEADER_LEN..cut.len() - CRC_LEN]
            );
            for lower in Level::ALL.into_iter().filter(|&l| l <= level) {
                assert_eq!(
                    truncate(&cut, lower).unwrap(),
                    pack(&packet, lower).unwrap()
                );
            }
        }
        if top < Level::E {
            assert!(truncate(&full, Level::E).is_err());
        }
    }
}

#[test]
fn corrupt_files_are_rejected() {
    let packet = synthetic_packet(Task::Binary, 2, 24, mixed, 5);
    let good = pack(&packet, Level::E).unwrap();
    assert!(unpack(&good).is_ok());
    let with = |at: usize, value: u8| {
        let mut b = good.clone();
        b[at] = value;
        unpack(&b)
    };
    assert_eq!(with(0, b'X'), Err(StatError::BadMagic));
    assert_eq!(with(4, 2), Err(StatError::Version));
    assert_eq!(with(5, 1), Err(StatError::Kind));
    assert_eq!(with(16, good[16] ^ 1), Err(StatError::Decoder));
    assert!(matches!(with(15, 1), Err(StatError::Header(_))));
    assert!(matches!(with(12, 9), Err(StatError::Header(_))));
    // CRC: a flipped trailer bit and a flipped payload bit.
    let last = good.len() - 1;
    assert_eq!(with(last, good[last] ^ 0x80), Err(StatError::Crc));
    let payload = HEADER_LEN + 3 + 24 + 2;
    assert_eq!(with(payload, good[payload] ^ 1), Err(StatError::Crc));
    // Trailing bytes and every truncation.
    let mut long = good.clone();
    long.push(0);
    assert_eq!(unpack(&long), Err(StatError::Trailing));
    assert_eq!(unpack(&good[..last]), Err(StatError::Truncated));
    for cut in 0..good.len() {
        assert!(
            unpack(&good[..cut]).is_err(),
            "prefix of {cut} bytes accepted"
        );
    }
    // No single-bit error anywhere in the file is accepted.
    for (at, &byte) in good.iter().enumerate() {
        for bit in 0..8 {
            assert!(
                with(at, byte ^ (1 << bit)).is_err(),
                "flip at byte {at} bit {bit}"
            );
        }
    }
    // Error text is static: it never quotes file content.
    let text = unpack(&long).unwrap_err().to_string();
    assert_eq!(text, "single-encoding: trailing bytes after the checksum");
    assert_eq!(decoder_sha8().len(), 8);
}

#[test]
fn budget_fit_stays_within_two_kib_up_to_256_columns() {
    let mut report = Vec::new();
    for (task, k) in tasks() {
        for p in [8usize, 24, 48, 64, 128, 256] {
            let packet = synthetic_packet(task, k, p, |_| Kind::Continuous, 13);
            let fit = fit_to_budget(&packet, 2048);
            assert!(fit.fits && fit.bytes <= 2048, "{task:?} K={k} P={p}");
            let bytes = pack(&fit.packet, fit.level).unwrap();
            assert_eq!(bytes.len(), fit.bytes);
            assert_eq!(unpack(&bytes).unwrap(), fit.packet);
            assert_eq!(fit, fit_to_budget(&packet, 2048));
            report.push(format!(
                "{task:?}/K{k}/P{p}:{}{}",
                fit.level.name(),
                fit.bytes
            ));
        }
    }
    println!("fit_to_budget(2048): {}", report.join(" "));
    let level_at = |task, k, p: usize, cap| {
        let fit = fit_to_budget(&synthetic_packet(task, k, p, |_| Kind::Continuous, 13), cap);
        (
            fit.level,
            fit.bytes,
            fit.step,
            fit.main_effect_cols,
            fit.target_cols,
        )
    };
    // The design's "what fits" rows.
    let dense = BudgetStep::Level;
    assert_eq!(
        level_at(Task::Regression, 0, 24, 2048),
        (Level::E, 853, dense, 24, 24)
    );
    assert_eq!(
        level_at(Task::Regression, 0, 48, 2048),
        (Level::E, 1624, dense, 48, 48)
    );
    assert_eq!(
        level_at(Task::Regression, 0, 64, 2048),
        (Level::D, 1834, dense, 64, 64)
    );
    assert_eq!(
        level_at(Task::Regression, 0, 24, 512),
        (Level::C, 466, dense, 0, 24)
    );
    assert_eq!(
        level_at(Task::Regression, 0, 24, 1024),
        (Level::E, 853, dense, 24, 24)
    );
    assert_eq!(
        level_at(Task::Regression, 0, 64, 4096),
        (Level::E, 2138, dense, 64, 64)
    );
    // Multiclass K = 5, P = 48: level D with S8 on the top 41 columns, exactly 2,048 B.
    let partial = BudgetStep::PartialMainEffects;
    assert_eq!(
        level_at(Task::Multiclass, 5, 48, 2048),
        (Level::D, 2048, partial, 41, 48)
    );
    // Wide and many-class: level A with weights for the top 9 columns only.
    let sparse = BudgetStep::SparseTarget;
    assert_eq!(
        level_at(Task::Multiclass, 32, 256, 2048),
        (Level::A, 2032, sparse, 0, 9)
    );
    // Beyond the format's reach the result is flagged, never silently cut.
    let wide = fit_to_budget(
        &synthetic_packet(Task::Regression, 0, 400, |_| Kind::Continuous, 1),
        2048,
    );
    assert!(!wide.fits && wide.step == BudgetStep::OverBudget && wide.bytes > 2048);
    assert_eq!(wide.packet.cols(), 400);
}

#[test]
fn budget_fallbacks_keep_the_most_important_columns() {
    let packet = synthetic_packet(Task::Multiclass, 5, 48, |_| Kind::Continuous, 21);
    let (p, o) = (packet.cols(), packet.outputs());
    let weight = |j: usize| -> u32 {
        (0..o)
            .map(|k| packet.target_weight[k * p + j].unsigned_abs() as u32)
            .sum()
    };
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by_key(|&j| (std::cmp::Reverse(weight(j)), j));
    let partial = fit_to_budget(&packet, 2048);
    let kept: Vec<usize> = (0..p).filter(|&j| partial.packet.main_mask[j]).collect();
    let mut want = order[..41].to_vec();
    want.sort_unstable();
    assert_eq!(kept, want);
    for &j in &kept {
        let cell = j * o * 4..(j + 1) * o * 4;
        assert_eq!(
            partial.packet.main_effect[cell.clone()],
            packet.main_effect[cell]
        );
    }
    let sparse = fit_to_budget(&packet, 420);
    assert_eq!(sparse.step, BudgetStep::SparseTarget);
    let kept: Vec<usize> = (0..p).filter(|&j| sparse.packet.target_mask[j]).collect();
    let mut want = order[..sparse.target_cols].to_vec();
    want.sort_unstable();
    assert_eq!(kept, want);
    assert!(sparse.packet.target_sparse && sparse.bytes <= 420);
    assert_eq!(
        unpack(&pack(&sparse.packet, Level::A).unwrap()).unwrap(),
        sparse.packet
    );
}

#[test]
fn binary16_conversion_is_exact_and_nearest() {
    for (x, bits) in [
        (1.0, 0x3c00u16),
        (-2.0, 0xc000),
        (0.1, 0x2e66),
        (65504.0, 0x7bff),
        (1e9, 0x7bff),
        (2f64.powi(-24), 0x0001),
        (0.0, 0),
    ] {
        assert_eq!(f64_to_f16(x), bits, "{x}");
    }
    for bits in (0u16..0x7c00).chain(0x8001..0xfc00) {
        assert_eq!(f64_to_f16(f16_to_f64(bits)), bits);
    }
    for i in 0..4000 {
        let x = -6.5 + 13.0 * i as f64 / 3999.0;
        let near = f16_to_f64(f64_to_f16(x));
        let ulp = 2f64.powi((x.abs().max(1e-3).log2().floor() as i32) - 10);
        assert!((near - x).abs() <= ulp / 2.0 + 1e-15, "{x}");
    }
}
