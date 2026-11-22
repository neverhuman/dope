//! Wire-level pins: the golden packets of the specification, canonical-form
//! checks under mutation with a valid checksum, and the budget boundaries.

use super::codec::{decoder_sha8, fit_to_budget, pack, unpack, BudgetStep};
use super::packet::{Level, Task, CRC_LEN};
use super::testkit::{fit_regression, synthetic_packet, to_unit, Kind};
use dope_core::mlc_v3::inflate::crc32_ieee;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The two packets quoted in section 10 of the specification.
#[test]
fn golden_packets_match_the_specification() {
    use sha2::{Digest, Sha256};
    assert_eq!(hex(&decoder_sha8()), "9542a24b5ce26f91");
    let fit = |raw: &[f64], n: usize, p: usize| {
        let (_, unit) = to_unit(raw, n, p);
        fit_regression(&unit, n, p)
    };
    let constant = pack(&fit(&[3.0; 16], 16, 1), Level::A).unwrap();
    assert_eq!(
        hex(&constant),
        "445345310100000001000000030400009542a24b5ce26f910000020000000000f46dd66e"
    );
    let raw: Vec<f64> = (0..32).flat_map(|i| [i as f64, (i % 2) as f64]).collect();
    let two = pack(&fit(&raw, 32, 2), Level::E).unwrap();
    assert_eq!(two.len(), 139);
    assert_eq!(
        hex(&Sha256::digest(&two)),
        "23cfd3aba4f988e3dd696507c72a3ebcc1b02d7f1ad4121bbb73db2fb1f9c11d"
    );
}

/// Rewrites the trailer so that only structural checks can reject the file.
fn with_valid_crc(mut bytes: Vec<u8>) -> Vec<u8> {
    let end = bytes.len() - CRC_LEN;
    let crc = crc32_ieee(&bytes[..end]);
    bytes[end..].copy_from_slice(&crc.to_le_bytes());
    bytes
}

#[test]
fn a_mutated_file_with_a_valid_checksum_is_rejected_or_canonical() {
    let kinds = |j: usize| match j % 4 {
        1 => Kind::Explicit(3 + j % 4),
        2 => Kind::Lattice(5 + j % 9),
        3 => Kind::Constant,
        _ => Kind::Continuous,
    };
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (task, k) in [(Task::Binary, 2u8), (Task::Multiclass, 4)] {
        let packet = synthetic_packet(task, k, 21, kinds, 77);
        let sparse = fit_to_budget(&packet, 250).packet;
        for source in [packet, sparse] {
            let good = pack(&source, source.level).unwrap();
            for at in 0..good.len() - CRC_LEN {
                for value in [
                    0u8,
                    0x7f,
                    0x80,
                    0xff,
                    good[at] ^ 1,
                    good[at].wrapping_add(1),
                ] {
                    let mut bytes = good.clone();
                    bytes[at] = value;
                    let bytes = with_valid_crc(bytes);
                    // Never a panic; an accepted file is the unique encoding of its packet.
                    match unpack(&bytes) {
                        Ok(back) => {
                            accepted += 1;
                            assert_eq!(pack(&back, back.level).unwrap(), bytes, "byte {at}");
                        }
                        Err(_) => rejected += 1,
                    }
                }
            }
        }
    }
    println!("mutation with valid CRC: {accepted} accepted as canonical, {rejected} rejected");
    assert!(accepted > 1000 && rejected > 1000);
}

#[test]
fn budget_boundaries_of_the_specification() {
    let at = |p: usize| {
        let fit = fit_to_budget(
            &synthetic_packet(Task::Regression, 0, p, |_| Kind::Continuous, 3),
            2048,
        );
        (fit.level, fit.bytes, fit.step, fit.target_cols, fit.fits)
    };
    // Level B is 34 + 11 P, dense level A 34 + 7 P, sparse level A 34 + 6 P + ceil(P / 8) + m.
    assert_eq!(at(183), (Level::B, 2047, BudgetStep::Level, 183, true));
    assert_eq!(at(184), (Level::A, 1322, BudgetStep::Level, 184, true));
    assert_eq!(at(287), (Level::A, 2043, BudgetStep::Level, 287, true));
    assert_eq!(
        at(288),
        (Level::A, 2048, BudgetStep::SparseTarget, 250, true)
    );
    assert_eq!(at(328), (Level::A, 2048, BudgetStep::SparseTarget, 5, true));
    assert_eq!(at(329), (Level::A, 2050, BudgetStep::OverBudget, 0, false));
}
