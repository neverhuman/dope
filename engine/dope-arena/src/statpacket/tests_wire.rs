//! Wire-level pins: the golden packets of the specification, canonical-form
//! checks under mutation with a valid checksum, and the budget boundaries.

use super::codec::{decoder_sha8, fit_to_budget, pack, unpack, BudgetStep};
use super::fit::{fit_packet, FitTarget, Size};
use super::packet::{PacketLevel, PacketTask, CRC_LEN};
use super::testkit::{fit_regression, synthetic_packet, to_unit, SynthKind};
use dope_core::mlc_v3::inflate::crc32_ieee;

const GOLDEN_SHA8: &str = "34db2b6a8d1846ef";
const GOLDEN_CONSTANT: &str =
    "4453453101000000010000000304000034db2b6a8d1846ef000002000000000093e67a93";
const GOLDEN_TWO_SHA256: &str = "52af756efbbfa3d127dff841542414dc42aed32406c314fb18ba91270b26a85e";
const GOLDEN_FITTED: &str = concat!(
    "4453453101000000020004000a06000034db2b6a8d1846ef51100005c44eec64",
    "9b65b965394100ef3c00007f105c5700005ca9000019379a0000000000000000",
    "df477ae1068201f7fb0509000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "000000875a4d48",
);

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The packets quoted in section 11 of the specification.
#[test]
fn golden_packets_match_the_specification() {
    use sha2::{Digest, Sha256};
    println!("decoder sha8 {}", hex(&decoder_sha8()));
    let fit = |raw: &[f64], n: usize, p: usize| {
        let (_, unit) = to_unit(raw, n, p);
        fit_regression(&unit, n, p)
    };
    // Feature sections only (the target sections stay zero).
    let constant = pack(&fit(&[3.0; 16], 16, 1), PacketLevel::A).unwrap();
    println!("constant {}", hex(&constant));
    let raw: Vec<f64> = (0..32).flat_map(|i| [i as f64, (i % 2) as f64]).collect();
    let two = pack(&fit(&raw, 32, 2), PacketLevel::E).unwrap();
    println!(
        "two columns {} bytes sha256 {}",
        two.len(),
        hex(&Sha256::digest(&two))
    );
    // Every section fitted: 64 rows, features `i` and `i mod 2`, target `2 i + 5 (i mod 2)`
    // plus a fixed zig-zag, through `fit_packet`.
    let raw: Vec<f64> = (0..64).flat_map(|i| [i as f64, (i % 2) as f64]).collect();
    let y: Vec<f64> = (0..64)
        .map(|i| 2.0 * i as f64 + 5.0 * (i % 2) as f64 + [0.0, 3.0, -2.0, 1.0][i % 4])
        .collect();
    let fitted = fit_packet(
        &raw,
        FitTarget::Regression(&y),
        PacketTask::Regression,
        0,
        Size::level(PacketLevel::E),
        64,
    )
    .unwrap();
    println!("fitted {}", hex(&fitted.bytes));
    assert_eq!(hex(&decoder_sha8()), GOLDEN_SHA8);
    assert_eq!(hex(&constant), GOLDEN_CONSTANT);
    assert_eq!(two.len(), 139);
    assert_eq!(hex(&Sha256::digest(&two)), GOLDEN_TWO_SHA256);
    assert_eq!(hex(&fitted.bytes), GOLDEN_FITTED);
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
        1 => SynthKind::Explicit(3 + j % 4),
        2 => SynthKind::Lattice(5 + j % 9),
        3 => SynthKind::Constant,
        _ => SynthKind::Continuous,
    };
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (task, k) in [(PacketTask::Binary, 2u8), (PacketTask::Multiclass, 4)] {
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
            &synthetic_packet(PacketTask::Regression, 0, p, |_| SynthKind::Continuous, 3),
            2048,
        );
        (fit.level, fit.bytes, fit.step, fit.target_cols, fit.fits)
    };
    // Level B is 34 + 11 P, dense level A 34 + 7 P, sparse level A 34 + 6 P + ceil(P / 8) + m.
    assert_eq!(
        at(183),
        (PacketLevel::B, 2047, BudgetStep::Level, 183, true)
    );
    assert_eq!(
        at(184),
        (PacketLevel::A, 1322, BudgetStep::Level, 184, true)
    );
    assert_eq!(
        at(287),
        (PacketLevel::A, 2043, BudgetStep::Level, 287, true)
    );
    assert_eq!(
        at(288),
        (PacketLevel::A, 2048, BudgetStep::SparseTarget, 250, true)
    );
    assert_eq!(
        at(328),
        (PacketLevel::A, 2048, BudgetStep::SparseTarget, 5, true)
    );
    assert_eq!(
        at(329),
        (PacketLevel::A, 2050, BudgetStep::OverBudget, 0, false)
    );
}
