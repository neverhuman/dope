//! Black-box probes for the draft 24-byte DSE8 v1 framing and native emitter.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const HEADER: usize = 24;
const SECTION_COUNT: usize = 20;

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let polynomial = if crc & 1 == 1 { 0xedb8_8320 } else { 0 };
            crc = (crc >> 1) ^ polynomial;
        }
    }
    !crc
}

fn sign(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&crc32(bytes).to_le_bytes());
}

fn sections(packet: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&packet[..5], b"DSE8\x01");
    let end = packet.len() - 4;
    assert_eq!(packet[end..], crc32(&packet[..end]).to_le_bytes());
    let mut parts = Vec::new();
    let mut remaining = &packet[HEADER..end];
    while !remaining.is_empty() {
        let size = usize::from(u16::from_le_bytes([remaining[1], remaining[2]]));
        let (section, tail) = remaining.split_at(3 + size);
        parts.push(section);
        remaining = tail;
    }
    assert_eq!(parts.len(), usize::from(packet[SECTION_COUNT]));
    parts
}

fn rebuild(packet: &[u8], parts: &[&[u8]]) -> Vec<u8> {
    let mut out = packet[..HEADER].to_vec();
    out[SECTION_COUNT] = u8::try_from(parts.len()).unwrap();
    out.extend(parts.iter().copied().flatten());
    sign(&mut out);
    out
}

fn malformed(packet: &[u8]) -> Vec<(&'static str, Vec<u8>)> {
    let mut vectors = Vec::new();
    for (name, at, value) in [
        ("version", 4, 2),
        ("task", 5, u8::MAX),
        ("active_slots", 7, 49),
        ("unknown_flag", 10, packet[10] | 0x80),
        ("section_count", SECTION_COUNT, packet[SECTION_COUNT] - 1),
        ("code_tier_without_code", 21, 1),
        ("reserved_header", 22, 1),
    ] {
        let mut bytes = packet[..packet.len() - 4].to_vec();
        bytes[at] = value;
        sign(&mut bytes);
        vectors.push((name, bytes));
    }
    let parts = sections(packet);
    let mut duplicate = parts.clone();
    duplicate.insert(0, parts[0]);
    vectors.push(("duplicate_section", rebuild(packet, &duplicate)));
    let mut reordered = parts.clone();
    reordered.swap(0, 1);
    vectors.push(("reordered_section", rebuild(packet, &reordered)));
    vectors.push(("missing_base_section", rebuild(packet, &parts[1..])));
    let mut invalid_len = packet[..packet.len() - 4].to_vec();
    invalid_len[HEADER + 1..HEADER + 3].copy_from_slice(&u16::MAX.to_le_bytes());
    sign(&mut invalid_len);
    vectors.push(("oversized_section", invalid_len));
    vectors
}

fn emit(binary: &Path, dir: &Path, args: &[&str]) -> Output {
    Command::new(binary)
        .current_dir(dir)
        .env_clear()
        .env("PATH", "")
        .args(args)
        .output()
        .unwrap()
}

fn matrix(width: u32, classes: u32) -> Vec<u8> {
    let rows = 256;
    let mut bytes = b"DOPEMAT1".to_vec();
    for field in [rows, width, 1, 0] {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    for row in 0..rows {
        let a = row & 1;
        let b = (row >> 1) & 1;
        for col in 0..width {
            let value = match col {
                0 => a as f32,
                1 => b as f32,
                _ => (row.wrapping_mul(37 + col) % 251) as f32 / 250.0,
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let label = if classes == 2 { a ^ b } else { row % classes };
        bytes.extend_from_slice(&(label as f32).to_le_bytes());
    }
    bytes
}

#[test]
fn independent_crc_matches_standard_check_value() {
    assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
}

#[test]
fn malformed_vectors_keep_valid_checksums_and_account_for_all_sections() {
    let mut header = [0; HEADER];
    header[..5].copy_from_slice(b"DSE8\x01");
    let parts: [&[u8]; 3] = [&[2, 2, 0, 17, 3], &[3, 1, 0, 42], &[4, 0, 0]];
    let packet = rebuild(&header, &parts);
    assert_eq!(sections(&packet), parts);
    assert_eq!(
        packet.len(),
        HEADER + parts.iter().map(|p| p.len()).sum::<usize>() + 4
    );
    let vectors = malformed(&packet);
    assert_eq!(vectors.len(), 11);
    for (name, invalid) in vectors {
        let end = invalid.len() - 4;
        assert_ne!(invalid, packet, "{name}");
        assert_eq!(
            invalid[end..],
            crc32(&invalid[..end]).to_le_bytes(),
            "{name}"
        );
    }
}

fn candidate(probe: &str) -> (PathBuf, PathBuf) {
    let binary = PathBuf::from(std::env::var_os("DOPE_V8_EMITTER").unwrap())
        .canonicalize()
        .unwrap();
    let expected = std::env::var("DOPE_V8_EMITTER_SHA256").unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(&binary).unwrap())),
        expected
    );
    let dir = std::env::current_exe()
        .unwrap()
        .ancestors()
        .nth(3)
        .unwrap()
        .join("dse8-conformance")
        .join(format!("{}-{probe}", std::process::id()));
    fs::create_dir_all(dir.join("fit")).unwrap();
    fs::create_dir_all(dir.join("recipient")).unwrap();
    (binary, dir)
}

fn fit(binary: &Path, dir: &Path, width: u32, classes: u32, model: &str, budget: &str) -> Output {
    fs::write(dir.join("fit/input.mat"), matrix(width, classes)).unwrap();
    emit(
        binary,
        dir,
        &[
            "fit",
            "--in",
            "fit/input.mat",
            "--task",
            if classes == 2 { "binary" } else { "multiclass" },
            "--normalized-only",
            "--budget",
            budget,
            "--out",
            model,
        ],
    )
}

#[test]
#[ignore = "requires a digest-pinned dope_v8_emit candidate using the draft framing"]
fn pinned_emitter_checks_wide_multiclass_shape_limits() {
    let (binary, dir) = candidate("shape");
    let admitted = fit(&binary, &dir, 48, 8, "wide.dse8", "2048");
    let mut failures = Vec::new();
    if let Ok(wide) = fs::read(dir.join("wide.dse8")) {
        assert!(admitted.status.success() && wide.len() <= 2048);
        assert_eq!((wide[7], wide[6]), (48, 8));
    } else {
        failures.push(format!(
            "48 slots and 8 classes at 2 KiB produced no artifact ({})",
            String::from_utf8_lossy(&admitted.stdout).trim()
        ));
    }
    for (width, classes) in [(49, 8), (48, 9)] {
        let name = format!("refused-{width}-{classes}.dse8");
        // Use the larger cap to distinguish shape rejection from budget refusal.
        let output = fit(&binary, &dir, width, classes, &name, "4096");
        if output.status.success() || dir.join(&name).exists() {
            failures.push(format!(
                "oversized shape {width}/{classes} lacked a shape refusal ({})",
                String::from_utf8_lossy(&output.stdout).trim()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "requires a digest-pinned dope_v8_emit candidate using the draft framing"]
fn pinned_emitter_preserves_the_legal_base_across_rungs() {
    let (binary, dir) = candidate("rungs");
    assert!(fit(&binary, &dir, 48, 8, "wide-4096.dse8", "4096")
        .status
        .success());
    let mut previous = fs::read(dir.join("wide-4096.dse8")).unwrap();
    assert!(previous.len() <= 4096);
    let base_bytes = HEADER
        + 4
        + sections(&previous)
            .iter()
            .filter(|part| part[0] <= 5)
            .map(|part| part.len())
            .sum::<usize>();
    let mut from = "wide-4096.dse8".to_owned();
    let mut refused = false;
    let mut headline = false;
    for cap in [2048, 1024, 512] {
        let model = format!("wide-{cap}.dse8");
        let output = emit(
            &binary,
            &dir,
            &[
                "truncate",
                "--model",
                &from,
                "--budget",
                &cap.to_string(),
                "--out",
                &model,
            ],
        );
        if !output.status.success() {
            assert!(!dir.join(&model).exists());
            println!(
                "refused {cap} B rung ({})",
                String::from_utf8_lossy(&output.stderr).trim()
            );
            refused = true;
            continue;
        }
        headline |= cap == 2048;
        assert!(
            !refused,
            "a lower rung must also refuse below the legal base"
        );
        let smaller = fs::read(dir.join(&model)).unwrap();
        assert!(smaller.len() <= cap && smaller.len() <= previous.len());
        let old = sections(&previous);
        let new = sections(&smaller);
        assert!(new.len() <= old.len());
        for (before, after) in old.iter().zip(&new) {
            assert_eq!(before[0], after[0], "section tag changed within the prefix");
            if before[0] <= 5 {
                assert_eq!(before, after, "mandatory section changed");
            }
        }
        previous = smaller;
        from = model;
    }
    assert!(
        headline,
        "the 4 KiB artifact has a {base_bytes} B legal base and cannot emit the 2 KiB rung"
    );
}

#[test]
#[ignore = "requires a digest-pinned dope_v8_emit candidate using the draft framing"]
fn pinned_emitter_is_deterministic_and_preserves_the_first_row() {
    let (binary, dir) = candidate("recipient");
    let narrow = "recipient/narrow.dse8";
    assert!(
        fit(&binary, &dir, 6, 2, narrow, "2048").status.success(),
        "narrow fit"
    );
    let packet = fs::read(dir.join(narrow)).unwrap();
    assert!(packet.len() <= 2048);
    assert_eq!(packet[7], 6);
    assert!(fit(&binary, &dir, 6, 2, "repeat.dse8", "2048")
        .status
        .success());
    assert_eq!(fs::read(dir.join("repeat.dse8")).unwrap(), packet);
    fs::rename(dir.join("fit/input.mat"), dir.join("fit/retained.mat")).unwrap();

    let recipient = dir.join("recipient");
    assert!(
        emit(&binary, &recipient, &["inspect", "--model", "narrow.dse8"])
            .status
            .success()
    );
    for rows in ["1", "256"] {
        let out = format!("rows-{rows}.mat");
        assert!(
            emit(
                &binary,
                &recipient,
                &[
                    "sample",
                    "--model",
                    "narrow.dse8",
                    "--rows",
                    rows,
                    "--seed",
                    "27",
                    "--out",
                    &out
                ]
            )
            .status
            .success(),
            "artifact sample"
        );
    }
    let one = fs::read(recipient.join("rows-1.mat")).unwrap();
    let many = fs::read(recipient.join("rows-256.mat")).unwrap();
    assert_eq!(&one[..8], b"DOPEMAT1");
    assert_eq!(&many[..8], b"DOPEMAT1");
    assert_eq!(one.len(), 24 + 7 * 4);
    assert_eq!(many.len(), 24 + 256 * 7 * 4);
    assert_eq!(one[24..], many[24..one.len()]);
}

#[test]
#[ignore = "requires a digest-pinned dope_v8_emit candidate using the draft framing"]
fn pinned_emitter_rejects_crc_valid_malformed_packets() {
    let (binary, dir) = candidate("malformed");
    let narrow = "recipient/narrow.dse8";
    assert!(
        fit(&binary, &dir, 6, 2, narrow, "2048").status.success(),
        "narrow fit"
    );
    let packet = fs::read(dir.join(narrow)).unwrap();
    let recipient = dir.join("recipient");
    let mut accepted = Vec::new();
    for (name, bytes) in malformed(&packet) {
        let model = format!("{name}.dse8");
        fs::write(recipient.join(&model), bytes).unwrap();
        if emit(&binary, &recipient, &["inspect", "--model", &model])
            .status
            .success()
        {
            accepted.push(name);
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted CRC-valid malformed packets {accepted:?}"
    );
}

#[test]
#[ignore = "requires a digest-pinned dope_v8_emit candidate"]
fn pinned_extractor_preserves_evidence_when_equal_fingerprints_reorder() {
    let (binary, dir) = candidate("physical-order");
    // Both feature vectors have the same column-permutation-invariant fingerprint.
    // Transposition reorders every physical row/target pair as one unit.
    for (name, transpose) in [("original", false), ("permuted", true)] {
        let mut bytes = b"DOPEMAT1".to_vec();
        for field in [256_u32, 2, 1, 0] {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        for position in 0..256 {
            let row = if transpose {
                (position % 16) * 16 + position / 16
            } else {
                position
            };
            let bit = row & 1;
            let target = (row % 13 + 4 * bit + 20 * (row % 4)) as f32;
            for value in [bit as f32, (1 - bit) as f32, target] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let input = format!("fit/{name}.mat");
        let output = format!("{name}.json");
        fs::write(dir.join(&input), bytes).unwrap();
        assert!(emit(
            &binary,
            &dir,
            &[
                "evidence",
                "--in",
                &input,
                "--task",
                "regression",
                "--tier",
                "t2",
                "--seed",
                "7",
                "--out",
                &output
            ]
        )
        .status
        .success());
    }
    let original = fs::read(dir.join("original.json")).unwrap();
    let permuted = fs::read(dir.join("permuted.json")).unwrap();
    assert!(
        original == permuted,
        "evidence changed when identical physical rows were reordered"
    );
}
