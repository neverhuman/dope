use dope_kernel::codec::{self, LoadedKernel, dpk4};
use dope_kernel::compiler::quantized_unit_packet;
use dope_kernel::model::{
    ColumnSchema, Dependence, Kernel, KernelProgram, LinearTerm, Marginal, Noise, ResidualV1,
    SchemaKind, Target, Task, Transform,
};
use dope_kernel::{SampleOptions, sample_kernel};
use sha2::{Digest, Sha256};

fn fixture(bits: u8, residual: bool) -> Kernel {
    Kernel {
        task: Task::Regression,
        rows_fitted: 100,
        features: 1,
        seed: 9,
        seed_policy: 0,
        quantization_bits: bits,
        compliant: false,
        schema: vec![ColumnSchema {
            kind: SchemaKind::Continuous,
            missing_probability: 0.0,
            impute: 0.5,
            transform: Transform::Identity,
        }],
        marginals: vec![Marginal::Constant { value: 0.5 }],
        program: KernelProgram::Symbolic {
            dependence: Dependence::Independent,
            target: Target::SparseLinear {
                intercept: 0.1,
                terms: vec![LinearTerm {
                    feature: 0,
                    coefficient: 0.7,
                }],
            },
            noise: if residual {
                Noise::ResidualV1 {
                    distribution: ResidualV1::StudentT4 {
                        location: 0.0,
                        scale: 0.125,
                    },
                }
            } else {
                Noise::Homoscedastic { sigma: 0.05 }
            },
        },
    }
}

fn unhex(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn resign(artifact: &mut [u8]) {
    let mut hash = Sha256::new();
    hash.update(&artifact[..32]);
    hash.update(&artifact[64..]);
    artifact[32..64].copy_from_slice(&hash.finalize());
}

fn rejected(bytes: &[u8], reason: &str) {
    let error = std::panic::catch_unwind(|| dpk4::decode(bytes))
        .expect("decoder must not panic")
        .unwrap_err()
        .to_string();
    assert!(error.contains(reason), "{error}");
    assert!(!error.contains("PRIVATE_SOURCE_HEADER"));
    assert!(dpk4::inspect(bytes).is_err());
}

#[test]
fn nibble_order_and_complete_wire_golden_are_platform_independent() {
    let packet = dpk4::Packet::Q4((0..16).collect());
    let bytes = dpk4::encode(std::slice::from_ref(&packet)).unwrap();
    let expected = unhex(concat!(
        "44504b3404004000010000000100000028000000000000004001000000000000",
        "8691db6fc7f9ffcb40cc5432778af2cc537d5a397e9f44f419c2f86ba3981770",
        "01000000100000000800000000000000400000000000000032e25d0000000000",
        "1032547698badcfe"
    ));
    assert_eq!(bytes, expected);
    assert_eq!(dpk4::decode(&expected).unwrap(), vec![packet]);
}

#[test]
fn q4_and_q8_charge_real_payload_widths_including_odd_and_empty_tensors() {
    for count in 0usize..=129 {
        let values = (0..count)
            .map(|index| (index % 16) as u8)
            .collect::<Vec<_>>();
        for bits in [4u8, 8] {
            let packet = if bits == 4 {
                dpk4::Packet::Q4(values.clone())
            } else {
                dpk4::Packet::Q8(values.clone())
            };
            let bytes = dpk4::encode(std::slice::from_ref(&packet)).unwrap();
            assert_eq!(dpk4::decode(&bytes).unwrap(), vec![packet]);
            let accounting = dpk4::inspect(&bytes).unwrap();
            let payload = if bits == 4 { count.div_ceil(2) } else { count };
            assert_eq!(accounting.packets[0].payload_bytes, payload);
            assert_eq!(
                accounting.packets[0].payload_bits,
                count * usize::from(bits)
            );
            assert_eq!(accounting.total_bytes, 64 + 32 + payload);
            assert_eq!(accounting.header_bytes, 64);
            assert_eq!(accounting.packets[0].framing_bytes, 32);
            if count % 2 == 0 {
                assert_eq!(count.div_ceil(2) * 2, count); // half q8
                assert_eq!(count.div_ceil(2) * 4, count * 2); // quarter u16
            }
        }
    }
    assert_eq!(dpk4::pack_levels(&[1, 2, 3], 4).unwrap(), [0x21, 0x03]);
    assert_eq!(dpk4::unpack_levels(&[0x21, 0x03], 3, 4).unwrap(), [1, 2, 3]);
}

#[test]
fn compiler_quantization_emits_usable_q4_and_q8_packets() {
    for bits in [4u8, 8] {
        let maximum = (1u16 << bits) - 1;
        let values = (0..=maximum)
            .map(|n| f32::from(n) / f32::from(maximum))
            .collect::<Vec<_>>();
        let packet = quantized_unit_packet(&values, bits).unwrap();
        let bytes = dpk4::encode(std::slice::from_ref(&packet)).unwrap();
        assert_eq!(dpk4::decode(&bytes).unwrap(), vec![packet.clone()]);
        let levels = match packet {
            dpk4::Packet::Q4(levels) | dpk4::Packet::Q8(levels) => levels,
            _ => panic!("compiler returned a non-quantized packet"),
        };
        assert_eq!(levels, (0..=maximum).map(|n| n as u8).collect::<Vec<_>>());
    }
    for value in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        assert!(quantized_unit_packet(&[value], 4).is_err());
    }
    assert!(quantized_unit_packet(&[0.5], 6).is_err());
    assert_eq!(
        quantized_unit_packet(&[0.5], 4).unwrap(),
        dpk4::Packet::Q4(vec![8])
    );
}

#[test]
fn candidate_models_reload_identical_semantics_and_downstream_samples() {
    for bits in [4u8, 8] {
        for residual in [false, true] {
            let mut kernel = fixture(bits, residual);
            let maximum = ((1u16 << bits) - 1) as f32;
            kernel.schema[0].impute = 8.0 / maximum;
            kernel.marginals[0] = Marginal::QuantileSpline {
                values: vec![0.0, 3.0 / maximum, 8.0 / maximum, 12.0 / maximum, 1.0],
            };
            let bytes = dpk4::encode_kernel(&kernel).unwrap();
            let reloaded = dpk4::decode_kernel(&bytes).unwrap();
            assert_eq!(kernel, reloaded);
            assert_eq!(dpk4::encode_kernel(&reloaded).unwrap(), bytes);
            let accounting = dpk4::inspect(&bytes).unwrap();
            assert!(accounting.packets[1].elements >= 7);
            assert!(accounting.packets[2].elements >= 2);
            assert_eq!(
                accounting.header_bytes
                    + accounting
                        .packets
                        .iter()
                        .map(|p| p.framing_bytes + p.payload_bytes)
                        .sum::<usize>(),
                bytes.len()
            );
            let options = SampleOptions {
                rows: 32,
                seed: Some(1729),
            };
            assert_eq!(
                sample_kernel(&LoadedKernel::V3(Box::new(kernel)), options).unwrap(),
                sample_kernel(&LoadedKernel::V3(Box::new(reloaded)), options).unwrap()
            );
        }
    }
}

#[test]
fn little_endian_floats_and_unaligned_parameters_keep_their_bits() {
    let packet = dpk4::Packet::F32(vec![-0.0, 1.0, -2.5, f32::MIN_POSITIVE]);
    let bytes = dpk4::encode(std::slice::from_ref(&packet)).unwrap();
    assert_eq!(&bytes[96..104], &[0, 0, 0, 128, 0, 0, 128, 63]);
    assert_eq!(dpk4::decode(&bytes).unwrap(), vec![packet]);
    let mut kernel = fixture(4, false);
    let KernelProgram::Symbolic {
        target: Target::SparseLinear { intercept, .. },
        ..
    } = &mut kernel.program
    else {
        unreachable!()
    };
    *intercept = -0.0;
    let bytes = dpk4::encode_kernel(&kernel).unwrap();
    let restored = dpk4::decode_kernel(&bytes).unwrap();
    assert_eq!(
        serde_json::to_vec(&kernel).unwrap(),
        serde_json::to_vec(&restored).unwrap()
    );
}

#[test]
fn malformed_headers_packets_and_integrity_fail_closed() {
    let valid = dpk4::encode(&[dpk4::Packet::Q4(vec![1, 2, 3])]).unwrap();
    for length in 0..valid.len() {
        assert!(
            std::panic::catch_unwind(|| dpk4::decode(&valid[..length]))
                .unwrap()
                .is_err()
        );
    }
    let mut bad = valid.clone();
    bad[0] ^= 1;
    rejected(&bad, "magic");
    let mut bad = valid.clone();
    bad[5] = 1;
    rejected(&bad, "version");
    let mut bad = valid.clone();
    bad[11] = 128;
    rejected(&bad, "features");
    let mut bad = valid.clone();
    bad[12..16].copy_from_slice(&65u32.to_le_bytes());
    rejected(&bad, "count");
    for offset in [16, 24] {
        let mut bad = valid.clone();
        bad[offset..offset + 8].fill(255);
        rejected(&bad, "length");
    }
    let mut bad = valid.clone();
    bad[32] ^= 1;
    rejected(&bad, "SHA-256");
    let mut bad = valid.clone();
    bad[96] ^= 1;
    rejected(&bad, "SHA-256");
    let mut bad = valid.clone();
    bad[64] = 255;
    resign(&mut bad);
    rejected(&bad, "unknown packet");
    let mut bad = valid.clone();
    bad[65] = 1;
    resign(&mut bad);
    rejected(&bad, "encoding");
    let mut bad = valid.clone();
    bad[95] = 1;
    resign(&mut bad);
    rejected(&bad, "flags");
    let mut bad = valid.clone();
    bad[68..72].fill(255);
    resign(&mut bad);
    rejected(&bad, "bound");
    for offset in [72, 80] {
        let mut bad = valid.clone();
        bad[offset..offset + 8].fill(255);
        resign(&mut bad);
        rejected(&bad, "length");
    }
    let mut bad = valid.clone();
    bad[88] ^= 1;
    resign(&mut bad);
    rejected(&bad, "CRC32");
    let mut bad = valid.clone();
    bad[96] ^= 1;
    resign(&mut bad);
    rejected(&bad, "CRC32");
    let mut bad = valid.clone();
    bad[8] = 2;
    resign(&mut bad);
    rejected(&bad, "feature mismatch");
    let mut bad = valid.clone();
    bad[12] = 2;
    resign(&mut bad);
    rejected(&bad, "truncated");
    let mut bad = valid.clone();
    bad.push(0);
    let length = (bad.len() - 64) as u64;
    bad[16..24].copy_from_slice(&length.to_le_bytes());
    bad[24..32].copy_from_slice(&(length * 8).to_le_bytes());
    resign(&mut bad);
    rejected(&bad, "trailing");
}

#[test]
fn invalid_numeric_inputs_and_noncanonical_padding_are_rejected() {
    assert!(dpk4::encode(&[]).is_err());
    assert!(dpk4::encode(&[dpk4::Packet::Q4(vec![16])]).is_err());
    assert!(dpk4::encode(&[dpk4::Packet::F32(vec![f32::NAN])]).is_err());
    assert!(dpk4::pack_levels(&[1], 16).is_err());
    assert!(dpk4::unpack_levels(&[], 1, 4).is_err());
    assert!(dpk4::unpack_levels(&[0xf1], 1, 4).is_err());
    assert!(dpk4::unpack_levels(&[], usize::MAX, 4).is_err());
    let packet = dpk4::Packet::Q4(vec![0; dpk4::MAX_VALUES]);
    assert!(dpk4::encode(&[packet, dpk4::Packet::Q8(vec![0])]).is_err());
    let mut padding = dpk4::encode(&[dpk4::Packet::Q4(vec![1, 2, 3])]).unwrap();
    padding[97] = 0xf3;
    // Independently computed IEEE CRC32 for [0x21, 0xf3].
    padding[88..92].copy_from_slice(&[0xba, 0xa4, 0xf2, 0xe9]);
    resign(&mut padding);
    rejected(&padding, "padding");
    let mut nan = dpk4::encode(&[dpk4::Packet::F32(vec![0.0])]).unwrap();
    nan[96..100].copy_from_slice(&[0, 0, 0xc0, 0x7f]);
    nan[88..92].copy_from_slice(&[0xff, 0x64, 0x04, 0x2a]);
    resign(&mut nan);
    rejected(&nan, "non-finite");
}

#[test]
fn model_errors_are_redacted_and_bad_semantics_do_not_reload() {
    let mut packets = dpk4::decode(&dpk4::encode_kernel(&fixture(4, false)).unwrap()).unwrap();
    packets[0] = dpk4::Packet::Model(b"{PRIVATE_SOURCE_HEADER: -987654321}".to_vec());
    let error = dpk4::decode_kernel(&dpk4::encode(&packets).unwrap())
        .unwrap_err()
        .to_string();
    assert!(error.contains("model layout"));
    assert!(!error.contains("PRIVATE_SOURCE_HEADER") && !error.contains("987654321"));
    let mut packets = dpk4::decode(&dpk4::encode_kernel(&fixture(4, false)).unwrap()).unwrap();
    let dpk4::Packet::Model(layout) = &mut packets[0] else {
        unreachable!()
    };
    let mut value: serde_json::Value = serde_json::from_slice(layout).unwrap();
    value["features"] = 0.into();
    *layout = serde_json::to_vec(&value).unwrap();
    assert!(dpk4::decode_kernel(&dpk4::encode(&packets).unwrap()).is_err());
    assert!(dpk4::decode_kernel(&dpk4::encode(&[dpk4::Packet::Q4(vec![])]).unwrap()).is_err());
}

#[test]
fn bounded_adversarial_corpus_never_panics() {
    let valid = dpk4::encode(&[dpk4::Packet::Q8((0..=255).collect())]).unwrap();
    for offset in 0..valid.len() {
        let mut bytes = valid.clone();
        bytes[offset] ^= 128;
        assert!(
            std::panic::catch_unwind(|| dpk4::decode(&bytes))
                .unwrap()
                .is_err()
        );
    }
    let mut state = 1729u64;
    for count in 0..512 {
        let mut bytes = vec![0; count];
        for byte in &mut bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        assert!(
            std::panic::catch_unwind(|| dpk4::decode(&bytes))
                .unwrap()
                .is_err()
        );
    }
}

#[test]
fn legacy_dpk32_and_dpk33_bytes_are_frozen_to_main() {
    // Origin/main 2bfee82: legacy codec/model source closure is unchanged.
    for residual in [false, true] {
        let bytes = codec::encode_kernel(&fixture(8, residual)).unwrap();
        let expected = unhex(if residual {
            concat!(
                "44504b3303030100080000000100000064000000000000000900000000000000",
                "26000000000000004972b9040c6c23616993c08823b81a52010701050000008000",
                "020401000080030100040900c09a0c0100c0b95505070401010090a10f"
            )
        } else {
            concat!(
                "44504b3303020100080000000100000064000000000000000900000000000000",
                "2200000000000000fb4d4808800ef039faea1859c4748fbd010701050000008000",
                "020401000080030100040900c09a0c0100c0b955050300cd0c"
            )
        });
        assert_eq!(bytes, expected);
        assert_eq!(bytes[4..6], [3, if residual { 3 } else { 2 }]);
        assert_eq!(
            codec::encode_kernel(&codec::decode_kernel(&expected).unwrap()).unwrap(),
            expected
        );
    }
}

#[test]
fn legacy_v1_json_reader_remains_unchanged() {
    let value = serde_json::json!({"format":"dope-kernel", "version":1, "seed":1729});
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/f01-evidence")
        .join(format!("v1-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let LoadedKernel::V1(loaded) = codec::load_kernel(&path).unwrap() else {
        panic!("legacy reader changed")
    };
    assert_eq!(loaded, value);
}
