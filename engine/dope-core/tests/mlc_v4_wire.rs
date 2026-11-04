use dope_core::mlc_v4::{
    self, basis, rd_pack, wire, GateValues, GroupModel, Level, Packet, PacketIdentity, TargetModel,
};
use dope_norm::{GroupKind, GroupSpec, TargetKind, UnitSchema};

fn packet() -> Packet {
    let schema = UnitSchema {
        schema: "dope.nc1-public/v1".into(),
        recipe: dope_norm::RECIPE.into(),
        columns: 1,
        groups: vec![GroupSpec {
            id: 0,
            kind: GroupKind::Numeric,
            start: 0,
            width: 1,
            has_missing: false,
        }],
        target: TargetKind::Regression,
        classes: 0,
    };
    Packet {
        contract: "dope.dml4-reference/v1".into(),
        level: Level::L2k,
        schema_sha256: schema.sha256().unwrap(),
        schema,
        identity: PacketIdentity {
            weights_sha256: "1".repeat(64),
            encoder_recipe_sha256: "2".repeat(64),
            conditioning_sha256: "3".repeat(64),
            fit_scope_sha256: "4".repeat(64),
        },
        gates: GateValues {
            linear: 0.3,
            asinh: 0.4,
            periodic: 0.5,
            hinge: 0.6,
            pair: 0.7,
            row_mean: 0.8,
        },
        groups: vec![GroupModel::Numeric {
            knots: vec![0.0, 1.0],
            missing_probability: 0.0,
            imputation: 0.5,
        }],
        target: TargetModel::Regression {
            intercept: 0.0,
            coefficients: vec![0.1; 5],
            noise_scale: 0.0,
        },
    }
}
#[test]
fn canonical_roundtrip_counts_whole_envelope() {
    let p = packet();
    let bytes = wire::encode(&p).unwrap();
    assert!(bytes.len() <= 2048);
    assert_eq!(&bytes[..4], b"DML4");
    assert_eq!(wire::encode(&wire::decode(&bytes).unwrap()).unwrap(), bytes);
    assert_eq!(wire::decode(&bytes).unwrap(), p);
    assert_eq!(wire::crc32(b"123456789"), 0xcbf43926);
}
#[test]
fn fitted_and_decoded_nondefault_gates_match_independent_math() {
    let p = packet();
    let decoded = wire::decode(&wire::encode(&p).unwrap()).unwrap();
    let x = 0.75f64;
    let expected = vec![
        0.3 * x,
        0.4 * x.asinh(),
        0.5 * (std::f64::consts::TAU * x).sin(),
        0.6 * (x - 0.5),
        0.8 * x,
    ];
    assert_eq!(basis::features(&[x], &p.gates).unwrap(), expected);
    assert_eq!(basis::features(&[x], &decoded.gates).unwrap(), expected);
    let mut omitted = decoded.gates;
    omitted.asinh = 0.0;
    assert_ne!(basis::features(&[x], &omitted).unwrap(), expected);
    assert_eq!(basis::REGISTRY.len(), 6);
}
#[test]
fn corrupt_truncated_oversized_and_unknown_fields_refuse() {
    let b = wire::encode(&packet()).unwrap();
    for length in [0, 1, 4, 12, b.len() - 1] {
        assert!(wire::decode(&b[..length]).is_err());
    }
    let mut bad = b.clone();
    bad[wire::HEADER_BYTES + 1] ^= 1;
    assert_eq!(wire::decode(&bad).unwrap_err().code, "checksum_mismatch");
    let mut bad = b.clone();
    bad[4] = 99;
    assert!(wire::decode(&bad).is_err());
    assert!(wire::decode(&vec![0; 65537]).is_err());
    let mut body: serde_json::Value = serde_json::from_slice(&b[wire::HEADER_BYTES..]).unwrap();
    body["private_header"] = serde_json::json!("CANARY");
    let body = serde_json::to_vec(&body).unwrap();
    let mut encoded = b[..5].to_vec();
    encoded.extend_from_slice(&(body.len() as u32).to_le_bytes());
    encoded.extend_from_slice(&wire::crc32(&body).to_le_bytes());
    encoded.extend_from_slice(&body);
    assert_eq!(
        wire::decode(&encoded).unwrap_err().code,
        "invalid_packet_json"
    );
}
#[test]
fn impossible_cap_refuses_instead_of_oversize_success() {
    let mut p = packet();
    p.schema.columns = 256;
    p.schema.groups = vec![GroupSpec {
        id: 0,
        kind: GroupKind::Nominal,
        start: 0,
        width: 256,
        has_missing: false,
    }];
    p.schema_sha256 = p.schema.sha256().unwrap();
    p.groups = vec![GroupModel::Categorical {
        probabilities: vec![1.0; 256],
    }];
    p.target = TargetModel::Regression {
        intercept: 0.0,
        coefficients: vec![0.0; 1280],
        noise_scale: 0.0,
    };
    assert_eq!(wire::encode(&p).unwrap_err().code, "over_budget");
    p.level = Level::L16k;
    assert!(wire::encode(&p).unwrap().len() <= 16384);
}
#[test]
fn strict_semantic_layout_and_finite_values_refuse() {
    let mut p = packet();
    p.schema_sha256 = "f".repeat(64);
    assert!(wire::encode(&p).is_err());
    let mut p = packet();
    p.gates.periodic = f64::NAN;
    assert!(wire::encode(&p).is_err());
    let mut p = packet();
    p.groups = vec![GroupModel::Categorical {
        probabilities: vec![1.0],
    }];
    assert!(wire::encode(&p).is_err());
    let mut p = packet();
    p.target = TargetModel::Regression {
        intercept: 0.0,
        coefficients: vec![],
        noise_scale: 0.0,
    };
    assert!(wire::encode(&p).is_err());
}
#[test]
fn rate_distortion_candidate_order_does_not_change_selection() {
    let a = rd_pack::Candidate {
        packet: packet(),
        fit_tune_loss: 1.0,
    };
    let mut other = packet();
    other.groups = vec![GroupModel::Numeric {
        knots: vec![0.0, 0.5, 1.0],
        missing_probability: 0.0,
        imputation: 0.5,
    }];
    let b = rd_pack::Candidate {
        packet: other,
        fit_tune_loss: 0.99,
    };
    assert_eq!(
        rd_pack::select(&[a.clone(), b.clone()], Level::L2k, 0.02).unwrap(),
        rd_pack::select(&[b, a], Level::L2k, 0.02).unwrap()
    );
    assert!(rd_pack::select(&[], Level::L2k, 0.0).is_err());
}
#[test]
fn scalar_basis_width_and_limits_are_explicit() {
    assert_eq!(basis::width(256).unwrap(), 1280);
    assert!(basis::width(257).is_err());
    assert!(basis::features(&[f64::INFINITY], &packet().gates).is_err());
}
#[test]
fn conformance_export_when_explicitly_requested() {
    if let Ok(path) = std::env::var("DOPE_DML4_CONFORMANCE_OUT") {
        let root = std::path::Path::new(&path);
        std::fs::create_dir_all(root).unwrap();
        let p = packet();
        let bytes = wire::encode(&p).unwrap();
        std::fs::write(root.join("packet.dml4"), &bytes).unwrap();
        std::fs::write(root.join("packet.json"), serde_json::to_vec(&p).unwrap()).unwrap();
        let samples = mlc_v4::sample(&p, 4, 42).unwrap();
        std::fs::write(
            root.join("samples.json"),
            serde_json::to_vec(&samples).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn candidate_scope_mismatch_cannot_compare_unrelated_losses() {
    let a = rd_pack::Candidate {
        packet: packet(),
        fit_tune_loss: 1.0,
    };
    let mut other = packet();
    other.identity.fit_scope_sha256 = "5".repeat(64);
    let b = rd_pack::Candidate {
        packet: other,
        fit_tune_loss: 0.01,
    };
    assert_eq!(
        rd_pack::select(&[a, b], Level::L2k, 0.0).unwrap_err().code,
        "candidate_scope_mismatch"
    );
    let registry: serde_json::Value = serde_json::from_str(basis::REGISTRY_JSON).unwrap();
    assert_eq!(
        registry["blocks"].as_array().unwrap().len(),
        basis::REGISTRY.len()
    );
}

fn framed_with_whitespace(packet: &Packet, total_len: usize) -> Vec<u8> {
    let canonical = wire::encode(packet).unwrap();
    assert!(total_len >= canonical.len());
    let mut body = canonical[wire::HEADER_BYTES..].to_vec();
    body.resize(body.len() + total_len - canonical.len(), b' ');
    let mut framed = canonical[..5].to_vec();
    framed.extend_from_slice(&(body.len() as u32).to_le_bytes());
    framed.extend_from_slice(&wire::crc32(&body).to_le_bytes());
    framed.extend_from_slice(&body);
    framed
}

fn larger_l64k_packet() -> Packet {
    let mut p = packet();
    p.level = Level::L64k;
    p.schema.columns = 80;
    p.schema.groups = (0..80)
        .map(|i| GroupSpec {
            id: i,
            kind: GroupKind::Numeric,
            start: i,
            width: 1,
            has_missing: false,
        })
        .collect();
    p.schema_sha256 = p.schema.sha256().unwrap();
    p.groups = (0..80)
        .map(|_| GroupModel::Numeric {
            knots: (0..65).map(|i| i as f64 / 64.0).collect(),
            missing_probability: 0.0,
            imputation: 0.5,
        })
        .collect();
    p.target = TargetModel::Regression {
        intercept: 0.0,
        coefficients: vec![0.0; basis::width(80).unwrap()],
        noise_scale: 0.0,
    };
    p
}

#[test]
fn every_level_encodes_and_roundtrips_with_complete_envelope_accounting() {
    for (level, cap) in [
        (Level::L2k, 2048usize),
        (Level::L16k, 16384),
        (Level::L64k, 65536),
    ] {
        let mut p = packet();
        p.level = level;
        let bytes = wire::encode(&p).unwrap();
        let declared = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), wire::HEADER_BYTES + declared);
        assert!(bytes.len() <= cap);
        assert_eq!(wire::decode(&bytes).unwrap(), p);
        assert_eq!(wire::encode(&wire::decode(&bytes).unwrap()).unwrap(), bytes);
    }
}

#[test]
fn exact_complete_cap_and_one_byte_over_refuse_are_contract_fixed() {
    for (level, cap) in [
        (Level::L2k, 2048usize),
        (Level::L16k, 16384),
        (Level::L64k, 65536),
    ] {
        let mut p = packet();
        p.level = level;
        let exact = framed_with_whitespace(&p, cap);
        assert_eq!(exact.len(), cap);
        assert_eq!(wire::decode(&exact).unwrap(), p);
        if level != Level::L64k {
            let over = framed_with_whitespace(&p, cap + 1);
            assert_eq!(over.len(), cap + 1);
            assert_eq!(wire::decode(&over).unwrap_err().code, "over_budget");
        } else {
            let over = framed_with_whitespace(&p, cap + 1);
            assert_eq!(over.len(), cap + 1);
            assert_eq!(wire::decode(&over).unwrap_err().code, "invalid_envelope");
        }
    }
}

#[test]
fn l64k_admits_a_materially_larger_valid_graph() {
    let p = larger_l64k_packet();
    let bytes = wire::encode(&p).unwrap();
    assert!(bytes.len() > Level::L16k.cap());
    assert!(bytes.len() <= Level::L64k.cap());
    assert_eq!(wire::decode(&bytes).unwrap(), p);
}

#[test]
fn framing_preflight_and_json_canary_refuse_without_disclosure() {
    let mut header = b"DML4\x01".to_vec();
    header.extend_from_slice(&u32::MAX.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(header.len(), wire::HEADER_BYTES);
    assert_eq!(wire::decode(&header).unwrap_err().code, "length_mismatch");

    let canonical = wire::encode(&packet()).unwrap();
    let mut body: serde_json::Value =
        serde_json::from_slice(&canonical[wire::HEADER_BYTES..]).unwrap();
    body["CANARY_PRIVATE_KEY"] = serde_json::json!("CANARY_PRIVATE_EXTREME_999999999");
    let body = serde_json::to_vec(&body).unwrap();
    assert!(body.len() + wire::HEADER_BYTES <= Level::L2k.cap());
    let mut encoded = canonical[..5].to_vec();
    encoded.extend_from_slice(&(body.len() as u32).to_le_bytes());
    encoded.extend_from_slice(&wire::crc32(&body).to_le_bytes());
    encoded.extend_from_slice(&body);
    let error = wire::decode(&encoded).unwrap_err();
    assert_eq!(error.code, "invalid_packet_json");
    let display = error.to_string();
    assert_eq!(display, "invalid_packet_json");
    assert!(!display.contains("CANARY_PRIVATE_KEY"));
    assert!(!display.contains("CANARY_PRIVATE_EXTREME_999999999"));
}

#[test]
fn correctly_framed_semantic_bound_refuses_with_stable_public_code() {
    let canonical = wire::encode(&packet()).unwrap();
    let mut body: serde_json::Value =
        serde_json::from_slice(&canonical[wire::HEADER_BYTES..]).unwrap();
    body["gates"]["linear"] = serde_json::json!(4.0000001);
    let body = serde_json::to_vec(&body).unwrap();
    let mut encoded = canonical[..5].to_vec();
    encoded.extend_from_slice(&(body.len() as u32).to_le_bytes());
    encoded.extend_from_slice(&wire::crc32(&body).to_le_bytes());
    encoded.extend_from_slice(&body);
    let error = wire::decode(&encoded).unwrap_err();
    assert_eq!(error.code, "invalid_gate");
    assert_eq!(error.to_string(), "invalid_gate");
}

#[test]
fn scalar_shape_preflights_refuse_without_large_allocations() {
    assert!(basis::width(256).is_ok());
    assert!(basis::width(257).is_err());
}
