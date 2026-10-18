//! Run `cargo test --locked --lib codec::repro_tests::p02_byte_anatomy_ledger`.
//! Measurements describe historical DPK3.2, not a successor packing repair.
use super::*;
use crate::{compiler::repro_tests as compiler, model::repro_tests as model};
use serde_json::json;

fn measure(name: &str, bytes: usize) -> Value {
    json!({"component": name, "bytes": bytes, "bits": bytes * 8})
}

fn spline(bits: u8) -> Marginal {
    // The legacy spline ladder admits 17 knots. Repeat the final lattice
    // point; the standalone Writer probe below measures exactly 16 fields.
    Marginal::QuantileSpline {
        values: compiler::q4_levels(bits).into_iter().chain([1.0]).collect(),
    }
}

fn fixture(bits: u8) -> Kernel {
    let mut kernel = model::fixture(
        Dependence::ChowLiu {
            root: 0,
            edges: vec![CopulaEdge {
                parent: 0,
                child: 1,
                correlation: 0.4,
            }],
        },
        2,
    );
    kernel.quantization_bits = bits;
    kernel.marginals = vec![spline(bits); 2];
    kernel.program = KernelProgram::Symbolic {
        dependence: Dependence::ChowLiu {
            root: 0,
            edges: vec![CopulaEdge {
                parent: 0,
                child: 1,
                correlation: 0.4,
            }],
        },
        target: Target::Ga2m {
            intercept: 0.5,
            logistic: false,
            main_terms: vec![],
            interactions: vec![InteractionTerm {
                left: 0,
                right: 1,
                left_knots: vec![0.0, 0.5, 1.0],
                right_knots: vec![0.0, 0.5, 1.0],
                values: vec![0.125; 9],
            }],
        },
        noise: Noise::Homoscedastic { sigma: 0.0 },
    };
    kernel
}

fn anatomy(bits: u8) -> Value {
    let kernel = fixture(bits);
    let artifact = encode_kernel(&kernel).expect("anatomy fixture encodes");
    assert_eq!(artifact[5], 2, "historical DPK3.2 fixture");
    let decoded = decode_kernel(&artifact).expect("anatomy fixture decodes");
    assert!(
        encode_kernel(&decoded).expect("reencode") == artifact,
        "canonical wire roundtrip"
    );
    let accounting = account_artifact(&artifact).expect("accounting");
    // Entropy coding pools raw components. These deliberately small sections
    // stay uncompressed, so every raw component has an exact wire attribution.
    assert!(
        accounting.sections.iter().all(|s| !s.compressed),
        "uncompressed fixture admission"
    );
    let (dependence, target, noise) = kernel.symbolic().expect("symbolic fixture");
    let raw_sections = [
        encode_schema(&kernel),
        encode_marginals(&kernel),
        encode_dependence(dependence),
        encode_target(target),
        encode_noise(noise),
    ];
    assert_eq!(accounting.sections.len(), raw_sections.len());
    let mut offset = HEADER_BYTES;
    for (index, (section, raw)) in accounting.sections.iter().zip(&raw_sections).enumerate() {
        assert_eq!(section.tag as usize, index + 1);
        assert_eq!(section.raw_bytes, raw.len());
        offset += section.framing_bytes;
        assert!(
            artifact[offset..offset + raw.len()] == *raw,
            "section matches actual payload"
        );
        offset += raw.len();
    }
    assert_eq!(offset, artifact.len());
    let mut without_interactions = target.clone();
    let Target::Ga2m { interactions, .. } = &mut without_interactions else {
        panic!("GA2M fixture")
    };
    interactions.clear();
    let target_base = encode_target(&without_interactions).len();
    let interaction_bytes = encode_target(target).len() - target_base;
    let mut edge_count = Writer::default();
    edge_count.varint(1);
    let mut edges = Writer::default();
    let Dependence::ChowLiu { edges: tree, .. } = dependence else {
        panic!("tree fixture")
    };
    encode_edges(&mut edges, tree);
    let mut schema_count = Writer::default();
    schema_count.varint(kernel.features as u64);
    let schema_components = vec![
        measure("column_count", schema_count.0.len()),
        measure("kind_opcodes", 2),
        measure("missing_probability_u16", 4),
        measure("impute_u16", 4),
        measure("identity_transform_opcodes", 2),
    ];
    let mut marginal_count = Writer::default();
    marginal_count.varint(kernel.marginals.len() as u64);
    let mut marginal_components = vec![measure("marginal_count", marginal_count.0.len())];
    for (i, marginal) in kernel.marginals.iter().enumerate() {
        let Marginal::QuantileSpline { values } = marginal else {
            panic!("spline fixture")
        };
        let mut count = Writer::default();
        count.varint(values.len() as u64);
        marginal_components.push(measure(
            &format!("spline_{i}_opcode_and_count"),
            1 + count.0.len(),
        ));
        marginal_components.push(measure(
            &format!("spline_{i}_unit_fields"),
            values.len() * 2,
        ));
    }
    let dependence_components = vec![
        measure("opcode_and_root", 2),
        measure("edge_count", edge_count.0.len()),
        measure("edge_endpoint_varints", 2),
        measure(
            "edge_correlation_u16",
            edges.0.len() - edge_count.0.len() - 2,
        ),
    ];
    let target_components = vec![
        measure(
            "opcode_intercept_logistic_main_and_interaction_counts",
            target_base,
        ),
        measure("noise_opcode_and_sigma_u16", encode_noise(noise).len()),
    ];
    let interaction_components = vec![
        measure("endpoints_and_knot_counts", 4),
        measure("knot_unit_fields", 12),
        measure("fixed_value_varints", interaction_bytes - 16),
    ];
    let header_components = vec![
        measure("magic", 4),
        measure("version_decoder_task", 4),
        measure("quantization_and_seed_policies", 2),
        measure("compliance_flags", 2),
        measure("feature_count", 4),
        measure("rows_fitted", 8),
        measure("seed", 8),
        measure("payload_length", 8),
        measure("checksum", 16),
    ];
    let framing_components = accounting
        .sections
        .iter()
        .map(|s| {
            measure(
                &format!("section_{}_tag_and_length", s.tag),
                s.framing_bytes,
            )
        })
        .collect::<Vec<_>>();
    let section = |bytes: usize, components: Vec<Value>| {
        assert_eq!(
            components
                .iter()
                .map(|v| v["bytes"].as_u64().expect("bytes") as usize)
                .sum::<usize>(),
            bytes,
            "component conservation"
        );
        json!({"bytes": bytes, "bits": bytes * 8, "components": components})
    };
    let sections = json!({
        "header": section(HEADER_BYTES, header_components),
        "schema": section(encode_schema(&kernel).len(), schema_components),
        "marginals": section(encode_marginals(&kernel).len(), marginal_components),
        "dependence": section(encode_dependence(dependence).len(), dependence_components),
        "interactions": section(interaction_bytes, interaction_components),
        "target": section(target_base + encode_noise(noise).len(), target_components),
        "framing": section(accounting.sections.iter().map(|s| s.framing_bytes).sum(), framing_components),
    });
    let section_sum = sections
        .as_object()
        .expect("sections")
        .values()
        .map(|v| v["bytes"].as_u64().expect("bytes") as usize)
        .sum::<usize>();
    assert_eq!(section_sum, artifact.len(), "section conservation");
    let mut units = Writer::default();
    let levels = compiler::q4_levels(bits);
    for value in &levels {
        units.unit(*value);
    }
    assert_eq!(units.0.len(), 32, "sixteen unit fields always use 32 bytes");
    let mut reader = Reader::new(&units.0);
    let decode_squared_error = levels
        .iter()
        .map(|v| {
            let decoded = reader.unit().expect("unit decode");
            let error = f64::from(decoded - v);
            assert!(
                error.abs() <= 0.5 / f64::from(u16::MAX) + 1e-7,
                "u16 decode precision"
            );
            error * error
        })
        .sum::<f64>()
        / levels.len() as f64;
    reader.finished().expect("all units consumed");
    json!({
        "profile_bits": bits, "format": "DPK3.2", "artifact_blake3": blake3::hash(&artifact).to_hex().to_string(),
        "sections": sections, "wire_sections": accounting.sections,
        "total_wire_bytes": artifact.len(), "total_wire_bits": artifact.len() * 8,
        "section_sum_bytes": section_sum, "sum_conserved": true,
        "unit_fields": 16, "unit_wire_bytes": units.0.len(), "unit_wire_bits_per_field": 16,
        "theoretical_profile_packed_bytes": (16 * usize::from(bits)).div_ceil(8),
        "theoretical_q4_packed_bytes": 8, "unit_decode_mean_squared_error": decode_squared_error,
        "model_json_bytes": serde_json::to_vec(&kernel).expect("fixture JSON").len(),
        "entropy_table_bytes": 0, "padding_bytes": 0, "codebook_bytes": 0,
        "scope": "interactions are records nested in target tag 4; target includes noise tag 5; framing counted once; no compressed component attribution"
    })
}

#[test]
fn p02_q4_unit_wire_width() {
    for bits in [4, 8, 16] {
        anatomy(bits);
    }
}

#[test]
fn p02_byte_anatomy_ledger() {
    let profiles = [4, 8, 16].into_iter().map(anatomy).collect::<Vec<_>>();
    let mut vine = model::vine_observation();
    vine["fitting"] = compiler::vine_fit_observation();
    let ledger = json!({
        "schema": "dope-f00-byte-anatomy-v1", "fixture_only": true,
        "claims": "historical behavior measured before F01/R03; no certification or release authorization",
        "findings": [
            {"id": "P02-REPRO-Q4", "status": "reproduced_before_repair", "profiles": profiles,
             "interpretation": "q4 describes numeric resolution; Writer::unit uses u16; theoretical packing is not legacy wire size"},
            compiler::copula_observation(), vine, compiler::poet_observation(),
            compiler::common_profile_observation(),
        ],
        "compressed_fixture": compressed_observation(),
        "bundle_scope": "DPK payload only; recipient closure and projection overhead await B00 interface; JSON model bytes are not a bundle projection measurement",
        "privacy": "synthetic fixtures; only counts, hashes and aggregate diagnostics; no source rows, headers or extrema"
    });
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("target/f00-byte-anatomy/byte-anatomy.json");
    fs::create_dir_all(path.parent().expect("ledger directory")).expect("create ledger directory");
    let mut bytes = serde_json::to_vec_pretty(&ledger).expect("serialize ledger");
    bytes.push(b'\n');
    fs::write(path, bytes).expect("write fixture ledger");
}

fn compressed_observation() -> Value {
    let mut kernel = model::fixture(Dependence::Independent, 256);
    kernel.marginals = vec![spline(4); 256];
    let artifact = encode_kernel(&kernel).expect("compressible fixture");
    let accounting = account_artifact(&artifact).expect("compressed accounting");
    assert!(
        accounting.sections.iter().any(|s| s.compressed),
        "exercise entropy framing"
    );
    assert_eq!(
        accounting.header_bytes
            + accounting
                .sections
                .iter()
                .map(|s| s.framing_bytes + s.encoded_bytes)
                .sum::<usize>(),
        artifact.len()
    );
    assert!(
        encode_kernel(&decode_kernel(&artifact).expect("decode compressed"))
            .expect("reencode compressed")
            == artifact
    );
    let mut offset = HEADER_BYTES;
    let sections = accounting
        .sections
        .iter()
        .map(|s| {
            offset += s.framing_bytes;
            let body = &artifact[offset..offset + s.encoded_bytes];
            offset += s.encoded_bytes;
            let components = if s.compressed {
                let mut reader = Reader::new(body);
                assert_eq!(reader.byte().expect("rANS version"), 1);
                let symbols = usize::from(reader.u16().expect("frequency count"));
                reader.take(symbols * 3).expect("frequency table");
                reader.take(4).expect("final rANS state");
                vec![
                    measure("coder_id", 1),
                    measure("frequency_count", 2),
                    measure("frequency_table", symbols * 3),
                    measure("final_state", 4),
                    measure("renormalized_stream", reader.remaining()),
                ]
            } else {
                vec![measure("raw_section", body.len())]
            };
            assert_eq!(
                components
                    .iter()
                    .map(|v| v["bytes"].as_u64().expect("bytes") as usize)
                    .sum::<usize>(),
                s.encoded_bytes,
                "entropy component conservation"
            );
            json!({"tag": s.tag, "raw_bytes": s.raw_bytes, "encoded_bytes": s.encoded_bytes,
            "encoded_bits": s.encoded_bytes * 8, "compressed": s.compressed,
            "framing_bytes": s.framing_bytes, "framing_bits": s.framing_bytes * 8,
            "components": components})
        })
        .collect::<Vec<_>>();
    assert_eq!(offset, artifact.len());
    json!({"width": 256, "header_bytes": HEADER_BYTES, "header_bits": HEADER_BYTES * 8,
        "sections": sections, "total_wire_bytes": artifact.len(), "total_wire_bits": artifact.len() * 8,
        "sum_conserved": true, "padding_bytes": 0,
        "scope": "pooled compressed section components; frequency table is the entropy codebook, counted once; no original-field compressed attribution"})
}

#[test]
fn p02_compressed_section_conservation() {
    compressed_observation();
}
