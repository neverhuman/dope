use super::*;
use dope_kernel::inspect::{inspect_bundle, inspect_charge};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_root() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/vnext/recipient-closure")
        .join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&root).unwrap();
    root
}

fn from_hex(value: &str) -> Vec<u8> {
    (0..value.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
        .collect()
}

fn stage_inventory(case: &Value) -> (PathBuf, String) {
    let root = unique_root();
    let manifest = serde_json::to_vec(&case["manifest"]).unwrap();
    fs::write(root.join("manifest.json"), &manifest).unwrap();
    for (name, encoded) in case["files_hex"].as_object().unwrap() {
        fs::write(root.join(name), from_hex(encoded.as_str().unwrap())).unwrap();
    }
    (root, content_fingerprint(&manifest))
}

fn inventory_fixture() -> Value {
    serde_json::from_slice(include_bytes!("../../schemas/bundle-golden-v1.json")).unwrap()
}

#[test]
fn recipient_cold_amortized_goldens_equal_stat_and_inspection() {
    let goldens: Value = serde_json::from_slice(include_bytes!("recipient-goldens.json")).unwrap();
    assert_eq!(
        goldens["legacy_fixture_sha256"],
        content_fingerprint(include_bytes!("../../schemas/bundle-golden-v1.json"))
    );
    for case in inventory_fixture()["cases"].as_array().unwrap() {
        let (root, digest) = stage_inventory(case);
        let limits = BundleLimits::default();
        assert_eq!(
            bundle::manifest_fingerprint(&root, &limits).unwrap(),
            digest
        );
        let verified = verify_bundle(&root, &digest, &limits).unwrap();
        let account = verified.recipient_accounting().unwrap();
        let inspected = inspect_bundle(
            &root,
            &digest,
            &dope_kernel::bundle::BundleLimits::default(),
        )
        .unwrap();
        assert_eq!(inspected, goldens["cases"][case["name"].as_str().unwrap()]);
        assert_eq!(inspect_charge(&root).unwrap(), inspected);
        let physical: u64 = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .sum();
        assert_eq!(account.physical_file_bytes, physical);
        assert_eq!(
            account.component_bytes + account.manifest_framing_bytes,
            physical
        );
        assert_eq!(
            account.cold_closure_bytes + account.nonrecipient_delivery_bytes,
            physical
        );
        assert_eq!(
            account.amortized_k_closure_bytes,
            account.cold_closure_bytes
        );
        assert_eq!(account.shared_source_independent_decoder_bytes, None);
    }
}

#[test]
fn hidden_projection_prior_checkpoint_and_digest_only_declarations_fail_closed() {
    let fixture = inventory_fixture();
    for name in ["projection.json", "private-prior.bin", "checkpoint.bin"] {
        let (root, digest) = stage_inventory(&fixture["cases"][0]);
        fs::write(root.join(name), b"PRIVATE_HEADER_97531,99999").unwrap();
        let error = verify_bundle(&root, &digest, &BundleLimits::default()).unwrap_err();
        assert_eq!(error.code(), "undeclared_component");
        let error = inspect_bundle(
            &root,
            &digest,
            &dope_kernel::bundle::BundleLimits::default(),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            error,
            "invalid or unsupported artifact: undeclared_component"
        );
        assert!(!error.contains(name));
        assert!(!error.contains("PRIVATE_HEADER_97531"));
        assert!(!error.contains(root.to_str().unwrap()));
    }
    for replacement in [None, Some(b"".as_slice()), Some(b"wrong-data".as_slice())] {
        let (root, digest) = stage_inventory(&fixture["cases"][0]);
        let path = root.join("shared.bin");
        if let Some(bytes) = replacement {
            fs::write(path, bytes).unwrap();
        } else {
            fs::remove_file(path).unwrap();
        }
        assert!(verify_bundle(&root, &digest, &BundleLimits::default()).is_err());
    }
}

#[test]
fn private_fitted_state_has_no_unverified_million_task_discount() {
    let mut case = inventory_fixture()["cases"][0].clone();
    case["manifest"]["amortization_count"] = json!(1000000);
    for (old, new) in [
        ("dictionary.bin", "codebook.bin"),
        ("transform.bin", "projection.json"),
        ("shared.bin", "private-prior.bin"),
        ("runtime.bin", "checkpoint.bin"),
    ] {
        let contents = case["files_hex"]
            .as_object_mut()
            .unwrap()
            .remove(old)
            .unwrap();
        case["files_hex"][new] = contents;
        for component in case["manifest"]["components"].as_array_mut().unwrap() {
            if component["path"] == old {
                component["path"] = json!(new);
            }
        }
    }
    let (root, digest) = stage_inventory(&case);
    let verified = verify_bundle(&root, &digest, &BundleLimits::default()).unwrap();
    let r = verified.recipient_accounting().unwrap();
    assert_eq!(
        r.cold_closure_bytes,
        fs::metadata(root.join("manifest.json")).unwrap().len() + 24
    );
    assert_eq!(r.dataset_specific_bytes, r.cold_closure_bytes);
    assert_eq!(r.amortized_k_closure_bytes, r.cold_closure_bytes);
    let inspected = inspect_charge(&root).unwrap();
    assert!(inspected["shared_source_independent_decoder_bytes"].is_null());
    for name in [
        "projection.json",
        "private-prior.bin",
        "checkpoint.bin",
        "codebook.bin",
    ] {
        assert!(!inspected.to_string().contains(name));
    }
}

#[test]
fn native_q4_label_does_not_replace_actual_wire_byte_measurement() {
    for bits in [4, 8, 16] {
        let mut kernel = symbolic_kernel();
        kernel.quantization_bits = bits;
        let bytes = codec::encode_kernel(&kernel).unwrap();
        let root = unique_root();
        let manifest = native_manifest("native-wire", "model.dpk", &bytes, "dpk3");
        fs::write(root.join("model.dpk"), &bytes).unwrap();
        fs::write(root.join("manifest.json"), &manifest).unwrap();
        let digest = content_fingerprint(&manifest);
        let verified = verify_bundle(&root, &digest, &BundleLimits::default()).unwrap();
        let account = verified.recipient_accounting().unwrap();
        assert_eq!(
            account.cold_closure_bytes,
            (bytes.len() + manifest.len()) as u64
        );
        let inspected = inspect_charge(&root).unwrap();
        assert_eq!(inspected["section_bytes"]["sampling_payload"], bytes.len());
        assert_eq!(
            inspected["physical_file_bytes"],
            account.physical_file_bytes
        );
    }
}
