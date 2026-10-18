//! D20: preserve the optional Torch ABI and exercise the patched archive boundary.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{Cursor, Read, Write};
use std::path::PathBuf;
use std::process::Command;
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn optional_torch_pin_and_registry_graph_reject_old_zip() {
    let output = Command::new("cargo")
        .args([
            "metadata",
            "--locked",
            "--offline",
            "--all-features",
            "--format-version",
            "1",
        ])
        .current_dir(root())
        .output()
        .expect("locked metadata unavailable");
    assert!(output.status.success(), "locked metadata failed");
    let graph: Value = serde_json::from_slice(&output.stdout).expect("metadata JSON invalid");
    let packages = graph["packages"]
        .as_array()
        .expect("metadata packages missing");
    let product = packages
        .iter()
        .find(|p| p["name"] == "dope-kernel")
        .unwrap();
    let dependency = product["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "tch")
        .expect("Torch declaration missing");
    assert_eq!(dependency["req"], "=0.20.0");
    assert_eq!(dependency["optional"], true);
    for name in ["tch", "torch-sys"] {
        let matches: Vec<_> = packages.iter().filter(|p| p["name"] == name).collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["version"], "0.20.0");
        assert!(
            matches[0]["source"].is_null(),
            "Torch patch must be local and reviewed"
        );
    }
    let archives: Vec<_> = packages.iter().filter(|p| p["name"] == "zip").collect();
    assert_eq!(archives.len(), 1);
    assert_eq!(archives[0]["version"], "9.0.5");
    assert!(
        archives[0]["source"]
            .as_str()
            .unwrap()
            .starts_with("registry+")
    );
}

#[test]
fn retained_torch_runtime_sources_match_authenticated_upstream_and_scoped_patch() {
    let vendor = root().join("ops/vendor");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(vendor.join("torch-zip-provenance.json")).unwrap())
            .unwrap();
    for (name, package) in manifest["packages"].as_object().unwrap() {
        assert_eq!(package["version"], "0.20.0");
        for (path, expected) in package["files"].as_object().unwrap() {
            let bytes = std::fs::read(vendor.join(name).join(path)).unwrap();
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected["sha256"]);
            if path == "Cargo.toml" {
                // Manifest patch is reviewed with Cargo's resolved dependency graph above.
                continue;
            }
            let original = if name == "tch" && path == "src/tensor/npy.rs" {
                let text = String::from_utf8(bytes).unwrap();
                assert_eq!(
                    text.matches("zip::write::SimpleFileOptions::default()")
                        .count(),
                    1
                );
                text.replace(
                    "zip::write::SimpleFileOptions::default()",
                    "zip::write::FileOptions::default()",
                )
                .replace(
                    "let mut reader = zip.by_index(i)?;",
                    "let mut reader = zip.by_index(i).unwrap();",
                )
                .replace("let name = reader.name()?;", "let name = reader.name();")
                .replace("unwrap_or(name.as_ref())", "unwrap_or(name)")
                .into_bytes()
            } else if name == "torch-sys" && path == "build.rs" {
                String::from_utf8(bytes)
                    .unwrap()
                    .replace("file.mangled_name()?", "file.mangled_name()")
                    .replace("file.name()?.ends_with", "file.name().ends_with")
                    .into_bytes()
            } else {
                bytes
            };
            if let Some(hash) = expected["upstream_sha256"].as_str() {
                assert_eq!(format!("{:x}", Sha256::digest(original)), hash);
            }
        }
    }
}

#[test]
fn npz_archive_api_keeps_stored_deflate_bzip2_and_zstd_roundtrips() {
    for compression in [
        CompressionMethod::Stored,
        CompressionMethod::Deflated,
        CompressionMethod::Bzip2,
        CompressionMethod::Zstd,
    ] {
        let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().compression_method(compression);
        archive.start_file("tensor.npy", options).unwrap();
        let payload = b"synthetic-npy-compatibility-fixture";
        archive.write_all(payload).unwrap();
        let mut archive = ZipArchive::new(archive.finish().unwrap()).unwrap();
        let mut decoded = Vec::new();
        archive
            .by_name("tensor.npy")
            .unwrap()
            .read_to_end(&mut decoded)
            .unwrap();
        assert_eq!(decoded, payload);
    }
}

#[cfg(unix)]
fn scratch(label: &str) -> PathBuf {
    let base = root().join("target");
    std::fs::create_dir_all(&base).unwrap();
    let path = base.join(format!("zip-security-{}-{label}", std::process::id()));
    std::fs::create_dir(&path).expect("fixture directory must be new");
    path
}

#[cfg(unix)]
#[test]
fn extraction_rejects_external_symlink_without_changing_its_target() {
    use std::os::unix::fs::PermissionsExt;
    let base = scratch("symlink");
    let output = base.join("out");
    std::fs::create_dir(&output).unwrap();
    let external = base.join("external");
    std::fs::write(&external, b"synthetic-external-target").unwrap();
    std::fs::set_permissions(&external, std::fs::Permissions::from_mode(0o600)).unwrap();
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .add_symlink("link", "../external", SimpleFileOptions::default())
        .unwrap();
    let mut reader = ZipArchive::new(writer.finish().unwrap()).unwrap();
    assert!(reader.extract(&output).is_err());
    assert!(std::fs::symlink_metadata(output.join("link")).is_err());
    assert_eq!(
        std::fs::read(&external).unwrap(),
        b"synthetic-external-target"
    );
    assert_eq!(
        std::fs::metadata(&external).unwrap().permissions().mode() & 0o7777,
        0o600
    );
    std::fs::remove_dir_all(base).unwrap();
}

#[cfg(unix)]
#[test]
fn extraction_masks_privileged_permission_bits_from_untrusted_central_directory() {
    use std::os::unix::fs::PermissionsExt;
    let base = scratch("permissions");
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file("fixture", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"synthetic-permission-fixture").unwrap();
    let mut bytes = writer.finish().unwrap().into_inner();
    let central = bytes.windows(4).position(|b| b == b"PK\x01\x02").unwrap();
    bytes[central + 5] = 3; // Unix origin, with attacker-supplied regular-file mode.
    bytes[central + 38..central + 42].copy_from_slice(&(0o106755u32 << 16).to_le_bytes());
    let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
    archive.extract(&base).unwrap();
    assert_eq!(
        std::fs::metadata(base.join("fixture"))
            .unwrap()
            .permissions()
            .mode()
            & 0o7000,
        0
    );
    std::fs::remove_dir_all(base).unwrap();
}
