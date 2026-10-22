// Private, bounded transport for the existing v1 evidence verifier.
// Policy pins are an executor input, never inferred from candidate receipts.
mod pilot_receipt_commands {
    use dope_kernel::evidence::{
        EvaluationIdentity, UnverifiedReceipt, VerificationPolicy, canonical_json, parse_json,
        verify_receipt,
    };
    use dope_kernel::{DopeError, Result};
    use rustix::fs::{Mode, OFlags, open, openat};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::fs::File;
    use std::io::Read;
    use std::os::unix::fs::MetadataExt;
    use std::path::{Component, Path};

    fn refused() -> DopeError {
        DopeError::Data("pilot_native_receipt_refused".into())
    }
    fn check(ok: bool) -> Result<()> {
        if ok { Ok(()) } else { Err(refused()) }
    }
    fn hash(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn string(v: &Value) -> Result<&str> {
        v.as_str().ok_or_else(refused)
    }
    fn fields(v: &Value, names: &[&str]) -> Result<()> {
        check(
            v.as_object().is_some_and(|m| {
                m.len() == names.len() && names.iter().all(|k| m.contains_key(*k))
            }),
        )
    }
    fn read_private(root: &Path, name: &str, limit: u64) -> Result<Vec<u8>> {
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        check(root.is_absolute())?;
        let mut dir = open("/", flags, Mode::empty()).map_err(|_| refused())?;
        for component in root.components() {
            match component {
                Component::RootDir => (),
                Component::Normal(part) => {
                    dir = openat(&dir, part, flags, Mode::empty()).map_err(|_| refused())?;
                }
                _ => return Err(refused()),
            }
        }
        let directory_meta = File::from(dir.try_clone().map_err(|_| refused())?)
            .metadata()
            .map_err(|_| refused())?;
        check(directory_meta.mode() & 0o077 == 0)?;
        check(
            !name.contains(['\\', '\0']) && name.split('/').all(|p| !matches!(p, "" | "." | "..")),
        )?;
        let mut parts = name.split('/').peekable();
        let fd = loop {
            let part = parts.next().ok_or_else(refused)?;
            if parts.peek().is_none() {
                break openat(
                    &dir,
                    part,
                    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map_err(|_| refused())?;
            }
            dir = openat(&dir, part, flags, Mode::empty()).map_err(|_| refused())?;
        };
        let file = File::from(fd);
        let meta = file.metadata().map_err(|_| refused())?;
        check(
            meta.is_file() && meta.len() <= limit && meta.nlink() == 1 && meta.mode() & 0o077 == 0,
        )?;
        let mut bytes = Vec::new();
        file.take(limit + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| refused())?;
        check(bytes.len() as u64 <= limit)?;
        Ok(bytes)
    }
    pub fn run(
        root: &Path,
        bundle: &str,
        policy: &str,
        expected_bundle: &str,
        expected_policy: &str,
        expected_executable: &str,
    ) -> Result<()> {
        // Collapse all native/schema/I/O diagnostics at this public CLI boundary.
        let result = verify(
            root,
            bundle,
            policy,
            expected_bundle,
            expected_policy,
            expected_executable,
        )
        .map_err(|_| refused())?;
        println!("{}", serde_json::to_string(&result).map_err(|_| refused())?);
        Ok(())
    }
    fn verify(
        root: &Path,
        bundle: &str,
        policy: &str,
        expected_bundle: &str,
        expected_policy: &str,
        expected_executable: &str,
    ) -> Result<Value> {
        let executable = std::env::current_exe().map_err(|_| refused())?;
        let executable_bytes = std::fs::read(executable).map_err(|_| refused())?;
        check(hash(&executable_bytes) == expected_executable)?;
        let raw = read_private(root, bundle, 16 << 20)?;
        let trusted = read_private(root, policy, 16 << 20)?;
        check(hash(&raw) == expected_bundle && hash(&trusted) == expected_policy)?;
        let bundle = parse_json(&raw)?;
        let policy = parse_json(&trusted)?;
        fields(&bundle, &["format", "receipts", "bindings"])?;
        fields(&policy, &["format", "bundle_sha256", "policies"])?;
        check(
            bundle["format"] == "dope-pilot-native-receipt-bundle-v1"
                && policy["format"] == "dope-pilot-native-receipt-policy-v1"
                && policy["bundle_sha256"] == expected_bundle,
        )?;
        let receipts = bundle["receipts"].as_array().ok_or_else(refused)?;
        let policies = policy["policies"].as_array().ok_or_else(refused)?;
        check(!receipts.is_empty() && receipts.len() <= 20 && policies.len() == receipts.len())?;
        fields(
            &bundle["bindings"],
            &["artifact", "learners", "drivers", "physical", "cell"],
        )?;
        let mut verified = Vec::new();
        let mut identities = std::collections::BTreeSet::new();
        for (receipt, policy) in receipts.iter().zip(policies) {
            fields(
                policy,
                &["identity", "executor_hash", "receipt_hash", "freeze_hash"],
            )?;
            check(
                receipt["identity"]["partition_role"] == "exploratory"
                    && receipt["freeze_hash"].is_null()
                    && policy["freeze_hash"].is_null(),
            )?;
            let deps = receipt["dependencies"].as_array().ok_or_else(refused)?;
            check(deps.len() <= 4096)?;
            for pin in bundle["bindings"].as_object().ok_or_else(refused)?.values() {
                check(deps.iter().any(|d| d["sha256"] == *pin))?;
            }
            // Preflight the same no-follow root before the existing verifier reads it.
            // The admitted executor owns this immutable private tree during verification.
            let mut total = 0_u64;
            for dep in deps {
                let bytes = read_private(root, string(&dep["path"])?, 256 << 20)?;
                total = total.checked_add(bytes.len() as u64).ok_or_else(refused)?;
                check(
                    total <= 1 << 30
                        && hash(&bytes) == string(&dep["sha256"])?
                        && dep["bytes"].as_u64() == Some(bytes.len() as u64),
                )?;
            }
            let trust = VerificationPolicy {
                identity: EvaluationIdentity::new(policy["identity"].clone())?,
                executor_hash: string(&policy["executor_hash"])?.into(),
                receipt_hash: string(&policy["receipt_hash"])?.into(),
                freeze_hash: None,
            };
            let result = verify_receipt(
                UnverifiedReceipt::from_bytes(&canonical_json(receipt)?)?,
                root,
                &trust,
            )?;
            let identity_sha256 = hash(&canonical_json(result.identity().value())?);
            check(identities.insert(identity_sha256.clone()))?;
            verified.push(
                json!({"receipt_sha256":result.receipt_hash(), "identity_sha256":identity_sha256}),
            );
        }
        Ok(
            json!({"format":"dope-pilot-native-verified-v1", "status":"verified",
                  "bundle_sha256":expected_bundle, "policy_sha256":expected_policy,
                  "executable_sha256":expected_executable, "receipts":verified,
                  "scientific_admission":false, "release_authorization":"not_authorized"}),
        )
    }
}
