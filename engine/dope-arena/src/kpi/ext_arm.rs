//! Arm file `dope.kpi-ext-arms/v1`: external generators run by `dope_kpi emit-cli`
//! (`docs/plans/v2-gap-closure/ext-arms.md`).
//!
//! An arm names an executable and two argument-vector templates (fit and
//! sample). The whole file is checked when it is loaded, so a mistake in any
//! arm is reported before any command runs. The executable of the selected
//! arm is hashed once; that digest identifies the generator in every row.

use super::ext_template::{Phase, Template, Values};
use super::jsonl::{file_error, file_sha256, read_file};
use dope_arena::{ArenaError, ArenaResult};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

pub const ARM_SCHEMA: &str = "dope.kpi-ext-arms/v1";
/// `arch_id` of every row emitted through an external arm.
pub const ARCH_ID: &str = "ext-cli/v1";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactClass {
    SourceFree,
    RowsAtSampler,
    Control,
}

impl ArtifactClass {
    pub fn name(self) -> &'static str {
        match self {
            ArtifactClass::SourceFree => "source_free",
            ArtifactClass::RowsAtSampler => "rows_at_sampler",
            ArtifactClass::Control => "control",
        }
    }
}

/// File format of `{fit_rows}` and `{out}`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArmIo {
    Csv,
    Mat,
}

impl ArmIo {
    /// The name in the arm file, also used as the scratch file extension.
    pub fn name(self) -> &'static str {
        match self {
            ArmIo::Csv => "csv",
            ArmIo::Mat => "mat",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmFile {
    schema: String,
    arms: Vec<ArmSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmSpec {
    arm_id: String,
    artifact_class: ArtifactClass,
    io: ArmIo,
    binary: String,
    #[serde(default)]
    binary_sha256: Option<String>,
    fit: Vec<String>,
    sample: Vec<String>,
    #[serde(default)]
    budget_bytes: Option<u64>,
    /// Free text for the reader of the file; never used.
    #[serde(default, rename = "note")]
    _note: Option<String>,
}

/// One selected arm, with its executable hashed and its templates parsed.
#[derive(Clone, Debug)]
pub struct Arm {
    pub arm_id: String,
    pub artifact_class: ArtifactClass,
    pub io: ArmIo,
    /// Absolute path of the executable (`{bin}`).
    pub binary: String,
    /// SHA-256 of the executable (`weights_sha256` of every row).
    pub weights_sha256: String,
    /// Length of the executable (`cold_bytes` of every row).
    pub cold_bytes: u64,
    pub budget_bytes: Option<u64>,
    fit: Template,
    sample: Template,
}

fn contract(message: String) -> ArenaError {
    ArenaError::Contract(message)
}

/// An arm id becomes a directory name: letters, digits, `.`, `_` and `-`,
/// starting with a letter or digit, at most 80 bytes.
pub fn valid_arm_id(id: &str) -> bool {
    let mut bytes = id.bytes();
    bytes.next().is_some_and(|b| b.is_ascii_alphanumeric())
        && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        && id.len() <= 80
}

fn templates(spec: &ArmSpec) -> ArenaResult<(Template, Template)> {
    let parse = |args: &[String], phase| {
        Template::parse(args, phase, spec.budget_bytes.is_some())
            .map_err(|message| contract(format!("arm {}: {message}", spec.arm_id)))
    };
    Ok((
        parse(&spec.fit, Phase::Fit)?,
        parse(&spec.sample, Phase::Sample)?,
    ))
}

/// Parse an arm file and check every arm in it: ids, templates, pins.
fn parse_arm_file(label: &str, bytes: &[u8]) -> ArenaResult<Vec<ArmSpec>> {
    // An arm file holds commands, not data, so the parser's message is safe to show.
    let file: ArmFile =
        serde_json::from_slice(bytes).map_err(|err| contract(format!("{label}: {err}")))?;
    if file.schema != ARM_SCHEMA {
        return Err(contract(format!("{label}: schema must be {ARM_SCHEMA}")));
    }
    let mut seen = BTreeSet::new();
    for spec in &file.arms {
        if !valid_arm_id(&spec.arm_id) {
            return Err(contract(format!("{label}: invalid arm_id")));
        }
        if !seen.insert(spec.arm_id.as_str()) {
            return Err(contract(format!("{label}: duplicate arm {}", spec.arm_id)));
        }
        templates(spec)?;
        let pin_ok = spec.binary_sha256.as_ref().is_none_or(|pin| {
            pin.len() == 64 && pin.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        });
        if spec.binary.is_empty() || !pin_ok {
            return Err(contract(format!("arm {}: invalid binary", spec.arm_id)));
        }
    }
    Ok(file.arms)
}

/// Load `arm_id` from an arm file. A relative `binary` is resolved against the
/// directory of the arm file; the executable is hashed here, once.
pub fn load_arm(path: &Path, arm_id: &str) -> ArenaResult<Arm> {
    let specs = parse_arm_file(&path.display().to_string(), &read_file(path)?)?;
    let Some(spec) = specs.iter().find(|spec| spec.arm_id == arm_id) else {
        return Err(contract(format!(
            "{}: no arm named {arm_id}",
            path.display()
        )));
    };
    let listed = Path::new(&spec.binary);
    let joined = match path.parent() {
        Some(base) if listed.is_relative() => base.join(listed),
        _ => listed.to_path_buf(),
    };
    let binary = std::fs::canonicalize(&joined).map_err(|e| file_error(&joined, &e))?;
    let meta = std::fs::metadata(&binary).map_err(|e| file_error(&binary, &e))?;
    let (true, Some(binary_text)) = (meta.is_file(), binary.to_str()) else {
        return Err(contract(format!(
            "arm {arm_id}: binary must be a regular file with a UTF-8 path"
        )));
    };
    let weights_sha256 = file_sha256(&binary)?;
    if spec
        .binary_sha256
        .as_ref()
        .is_some_and(|pin| *pin != weights_sha256)
    {
        return Err(contract(format!(
            "arm {arm_id}: binary does not match binary_sha256"
        )));
    }
    let (fit, sample) = templates(spec)?;
    Ok(Arm {
        arm_id: spec.arm_id.clone(),
        artifact_class: spec.artifact_class,
        io: spec.io,
        binary: binary_text.to_string(),
        weights_sha256,
        cold_bytes: meta.len(),
        budget_bytes: spec.budget_bytes,
        fit,
        sample,
    })
}

impl Arm {
    /// The argument vector of one command; element 0 is the program.
    pub fn command(&self, phase: Phase, values: &Values) -> Vec<String> {
        match phase {
            Phase::Fit => self.fit.render(values),
            Phase::Sample => self.sample.render(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str =
        include_str!("../../../../docs/plans/v2-gap-closure/ext-arms.example.json");
    const FIT: &str = r#"["{bin}","fit","{fit_rows}","{budget}","{packet}"]"#;
    const SAMPLE: &str = r#"["{bin}","sample","{packet}","{out}","{rows}"]"#;

    fn arm_json(arm_id: &str, extra: &str, fit: &str) -> String {
        format!(
            "{{\"arm_id\":\"{arm_id}\",\"artifact_class\":\"control\",\"io\":\"mat\",\
             \"binary\":\"bin/gen.sh\",\"budget_bytes\":4096,{extra}\"fit\":{fit},\"sample\":{SAMPLE}}}"
        )
    }

    fn file_json(arms: &[String]) -> String {
        format!(
            "{{\"schema\":\"{ARM_SCHEMA}\",\"arms\":[{}]}}",
            arms.join(",")
        )
    }

    fn check(json: &str) -> String {
        match parse_arm_file("arms.json", json.as_bytes()) {
            Ok(_) => String::new(),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn every_arm_of_a_file_is_checked() {
        let good = file_json(&[
            arm_json("a", "", FIT),
            arm_json("b", "\"note\":\"n\",", FIT),
        ]);
        assert_eq!(check(&good), "");
        let expect = |json: String, needle: &str| {
            let err = check(&json);
            assert!(err.contains(needle), "{needle}: {err}");
        };
        // A template mistake in an arm that is not selected is still reported.
        let typo = arm_json("b", "", r#"["{bin}","{pakcet}"]"#);
        expect(
            file_json(&[arm_json("a", "", FIT), typo]),
            "arm b: fit[1]: unknown placeholder {pakcet}",
        );
        expect(
            good.replace("\"budget_bytes\":4096,", ""),
            "arm a: fit[3]: {budget} needs budget_bytes",
        );
        expect(
            file_json(&[arm_json("a", "", FIT), arm_json("a", "", FIT)]),
            "duplicate arm a",
        );
        expect(good.replace("ext-arms/v1", "ext-arms/v2"), "schema must be");
        expect(good.replace("\"control\"", "\"open\""), "unknown variant");
        expect(good.replace("\"mat\"", "\"tsv\""), "unknown variant");
        expect(good.replace("\"note\"", "\"budget\""), "unknown field");
        expect(good.replace("\"a\"", "\"../a\""), "invalid arm_id");
        expect(good.replace("bin/gen.sh", ""), "arm a: invalid binary");
        expect(
            file_json(&[arm_json("a", "\"binary_sha256\":\"ABC\",", FIT)]),
            "arm a: invalid binary",
        );
        expect("{\"schema\":".to_string(), "arms.json: EOF");
        for id in ["a", "dope-1.3.0-small", "A_b.9"] {
            assert!(valid_arm_id(id), "{id}");
        }
        for id in ["", ".hidden", "-x", "a/b", "a b", "é", &"a".repeat(81)] {
            assert!(!valid_arm_id(id), "{id}");
        }
    }

    #[test]
    fn loading_resolves_and_hashes_the_binary() {
        let dir = crate::kpi::unit_test_dir("ext-arm-load");
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::write(dir.join("bin/gen.sh"), b"#!/bin/sh\nexit 0\n").unwrap();
        let sha = file_sha256(&dir.join("bin/gen.sh")).unwrap();
        let file = dir.join("arms.json");
        let pin = format!("\"binary_sha256\":\"{sha}\",");
        let json = file_json(&[arm_json("other", "", FIT), arm_json("gen", &pin, FIT)]);
        std::fs::write(&file, &json).unwrap();

        let arm = load_arm(&file, "gen").unwrap();
        // Relative to the arm file, not to the working directory.
        let binary = std::fs::canonicalize(dir.join("bin/gen.sh")).unwrap();
        assert_eq!(arm.binary, binary.to_str().unwrap());
        assert_eq!(arm.weights_sha256, sha);
        assert_eq!(arm.cold_bytes, 17);
        assert_eq!(arm.budget_bytes, Some(4096));
        assert_eq!(
            (arm.io, arm.artifact_class),
            (ArmIo::Mat, ArtifactClass::Control)
        );
        let values = Values {
            bin: &arm.binary,
            fit_rows: "/w/fit.mat",
            packet: "/p/x.packet",
            card: "/p/x.card",
            out: "/w/out.mat",
            rows: 64,
            seed: 7,
            fit_seed: 9,
            task: "binary",
            classes: 2,
            budget: arm.budget_bytes,
        };
        assert_eq!(
            arm.command(Phase::Fit, &values),
            [
                arm.binary.as_str(),
                "fit",
                "/w/fit.mat",
                "4096",
                "/p/x.packet"
            ]
        );
        assert_eq!(
            arm.command(Phase::Sample, &values),
            [
                arm.binary.as_str(),
                "sample",
                "/p/x.packet",
                "/w/out.mat",
                "64"
            ]
        );
        let absolute = json.replace("bin/gen.sh", binary.to_str().unwrap());
        std::fs::write(&file, &absolute).unwrap();
        assert_eq!(load_arm(&file, "other").unwrap().binary, arm.binary);

        let err = |arm_id: &str, json: &str| {
            std::fs::write(&file, json).unwrap();
            load_arm(&file, arm_id).unwrap_err().to_string()
        };
        assert!(err("absent", &json).contains("no arm named absent"));
        let pinned = json.replace(&sha, &"0".repeat(64));
        assert!(err("gen", &pinned).contains("arm gen: binary does not match binary_sha256"));
        let missing = json.replace("bin/gen.sh", "bin/absent.sh");
        let missing = err("gen", &missing);
        assert!(missing.contains("absent.sh") && missing.contains("not found"));
        let directory = json.replace("bin/gen.sh", "bin");
        assert!(err("gen", &directory).contains("regular file"));
        std::fs::remove_file(&file).unwrap();
        assert!(load_arm(&file, "gen").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_documented_example_file_is_valid() {
        let specs = parse_arm_file("ext-arms.example.json", EXAMPLE.as_bytes()).unwrap();
        let ids: Vec<&str> = specs.iter().map(|spec| spec.arm_id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "dope-1.3.0-small",
                "dope-1.3.0-medium",
                "dope-1.3.0-large",
                "packet-2k"
            ]
        );
        let budgets: Vec<Option<u64>> = specs.iter().map(|spec| spec.budget_bytes).collect();
        assert_eq!(budgets, [Some(2048), Some(4096), Some(16384), Some(2048)]);
        assert!(specs[..3].iter().all(|spec| spec.io == ArmIo::Csv));
        assert_eq!(specs[3].io, ArmIo::Mat);
        assert!(specs
            .iter()
            .all(|spec| spec.artifact_class == ArtifactClass::SourceFree));
        for (spec, size) in specs.iter().zip(["small", "medium", "large"]) {
            assert_eq!(spec.fit[7..9], ["--size", size]);
            assert_eq!(spec.binary_sha256.as_ref().map(String::len), Some(64));
        }
        assert!(specs[3].fit.contains(&"{budget}".to_string()));
    }
}
