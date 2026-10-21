use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn collect(root: &Path, directory: &Path, inputs: &mut BTreeMap<String, String>) -> io::Result<()> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(root, &path, inputs)?;
        } else if path.is_file() {
            add(root, &path, inputs)?;
        }
    }
    Ok(())
}
fn add(root: &Path, path: &Path, inputs: &mut BTreeMap<String, String>) -> io::Result<()> {
    inputs.insert(
        path.strip_prefix(root)
            .map_err(io::Error::other)?
            .to_string_lossy()
            .into_owned(),
        format!("{:x}", Sha256::digest(fs::read(path)?)),
    );
    Ok(())
}
pub(super) fn inputs(root: &Path) -> io::Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "rust-toolchain.toml",
    ] {
        add(root, &root.join(name), &mut result)?;
    }
    for name in [
        "rust",
        ".cargo",
        "schemas",
        "contracts",
        "production",
        "tests/fixtures",
        "tests/golden",
    ] {
        collect(root, &root.join(name), &mut result)?;
    }
    let tooling = root.join("ops/merge_guard");
    if tooling.exists() {
        for entry in fs::read_dir(&tooling)? {
            let path = entry?.path();
            if path.is_file()
                && matches!(
                    path.extension().and_then(|x| x.to_str()),
                    Some("rs" | "toml" | "lock")
                )
            {
                add(root, &path, &mut result)?;
            }
        }
        collect(root, &tooling.join("audit_proof"), &mut result)?;
    }
    Ok(result)
}
pub(super) fn selected(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .filter(|name| {
            !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == ':')
        })
        .map(str::to_owned)
        .collect()
}
pub(super) fn executed(text: &str) -> bool {
    let mut passed = 0_u64;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("test result: ok. ") {
            let fields: Vec<_> = rest.split(';').collect();
            if fields.len() < 2 {
                continue;
            }
            let p = fields[0]
                .trim()
                .strip_suffix(" passed")
                .and_then(|x| x.parse::<u64>().ok());
            let f = fields[1]
                .trim()
                .strip_suffix(" failed")
                .and_then(|x| x.parse::<u64>().ok());
            if let (Some(p), Some(f)) = (p, f) {
                if f != 0 {
                    return false;
                }
                passed = passed.saturating_add(p);
            }
        }
    }
    passed > 0
}
pub(super) fn run_with<F>(root: &Path, filter: &str, mut execute: F) -> io::Result<(i32, PathBuf)>
where
    F: FnMut(&[String], &Path) -> io::Result<i32>,
{
    if filter.is_empty() || filter.starts_with('-') || filter.chars().any(char::is_control) {
        return Err(io::Error::other(
            "one nonempty Rust test filter is required",
        ));
    }
    let output = root.join("target/focus");
    fs::create_dir_all(&output)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_nanos();
    let destination = output.join(format!("run-{stamp}-{}", std::process::id()));
    fs::create_dir(&destination)?;
    let before = inputs(root)?;
    let started = Instant::now();
    let mut receipt = json!({"schema_version":1,"selection":filter,"source_inputs":before,
        "commands":[],"status":"failed","full_gate":false,
        "cache_policy":"reuse existing Cargo target and inherited RUSTC_WRAPPER",
        "rustc_wrapper":std::env::var("RUSTC_WRAPPER").ok()});
    let mut code = 1;
    for phase in ["list", "test"] {
        let mut argv: Vec<String> = [
            "test",
            "-p",
            "dope-kernel",
            "--locked",
            "--lib",
            filter,
            "--",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        if phase == "list" {
            argv.push("--list".into());
        }
        let log = destination.join(format!("{phase}.log"));
        let result = match execute(&argv, &log) {
            Ok(result) => result,
            Err(_) => {
                receipt["reason"] = json!("cargo invocation failed");
                break;
            }
        };
        let raw = match fs::read(&log) {
            Ok(raw) => raw,
            Err(_) => {
                receipt["reason"] = json!("cargo log unavailable");
                break;
            }
        };
        receipt["commands"].as_array_mut().expect("commands initialized as array").push(json!({
            "argv":std::iter::once("cargo".to_owned()).chain(argv).collect::<Vec<_>>(),
            "exit_code":result,"log":format!("{phase}.log"),"sha256":format!("{:x}",Sha256::digest(&raw))}));
        if result != 0 {
            code = result;
            receipt["reason"] = json!("cargo command failed");
            break;
        }
        let text = String::from_utf8_lossy(&raw);
        if phase == "list" {
            let names = selected(&text);
            let empty = names.is_empty();
            receipt["selected_tests"] = json!(names);
            if empty {
                receipt["reason"] = json!("filter selected no tests");
                break;
            }
        } else if !executed(&text) {
            receipt["reason"] = json!("no executed passing test in result");
            break;
        } else {
            code = 0;
        }
    }
    if inputs(root).ok().as_ref() != Some(&before) {
        code = 1;
        receipt["reason"] = json!("source inputs changed or became unreadable");
    }
    if code == 0 {
        receipt["status"] = Value::String("passed".into());
    }
    receipt["exit_code"] = json!(code);
    receipt["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    fs::write(
        destination.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok((code, destination))
}
pub(super) fn run(filter: &str) -> io::Result<i32> {
    let root = std::env::current_dir()?;
    let (code, destination) = run_with(&root, filter, |argv, log| {
        let file = fs::File::create(log)?;
        Ok(Command::new("cargo")
            .args(argv)
            .current_dir(&root)
            .stdout(Stdio::from(file.try_clone()?))
            .stderr(Stdio::from(file))
            .status()?
            .code()
            .unwrap_or(1))
    })?;
    println!(
        "Focused proof: {}",
        destination.join("receipt.json").display()
    );
    Ok(code)
}
