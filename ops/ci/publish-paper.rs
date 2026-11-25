//! Publish the rebuilt v3 PDF from the protected main CI run to a PDF-only ref.
use std::env;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const MAIN: &str = "refs/heads/main";
const PUBLICATION: &str = "refs/heads/paper/v3-pdf";
const TRACKED_PDF: &str = "docs/whitepaper-v3/dope-v3.pdf";

fn git(args: &[&str], input: Option<&str>) -> Result<String, String> {
    let mut cmd = Command::new("git");
    cmd.args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if input.is_some() {
        cmd.stdin(Stdio::piped());
    }
    let mut child = cmd.spawn().map_err(|error| error.to_string())?;
    if let Some(data) = input {
        let mut stdin = child.stdin.take().ok_or("git stdin unavailable")?;
        stdin
            .write_all(data.as_bytes())
            .map_err(|error| error.to_string())?;
    }
    let out = child
        .wait_with_output()
        .map_err(|error| error.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args[0],
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8(out.stdout)
        .map_err(|error| error.to_string())?
        .trim()
        .to_owned())
}

fn remote_head(reference: &str) -> Result<Option<String>, String> {
    let line = git(&["ls-remote", "--refs", "origin", reference], None)?;
    if line.is_empty() {
        return Ok(None);
    }
    let (oid, name) = line.split_once('\t').ok_or("invalid remote ref response")?;
    if name != reference || !valid_oid(oid) {
        return Err("invalid remote ref response".into());
    }
    Ok(Some(oid.to_owned()))
}

fn valid_oid(oid: &str) -> bool {
    oid.len() == 40
        && oid
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn publish(pdf: &str) -> Result<(), String> {
    let source = env::var("GITHUB_SHA").map_err(|_| "missing source commit")?;
    if env::var("GITHUB_REPOSITORY").as_deref() != Ok("neverhuman/dope")
        || env::var("GITHUB_REF").as_deref() != Ok(MAIN)
        || !valid_oid(&source)
        || git(&["rev-parse", "HEAD"], None)? != source
    {
        return Err("publication requires the exact main CI checkout".into());
    }
    if remote_head(MAIN)?.as_deref() != Some(&source) {
        println!("A newer source commit superseded {source}");
        return Ok(());
    }
    if !Path::new(pdf).is_file() {
        return Err("missing rebuilt PDF".into());
    }
    let rebuilt = git(&["hash-object", "--", pdf], None)?;
    let expected = git(&["rev-parse", &format!("{source}:{TRACKED_PDF}")], None)?;
    if rebuilt != expected {
        return Err("rebuilt PDF differs from the source commit's verified PDF".into());
    }
    let parent = remote_head(PUBLICATION)?;
    if let Some(parent) = &parent {
        git(&["fetch", "--quiet", "--no-tags", "origin", parent], None)?;
        let old = git(&["ls-tree", parent, "--", "dope-v3.pdf"], None)?;
        if old == format!("100644 blob {rebuilt}\tdope-v3.pdf") {
            println!("The verified PDF is already published at {parent}");
            return Ok(());
        }
    }
    let saved = git(&["hash-object", "-w", "--", pdf], None)?;
    if saved != expected {
        return Err("rebuilt PDF changed during publication".into());
    }
    let entry = format!("100644 blob {saved}\tdope-v3.pdf\n");
    let tree = git(&["mktree"], Some(&entry))?;
    let mut args = vec![
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit-tree",
        &tree,
    ];
    if let Some(parent) = &parent {
        args.extend(["-p", parent]);
    }
    let message = format!("Publish rebuilt DOPE v3 paper\n\nSource commit: {source}\n");
    let commit = git(&args, Some(&message))?;
    git(
        &[
            "-c",
            "credential.helper=!gh auth git-credential",
            "push",
            "origin",
            &format!("{commit}:{PUBLICATION}"),
        ],
        None,
    )?;
    println!("Published the rebuilt PDF at {commit} from {source}");
    Ok(())
}

fn main() -> std::process::ExitCode {
    let mut args = env::args().skip(1);
    let result = match (args.next(), args.next()) {
        (Some(pdf), None) => publish(&pdf),
        _ => Err("usage: publish-paper REBUILT_PDF".into()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("publish-paper: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}
