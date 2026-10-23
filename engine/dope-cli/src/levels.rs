use crate::args::LevelArgs;
use dope_core::{
    levels::{digest, fit_levels, Level, LevelOptions},
    schema::Task,
    FitOptions, RawTable,
};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

fn absolute_new(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|_| "output parent must already exist")?;
    let name = path.file_name().ok_or("invalid output path")?;
    Ok(parent.join(name))
}

pub(crate) fn run(a: LevelArgs) -> Result<(), String> {
    let out = absolute_new(&a.out)?;
    if out.symlink_metadata().is_ok() {
        return Err("release directory already exists".into());
    }
    let private = a
        .private_report
        .as_ref()
        .map(|p| absolute_new(p))
        .transpose()?;
    if private
        .as_ref()
        .is_some_and(|p| p.starts_with(&out) || p.symlink_metadata().is_ok())
    {
        return Err("private report needs a new path outside the release directory".into());
    }
    let levels = Level::parse_list(&a.export_levels).map_err(|e| e.to_string())?;
    let through = Level::parse(&a.compute_through).ok_or("invalid compute-through level")?;
    if a.combine && levels.len() < 2 {
        return Err("combined descriptor requires at least two released levels".into());
    }
    let raw = RawTable::read(&a.csv).map_err(|e| e.to_string())?;
    let target = raw.column_index(&a.target).map_err(|e| e.to_string())?;
    let options = LevelOptions {
        compute_through: through,
        export_levels: levels,
        descriptors: a.descriptors,
        fit: FitOptions {
            seed: a.seed,
            task: a.task.as_deref().map(|t| {
                if t == "regression" {
                    Task::Regression
                } else {
                    Task::Classification
                }
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let session = fit_levels(&raw, target, &options).map_err(|e| e.to_string())?;
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut objects = Vec::new();
    let mut blocks = Vec::new();
    for output in &session.outputs {
        let name = format!("{}.dope", output.level.name());
        objects.push(json!({"path":name,"level":output.level,"kind":"generator","bytes":output.bytes.len(),"cap":output.level.cap(),"sha256":digest(&output.bytes),"schema_sha256":digest(&serde_json::to_vec(&output.program.schema).map_err(|_|"cannot encode schema")?)}));
        files.push((name, output.bytes.clone()));
        if let Some(descriptor) = &output.descriptor {
            let name = format!("{}.embedding.json", output.level.name());
            objects.push(json!({"path":name,"level":output.level,"kind":"artifact_descriptor","bytes":descriptor.len(),"sha256":digest(descriptor),"source_sha256":digest(&output.bytes)}));
            let data: Value =
                serde_json::from_slice(descriptor).map_err(|_| "invalid native descriptor")?;
            blocks.push(json!({"level":output.level,"source_sha256":digest(&output.bytes),"values":data["values"],"mask":data["mask"],"panel":data["panel"],"options":data["options"]}));
            files.push((name, descriptor.clone()));
        }
    }
    if a.combine {
        let values: Vec<Value> = blocks
            .iter()
            .flat_map(|b| b["values"].as_array().unwrap().clone())
            .collect();
        let masks: Vec<Value> = blocks
            .iter()
            .flat_map(|b| b["mask"].as_array().unwrap().clone())
            .collect();
        let bytes=serde_json::to_vec(&json!({"format":"authorized-concat-v1","learned":false,"blocks":blocks,"values":values,"mask":masks})).map_err(|_|"cannot encode combined descriptor")?;
        objects.push(json!({"path":"combined.embedding.json","kind":"authorized_concatenation","bytes":bytes.len(),"sha256":digest(&bytes)}));
        files.push(("combined.embedding.json".into(), bytes));
    }
    let payload: usize = files.iter().map(|(_, b)| b.len()).sum();
    let mut inventory = json!({"format":"portable-levels-pilot-v1","learned":false,"external_transfer":"not_measured","objects":objects,"payload_bytes":payload,"total_bytes":0});
    let manifest = loop {
        let bytes = serde_json::to_vec_pretty(&inventory).map_err(|_| "cannot encode inventory")?;
        let total = payload + bytes.len();
        if inventory["total_bytes"].as_u64() == Some(total as u64) {
            break bytes;
        }
        inventory["total_bytes"] = json!(total);
    };
    files.push(("inventory.json".into(), manifest));
    // Exclusive directory ownership, create_new files, inventory LAST. Readers
    // must require/verify the inventory. No existing directory is replaced.
    fs::create_dir(&out).map_err(|_| "cannot exclusively create release directory")?;
    let mut created = Vec::new();
    for (name, bytes) in &files {
        let path = out.join(name);
        let result = (|| -> std::io::Result<()> {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            created.push(path.clone());
            f.write_all(bytes)?;
            f.sync_all()
        })();
        if result.is_err() {
            for path in created.iter().rev() {
                let _ = fs::remove_file(path);
            }
            let _ = fs::remove_dir(&out);
            return Err("release publication failed".into());
        }
    }
    if let Some(path) = private {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|_| "release saved; private report could not be created")?;
        serde_json::to_writer_pretty(&mut file, session.private_report())
            .map_err(|_| "release saved; private report could not be written")?;
        file.sync_all()
            .map_err(|_| "release saved; private report could not be synced")?;
        eprintln!(
            "private selection report saved separately; it is not external transfer certification"
        );
    }
    println!(
        "released {} level(s), {} total bytes",
        session.outputs.len(),
        inventory["total_bytes"]
    );
    Ok(())
}
