use std::fs;
use std::process::Command;

#[test]
fn demo_producer_schema_and_integer_csv_are_admitted_by_native_reader() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = root
        .join("target/public-demo-native-fixture")
        .join(std::process::id().to_string());
    fs::create_dir_all(&output).unwrap();
    let code = r#"
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('public_demo',sys.argv[1])
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
out=Path(sys.argv[2])
rows=[[float(f),0.0,0.1,20.0,0.01,float(f+100)] for f in [100,200,300]]
(out/'schema.json').write_text(json.dumps(demo.schema()))
demo.write_fit_csv(out/'fit.csv',rows,[0,1,2])
"#;
    let result = Command::new("python3")
        .args(["-I", "-B", "-c", code])
        .arg(root.join("examples/public-regression/run.py"))
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "public demo fixture producer failed"
    );
    let schema = dope_kernel::schema::TableSchemaV1::from_private_json(
        &fs::read(output.join("schema.json")).unwrap(),
    )
    .unwrap();
    let table = dope_kernel::typed_data::TypedTable::read_csv(
        fs::File::open(output.join("fit.csv")).unwrap(),
        schema.clone(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(table.rows(), 3);
    let old_csv = fs::read_to_string(output.join("fit.csv"))
        .unwrap()
        .replacen("100,", "100.0,", 1);
    let error = dope_kernel::typed_data::TypedTable::read_csv(
        old_csv.as_bytes(),
        schema,
        Default::default(),
    )
    .unwrap_err();
    assert_eq!(error.code, "integer_precision_or_type_mismatch");
    assert_eq!((error.row, error.column), (Some(2), Some(1)));
}
