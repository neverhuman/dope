//! Small hostile containers: no fitting, large allocation or OOM fixture.
use dope_core::artifact::{container::repack, decode, encode, unpack};
use dope_core::noise::RESIDUAL_KNOTS;
use dope_core::quant::QTensor;
use dope_core::target::QuantNet;
use dope_core::{sample, SampleOptions};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

fn fixture() -> (Value, BTreeMap<String, Vec<u8>>) {
    let meta = json!({"v":1,"cols":[["x","n",0],["z","n",0]],"y":["y","n",0],
        "task":"regression","k":2,"dep":"independent","regimes":1,
        "s":[0],"net":[1,1,1,2,2,2,2],"t":0});
    let f32s = |v: &[f32]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let f16s = |v: &[f64]| {
        v.iter()
            .flat_map(|v| half::f16::from_f64(*v).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut noise = vec![0.0; 5 + RESIDUAL_KNOTS];
    *noise.last_mut().unwrap() = 1.0;
    let sections = BTreeMap::from([
        ("schema.affine".into(), f32s(&[0., 1., 0., 1., 0., 1.])),
        ("marginal.range".into(), f16s(&[-1., 1., -1., 1.])),
        ("net.codes".into(), vec![2; 4]),
        ("net.params".into(), f32s(&[1., 1., 1., 1., 0.])),
        ("noise".into(), f16s(&noise)),
    ]);
    (meta, sections)
}
fn refuse(meta: &Value, sections: &BTreeMap<String, Vec<u8>>) {
    let bytes = repack(meta, sections).unwrap();
    let result = std::panic::catch_unwind(|| decode(&bytes));
    assert!(
        result.is_ok(),
        "decoder panicked on a small hostile fixture"
    );
    assert!(result.unwrap().is_err());
}

#[test]
fn small_program_encode_decode_and_sampling_are_unchanged() {
    let (m, s) = fixture();
    let bytes = repack(&m, &s).unwrap();
    let p = decode(&bytes).unwrap();
    let q = QuantNet {
        inputs: 1,
        hidden: 1,
        outputs: 1,
        w: QTensor::quantize(&[0.], 2),
        b: QTensor::quantize(&[0.], 2),
        beta: QTensor::quantize(&[0.], 2),
        v: QTensor::quantize(&[0.], 2),
        a: vec![0.],
    };
    let encoded = encode(&p, &q, 2).unwrap();
    assert_eq!(bytes, encoded);
    let back = decode(&encoded).unwrap();
    assert_eq!(p, back);
    let opts = SampleOptions {
        rows: 4,
        seed: 7,
        ..Default::default()
    };
    let a = sample(&p, &opts).unwrap();
    let b = sample(&back, &opts).unwrap();
    assert_eq!(a.x, b.x);
    assert_eq!(a.y, b.y);
}

#[test]
fn all_network_widths_are_checked_before_narrowing_or_unpacking() {
    for index in 3..7 {
        for bits in [0, 1, 9, 255, 256, u64::MAX] {
            let (mut m, s) = fixture();
            m["net"][index] = json!(bits);
            refuse(&m, &s);
        }
    }
    for bits in 2..=8 {
        let (mut m, mut s) = fixture();
        for index in 3..7 {
            m["net"][index] = json!(bits);
        }
        s.insert("net.codes".into(), vec![1 << (bits - 1); 4]);
        assert!(decode(&repack(&m, &s).unwrap()).is_ok());
    }
}

#[test]
fn numeric_metadata_is_strict_and_dimensions_are_bounded() {
    for value in [
        json!(-1),
        json!(1.5),
        json!("1"),
        Value::Null,
        json!(u64::MAX),
    ] {
        for key in ["net", "s", "forest"] {
            let (mut m, s) = fixture();
            if key == "forest" {
                m[key] = json!([1, 1, 2]);
            }
            m[key][0] = value.clone();
            refuse(&m, &s);
        }
    }
    for index in 0..3 {
        let (mut m, s) = fixture();
        m["net"][index] = json!(u64::MAX);
        refuse(&m, &s);
    }
    let (mut m, s) = fixture();
    m["net"][1] = json!(3_000_000); // bounded materialization fails; tiny byte body
    refuse(&m, &s);
    for screen in [json!([2]), json!([0, 0]), json!([0, "bad"])] {
        let (mut m, s) = fixture();
        m["s"] = screen;
        refuse(&m, &s);
    }
    for key in ["disc", "drop", "forest"] {
        let (mut m, s) = fixture();
        m[key] = json!("invalid");
        refuse(&m, &s);
    }
    let (mut m, s) = fixture();
    m["disc"] = json!({"0":0});
    refuse(&m, &s);
    let (mut m, s) = fixture();
    m["t"] = json!(u64::MAX);
    refuse(&m, &s);
    let (mut m, s) = fixture();
    m["k"] = json!(u64::MAX);
    refuse(&m, &s);
}

#[test]
fn copula_lengths_ranks_parents_and_cycles_are_rejected() {
    for dep in [
        "full",
        "factor0",
        "factor3",
        "factor18446744073709551615",
        "unknown",
    ] {
        let (mut m, mut s) = fixture();
        m["dep"] = json!(dep);
        s.insert("copula.corr".into(), vec![]);
        s.insert("copula.loadings".into(), vec![]);
        refuse(&m, &s);
    }
    for parents in [vec![1u16, 0], vec![2, 1], vec![3, 0], vec![0]] {
        let (mut m, mut s) = fixture();
        m["dep"] = json!("tree");
        s.insert(
            "copula.parent".into(),
            parents.iter().flat_map(|p| p.to_le_bytes()).collect(),
        );
        s.insert("copula.rho".into(), vec![0; 2]);
        refuse(&m, &s);
    }
    for dep in ["full", "factor1", "tree"] {
        let (mut m, mut s) = fixture();
        m["dep"] = json!(dep);
        s.insert("copula.corr".into(), vec![0]);
        s.insert("copula.loadings".into(), vec![0; 2]);
        s.insert("copula.parent".into(), vec![0, 0, 1, 0]);
        s.insert("copula.rho".into(), vec![0; 2]);
        assert!(decode(&repack(&m, &s).unwrap()).is_ok());
    }
    let (mut m, mut s) = fixture();
    m["cols"] = json!(vec![json!(["x", "n", 0]); 4097]);
    m["dep"] = json!("full");
    s.insert("copula.corr".into(), vec![]);
    refuse(&m, &s); // feature limit before affine expansion or square allocation
}

#[test]
fn section_lengths_and_forest_cost_are_checked_before_use() {
    for name in [
        "schema.affine",
        "marginal.range",
        "net.params",
        "net.codes",
        "noise",
    ] {
        let (m, mut s) = fixture();
        s.get_mut(name).unwrap().push(0);
        refuse(&m, &s);
    }
    for dims in [
        json!([1, 1, 0]),
        json!([1, 9, 2]),
        json!([u64::MAX, 8, 8]),
        json!([1, 1, 256]),
        json!([1, 1, 2]),
    ] {
        let (mut m, mut s) = fixture();
        m["forest"] = dims;
        s.insert("forest".into(), vec![]);
        refuse(&m, &s);
    }
    let (mut m, mut s) = fixture();
    m["regimes"] = json!(1);
    s.insert("regime.weights".into(), vec![0]);
    refuse(&m, &s);
}

#[test]
fn container_requires_exactly_one_u8_vector_and_accounts_all_bytes() {
    let (m, s) = fixture();
    let bytes = repack(&m, &s).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let data = st.tensor("dope").unwrap().data();
    let info = || Some(HashMap::from([("dope".into(), "1".into())]));
    let extra = safetensors::serialize(
        [
            (
                "dope",
                TensorView::new(Dtype::U8, vec![data.len()], data).unwrap(),
            ),
            ("extra", TensorView::new(Dtype::U8, vec![1], &[0]).unwrap()),
        ],
        info(),
    )
    .unwrap();
    assert!(unpack(&extra).is_err());
    let wrong = safetensors::serialize(
        [(
            "dope",
            TensorView::new(Dtype::I8, vec![data.len()], data).unwrap(),
        )],
        info(),
    )
    .unwrap();
    assert!(unpack(&wrong).is_err());
    let matrix = safetensors::serialize(
        [(
            "dope",
            TensorView::new(Dtype::U8, vec![1, data.len()], data).unwrap(),
        )],
        info(),
    )
    .unwrap();
    assert!(unpack(&matrix).is_err());
    let good = unpack(&bytes).unwrap();
    assert_eq!(good.header + good.stored, bytes.len());
    let mut many = BTreeMap::new();
    for i in 0..65 {
        many.insert(format!("section{i}"), vec![]);
    }
    assert!(unpack(&repack(&json!({}), &many).unwrap()).is_err());
}
