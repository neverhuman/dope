#[test]
fn canonical_json_matches_python_golden_and_rejects_duplicates() {
    let value = json!({"z":-0.0,"b":0.3,"a":[true,null,"é"]});
    assert_eq!(
        String::from_utf8(canonical_json(&value).unwrap()).unwrap(),
        "{\"a\":[true,null,\"é\"],\"b\":2.99999999999999989e-1,\"z\":-0.00000000000000000e0}"
    );
    assert_eq!(
        digest(&json!({"a":1,"b":2})).unwrap(),
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    );
    for data in [
        br#"{"a":1,"a":2}"#.as_slice(),
        br#"{"x":{"a":1,"a":2}}"#,
        b"NaN",
        b"Infinity",
        b"1e999",
        b"9007199254740992",
        b"184467440737095516160",
        br#""\ud800""#,
    ] {
        assert!(parse_json(data).is_err());
    }
}

#[test]
fn numeric_parser_roundtrips_extremes_and_preserves_json_grammar() {
    let original = json!([
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        -0.0,
        0.3,
        1e-100,
        1e100
    ]);
    let encoded = canonical_json(&original).unwrap();
    assert_eq!(
        canonical_json(&parse_json(&encoded).unwrap()).unwrap(),
        encoded
    );
    assert_eq!(
        canonical_json(&parse_json(br#"{"z":1.0,"a":2,"digit\"1":3.5}"#).unwrap()).unwrap(),
        br#"{"a":2,"digit\"1":3.50000000000000000e0,"z":1.00000000000000000e0}"#
    );
    for input in [
        b"01".as_slice(),
        b"1.",
        b"-.1",
        b"1e",
        b"--1",
        b"[1 2]",
        b"[1,]",
        b"1 true",
    ] {
        assert!(parse_json(input).is_err());
    }
}
