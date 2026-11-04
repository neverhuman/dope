//! Public synthetic producer fixtures; no model/source/card at recipient.
use dope_core::mlc_v4::{decode, sample_with_draws, Packet};
use serde_json::Value;

#[test]
fn producer_packets_are_accepted_and_match_fixed_draws_without_encoder() {
    let fixtures = [
        (
            include_bytes!("fixtures/v4_producer/regression.dml4").as_slice(),
            include_str!("fixtures/v4_producer/regression.draws.json"),
            include_str!("fixtures/v4_producer/regression.samples.json"),
        ),
        (
            include_bytes!("fixtures/v4_producer/multiclass.dml4").as_slice(),
            include_str!("fixtures/v4_producer/multiclass.draws.json"),
            include_str!("fixtures/v4_producer/multiclass.samples.json"),
        ),
    ];
    for (wire, draws, expected) in fixtures {
        let packet: Packet = decode(wire).unwrap();
        let draws: Vec<Vec<f64>> = serde_json::from_str(draws).unwrap();
        let expected: Value = serde_json::from_str(expected).unwrap();
        let sampled = sample_with_draws(&packet, &draws).unwrap();
        assert_eq!(sampled.x.len(), expected["x"].as_array().unwrap().len());
        for (actual, expected) in sampled.x.iter().zip(expected["x"].as_array().unwrap()) {
            assert_eq!(actual.len(), expected.as_array().unwrap().len());
            for (a, b) in actual.iter().zip(expected.as_array().unwrap()) {
                assert!((a - b.as_f64().unwrap()).abs() <= 1e-11);
            }
        }
        assert_eq!(sampled.y.len(), expected["y"].as_array().unwrap().len());
        for (a, b) in sampled.y.iter().zip(expected["y"].as_array().unwrap()) {
            assert!((a - b.as_f64().unwrap()).abs() <= 1e-10);
        }
        packet.validate().unwrap();
    }
}
