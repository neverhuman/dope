//! Regression tests for DSE7 binary custom-label encoding (`{-1, 1}`, `{1, 2}`).

use dope_core::rng::Rng;
use dope_core::single_encoding::{fit_dse7_packet, unpack_dse7_packet};

#[test]
fn dse7_binary_preserves_non_canonical_labels_minus1_plus1_and_1_2() {
    let n = 120;
    let mut rng = Rng::new(0xB1A5_ED12);
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    for _ in 0..n {
        x0.push(rng.normal());
        x1.push(rng.normal());
    }
    for (lo, hi) in [(-1.0f64, 1.0f64), (1.0f64, 2.0f64)] {
        let y: Vec<f64> = (0..n)
            .map(|i| if x0[i] + 0.3 * x1[i] > 0.0 { hi } else { lo })
            .collect();
        let pkt_bytes = fit_dse7_packet(&[x0.clone(), x1.clone()], &y, 1, 2048)
            .expect("binary DSE7 fit must succeed for custom class labels");
        assert!(pkt_bytes.len() <= 2048);
        let (pkt, hdr, card) = unpack_dse7_packet(&pkt_bytes).expect("unpack binary DSE7");
        assert_eq!(hdr.task_id, 1);
        assert_eq!(hdr.k_eff, 2);
        assert_eq!(card.classes, vec![lo as i32, hi as i32]);
        assert!(
            pkt.w_base[0][0] > 0.2,
            "expected positive slope on x0 for ({lo}, {hi}), got {}",
            pkt.w_base[0][0]
        );
    }
}
