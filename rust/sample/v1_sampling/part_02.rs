

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompileOptions, compile_kernel_from_arrays};

    #[test]
    fn seeded_sampling_is_exact_and_target_last() {
        let rows = 50;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| [0.2, row as f32 / rows as f32])
            .collect();
        let target: Vec<f32> = (0..rows).map(|row| row as f32 / rows as f32).collect();
        let compiled = compile_kernel_from_arrays(
            &features,
            &target,
            rows,
            2,
            Task::Regression,
            &CompileOptions::default(),
        )
        .unwrap();
        let loaded = LoadedKernel::V2(Box::new(
            crate::codec::decode_kernel(&compiled.artifact).unwrap(),
        ));
        let first = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 100,
                seed: Some(8),
            },
        )
        .unwrap();
        let second = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 100,
                seed: Some(8),
            },
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 300);
        assert!(first.chunks_exact(3).all(|row| row[2].is_finite()));
    }

    #[test]
    fn binary_targets_remain_binary() {
        let rows = 80;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| [row as f32 / rows as f32, (row % 3) as f32 / 2.0])
            .collect();
        let target: Vec<f32> = (0..rows).map(|row| (row > 40) as u8 as f32).collect();
        let compiled = compile_kernel_from_arrays(
            &features,
            &target,
            rows,
            2,
            Task::Binary,
            &CompileOptions::default(),
        )
        .unwrap();
        let loaded = LoadedKernel::V2(Box::new(
            crate::codec::decode_kernel(&compiled.artifact).unwrap(),
        ));
        let sampled = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 200,
                seed: Some(11),
            },
        )
        .unwrap();
        assert!(
            sampled
                .chunks_exact(3)
                .all(|row| row[2] == 0.0 || row[2] == 1.0)
        );
    }

    #[test]
    fn missingness_is_explicit_and_deterministic() {
        let rows = 100;
        let features: Vec<f32> = (0..rows)
            .flat_map(|row| {
                [
                    if row % 4 == 0 {
                        f32::NAN
                    } else {
                        row as f32 / rows as f32
                    },
                    0.3,
                ]
            })
            .collect();
        let target: Vec<f32> = (0..rows).map(|row| row as f32 / rows as f32).collect();
        let compiled = compile_kernel_from_arrays(
            &features,
            &target,
            rows,
            2,
            Task::Regression,
            &CompileOptions::default(),
        )
        .unwrap();
        let loaded = LoadedKernel::V2(Box::new(
            crate::codec::decode_kernel(&compiled.artifact).unwrap(),
        ));
        let first = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 1_000,
                seed: Some(12),
            },
        )
        .unwrap();
        let second = sample_kernel(
            &loaded,
            SampleOptions {
                rows: 1_000,
                seed: Some(12),
            },
        )
        .unwrap();
        assert!(
            first
                .iter()
                .zip(&second)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        let missing = first.chunks_exact(3).filter(|row| row[0].is_nan()).count();
        assert!((180..=320).contains(&missing));
        assert!(
            first
                .chunks_exact(3)
                .all(|row| row[2].is_finite() && (0.0..=1.0).contains(&row[2]))
        );
    }
}
