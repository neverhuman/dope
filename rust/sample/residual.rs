fn residual_v1(distribution: &ResidualV1, uniform: f64, normal: f64) -> f32 {
    match distribution {
        ResidualV1::NoNoise => 0.0,
        ResidualV1::StudentT4 { location, scale } => {
            use statrs::distribution::StudentsT;
            let t = StudentsT::new(0.0, 1.0, 4.0).expect("fixed standard t4");
            *location + *scale * t.inverse_cdf(uniform.clamp(1e-12, 1.0 - 1e-12)) as f32
        }
        ResidualV1::GaussianMixture2 {
            weight,
            locations,
            scales,
        } => {
            let component = usize::from(uniform >= f64::from(*weight));
            locations[component] + scales[component] * inverse_normal(normal) as f32
        }
        ResidualV1::MonotoneQuantile17 { values } => {
            let position = (uniform * 16.0).clamp(0.0, 16.0);
            let lower = position.floor() as usize;
            let upper = (lower + 1).min(16);
            values[lower] + (values[upper] - values[lower]) * (position - lower as f64) as f32
        }
    }
}
