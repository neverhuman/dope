//! Explicit development residual operators. Historical noise opcodes stay fixed.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualProfile {
    NoNoise,
    StudentT4,
    GaussianMixture2,
    MonotoneQuantile17,
}

impl ResidualProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "no_noise_v1" => Some(Self::NoNoise),
            "student_t4_v1" => Some(Self::StudentT4),
            "gaussian_mixture2_v1" => Some(Self::GaussianMixture2),
            "monotone_quantile17_v1" => Some(Self::MonotoneQuantile17),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoNoise => "no_noise_v1",
            Self::StudentT4 => "student_t4_v1",
            Self::GaussianMixture2 => "gaussian_mixture2_v1",
            Self::MonotoneQuantile17 => "monotone_quantile17_v1",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResidualV1 {
    NoNoise,
    StudentT4 {
        location: f32,
        scale: f32,
    },
    GaussianMixture2 {
        weight: f32,
        locations: [f32; 2],
        scales: [f32; 2],
    },
    MonotoneQuantile17 {
        values: Vec<f32>,
    },
}

impl ResidualV1 {
    pub fn validate(&self) -> std::result::Result<(), String> {
        let location = |v: f32| v.is_finite() && v.abs() <= 4096.0;
        let scale = |v: f32| v.is_finite() && (0.0..=4096.0).contains(&v);
        let valid = match self {
            Self::NoNoise => true,
            Self::StudentT4 {
                location: mean,
                scale: sigma,
            } => location(*mean) && scale(*sigma),
            Self::GaussianMixture2 {
                weight,
                locations,
                scales,
            } => {
                weight.is_finite()
                    && (0.0..=1.0).contains(weight)
                    && locations.iter().all(|v| location(*v))
                    && scales.iter().all(|v| scale(*v))
            }
            Self::MonotoneQuantile17 { values } => {
                values.len() == 17
                    && values.iter().all(|v| location(*v))
                    && values.windows(2).all(|v| v[0] <= v[1])
            }
        };
        if valid {
            Ok(())
        } else {
            Err("invalid development residual v1".into())
        }
    }
}
