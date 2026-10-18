//! Explicit development residual operators. Historical noise opcodes stay fixed.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualProfile {
    NoNoise,
    GaussianHomoscedastic,
    GaussianHeteroscedastic,
    StudentT4,
    GaussianMixture2,
    MonotoneQuantile17,
}

impl ResidualProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "no_noise_v1" => Some(Self::NoNoise),
            "gaussian_homoscedastic_v1" => Some(Self::GaussianHomoscedastic),
            "gaussian_heteroscedastic_v1" => Some(Self::GaussianHeteroscedastic),
            "student_t4_v1" => Some(Self::StudentT4),
            "gaussian_mixture2_v1" => Some(Self::GaussianMixture2),
            "monotone_quantile17_v1" => Some(Self::MonotoneQuantile17),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoNoise => "no_noise_v1",
            Self::GaussianHomoscedastic => "gaussian_homoscedastic_v1",
            Self::GaussianHeteroscedastic => "gaussian_heteroscedastic_v1",
            Self::StudentT4 => "student_t4_v1",
            Self::GaussianMixture2 => "gaussian_mixture2_v1",
            Self::MonotoneQuantile17 => "monotone_quantile17_v1",
        }
    }

    /// Residual scales need cross-fitted errors; the deterministic control does not.
    pub const fn minimum_rows(self) -> usize {
        match self {
            Self::NoNoise => 0,
            Self::GaussianHomoscedastic
            | Self::GaussianHeteroscedastic
            | Self::StudentT4
            | Self::GaussianMixture2
            | Self::MonotoneQuantile17 => 24,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_preregistered_residual_profiles_round_trip_and_enforce_fit_support() {
        let profiles = [
            ("no_noise_v1", ResidualProfile::NoNoise, 0),
            (
                "gaussian_homoscedastic_v1",
                ResidualProfile::GaussianHomoscedastic,
                24,
            ),
            (
                "gaussian_heteroscedastic_v1",
                ResidualProfile::GaussianHeteroscedastic,
                24,
            ),
            ("student_t4_v1", ResidualProfile::StudentT4, 24),
            (
                "gaussian_mixture2_v1",
                ResidualProfile::GaussianMixture2,
                24,
            ),
            (
                "monotone_quantile17_v1",
                ResidualProfile::MonotoneQuantile17,
                24,
            ),
        ];
        for (id, profile, minimum_rows) in profiles {
            assert_eq!(ResidualProfile::parse(id), Some(profile));
            assert_eq!(profile.as_str(), id);
            assert_eq!(profile.minimum_rows(), minimum_rows);
        }
        assert_eq!(ResidualProfile::parse("unregistered_v1"), None);

        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../research/transfer/ablations/residuals.json"
        ))
        .unwrap();
        let planned = manifest["candidate_set"].as_array().unwrap();
        assert_eq!(planned.len(), profiles.len());
        for candidate in planned {
            let id = candidate["profile"].as_str().unwrap();
            assert!(ResidualProfile::parse(id).is_some(), "{id}");
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
