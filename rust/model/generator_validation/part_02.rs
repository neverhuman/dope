

fn validate_target(target: &Target, width: usize) -> std::result::Result<(), String> {
    let linear_terms_valid = |terms: &[LinearTerm]| {
        terms.len() <= 64
            && terms
                .iter()
                .all(|term| (term.feature as usize) < width && term.coefficient.is_finite())
    };
    match target {
        Target::SparseLinear { intercept, terms } | Target::SparseLogistic { intercept, terms }
            if intercept.is_finite() && linear_terms_valid(terms) =>
        {
            Ok(())
        }
        Target::SparseGam {
            intercept, terms, ..
        } if intercept.is_finite()
            && terms.len() <= 64
            && terms.iter().all(|term| validate_additive(term, width)) =>
        {
            Ok(())
        }
        Target::Ga2m {
            intercept,
            main_terms,
            interactions,
            ..
        } if intercept.is_finite()
            && main_terms.len() <= 64
            && main_terms.iter().all(|term| validate_additive(term, width))
            && interactions.len() <= 64
            && interactions.iter().all(|term| {
                (term.left as usize) < width
                    && (term.right as usize) < width
                    && term.left < term.right
                    && (3..=9).contains(&term.left_knots.len())
                    && (3..=9).contains(&term.right_knots.len())
                    && increasing(&term.left_knots)
                    && increasing(&term.right_knots)
                    && term.values.len() == term.left_knots.len() * term.right_knots.len()
                    && term.values.iter().all(|value| value.is_finite())
            }) =>
        {
            Ok(())
        }
        Target::Mars {
            intercept, terms, ..
        } if intercept.is_finite()
            && terms.len() <= 128
            && terms.iter().all(|term| {
                term.coefficient.is_finite()
                    && (1..=3).contains(&term.factors.len())
                    && term.factors.iter().all(|factor| {
                        (factor.feature as usize) < width
                            && factor.knot.is_finite()
                            && matches!(factor.direction, -1 | 1)
                    })
            }) =>
        {
            Ok(())
        }
        Target::ObliviousTree {
            features,
            thresholds,
            leaves,
            ..
        } if features.len() <= 8
            && features.len() == thresholds.len()
            && leaves.len() == 1usize << features.len()
            && features.iter().all(|feature| (*feature as usize) < width)
            && thresholds.iter().all(|value| value.is_finite())
            && leaves.iter().all(|value| value.is_finite()) =>
        {
            Ok(())
        }
        Target::CompactNeuralResidual {
            intercept,
            linear_terms,
            hidden_features,
            hidden_width,
            input_weights,
            hidden_biases,
            output_weights,
            ..
        } if intercept.is_finite()
            && linear_terms_valid(linear_terms)
            && (1..=24).contains(&hidden_features.len())
            && (1..=16).contains(hidden_width)
            && hidden_features
                .iter()
                .all(|feature| (*feature as usize) < width)
            && {
                let mut unique = hidden_features.clone();
                unique.sort_unstable();
                unique.dedup();
                unique.len() == hidden_features.len()
            }
            && input_weights.len() == hidden_features.len() * usize::from(*hidden_width)
            && hidden_biases.len() == usize::from(*hidden_width)
            && output_weights.len() == usize::from(*hidden_width)
            && input_weights.iter().all(|value| value.is_finite())
            && hidden_biases.iter().all(|value| value.is_finite())
            && output_weights.iter().all(|value| value.is_finite()) =>
        {
            Ok(())
        }
        _ => Err("invalid bounded target operator".into()),
    }
}

fn validate_noise(noise: &Noise, width: usize) -> std::result::Result<(), String> {
    match noise {
        Noise::Homoscedastic { sigma } if valid_probability(*sigma) => Ok(()),
        Noise::BernoulliCalibration { base_rate } if valid_probability(*base_rate) => Ok(()),
        Noise::Heteroscedastic {
            intercept,
            terms,
            minimum_sigma,
            maximum_sigma,
        } if intercept.is_finite()
            && valid_probability(*minimum_sigma)
            && valid_probability(*maximum_sigma)
            && minimum_sigma <= maximum_sigma
            && terms.len() <= 64
            && terms
                .iter()
                .all(|term| (term.feature as usize) < width && term.coefficient.is_finite()) =>
        {
            Ok(())
        }
        Noise::IsotonicCalibration {
            knots,
            probabilities,
        } if (2..=33).contains(&knots.len())
            && knots.len() == probabilities.len()
            && increasing(knots)
            && increasing(probabilities)
            && probabilities.iter().all(|value| valid_probability(*value)) =>
        {
            Ok(())
        }
        _ => Err("invalid bounded noise or calibration operator".into()),
    }
}
