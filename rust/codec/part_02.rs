

fn encode_dependence(dependence: &Dependence) -> Vec<u8> {
    let mut writer = Writer::default();
    encode_dependence_value(&mut writer, dependence);
    writer.0
}

fn encode_additive(writer: &mut Writer, term: &AdditiveTerm) {
    writer.varint(u64::from(term.feature));
    writer.varint(term.knots.len() as u64);
    for knot in &term.knots {
        writer.unit(*knot);
    }
    for value in &term.values {
        writer.fixed(*value);
    }
}

fn encode_target(target: &Target) -> Vec<u8> {
    let mut writer = Writer::default();
    match target {
        Target::SparseLinear { intercept, terms } | Target::SparseLogistic { intercept, terms } => {
            writer.byte(matches!(target, Target::SparseLogistic { .. }) as u8);
            writer.fixed(*intercept);
            writer.varint(terms.len() as u64);
            let mut ordered = terms.clone();
            ordered.sort_by_key(|term| term.feature);
            let mut prior = 0u32;
            for (index, term) in ordered.iter().enumerate() {
                let delta = if index == 0 {
                    term.feature
                } else {
                    term.feature - prior
                };
                writer.varint(u64::from(delta));
                writer.fixed(term.coefficient);
                prior = term.feature;
            }
        }
        Target::SparseGam {
            intercept,
            logistic,
            terms,
        } => {
            writer.byte(2);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(terms.len() as u64);
            for term in terms {
                encode_additive(&mut writer, term);
            }
        }
        Target::Ga2m {
            intercept,
            logistic,
            main_terms,
            interactions,
        } => {
            writer.byte(3);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(main_terms.len() as u64);
            for term in main_terms {
                encode_additive(&mut writer, term);
            }
            writer.varint(interactions.len() as u64);
            for term in interactions {
                writer.varint(u64::from(term.left));
                writer.varint(u64::from(term.right));
                writer.varint(term.left_knots.len() as u64);
                writer.varint(term.right_knots.len() as u64);
                for knot in &term.left_knots {
                    writer.unit(*knot);
                }
                for knot in &term.right_knots {
                    writer.unit(*knot);
                }
                for value in &term.values {
                    writer.fixed(*value);
                }
            }
        }
        Target::Mars {
            intercept,
            logistic,
            terms,
        } => {
            writer.byte(4);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(terms.len() as u64);
            for term in terms {
                writer.fixed(term.coefficient);
                writer.varint(term.factors.len() as u64);
                for factor in &term.factors {
                    writer.varint(u64::from(factor.feature));
                    writer.unit(factor.knot);
                    writer.byte((factor.direction > 0) as u8);
                }
            }
        }
        Target::ObliviousTree {
            logistic,
            features,
            thresholds,
            leaves,
        } => {
            writer.byte(5);
            writer.fixed(0.0);
            writer.byte(*logistic as u8);
            writer.varint(features.len() as u64);
            for (feature, threshold) in features.iter().zip(thresholds) {
                writer.varint(u64::from(*feature));
                writer.unit(*threshold);
            }
            for leaf in leaves {
                writer.fixed(*leaf);
            }
        }
        Target::CompactNeuralResidual {
            intercept,
            logistic,
            linear_terms,
            hidden_features,
            hidden_width,
            input_weights,
            hidden_biases,
            output_weights,
        } => {
            writer.byte(6);
            writer.fixed(*intercept);
            writer.byte(*logistic as u8);
            writer.varint(linear_terms.len() as u64);
            for term in linear_terms {
                writer.varint(u64::from(term.feature));
                writer.fixed(term.coefficient);
            }
            writer.varint(hidden_features.len() as u64);
            writer.byte(*hidden_width);
            for feature in hidden_features {
                writer.varint(u64::from(*feature));
            }
            for value in input_weights {
                writer.fixed(*value);
            }
            for value in hidden_biases {
                writer.fixed(*value);
            }
            for value in output_weights {
                writer.fixed(*value);
            }
        }
    }
    writer.0
}

fn encode_noise(noise: &Noise) -> Vec<u8> {
    let mut writer = Writer::default();
    match noise {
        Noise::Homoscedastic { sigma } => {
            writer.byte(0);
            writer.unit(*sigma);
        }
        Noise::BernoulliCalibration { base_rate } => {
            writer.byte(1);
            writer.unit(*base_rate);
        }
        Noise::Heteroscedastic {
            intercept,
            terms,
            minimum_sigma,
            maximum_sigma,
        } => {
            writer.byte(2);
            writer.fixed(*intercept);
            writer.unit(*minimum_sigma);
            writer.unit(*maximum_sigma);
            writer.varint(terms.len() as u64);
            for term in terms {
                writer.varint(u64::from(term.feature));
                writer.fixed(term.coefficient);
            }
        }
        Noise::IsotonicCalibration {
            knots,
            probabilities,
        } => {
            writer.byte(3);
            writer.varint(knots.len() as u64);
            for knot in knots {
                writer.unit(*knot);
            }
            for probability in probabilities {
                writer.unit(*probability);
            }
        }
    }
    writer.0
}

const RANS_SCALE_BITS: u32 = 12;
const RANS_TOTAL: u32 = 1 << RANS_SCALE_BITS;
const RANS_LOWER_BOUND: u64 = 1 << 23;

fn normalized_frequencies(raw: &[u8]) -> [u16; 256] {
    let mut counts = [0usize; 256];
    for &byte in raw {
        counts[byte as usize] += 1;
    }
    let mut frequencies = [0u16; 256];
    for symbol in 0..256 {
        if counts[symbol] > 0 {
            frequencies[symbol] =
                ((counts[symbol] * RANS_TOTAL as usize / raw.len()).max(1)) as u16;
        }
    }
    let mut total = frequencies
        .iter()
        .map(|value| u32::from(*value))
        .sum::<u32>();
    while total > RANS_TOTAL {
        let symbol = (0..256)
            .filter(|symbol| frequencies[*symbol] > 1)
            .max_by_key(|symbol| (frequencies[*symbol], counts[*symbol], 255 - *symbol))
            .expect("normalization has a decrementable frequency");
        frequencies[symbol] -= 1;
        total -= 1;
    }
    while total < RANS_TOTAL {
        let symbol = (0..256)
            .filter(|symbol| counts[*symbol] > 0)
            .max_by_key(|symbol| (counts[*symbol], 255 - *symbol))
            .expect("non-empty input has a symbol");
        frequencies[symbol] += 1;
        total += 1;
    }
    frequencies
}

fn rans_encode(raw: &[u8]) -> Vec<u8> {
    let frequencies = normalized_frequencies(raw);
    let mut cumulative = [0u32; 256];
    let mut sum = 0u32;
    for symbol in 0..256 {
        cumulative[symbol] = sum;
        sum += u32::from(frequencies[symbol]);
    }
    let mut state = RANS_LOWER_BOUND;
    let mut renormalized = Vec::new();
    for &byte in raw.iter().rev() {
        let frequency = u64::from(frequencies[byte as usize]);
        let threshold = ((RANS_LOWER_BOUND >> RANS_SCALE_BITS) << 8) * frequency;
        while state >= threshold {
            renormalized.push(state as u8);
            state >>= 8;
        }
        state = ((state / frequency) << RANS_SCALE_BITS)
            + state % frequency
            + u64::from(cumulative[byte as usize]);
    }
    let mut encoded = Vec::new();
    encoded.push(1); // canonical static byte-rANS
    let nonzero = frequencies.iter().filter(|value| **value > 0).count() as u16;
    encoded.extend_from_slice(&nonzero.to_le_bytes());
    for (symbol, frequency) in frequencies.iter().enumerate() {
        if *frequency > 0 {
            encoded.push(symbol as u8);
            encoded.extend_from_slice(&frequency.to_le_bytes());
        }
    }
    encoded.extend_from_slice(&(state as u32).to_le_bytes());
    renormalized.reverse();
    encoded.extend_from_slice(&renormalized);
    encoded
}

fn rans_decode(encoded: &[u8], raw_length: usize) -> Result<Vec<u8>> {
    let mut reader = Reader::new(encoded);
    if reader.byte()? != 1 {
        return Err(DopeError::Codec("unknown range-coder identifier".into()));
    }
    let count = reader.u16()? as usize;
    if count == 0 || count > 256 {
        return Err(DopeError::Codec("invalid rANS frequency table".into()));
    }
    let mut frequencies = [0u16; 256];
    for _ in 0..count {
        let symbol = reader.byte()? as usize;
        let frequency = reader.u16()?;
        if frequency == 0 || frequencies[symbol] != 0 {
            return Err(DopeError::Codec(
                "non-canonical rANS frequency table".into(),
            ));
        }
        frequencies[symbol] = frequency;
    }
    if frequencies
        .iter()
        .map(|value| u32::from(*value))
        .sum::<u32>()
        != RANS_TOTAL
    {
        return Err(DopeError::Codec(
            "rANS frequencies do not sum to scale".into(),
        ));
    }
    let mut cumulative = [0u32; 256];
    let mut lookup = [0u8; RANS_TOTAL as usize];
    let mut sum = 0u32;
    for symbol in 0..256 {
        cumulative[symbol] = sum;
        for slot in sum..sum + u32::from(frequencies[symbol]) {
            lookup[slot as usize] = symbol as u8;
        }
        sum += u32::from(frequencies[symbol]);
    }
    let state_bytes: [u8; 4] = reader.take(4)?.try_into().expect("four bytes requested");
    let mut state = u64::from(u32::from_le_bytes(state_bytes));
    let mut output = Vec::with_capacity(raw_length);
    for _ in 0..raw_length {
        let residue = (state & u64::from(RANS_TOTAL - 1)) as u32;
        let symbol = lookup[residue as usize];
        output.push(symbol);
        let frequency = u64::from(frequencies[symbol as usize]);
        state = frequency * (state >> RANS_SCALE_BITS)
            + u64::from(residue - cumulative[symbol as usize]);
        while state < RANS_LOWER_BOUND {
            state = (state << 8) | u64::from(reader.byte()?);
        }
    }
    if reader.remaining() != 0 {
        return Err(DopeError::Codec("trailing bytes in rANS stream".into()));
    }
    Ok(output)
}

include!("section_encoding.rs");
include!("decoding.rs");
