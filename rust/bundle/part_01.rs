#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleError {
    code: &'static str,
}

impl BundleError {
    pub const fn code(self) -> &'static str {
        self.code
    }
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.code)
    }
}

impl std::error::Error for BundleError {}

fn err(code: &'static str) -> BundleError {
    BundleError { code }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleLimits {
    pub max_manifest_bytes: usize,
    pub max_component_bytes: usize,
    pub max_components: usize,
    pub max_expanded_section_bytes: usize,
    pub max_features: usize,
    pub max_sample_rows: usize,
    pub max_sample_bytes: usize,
    pub max_total_bytes: usize,
}

impl Default for BundleLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 1024 * 1024,
            max_component_bytes: 16 * 1024 * 1024,
            max_components: 32,
            max_expanded_section_bytes: 16 * 1024 * 1024,
            max_features: 4096,
            max_sample_rows: 100_000,
            max_sample_bytes: 32 * 1024 * 1024,
            max_total_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportFraming {
    Unmeasured,
    Included,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SamplingAccounting {
    pub sampling_payload_bytes: u64,
    pub shared_learned_state_bytes: u64,
    pub standalone_learned_state_bytes: u64,
    pub amortized_learned_state_bytes: u64,
    pub runtime_package_bytes: u64,
    pub evidence_authorization_bytes: u64,
    pub materialized_rows_bytes: u64,
    pub transform_bytes: u64,
    pub dictionary_bytes: u64,
    pub manifest_bytes: u64,
    pub charged_sampling_bytes: u64,
    pub amortization_count: u64,
    pub transport_framing: TransportFraming,
    pub complete_delivery_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DpkSummary {
    pub features: u32,
    pub minor: u8,
    pub v3: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    Dictionary,
    Evidence,
    Rows,
    Runtime,
    Payload,
    Shared,
    Transform,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Dpk2,
    Dpk3,
    Opaque,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplingHandler {
    Inventory,
    Native,
}

#[derive(Clone, Debug)]
struct Component {
    id: String,
    role: Role,
    path: String,
    sha256: String,
    bytes: u64,
    kind: Kind,
    required: bool,
    consumed_by_sampler: bool,
    depends_on: Vec<String>,
    measured_bytes: u64,
}

#[derive(Debug)]
pub struct VerifiedBundle {
    accounting: SamplingAccounting,
    payload: Option<Vec<u8>>,
    features: Option<u32>,
    handler: SamplingHandler,
}

impl VerifiedBundle {
    pub fn accounting(&self) -> &SamplingAccounting {
        &self.accounting
    }

    pub fn retained_payload(&self) -> Option<&[u8]> {
        self.payload.as_deref()
    }

    pub fn sample(
        &self,
        rows: usize,
        seed: Option<u64>,
        limits: &BundleLimits,
    ) -> Result<Vec<f32>, BundleError> {
        if self.handler != SamplingHandler::Native {
            return Err(err("sampling_handler_refused"));
        }
        let payload = self
            .payload
            .as_deref()
            .ok_or_else(|| err("sampler_closure_mismatch"))?;
        let features = self
            .features
            .ok_or_else(|| err("sampler_closure_mismatch"))?;
        if rows == 0 || features == 0 {
            return Err(err("invalid_dimensions"));
        }
        if rows > limits.max_sample_rows {
            return Err(err("sample_cap_exceeded"));
        }
        let bytes = u128::from(u64::try_from(rows).map_err(|_| err("sample_cap_exceeded"))?)
            * (u128::from(features) + 1)
            * 4;
        if bytes > u128::from(limits.max_sample_bytes as u64) {
            return Err(err("sample_cap_exceeded"));
        }
        preflight_dpk(payload, limits)?;
        let kernel = crate::codec::decode_kernel(payload).map_err(|_| err("artifact_rejected"))?;
        let loaded = if payload.starts_with(crate::codec::MAGIC) {
            crate::codec::LoadedKernel::V3(Box::new(kernel))
        } else if payload.starts_with(crate::codec::V2_MAGIC) {
            crate::codec::LoadedKernel::V2(Box::new(kernel))
        } else {
            return Err(err("unsupported_dpk_version"));
        };
        let values =
            crate::sample::sample_kernel(&loaded, crate::sample::SampleOptions { rows, seed })
                .map_err(|_| err("sample_rejected"))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(err("nonfinite_value"));
        }
        Ok(values)
    }
}

pub fn charged_sampling_bytes(bundle: &VerifiedBundle) -> u64 {
    bundle.accounting.charged_sampling_bytes
}

pub fn within_budget(charged: u64, cap: u64) -> bool {
    charged <= cap
}

pub fn content_fingerprint(data: &[u8]) -> String {
    hex(&sha256(data))
}

fn checked_add(left: u64, right: u64) -> Result<u64, BundleError> {
    left.checked_add(right)
        .ok_or_else(|| err("integer_overflow"))
}

fn sha256(message: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = u64::try_from(message.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(8);
    let mut data = Vec::with_capacity(message.len() + 72);
    data.extend_from_slice(message);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in data.chunks_exact(64) {
        let mut words = [0u32; 64];
        for index in 0..16 {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&chunk[index * 4..index * 4 + 4]);
            words[index] = u32::from_be_bytes(bytes);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    let mut out = [0u8; 32];
    for (index, value) in state.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    out
}
