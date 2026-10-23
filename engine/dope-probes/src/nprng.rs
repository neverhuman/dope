//! Bit-exact port of NumPy's `np.random.default_rng(seed).permutation(n)` for a seed below
//! 2^32: `SeedSequence` (pool of four words) seeds `PCG64` (128-bit LCG, XSL-RR output), and
//! `Generator.shuffle` runs a Fisher–Yates pass with masked rejection sampling.
//!
//! The probes split their inputs with this permutation, so the Rust split selects the same
//! rows in the same order as the original probe files.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;
const PCG_MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `SeedSequence(seed).generate_state(n_words, np.uint64)`.
pub fn seed_sequence_u64(seed: u32, n_words: usize) -> Vec<u64> {
    // A non-negative integer below 2^32 coerces to the single entropy word [seed].
    let entropy = [seed];
    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        let word = entropy.get(i).copied().unwrap_or(0);
        *slot = hashmix(word, &mut hash_const);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                let h = hashmix(pool[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
    }
    let mut hash_const = INIT_B;
    let words: Vec<u32> = (0..n_words * 2)
        .map(|i| {
            let mut v = pool[i % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^ (v >> XSHIFT)
        })
        .collect();
    words
        .chunks(2)
        .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
        .collect()
}

/// NumPy's `PCG64` bit generator, including its buffered 32-bit draws.
pub struct Pcg64 {
    state: u128,
    inc: u128,
    spare: Option<u32>,
}

impl Pcg64 {
    /// `np.random.PCG64(seed)` for an integer seed below 2^32.
    pub fn new(seed: u32) -> Self {
        let s = seed_sequence_u64(seed, 4);
        let initstate = (u128::from(s[0]) << 64) | u128::from(s[1]);
        let initseq = (u128::from(s[2]) << 64) | u128::from(s[3]);
        let mut rng = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            spare: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    pub fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let next = self.next_u64();
        self.spare = Some((next >> 32) as u32);
        next as u32
    }

    /// NumPy's `random_interval`: uniform on `0..=max` by masked rejection.
    pub fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for shift in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> shift;
        }
        loop {
            let v = if max <= u64::from(u32::MAX) {
                u64::from(self.next_u32()) & mask
            } else {
                self.next_u64() & mask
            };
            if v <= max {
                return v;
            }
        }
    }
}

/// `np.random.default_rng(seed).permutation(n)`.
pub fn permutation(seed: u32, n: usize) -> Vec<usize> {
    let mut rng = Pcg64::new(seed);
    let mut out: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.interval(i as u64) as usize;
        out.swap(i, j);
    }
    out
}
