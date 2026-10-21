# Rust deployment runtime

`dope-runtime` is an **additive development runtime**, built separately from the
existing native CLI. Application inference is Rust; it starts no Python process
and links no libtorch. SafeTensors is the transport. Candle supplies tensor
operations. System libraries and, for CUDA builds, the GPU driver/toolkit are
external runtime dependencies; this is not a universally static OS image.

The default native `dope-kernel` CPU executable remains the complete raw/mixed
compiler and sampler. Use it for your source table and the public Airfoil example.
This runtime consumes native numeric DPK exports. It does **not** consume a raw
CSV, invent a schema, load arbitrary Python checkpoints, or export raw/mixed
envelopes. Native DPK samples stay in their existing normalized coordinates;
retain and charge a required historical projection separately. Raw-unit users
should use the complete native raw artifact workflow until a versioned envelope
export is integrated.

| Carrier program | `cpu` | `auto` | `cuda` |
| --- | --- | --- | --- |
| Symbolic DPK | Native Rust | Native Rust | Explicit unsupported error |
| TVAE, `legacy-v1` activation | Native Rust, original draw stream | Batched Candle CUDA when available in a CUDA build; native CPU otherwise | Candle CUDA or explicit unavailable error |
| TVAE, `tanh-v2` activation | Candle CPU | Candle CUDA when available; Candle CPU otherwise | Candle CUDA or explicit unavailable error |
| TabSyn, TabDDPM, autoregressive transformer | Native Rust | Native Rust | Explicit unsupported error |

CUDA is source-integrated but **not hardware-qualified by this PR**. A CUDA-enabled
binary requires its platform CUDA dependencies to load. Once the process starts,
`auto` falls back on device-initialization failure. Allocation or execution failures
return an error; they are not silently retried on another backend. Inspect the
actual `device_used` receipt. CPU and GPU floating-point reductions need tolerance
parity, not a claim of bit-identical GPU samples.

## Build, export, inspect, sample

From the repository root:

```sh
cargo build --manifest-path deploy/Cargo.toml --locked --release
target_runtime=deploy/target/release/dope-runtime

# INPUT.dpk is an existing native DPK, not a raw CSV or raw-unit envelope.
"$target_runtime" pack --artifact INPUT.dpk --out model.safetensors
"$target_runtime" inspect --artifact model.safetensors
"$target_runtime" sample --artifact model.safetensors --rows 1000 --seed 91 \
  --device auto --out synthetic.csv
```

For a CUDA build on a provisioned CUDA host:

```sh
cargo build --manifest-path deploy/Cargo.toml --locked --release --features cuda
```

The dedicated lock includes the deployment dependency graph; it does not modify
the existing product lock, frozen contracts or default dependency notices.
`deploy/native` builds `rust/runtime_lib.rs`, an inference-only entry over the
canonical codec, model, neural and sampling modules. It avoids compiling the
compiler, campaign, training and service modules, while retaining one sampler
implementation. Their full unit suites remain in the root package; deployment
tests exercise this shared inference build through complete carriers and draws.
Verify deployment graph licenses and produce a separate SBOM before distributing
this experimental binary. Build without `target-cpu=native` for portable CPU
artifacts; reserve native CPU tuning for a named local build. Fat LTO and symbol
stripping are configured. The runtime caps artifact reads at 64 MiB, samples at
100,000 rows and one million cells, uses exclusive 0600 outputs on Unix, and emits
fixed error codes without source values or paths. It opens no network connection.
It is an ordinary local process, not the isolated source-free certification worker.

## Numerical compatibility and the GELU correction

The training paths use PyTorch `gelu("tanh")`, whose coefficient is
`sqrt(2/pi)`. Historical native inference uses `2/sqrt(pi)`. At input one,
these give different activations. Reinterpreting an old DPK would change its
distribution, replay digest and benchmark evidence.

Packing defaults to `--activation legacy-v1`. The additive
`--activation tanh-v2` flag is accepted only for TVAE and records the corrected
formula in the carrier. The program and training weights are otherwise the same;
the new behavior needs new transfer measurements. Tests use nonzero weights,
a nonidentity feature permutation and multiple batches, not just zero networks.
The old GPU parity test in `rust/neural_train/tests.rs` also compares the historical
activation to PyTorch's training formula: rerun it on hardware and retain any
failure as historical evidence; do not loosen its tolerance to hide the mismatch.

## Byte accounting and model exchange

Version one stores a `native_dpk` U8 tensor plus typed TVAE I8 weight matrices,
F32 per-output scales and F32 biases. The loader verifies every named tensor,
shape, dtype and byte against the native graph and refuses extras or mismatches.
The tensors are therefore actual model tensors, rather than an arbitrary pickle
hidden behind a file extension. Native code still owns graph semantics, random
streams, feature order, inverse marginals, missingness and target support.

The initial format intentionally duplicates decoder parameters. The **whole
SafeTensors file** is the charged payload, including header and duplicates.
Inspection reports the full length and SHA256; cold closure remains null until
measured. This is an interoperability and numerical-boundary implementation,
not a compression claim. The small symbolic sampler should continue using DPK.
A future nonduplicating tensor carrier needs a versioned manifest and independent
replay proof before it can replace this reference format.

Python DL training may produce and quantize a native graph through the existing
training/export implementation. Only the declared graph and tensors cross into
the runtime. Arbitrary SafeTensors files lack DOPE's architecture, marginal maps,
quantization scales and variance rules, and are refused.

## Verification and promotion

```sh
cargo test --manifest-path deploy/Cargo.toml --locked
cargo clippy --manifest-path deploy/Cargo.toml --locked --all-targets -- -D warnings
cargo tree --manifest-path deploy/Cargo.toml --locked -e normal
```

CPU CI exercises carrier corruption, exact historical CPU replay, nonzero tensor
head parity, explicit corrected behavior, bounds and unavailable-device refusal.
An actual CLI integration test exports, inspects and samples a native symbolic
fixture, verifies the full charge, and checks exclusive outputs and redacted errors.
The independent deployment graph is checked for forbidden training dependencies.
This does not qualify GPU execution, a sealed benchmark or release authorization.
Require `Rust deployment conformance / cpu` and the existing repository gates
before promoting this workspace. The new lane's cold-run duration is unmeasured;
its 20-minute cap is a bound, not a demonstrated performance claim.

Before promoting a CUDA binary, run CPU/CUDA decoder and sample comparisons on
nonzero trained profiles, batch sizes 1/32/256/4096, tail latents and all support
types. Test missing driver, unavailable device, OOM and execution failure.
Measure startup, export/load time, rows/s, peak RSS/VRAM, actual mapped libraries,
binary size and complete cold closure. Require finite outputs, declared tolerances
and unchanged categorical/missingness decisions or report their disagreements.
Run downstream transfer for both activation profiles as separate methods.
Requested compute and the finite screening plan are in
[`docs/technical-upgrade.md`](../docs/technical-upgrade.md).
