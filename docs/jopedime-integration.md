# Pinned downstream DOPE encoder

Use the existing GitHub release weights with a separately pinned source checkout.
`DopeEncoder629` embeds the High router and column-gate head in a resident PyTorch
process. It returns features for downstream projections and rankers; downstream
preprocessing, fitted model state, program selection, and predictions remain the
consumer's responsibility. No JopeDimeMicro runtime changes are required here.

## Pins and distribution

| Item | Pin |
| --- | --- |
| Repository | `https://github.com/neverhuman/dope` |
| Compatible encoder source commit | `8b12a262ef43d30538da8274cd2a758b3740881e` |
| Source engine workspace version | `1.1.0` (source enhancement; no new release published) |
| Model release | [`v1.0.0`](https://github.com/neverhuman/dope/releases/tag/v1.0.0) |
| Canonical file | `dope_mlc_v3_multitask_best.safetensors` |
| Exact bytes | `1,781,800` |
| SHA-256 | `d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c` |
| Reference environment | Python `3.12.3`, PyTorch `2.7.0`, NumPy `1.26.4`, SafeTensors `0.7.0` |
| Existing research requirements | [requirements.txt](../research/levels/requirements.txt), including `packaging==25.0` |
| Rust dependency graph | [engine/Cargo.lock](../engine/Cargo.lock) at the source pin |

The model release `v1.0.0` and the signed engine release
[`dope-engine-v1.0.0`](https://github.com/neverhuman/dope/releases/tag/dope-engine-v1.0.0)
are distinct. The engine release carries binaries, not these model weights.
The `v1.0.0` source tag predates `research/levels/multilevel_model.py`, the Python
model source used by Micro. Checking out that tag does not provide the encoder.
Use the exact source commit above, separately from the weight asset URL.

The 1.70 MiB checkpoint fits comfortably within GitHub's per-asset limit of under
2 GiB ([GitHub release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)).
Additional hosting is unnecessary. Keep published tags and assets intact. A checksum
compares bytes against the reviewed pin; it does not authenticate release provenance.
The release API reported `immutable: false` when checked on 2026-10-27. Keep the
reviewed digest in downstream configuration rather than trusting a newly downloaded
checksum manifest as an independent authority.

DOPE source and the released model manifest declare MIT. Preserve the copyright,
permission notice, and [LICENSE](../LICENSE) in bundled copies. Include
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the actual dependency notices
for the distributed runtime. MIT for DOPE does not relicense PyTorch, NumPy,
SafeTensors, datasets, or unrelated downstream model assets.

## Checkout and verified download

Run from a downstream repository. Record the checked-out commit in its lock or
submodule gitlink; do not follow a moving branch automatically.

```bash
set -euo pipefail
git clone https://github.com/neverhuman/dope.git vendor/dope
git -C vendor/dope checkout --detach 8b12a262ef43d30538da8274cd2a758b3740881e
# Alternatively, instead of the clone above:
# git submodule add https://github.com/neverhuman/dope.git vendor/dope
# git -C vendor/dope checkout --detach 8b12a262ef43d30538da8274cd2a758b3740881e
# git add .gitmodules vendor/dope

python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.0'
.venv/bin/python -m pip install 'numpy==1.26.4' 'safetensors==0.7.0' 'packaging==25.0'
# CUDA hosts: use the compatible PyTorch 2.7.0 CUDA build for the host driver.

mkdir -p assets/models
curl --fail --location --silent --show-error \
  https://github.com/neverhuman/dope/releases/download/v1.0.0/dope_mlc_v3_multitask_best.safetensors \
  --output assets/models/dope_mlc_v3_multitask_best.safetensors
printf '%s  %s\n' \
  d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c \
  assets/models/dope_mlc_v3_multitask_best.safetensors | sha256sum --check -
test "$(wc -c < assets/models/dope_mlc_v3_multitask_best.safetensors)" -eq 1781800
export PYTHONPATH="$PWD/vendor/dope${PYTHONPATH:+:$PYTHONPATH}"
```

Bundle this verified asset and the pinned source with the application for offline
startup. `from_pretrained` reads a local file only. Its default checksum is the
canonical multitask digest, verified **before** parsing SafeTensors. It never
loads pickle. It validates all 56 checkpoint tensors (444,168 parameters) and
shapes strictly as finite FP32, then retains only 40 tensors / 353,095 parameters: 38 router tensors / 352,949
parameters and 2 column-gate tensors / 146 parameters. Extra keys, missing keys,
changed shapes or dtypes are refused even in unused heads. An explicit alternate
digest permits a compatible full checkpoint, but downstream qualification must be
repeated. Pruned checkpoints and legacy `.pt` files are outside this loader's API.

## Tensor and preprocessing contract

```text
encoder(dataset_features, col_landmarks) -> features
FP32 [B,256] + FP32 [B,64,17] -> FP32 [B,629], B >= 1
```

Both inputs and every encoder parameter must share one CPU or CUDA device.
Use dense tensors and disable autocast; FP16/BF16 composition and other tiers are
outside the supported contract. Wrong widths, ranks, batches, dtypes, or devices
raise generic errors without input values. The interface does not crop, broadcast,
move, or cast inputs. Nonfinite outputs become zero at the **end**, as in Micro;
there is no input sanitization that would alter the router's original computation.

| Output slice | Width | Meaning |
| --- | ---: | --- |
| `[0:256]` | 256 | Conditioning vector |
| `[256:384]` | 128 | Task latent |
| `[384:512]` | 128 | Distribution latent |
| `[512:526]` | 14 | Alphabetical gates below |
| `[526:549]` | 23 | Landmark predictions |
| `[549:565]` | 16 | Building-block predictions |
| `[565:629]` | 64 | Column gates |

Gate order is fixed independently of the router dictionary's insertion order:

```text
asinh_atan_w, copula_retention, het_scale_w, hinge_w, monotone_w,
neural_w, noise_scale, pair_w, periodic_fract_w, regime_sharpness,
rff_w, ridge_alpha, row_summary_w, tree_w
```

Column gate `j` is `1 + 0.25*tanh(Linear(cat(task_latent, col_landmarks[j])))`.
The caller prepares a zero-filled `[B,64,17]` buffer, copies the first
`min(p,64)` columns and the first 17 statistics of the train-derived 24-statistic
column descriptors, and leaves the remaining column rows zero. Do not pad the
resulting **gates** with zero or one: padded columns still receive the learned
head applied to the task latent and zero descriptors.

Use the pinned [High conditioning builder](../research/levels/multilevel_model.py)
and [building-block statistics](../research/levels/building_block_landmarks.py).
Preserve the 256-coordinate schema, landmark masks, scaling, categorical mapping,
feature order, and column identities. Check the declared input schema before
turning data into tensors: the encoder can refuse dimension drift, but cannot
detect a same-width semantic permutation. Fit imputers/scalers/categories,
landmarks, target-related probes, SVD bases, and normalization statistics on the
admitted training partition only; freeze them for evaluation. Query targets must
never enter query features, context selection, or ranker inputs. Retain physical
row identities and follow the [target-last contract](data-contract.md), resolving
the target from an explicit declaration and normalizing external CSVs once through
the shared intake. Never infer the target from the last source column.

## Offline inference and downstream composition

Run with the `PYTHONPATH` and local asset prepared above. Zero inputs below only
exercise the interface; replace them with admitted train-derived descriptors.

```python
import torch
from research.levels.encoder import DopeEncoder629

torch.set_num_threads(1)
encoder = DopeEncoder629.from_pretrained(
    "assets/models/dope_mlc_v3_multitask_best.safetensors", device="cpu",
)
vec = torch.zeros(1, 256, dtype=torch.float32)
columns = torch.zeros(1, 64, 17, dtype=torch.float32)
with torch.no_grad():
    features = encoder(vec, columns)
assert features.shape == (1, 629)
```

Compose the encoder directly with the downstream module. This example uses a
representative projection/ranker to exercise export and capture, with randomly
initialized downstream weights. A production consumer must separately pin and
load its actual projection/ranker state, including ordered gate keys,
DOPE-feature mean/std and masks, token projections, SVD basis and landmark
normalizers, candidate/prior builders, anchors, policy weights and fitted state.
The DOPE pin supplies none of those downstream assets.

```python
class Composite(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projection = torch.nn.Linear(629, 48)
        self.ranker = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(48, 3))

    def forward(self, vec, columns):
        return self.ranker(self.projection(self.encoder(vec, columns)))

composite = Composite(encoder).requires_grad_(False).eval()
```

### CPU export

The PyTorch 2.7.0 export round trip below was exercised on CPU for batches 1 and 4.
Use an example batch larger than one when declaring a dynamic batch dimension.
The exported program captures batch constraints and an FP32 CPU computation;
Python validation belongs to the eager API. Keep shape, dtype, device, and schema
validation at the consumer's input boundary.

```python
v4 = torch.zeros(4, 256)
c4 = torch.zeros(4, 64, 17)
batch = torch.export.Dim("batch", min=1, max=8)
with torch.no_grad():
    program = torch.export.export(
        composite, (v4, c4), dynamic_shapes=({0: batch}, {0: batch}),
    )
    torch.export.save(program, "composite.pt2")
    offline = torch.export.load("composite.pt2").module()
    torch.testing.assert_close(offline(vec, columns), composite(vec, columns))
```

### CUDA Graph replay

Tested on an NVIDIA GeForce RTX 3090 with PyTorch 2.7.0 (CUDA 12.8), FP32, autocast off.
Capture the final composed module on fixed shapes. Replay reuses static output
storage; copy a result if it must survive a later replay. The caller owns stream
ordering and input validation on subsequent replays.

```python
composite = composite.cuda().eval()
static_vec = torch.zeros(1, 256, device="cuda")
static_columns = torch.zeros(1, 64, 17, device="cuda")
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.no_grad(), torch.cuda.stream(stream):
    for _ in range(3):
        composite(static_vec, static_columns)
torch.cuda.current_stream().wait_stream(stream)
graph = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(graph):
    static_scores = composite(static_vec, static_columns)
# In a real call, copy prepared tensors with these exact shapes/dtypes/devices.
static_vec.copy_(torch.ones_like(static_vec))
with torch.no_grad():
    expected = composite(static_vec, static_columns)
graph.replay()
torch.cuda.synchronize()
torch.testing.assert_close(static_scores, expected, rtol=1e-5, atol=1e-6)
```

The CPU export and CUDA replay tests compare numerical outputs; they do not
measure latency or throughput. These examples do not qualify ONNX, TensorRT,
mixed precision, a particular thread/NUMA policy, or the private downstream ranker.
The nonlinear router cannot be collapsed into a single projection matrix.

## Rust composition

The additive `MlcV3Model::encode_features_629` and `encode_features_629_batch`
methods use the same layout, High-only level, exact 256/64/17 dimensions, and
final nonfinite-to-zero behavior. Rust types enforce FP32; this implementation
runs on CPU and retains the full model. The batch must be nonempty with matching
lengths. Legacy routing/sampling methods retain their existing contracts.

The [complete Rust example](../engine/dope-core/examples/encoder_629.rs) verifies
the canonical digest on raw bytes before parsing. Compile and run from the pinned
DOPE checkout:

```bash
cargo run --manifest-path engine/Cargo.toml --locked -p dope-core \
  --example encoder_629 -- /absolute/path/to/dope_mlc_v3_multitask_best.safetensors
```

Downstream Cargo projects may use
`dope-core = { path = "vendor/dope/engine/dope-core" }` with the source gitlink
pinned above and their own committed Cargo lockfile.

## Proposed downstream CPU projection optimization

Micro Stage 1A computes
`lm48 = clamp((vec256 - step2_lm_mean) / step2_lm_std, -6, 6) @ step2_svd_basis_48`
on GPU, alongside the DOPE features. Stage 1B standardizes it as
`lm48_z = clamp((lm48 - mu_lm48) / sd_lm48, -6, 6)`.
The fresh-table candidate builder uses `lm48`/`lm48_z` and the training table,
while the neural ranker consumes the DOPE output later.

Moving the frozen landmark projection and normalization onto CPU before building
candidates could permit one subsequent composed GPU replay. This is a proposal
requiring downstream qualification. No such runtime restructuring is implemented
here, and numerical closeness of the projection alone is insufficient. Compare
ordered candidate hashes, anchor coefficients and diagnostics, prior scores,
selected program/recipe identity, and final held-out predictions. Measure latency
and throughput separately against the existing path and asynchronous overlap.
CPU/GPU rounding can change discrete selection at ties; any discrepancy must be
resolved before adopting the optimization.

## Reproduce the focused evidence

From the pinned DOPE checkout with the reference dependencies installed:

```bash
SHARED_PYTHON=python3 bash ops/ci/shared-levels.sh torch test_encoder_629
export DOPE_ENCODER_WEIGHTS="$PWD/target/encoder-629/dope_mlc_v3_multitask_best.safetensors"
export DOPE_ENCODER_PARITY_FIXTURE="$PWD/target/encoder-629/micro-reference.json"
cargo test --manifest-path engine/Cargo.toml --locked -p dope-core \
  --test mlc_v3_safetensors -- --nocapture
```

The CPU shard fetches and verifies the canonical file and fails if it is absent or
changed. It tests every block against Micro's original per-row formula for batches
1 and 4 with padded columns, nonfinite behavior, malformed inputs, checksum failures,
checkpoint schema drift, and dynamic CPU export through a composed module. CUDA
capture/replay is additionally exercised when available. Raw logs and numerical
differences belong under `target/encoder-629`; performance remains unmeasured.

Local numerical checks on 2026-10-27 used the canonical weights:

| Comparison | Maximum absolute difference |
| --- | ---: |
| CPU encoder vs Micro per-row formula, batches 1 and 4 | `0` |
| CPU exported composite vs eager, batches 1 and 4 | `0` |
| Actual Micro Stage 1A CUDA capture vs encoder, batch 1 | `0` |
| Actual Micro Stage 1A CUDA capture vs encoder, batch 4 | `4.48e-7` |
| CUDA Graph replay vs eager composite | `0` |
| Rust vs canonical Micro CPU reference, all feature blocks | `1.41e-6` |

These are fixture-specific numerical observations, not performance or downstream
quality measurements. The actual private Micro file was exercised read-only;
its Stage 1A source SHA-256 was
`6c785a1ccc1c16b6c12c8367b79c01df26de61ed40cbeedc5fce78e3612e9a1a`.
CI uses the self-contained per-row formula reference, so it does not depend on a
private Micro checkout. Micro's vendored full-model source matched the pinned
DOPE model source byte-for-byte, and its 56 multitask `.pt` tensors matched the
released SafeTensors tensors exactly.
