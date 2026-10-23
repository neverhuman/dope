# Public v2 operator fixtures

These tiny graphs contain only owned synthetic engineering data from
`research.levels.fixtures`; they are not trained production models or private
dataset exports. Each graph receives one ordinary and one frozen-base residual
optimizer update. The graph width is four, with two ordinary and two residual
queries for slot graphs. The code widths remain the full v2 contract.

`slot`, `matched` and `pooled` cover the original slot readout, parameter-matched pooled
readout and width-only pooled control respectively. The corresponding `-bounded`
directories cover separately versioned tanh readouts and range-refusing readers;
they do not replace the original fixtures. `fixture.json` binds the
generator source fingerprint, public schema and operator tolerance. Each model
manifest binds its exact encoder and decoder bytes. Python generated the stored
FP16 code and FP32 decoder operators; native tests independently replay every
tier, exact prefix exports and decoder-only recipient sampling.

Regenerate into a fresh ignored directory using the pinned reference runtime:

```sh
python3 -B -m research.levels.fixtures --graph slot-code-diffusion-v2 --out target/slot-fixture
python3 -B -m research.levels.fixtures --graph pooled-control-code-diffusion-v2 --out target/matched-fixture
python3 -B -m research.levels.fixtures --graph pooled-code-diffusion-v2 --out target/pooled-fixture
```

Retain the generated optimizer checkpoint in ignored private evidence, outside
this fixture tree. Copy only model, parity, fixture and selected code artifacts
after conformance review. Fixtures do not qualify model utility or a tier claim.
