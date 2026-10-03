# DP-CTGAN author-source admission audit

DP-CTGAN remains `source_audit_pending`. No runtime or fit/sample job is admitted;
no author modules were imported or executed. Official tests stay sealed and
MFS-v2/PTF-v1/release/superiority remain null.

The [author paper](https://ml-research.github.io/papers/fang2022dpctgan.pdf) links
the `DP` branch of [juliecious/CTGAN](https://github.com/juliecious/CTGAN/tree/DP).
That branch is pinned at `4fb93e7727fbe400c28d642111347d9b517004d4`, with verified
MIT root source license. The bundled TensorFlow accountant carries an
Apache-2.0 notice, bound separately in the audit; the selected source license
expression records both. Full third-party/runtime rights remain pending. [dpctgan-source.audit.json](dpctgan-source.audit.json) binds
27 selected source/configuration files to Git blobs and SHA256. Selection is
not a complete executable or third-party rights inventory. Citation: `fang2022dp`.

## Defaults and author-native outcome

The constructor declares batch 500, embedding 128, two 256-unit generator and
discriminator layers, clipping coefficient 0.1, noise multiplier 1, epsilon 3
and delta 1e-5. Its nominal 300 epochs argument does not bound `fit`: training
uses an epoch-end privacy-accountant loop. Record actual spent epsilon and any
overshoot; the external 600-second whole-phase timer remains required. Protocol
epsilon/delta comparisons remain separate.

The paper measures classifier-averaged AUROC and AUPRC, maximized. Released
classification evaluation also returns accuracy, F1 and multiclass log loss.
No single validation scalar/grid/ties or author regression objective is frozen.
Regression cells use author defaults with tuning inapplicable unless an author
regression objective is independently verified before selection. Our shared
retention KPI supplies no substitute tuning objective.

## Artifact, privacy and runtime gates

The original sampler retains transformed training rows and reads them for
categorical generation. The original save method pickles the entire object.
Verify a safe inference codec, removal of original rows, all charged model/
transformer/projection/configuration bytes and artifact-only replay. Any source
adaptation or change of categorical conditioning needs an explicit version and
label. Fit/sample wrappers must bind NumPy/PyTorch seeds and generation mode.

Static inspection finds per-parameter noise assigned before the following
`optimizerD.zero_grad()`, backward pass and step. This ordering does not establish
a privatized discriminator update. The normal standard deviation argument is
squared, and preprocessing/selection accounting and mechanism/accountant
equivalence remain unverified. No formal DP claim, patched mechanism or privacy
reproduction is admitted.

The declared package requires Python >=3.6,<3.9 and older NumPy/pandas/PyTorch/RDT
ranges. Full interpreter/import-root/bytecode/library rights/runtime custody and
fresh CPU/RAM/scratch/GPU admission remain pending. Data, notebooks, figures,
bytecode and weights were not opened; no reported experiment is our reproduction.
