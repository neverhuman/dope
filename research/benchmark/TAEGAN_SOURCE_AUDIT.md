# TAEGAN source availability

TAEGAN is unavailable for this study at author commit
`19d17f65cfa52ad5d84c0845830ef7c418e4397a`.
The [published paper](https://proceedings.mlr.press/v304/li26c.html) links the
[author repository](https://github.com/BetterdataLabs/taegan). Its complete pinned
tree has no license/notice file. The inspected README and four Python files
contain no verified grant; GitHub license metadata is null and the pinned license
endpoint returns 404. All five selected files match their Git blobs and SHA256.
Demo rows, tensors, figures, bytecode and weights were not opened; no author
modules were imported or executed.

The author README says enterprise preprocessing and inverse recovery are
withheld. Static AST inspection confirms that `prepare` and `recover` contain
only docstrings and ellipsis stubs. Generated tensors alone do not establish a
complete tabular fit/sample contract. The suggestion to reconstruct preprocessing
from CTAB-GAN+ supplies no verified rights grant or reproduced implementation
for this study.

The [unavailable receipt](results/taegan-unavailable.json) and
[schema](results/taegan-unavailable.schema.json) bind the evidence. Unavailable
cells contribute no DOPE win and are excluded from the win denominator. Future
admission needs an explicit rights grant and complete pipeline, or a licensed
independent implementation reproducing two reported experiments, followed by
runtime, artifact-only fit/sample and author-native validation selection gates.

Official tests remain sealed. MFS-v2, PTF-v1, release-safe L3 and paired
superiority remain null. Citation: `li2025taegan` from the corrected bibliography;
that key retains the earlier preprint title, while the published paper is titled
“TAEGAN: Revisit GANs for Tabular Data Generation”. No reported result is our
reproduction evidence.
