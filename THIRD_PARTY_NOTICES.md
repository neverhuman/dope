# Third-party notices

DOPE's own code is MIT licensed. That license does not relicense dependencies,
optional research tools, datasets or model weights. This document identifies
important boundaries; release bundles must carry their actual dependency notices
and the complete pinned license texts. Native notices are shipped in
[scripts/native-third-party.NOTICE](scripts/native-third-party.NOTICE) and verified by the
`dope-native-notices` tool. Cargo.lock and the Python requirements identify the
resolved dependency versions; release SBOMs belong with the release artifacts.

| Component | Declared terms | Scope |
|---|---|---|
| NumPy, SciPy, pandas, scikit-learn | BSD family | Python research runtime; retain distribution notices |
| Rich | MIT | Python CLI presentation |
| PyTorch | BSD-3-Clause and bundled notices | Optional training/runtime dependency |
| CatBoost, XGBoost | Apache-2.0 | Candidate auditors; qualification still required |
| LightGBM, RealMLP/pytabkit, InterpretML/EBM | MIT | Candidate auditors; qualification still required |
| Matplotlib | Matplotlib license and bundled notices | Paper/figure tooling |
| [synthpop](https://cran.r-project.org/web/packages/synthpop/index.html) | GPL-2 or GPL-3 | Optional research adapter, outside the MIT product runtime |

Auditor and model terms are separate for source code and weights. All TDL
candidates in [auditors.json](research/transfer/registries/auditors.json) are
specified and unqualified. Declared upstream terms are metadata; D-07 admission
requires pinned texts, dependency closure, native parity, resources and an owner
license decision. This change acquires no model weights.

| Planned model | Code / weight boundary | Primary source |
|---|---|---|
| Kumo Tabular | Structured Data Models code: Apache-2.0; weights: OpenMDW-1.1 | [code license](https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE), [model card](https://huggingface.co/nvidia/Kumo-Tabular) |
| Mitra-v2 | AutoGluon code and declared model card terms: Apache-2.0; exact snapshot review pending | [model card](https://huggingface.co/autogluon/mitra-classifier-2) |
| TabICLv2 | Declared BSD-3-Clause for code/model distribution; exact checkpoint license review pending | [code license](https://github.com/soda-inria/tabicl/blob/main/LICENSE), [model card](https://huggingface.co/jingang/TabICL) |
| TabPFN-3.x | Apache-2.0 code does not grant rights to the separately licensed non-commercial weights | [upstream license guidance](https://github.com/PriorLabs/TabPFN), [weight license](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE) |
| LimiX | Code and model weight terms require separate snapshot review; model license adds attribution/naming terms | [upstream](https://github.com/limix-ldm-ai/LimiX), [model card](https://huggingface.co/stable-ai/LimiX-2M) |
| TabDPT | Exact code and weight terms pending review | [upstream](https://github.com/layer6ai-labs/TabDPT) |
| TabFM | Apache-2.0 code; separate non-commercial, non-production weight license | [upstream](https://github.com/google-research/tabfm), [model card](https://huggingface.co/google/tabfm-1.0.0-jax) |

No TabPFN-derived weights are distributed by this PR. Weight redistribution,
commercial use and product incorporation require the applicable model license;
MIT repository licensing alone grants none of those permissions.

Dataset rights are per source. The retained private-catalog inventory includes
an entry marked **non-commercial purposes only**. It is excluded from public
headline and product distribution until rights are reviewed. An inventory entry
is not permission to download, train, redistribute, or include rows in a release.
Unknown rights and pretraining contamination remain unresolved in the
[dataset manifest](research/transfer/manifests/datasets.json). PMLB is a mechanism
lab, with each upstream dataset's own terms; it is not an observational headline
cohort. No dataset downloads or result deletions are part of the tree scrub.
