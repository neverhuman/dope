# Historical binder recovery

The scalar binders, toy fixture tests and evaluator metadata were recovered exactly
from commit 0dae88be6ab574bbc8fe022a5f6f0ed23b968c54. They preserve historical source
identities and the original development-validation evidence. They supply no new
numerical replay, independent empirical reproduction, current model admission,
privacy assurance, sealed access or release authorization.

Run only the three `test_bind_privacy_cells`, `test_bind_tabsyn_cells` and
`test_tabsyn_evaluator_replay` modules on their public toy fixtures. The real replay
entry is retained historical source and is not wired into automatic CI. Do not
reconstruct missing raw partitions, workers or custody locations to run it.
