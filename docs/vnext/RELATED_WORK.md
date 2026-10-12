# Related work for the vNext draft

This note scopes prior art for the numeric-regression transfer draft. It does not rank methods and it does not carry measurements. Bibliographic years live only in `docs/whitepaper/vnext/references.bib`.

## What the compiler is compared with

The historical neural anchors are conditional generative adversarial networks and variational autoencoders for tables (Xu, Skoularidou, Cuesta-Infante, and Veeramachaneni). Tabular diffusion in the Kotelnikov, Baranchuk, Rubachev, and Babenko construction is a separate neural recipe. Adversarial random forests (Watson, Blesch, Kapar, and Wright) are a structured generator. Dependence trees (Chow and Liu) and pair-copula constructions (Aas, Czado, Frigessi, and Bakken) are circuit baselines. synthpop (Nowok, Raab, and Dibben) is a tree synthesizer, not an interchangeable joint model. CatBoost (Prokhorenkova and colleagues) is an example downstream learner, not a definition of utility.

The Penn Machine Learning Benchmarks collection (Olson and colleagues; Romano and colleagues) is a historical source pool. A new split name on that pool is not a fresh source family.

TabSyn, TabDiff, and Forest-Flow are adapter candidates named in the release specification. This draft does not score them.

## Alternatives to generation

Four uses of the same byte constraint are predeclared controls, not leaderboard gaps.

- A direct predictor ships the development predictor instead of a generator.
- A coreset ships compressed real rows.
- Distillation ships synthetic prototypes with the labels, weights, and recipe they require.
- A circuit ships a symbolic dependence program.

If a control accounts for the withheld change, the generator hypothesis narrows. The control row stays in the claim ladder either way.

Stadler, Oprisanu, and Troncoso argue that synthetic release is not de-identification. That limit is why this draft refuses a de-identification conclusion.

## Non-claims

This draft does not claim a state-of-the-art result.
This draft does not claim equivalence.
This draft does not claim a privacy result, a differential-privacy guarantee, or a HIPAA determination.
This draft does not claim a causal result.
This draft does not claim a pretraining result.
This draft does not claim to be the first method of this kind.
