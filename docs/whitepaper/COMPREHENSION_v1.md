# Page-1 comprehension check, version 1 (frozen before review)

This check is frozen before any reviewer sees the rendered page 1 of the current source. It tests whether page 1 (title, abstract and Figure 1) communicates the paper. It is a readability gate, not evidence that any result is valid. Once a score is seen, nothing here changes; any successor is a new file with a new version number, and the scores under both versions are reported.

## Procedure

1. The reader is a fresh agent or person with no repository, source or conversation context. It receives only the 150 dpi PNG of page 1 of `dope-mfs.pdf` and the five questions below. The anonymous PDF works the same way.
2. The reader has five minutes and answers each question in at most two sentences, without seeing the answer key.
3. A second party scores each answer against the key as **correct**, **partly correct** or **wrong**. That party is not the author of the page; the paper reviewer (`dope_paper_grok`) qualifies. The reader's text, the PNG hash and the scores go into the visual-QA receipt.

## Questions and answer key

| # | Question | A correct answer states |
| --- | --- | --- |
| Q1 Object | What does DOPE produce from a table? | A small sampling program (artifact) compiled from one numeric regression table; new rows are sampled from the artifact alone. |
| Q2 Preserved behavior | What is it trying to preserve, and how is that measured? | The table's predictive structure: how much of a real-trained learner's gain a learner trained on synthetic rows recovers on real validation rows (retention). It is not a general store of knowledge. |
| Q3 Comparison | How does DOPE compare with the strongest generator under CatBoost at 4n? | The paired difference with TabSyn (scaled schedule) has an interval that contains zero, so the two cannot be separated. That is not a win, and it is not equivalence. |
| Q4 Size meaning | What does the byte axis show, and what does it not show? | Every byte that sampling reads, with DOPE's median under the 10,240-byte cap, about four orders of magnitude below TabSyn. Smallness alone shows nothing about learning. |
| Q5 Limitations | Name one limitation stated on page 1. | Any one of: estimates are on validation rows and selection-exposed (profile chosen on them); the official test split is reserved, not run; Forest-Flow is a separate six-lineage block on which DOPE is below; resampled real rows stay above DOPE. |

## Pass rule

- **Pass:** at least four correct answers, Q3 and Q5 both correct, and no answer that claims superiority over TabSyn, equivalence, privacy, or a general knowledge or intelligence measure.
- **Fail:** anything else. A fail triggers a page-1 redesign (figure, caption or abstract wording) and a fresh reader. It never triggers a weaker key.

Wording, figure and caption changes made after a fail must keep every number generated and every claim inside `claim-registry.json`.
