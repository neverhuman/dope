# DOI and URL check

Recorded 2026-10-06 from the bibliography on `paper-ieee-w1`.

Each DOI was requested from `https://doi.org/` with `Accept: application/vnd.citationstyles.csl+json`. A 200 response with citation JSON means the DOI resolved. Publisher HTML was not used: several of those sites return 403 to a script after doi.org has already redirected.

An earlier Crossref bibliographic search returned HTTP 429 and, when it answered, the wrong top hit. Those hits were not written into the bibliography.

Unique DOI fields: 34. Resolved: 34. Unresolved: 0.

| DOI | Status |
| --- | --- |
| `10.1109/DSAA.2016.49` | 200 |
| `10.1109/TIT.1968.1054142` | 200 |
| `10.18637/jss.v074.i11` | 200 |
| `10.1186/s13040-017-0154-4` | 200 |
| `10.1093/bioinformatics/btab727` | 200 |
| `10.3389/fdata.2023.1296508` | 200 |
| `10.48550/ARXIV.2302.02041` | 200 |
| `10.14778/3231751.3231757` | 200 |
| `10.1145/3442381.3449999` | 200 |
| `10.52202/075280-2005` | 200 |
| `10.1007/978-981-96-8186-0_20` | 200 |
| `10.1145/2588555.2588573` | 200 |
| `10.29012/jpc.778` | 200 |
| `10.14778/3551793.3551817` | 200 |
| `10.14778/3476249.3476272` | 200 |
| `10.29012/jpc.748` | 200 |
| `10.1561/0400000042` | 200 |
| `10.1145/2976749.2978318` | 200 |
| `10.1109/sp.2017.41` | 200 |
| `10.2478/popets-2019-0008` | 200 |
| `10.1111/rssa.12358` | 200 |
| `10.29012/jpc.v7i3.407` | 200 |
| `10.1145/3339252.3339281` | 200 |
| `10.1038/s41586-024-08328-6` | 200 |
| `10.52202/068431-0037` | 200 |
| `10.1016/j.inffus.2021.11.011` | 200 |
| `10.1145/3769764` | 200 |
| `10.1016/j.neucom.2025.131655` | 200 |
| `10.1016/j.neucom.2022.04.053` | 200 |
| `10.1186/s40537-023-00792-7` | 200 |
| `10.1109/tnnls.2022.3229161` | 200 |
| `10.7551/mitpress/1114.003.0004` | 200 |
| `10.1214/aos/1176346150` | 200 |
| `10.1016/j.neucom.2019.12.136` | 200 |

Entries that had a URL and no DOI were requested directly. All 14 returned HTTP 200: seven OpenReview forum pages, two NeurIPS abstracts, the SDMetrics docs page, two JMLR papers, and one arXiv abstract. OpenReview and JMLR entries stay URL-only. No DOI was invented for them.

Eleven entries had neither a DOI nor a URL. Each URL below was taken from the venue page whose title matched the bib entry, then added:

| Key | Locator |
| --- | --- |
| `xu2019ctgan` | NeurIPS 2019 abstract, title "Modeling Tabular data using Conditional GAN" |
| `prokhorenkova2018catboost` | NeurIPS 2018 abstract, title "CatBoost: unbiased boosting with categorical features" |
| `watson2023arf` | `proceedings.mlr.press/v206/watson23a.html` |
| `kotelnikov2023tabddpm` | `proceedings.mlr.press/v202/kotelnikov23a.html` |
| `lee2023codi` | `proceedings.mlr.press/v202/lee23i.html` |
| `jolicoeurmartineau2024forestdiffusion` | `proceedings.mlr.press/v238/jolicoeur-martineau24a.html` |
| `zhao2021ctabgan` | `proceedings.mlr.press/v157/zhao21a.html` |
| `vanbreugel2023domias` | `proceedings.mlr.press/v206/breugel23a.html` |
| `alaa2022faithful` | `proceedings.mlr.press/v162/alaa22a.html` |
| `choi2017medgan` | `proceedings.mlr.press/v68/choi17a.html` |
| `zagardo2026gatd` | `proceedings.mlr.press/v306/zagardo26a.html` |

`chawla2002smote` was added because the manuscript names SMOTE. doi.org returned CSL JSON for `10.1613/jair.953`: Journal of Artificial Intelligence Research, volume 16, pages 321-357, 2002. The author initials are the ones in that record.

Every bib entry has an author, a title, a year, a venue, and either a DOI or a URL. Pages are present when the venue record states them. ICLR OpenReview papers have no pages in the bib, and none were filled in.

## 2026-10-06 later pass on `related.bib`

The 21 entries that still had no DOI were checked again. A DOI was written only when `https://doi.org/` returned citation JSON whose title and authors match the bib entry. Crossref's bibliographic top hit was not used: it returned unrelated papers for most of these titles.

Proceedings DOIs, each HTTP 200:

| Key | DOI | Record |
| --- | --- | --- |
| `qian2023synthcity` | `10.52202/075280-0140` | NeurIPS 36, 2023, Qian, Davis, van der Schaar |
| `mcelfresh2023tabzilla` | `10.52202/075280-3337` | NeurIPS 36, 2023, McElfresh and coauthors |
| `erickson2026tabarena` | `10.52202/085713-0519` | NeurIPS 38 proceedings article, 2025, Erickson and coauthors. Same title and authors as the bib entry |

arXiv DOIs, each HTTP 200, used because the venue page does not register its own DOI:

| Key | DOI |
| --- | --- |
| `zhang2024tabsyn` | `10.48550/ARXIV.2310.09656` |
| `kim2023stasy` | `10.48550/ARXIV.2210.04018` |
| `lee2023codi` | `10.48550/ARXIV.2304.12654` |
| `shi2025tabdiff` | `10.48550/ARXIV.2410.20626` |
| `jolicoeurmartineau2024forestdiffusion` | `10.48550/ARXIV.2309.09968` |
| `mueller2025cdtd` | `10.48550/ARXIV.2312.10431` |
| `zagardo2026gatd` | `10.48550/ARXIV.2606.02607` |
| `borisov2023great` | `10.48550/ARXIV.2210.06280` |
| `choi2017medgan` | `10.48550/ARXIV.1703.06490` |
| `vanbreugel2023domias` | `10.48550/ARXIV.2302.12580` |
| `alaa2022faithful` | `10.48550/ARXIV.2102.08921` |
| `han2016deepcompression` | `10.48550/ARXIV.1510.00149` |

Venue-only, marked in `related.bib`. No DOI was written:

| Key | Why |
| --- | --- |
| `zhao2021ctabgan` | PMLR page has no DOI. The arXiv record adds an author who is not in the PMLR entry |
| `ma2023tabpfgen` | 2023 workshop page has no DOI. A 2024 arXiv preprint was not substituted |
| `liu2023goggle` | OpenReview page has no DOI, and no resolving record matched the title |
| `zhang2020sdmetrics` | Software documentation. No DOI |
| `demsar2006statistical` | JMLR page has no DOI |
| `benavoli2017time` | JMLR page has no DOI. The 2016 preprint DOI was not substituted for the journal article |
