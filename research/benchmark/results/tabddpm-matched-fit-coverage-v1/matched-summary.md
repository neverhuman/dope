# Matched retained TabDDPM measurements

Training-derived validation only. Average the three sample seeds within each lineage before
forming a paired dataset summary. Fit seed 11 only; no confidence interval or superiority claim.

| TabDDPM selection | Reference | Size | Metric | Supported lineages | TabDDPM median | Reference median | Median paired difference |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| author_default | ARF | 1 | catboost_retention | 10 | 0.8754008762421992 | 0.7697690119503235 | 0.025378007641695233 |
| author_default | ARF | 1 | marginal_error_mean | 10 | 0.12801534536233705 | 0.13516613227712812 | 0.004816919191919199 |
| author_default | ARF | 1 | c2st_catboost_auc | 10 | 0.578936522582356 | 0.6171527798294515 | -0.01984876557193449 |
| author_default | ARF | 1 | distance_mia_auc | 10 | 0.5312792139077853 | 0.5118298108095031 | 0.02054810809503077 |
| author_default | ARF | 4 | catboost_retention | 10 | 0.9436357502507039 | 0.8388514608153798 | 0.03494885431313033 |
| author_default | ARF | 4 | marginal_error_mean | 10 | 0.11754378097004689 | 0.12897958927134812 | 0.0033406780350917487 |
| author_default | ARF | 4 | c2st_catboost_auc | 10 | 0.5950422455630788 | 0.5809598891390686 | -0.045766349924991234 |
| author_default | ARF | 4 | distance_mia_auc | 10 | 0.5258677190690473 | 0.5627434903047092 | 0.0036946820146744885 |
| author_default | TabSyn | 1 | catboost_retention | 10 | 0.8754008762421992 | 0.8853451077627061 | -0.03925427255488628 |
| author_default | TabSyn | 1 | marginal_error_mean | 10 | 0.12801534536233705 | 0.1381623437031813 | -0.014298066373895926 |
| author_default | TabSyn | 1 | c2st_catboost_auc | 10 | 0.578936522582356 | 0.6251044759689397 | -3.825592454859139e-05 |
| author_default | TabSyn | 1 | distance_mia_auc | 10 | 0.5312792139077853 | 0.5782863567649281 | -0.06129999999999994 |
| author_default | TabSyn | 4 | catboost_retention | 8 | 0.9436357502507039 | 0.9112758346655363 | -0.015404731733389243 |
| author_default | TabSyn | 4 | marginal_error_mean | 8 | 0.11754378097004689 | 0.13599837200611115 | -0.01678117772346354 |
| author_default | TabSyn | 4 | c2st_catboost_auc | 8 | 0.5950422455630788 | 0.669789460324677 | -0.02549826029248753 |
| author_default | TabSyn | 4 | distance_mia_auc | 8 | 0.5258677190690473 | 0.571424074074074 | -0.04361886336559864 |
| native_selected | ARF | 1 | catboost_retention | 12 | 0.9149074674658605 | 0.47556855828820627 | 0.3185411227003236 |
| native_selected | ARF | 1 | marginal_error_mean | 12 | 0.10386654116080118 | 0.1348869989972863 | 0.0014248162832003544 |
| native_selected | ARF | 1 | c2st_catboost_auc | 12 | 0.5843468604175692 | 0.6161242377420745 | -0.04611717865059867 |
| native_selected | ARF | 1 | distance_mia_auc | 12 | 0.5514655791525598 | 0.5404768471476267 | 0.03642248384118188 |
| native_selected | ARF | 4 | catboost_retention | 12 | 0.9738846348524721 | 0.6054875158617974 | 0.30148460588738024 |
| native_selected | ARF | 4 | marginal_error_mean | 12 | 0.0990217266021208 | 0.13526959759003318 | -0.004763777850575329 |
| native_selected | ARF | 4 | c2st_catboost_auc | 12 | 0.5797380295760863 | 0.6212442535573286 | -0.034296102235896575 |
| native_selected | ARF | 4 | distance_mia_auc | 12 | 0.5712534626038781 | 0.5380001466333243 | 0.06288333333333335 |
| native_selected | TabSyn | 1 | catboost_retention | 11 | 0.937106667464301 | 0.8365100778296714 | 0.03880114786883426 |
| native_selected | TabSyn | 1 | marginal_error_mean | 11 | 0.11165283540802214 | 0.13950133376884968 | -0.01777254210882527 |
| native_selected | TabSyn | 1 | c2st_catboost_auc | 11 | 0.570209485730319 | 0.6486095251301194 | -0.10172084943553661 |
| native_selected | TabSyn | 1 | distance_mia_auc | 11 | 0.5581481481481482 | 0.5859 | -0.036633333333333296 |
| native_selected | TabSyn | 4 | catboost_retention | 8 | 0.9894678389958063 | 0.9112758346655363 | 0.06025594609461621 |
| native_selected | TabSyn | 4 | marginal_error_mean | 8 | 0.09672848165262585 | 0.13599837200611115 | -0.019102580001513998 |
| native_selected | TabSyn | 4 | c2st_catboost_auc | 8 | 0.5778624483671718 | 0.669789460324677 | -0.09731241730291729 |
| native_selected | TabSyn | 4 | distance_mia_auc | 8 | 0.5712534626038781 | 0.571424074074074 | -0.0005833333333332691 |
