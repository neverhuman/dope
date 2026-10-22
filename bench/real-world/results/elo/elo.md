# Budgeted transfer Elo

Selection-exposed development evidence; no sealed-test or D80 claim.

BTR-AUC: area under clipped worst-family retention versus log byte budget (0–100).
This is the envelope of tested configurations; it does not predict validation-selected deployment performance.
Common complete cohort: 7/16 datasets. Tie margin: 1 BTR-AUC points. 95% intervals resample datasets.

| Alternative | Elo [95% interval] | Mean BTR-AUC | W / T / L |
|---|---:|---:|---:|
| DOPE | 1837 [1787, 1912] | 92.9 | 36 / 3 / 3 |
| Real rows (xz) | 1825 [1737, 1948] | 91.9 | 36 / 2 / 4 |
| DOPE research kernel | 1640 [1520, 1766] | 62.4 | 26 / 5 / 11 |
| ARF | 1398 [1326, 1458] | 17.7 | 13 / 6 / 23 |
| Gaussian copula | 1380 [1289, 1459] | 17.4 | 12 / 6 / 24 |
| TVAE | 1253 [1180, 1309] | 9.4 | 4 / 9 / 29 |
| CTGAN | 1167 [1043, 1263] | 2.8 | 0 / 9 / 33 |

![Elo ratings and paired dataset intervals](elo.svg)

## Individual dataset rankings

Local Elo uses only the alternatives complete on that dataset. Its 1500 center is local, so compare alternatives within a row group, not scores across datasets.

![Per-dataset local Elo with missing alternatives explicit](datasets.svg)

| Dataset | Alternative | BTR-AUC | Local Elo | Status |
|---|---|---:|---:|---|
| abalone | dope | 97.9 | 1718 | complete |
| abalone | kernel | 94.5 | 1642 | complete |
| abalone | rows | 79.1 | 1570 | complete |
| abalone | copula | 57.5 | 1500 | complete |
| abalone | arf | 31.9 | 1430 | complete |
| abalone | tvae | 28.7 | 1358 | complete |
| abalone | ctgan | 19.3 | 1282 | complete |
| adult | dope | 88.7 | 1669 | complete |
| adult | kernel | 82.4 | 1582 | complete |
| adult | rows | 75.9 | 1500 | complete |
| adult | arf | 18.9 | 1418 | complete |
| adult | copula | 11.6 | 1331 | complete |
| adult | ctgan | — | — | incomplete |
| adult | tvae | — | — | incomplete |
| airfoil | dope | 95.8 | 1679 | complete |
| airfoil | rows | 95.4 | 1679 | complete |
| airfoil | kernel | 76.4 | 1570 | complete |
| airfoil | arf | 35.2 | 1500 | complete |
| airfoil | copula | 8.4 | 1431 | complete |
| airfoil | tvae | 4.6 | 1359 | complete |
| airfoil | ctgan | 0.0 | 1282 | complete |
| bank | dope | 84.7 | 1669 | complete |
| bank | rows | 64.1 | 1582 | complete |
| bank | kernel | 60.9 | 1500 | complete |
| bank | arf | 17.4 | 1418 | complete |
| bank | copula | 5.0 | 1331 | complete |
| bank | ctgan | — | — | incomplete |
| bank | tvae | — | — | incomplete |
| bike | dope | 93.8 | 1669 | complete |
| bike | rows | 81.6 | 1582 | complete |
| bike | kernel | 46.9 | 1500 | complete |
| bike | copula | 19.2 | 1418 | complete |
| bike | arf | 15.4 | 1331 | complete |
| bike | ctgan | — | — | incomplete |
| bike | tvae | — | — | incomplete |
| california | dope | 93.2 | 1669 | complete |
| california | kernel | 82.4 | 1582 | complete |
| california | rows | 77.0 | 1500 | complete |
| california | copula | 47.9 | 1418 | complete |
| california | arf | 18.7 | 1331 | complete |
| california | ctgan | — | — | incomplete |
| california | tvae | — | — | incomplete |
| car | rows | 97.7 | 1712 | complete |
| car | dope | 96.3 | 1636 | complete |
| car | arf | 0.0 | 1430 | complete |
| car | copula | 0.0 | 1430 | complete |
| car | ctgan | 0.0 | 1430 | complete |
| car | kernel | 0.0 | 1430 | complete |
| car | tvae | 0.0 | 1430 | complete |
| concrete | rows | 92.0 | 1717 | complete |
| concrete | dope | 87.0 | 1640 | complete |
| concrete | kernel | 45.2 | 1568 | complete |
| concrete | arf | 4.8 | 1499 | complete |
| concrete | copula | 0.0 | 1358 | complete |
| concrete | ctgan | 0.0 | 1358 | complete |
| concrete | tvae | 0.0 | 1358 | complete |
| covertype | rows | 61.6 | 1666 | complete |
| covertype | dope | 26.6 | 1581 | complete |
| covertype | arf | 0.0 | 1418 | complete |
| covertype | copula | 0.0 | 1418 | complete |
| covertype | kernel | 0.0 | 1418 | complete |
| covertype | ctgan | — | — | incomplete |
| covertype | tvae | — | — | incomplete |
| credit | dope | 90.1 | 1669 | complete |
| credit | kernel | 81.7 | 1582 | complete |
| credit | rows | 61.3 | 1500 | complete |
| credit | copula | 12.8 | 1418 | complete |
| credit | arf | 0.0 | 1331 | complete |
| credit | ctgan | — | — | incomplete |
| credit | tvae | — | — | incomplete |
| energy | dope | 99.1 | 1679 | complete |
| energy | rows | 99.9 | 1679 | complete |
| energy | kernel | 53.6 | 1570 | complete |
| energy | arf | 48.4 | 1500 | complete |
| energy | tvae | 32.9 | 1431 | complete |
| energy | copula | 27.5 | 1359 | complete |
| energy | ctgan | 0.0 | 1282 | complete |
| german | rows | 86.5 | 1717 | complete |
| german | dope | 77.1 | 1640 | complete |
| german | kernel | 70.3 | 1568 | complete |
| german | copula | 23.2 | 1499 | complete |
| german | arf | 0.0 | 1358 | complete |
| german | ctgan | 0.0 | 1358 | complete |
| german | tvae | 0.0 | 1358 | complete |
| mushroom | rows | 98.3 | 1669 | complete |
| mushroom | dope | 94.8 | 1582 | complete |
| mushroom | kernel | 93.2 | 1500 | complete |
| mushroom | arf | 40.2 | 1418 | complete |
| mushroom | copula | 23.4 | 1331 | complete |
| mushroom | ctgan | — | — | incomplete |
| mushroom | tvae | — | — | incomplete |
| news | dope | 95.5 | 1669 | complete |
| news | copula | 46.9 | 1582 | complete |
| news | rows | 28.1 | 1500 | complete |
| news | kernel | 24.0 | 1418 | complete |
| news | arf | 0.0 | 1331 | complete |
| news | ctgan | — | — | incomplete |
| news | tvae | — | — | incomplete |
| student | dope | 96.9 | 1679 | complete |
| student | kernel | 96.9 | 1679 | complete |
| student | rows | 92.6 | 1570 | complete |
| student | copula | 5.0 | 1500 | complete |
| student | arf | 3.6 | 1430 | complete |
| student | ctgan | 0.0 | 1321 | complete |
| student | tvae | 0.0 | 1321 | complete |
| wine | dope | 75.7 | 1669 | complete |
| wine | rows | 65.0 | 1582 | complete |
| wine | kernel | 54.3 | 1500 | complete |
| wine | copula | 33.9 | 1418 | complete |
| wine | arf | 19.8 | 1331 | complete |
| wine | ctgan | — | — | incomplete |
| wine | tvae | — | — | incomplete |
