# Data

Every raw file needed to rerun the experiments is in `data/raw/`.

| Dataset | File | Source |
|---|---|---|
| Adult | `raw/adult/adult.data`, `raw/adult/adult.test` | UCI Machine Learning Repository |
| COMPAS | `raw/compas/compas-scores-two-years.csv` | ProPublica, `propublica/compas-analysis` |
| HSLS | `raw/hsls/hsls_df_knn_impute_past_v2.pkl` | Authors' official repository (see below) |
| HSLS codebooks | `raw/hsls/*.txt` | NCES HSLS:09 documentation (reference only) |

## Preprocessing

`src/data/datasets.py` rebuilds the exact tables used in the paper. Its
output matches the authors' cleaned files (`adult_cleaned.csv`,
`compas_cleaned_labeled.csv`, `hsls_cleaned.csv`) value for value:

| Dataset | Rows | Model inputs (incl. intercept) | Sensitive attribute | Target |
|---|---|---|---|---|
| Adult | 46,447 | 21 | race: White (1) / Black (0) | income > $50K |
| COMPAS | 5,278 | 7 | race: Caucasian (1) / African-American (0) | is_recid |
| HSLS | 14,509 | 59 | racebin: White/Asian (1) / other (0) | gradebin |

`load_adult`, `load_compas` and `load_hsls` (in `src/data/`) return a random
80/20 split with every row scaled to norm <= 1 and an intercept column, as
in the official code.

## HSLS file and attribution

`hsls_df_knn_impute_past_v2.pkl` is the preprocessed High School Longitudinal
Study table used by Oesterling et al. (2024). It derives from the NCES HSLS:09
public-use data, with the preprocessing and k-NN imputation of Jeong, Wang and
Calmon, "Fairness without Imputation: A Decision Tree Approach for Fair
Prediction with Missing Values" (AAAI 2022).

It is copied unmodified from the official code release:

> Oesterling, Ma, Calmon, Lakkaraju. *Fair Machine Unlearning: Data Removal
> while Mitigating Disparities.* AISTATS 2024.
> https://github.com/AI4LIFE-GROUP/fair-unlearning
> Licensed under the Apache License, Version 2.0.

The preprocessing in `src/data/datasets.py` reimplements the official
`prepare_adult.py`, `prepare_compas.py`, `prepare_hsls.py` and
`Dataloader.py` from that repository.
