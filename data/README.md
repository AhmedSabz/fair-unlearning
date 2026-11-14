# Data

| Dataset | File | Source | In git |
|---|---|---|---|
| Adult | `raw/adult/adult.data`, `raw/adult/adult.test` | UCI Machine Learning Repository | yes |
| COMPAS | `raw/compas/compas-scores-two-years.csv` | ProPublica, `propublica/compas-analysis` | yes |
| HSLS (authors' table) | `raw/hsls/hsls_df_knn_impute_past_v2.pkl` | Authors' official repository (see below) | yes |
| HSLS (NCES original) | `raw/hsls/nces_2016_public_use/hsls_16_student_v1_0.csv` | NCES HSLS:09 public-use files, 2016 release | **no** (~800 MB, gitignored) |
| HSLS codebooks | `raw/hsls/*.txt` | NCES HSLS:09 documentation | yes |

Everything needed for the paper's experiments is in git. The NCES file is
needed only for the independently built HSLS table (below). Download the
HSLS:09 public-use student data from NCES
(https://nces.ed.gov/surveys/hsls09/) and place it at the path above; the
school-level file is not used.

## Preprocessing used for the paper's experiments

`src/data/datasets.py` rebuilds the tables used in the paper:

| Dataset | Rows | Model inputs (incl. intercept) | Sensitive attribute | Target |
|---|---|---|---|---|
| Adult | 46,447 | 21 | race: White (1) / Black (0) | income > $50K |
| COMPAS | 5,278 | 7 | race: Caucasian (1) / African-American (0) | is_recid |
| HSLS | 14,509 | 59 | racebin: White/Asian (1) / other (0) | gradebin: 9th-grade math test score above median |

`load_adult`, `load_compas` and `load_hsls` (in `src/data/`) return a random
80/20 split with every row divided by the largest training-row norm (so
training rows have norm <= 1 before the intercept; with the intercept the
norm is at most sqrt(2), and test rows can exceed 1) and an intercept
column, as in the official code.

### HSLS: equality with the authors' cleaned table (verified)

`prepare_hsls()` was compared with the authors' published
`datasets/HSLS/hsls_cleaned.csv` (official repository, commit `8d7a019`,
SHA-256 starting `58b5b8f5c4e2ec2a`):

- same 14,509 rows, aligned by row id (the index of both tables is the row
  position in the NCES student file), and the same 59 columns in the same
  order;
- `gradebin`, `racebin` and `sexbin` identical;
- every feature value equal to within 1.1e-16 (CSV rounding).

**Scope of this check:** `prepare_hsls()` starts from the authors' already
k-NN-imputed pickle, so the equality validates only the steps after it
(missing-code masking, deriving `gradebin`/`racebin`/`sexbin`, MinMax
scaling). It says nothing about how the pickle was built from NCES data; for
that, see the independent pipeline below.

To re-run the check, clone the official repository at that commit into
`external/fair-unlearning-official/` (`python -m validation.compare_official`
does this) and run `python -m pytest tests/test_reproduction.py`; the test
`test_prepare_hsls_matches_authors_cleaned_csv` is skipped when the clone is
absent. `prepare_adult()` and `prepare_compas()` were compared once with the
authors' `adult_cleaned.csv` and `compas_cleaned_labeled.csv` at the same
commit: same rows, index and column order, and 0 differing cells (975,387
and 36,946 cells). That check is not part of the automated tests.

## HSLS built independently from NCES data

`src/data/hsls_nces.py` builds HSLS from the NCES public-use student file
using only NCES documentation (the codebook in `raw/hsls/`). Entry points:
`load_hsls_nces()` returns the same dictionary as `load_dataset("hsls")`
(plus row ids and fitted parameters); `prepare_hsls_nces()` returns the
table. Tests: `tests/test_hsls_nces.py` (skipped when the NCES file is
absent).

**Rows.** Row ids are the 0-based row positions in the student file (sorted
by `STU_ID`); the authors' pickle uses the same ids.

| `row_set` | Rows | Rule | Role |
|---|---|---|---|
| `"principled"` | 14,843 | `X1SEX` and `X1RACE` present; base-year and first-follow-up math scores (`X1TXMSCR`, `X2TXMSCR`) present; base-year student and parent questionnaires completed (`X1SQSTAT` ≠ 8, `X1PQSTAT` ≠ 8) | default |
| `"s1m8grade"` | 14,584 | above, plus 8th-grade math grade `S1M8GRADE` present | diagnostic: this rule was found by matching the authors' rows, not derived independently |
| `"authors_ids"` | 14,509 | exactly the authors' rows (their pickle's index) | diagnostic control, not a filtering rule |

Requiring both questionnaires is equivalent to excluding rows with any
unit-non-response code (`-8`) in the features. Requiring `X2TXMSCR` follows
the authors' table, which carries an 11th-grade outcome; dropping that
requirement would give 16,429 rows.

**75 unexplained exclusions.** The authors' 14,509 rows are a subset of the
14,584. The remaining 75 students satisfy every rule above; in aggregate,
nearly all of their parent-questionnaire items are missing (at least 89% for
each of them; household size is NCES-imputed for 96% of them versus 7.5% of
the authors' rows), but
the authors kept 600–900 students with similarly sparse parent data, so no
documented rule explains the exclusion. We keep them. The 75 and the 259
students without `S1M8GRADE` differ from the authors' rows (label = 1 for 21%
and 28% versus 50%; White/Asian 48% and 57% versus 65.5%), so row selection
may affect fairness metrics.

**Features.** The same 56 base-year variables (`X1*`, `S1*`, `P1*`) as the
authors' table, so both sources give models the same inputs, plus `sexbin`
and `racebin`. The feature list is the one choice taken from the authors'
table rather than derived independently.

**Label.** `gradebin = 1[X1TXMTH > threshold]`. `X1TXMTH` is the base-year
(fall 2009, 9th grade) mathematics IRT theta score, an assessment score, not
a course grade. The threshold is the median over the training rows. NCES
imputed the score for 485 of the 14,843 selected students (3.3%,
`X1TXMATH_IM`); they are kept, as in the authors' table. Given the same rows and full-table fitting, this
reproduces the authors' labels exactly (their threshold: 0.1540).

**Sensitive attribute and sex.** `racebin = 1` for `X1RACE` in {2 Asian,
8 White}, else 0. `sexbin = 1` for `X1SEX = 2` (female), as in the authors'
table.

**Missing values.** Only the NCES reserve codes -9 to -3 (missing, unit
non-response, legitimate skip, not applicable, suppressed, not administered,
carry-through) become missing. `X1MTHID`, `X1MTHEFF` and `X1MTHINT` are
continuous scales with valid negative values (down to about -2.9); those are
data and are kept. The authors' table instead treats them as missing and
imputes them: among its 14,509 rows, 5,152 / 5,173 / 5,564 valid negative
scores of these three scales were replaced
(`test_authors_table_drops_negative_scale_scores` documents this).

**Category coding (unchanged from the authors).** Every feature is used as
one numeric column, as in the authors' table. That means nominal variables
(for example `X1PAR1RACE`, `P1USBORN1`, `X1P1RELATION`, `S1LANG1ST`) are
treated as ordered numbers, and some "don't know" answers are ordinary
positive category codes rather than reserve codes, so they are kept as values,
not imputed: `S1APCALC` = 4 and `S1IBCALC` = 4 ("don't know what AP/IB
calculus is"; 2,008 and 5,549 of the 14,843 students), `P1ELLEVER` = 3 (196)
and `P1EDUEXPECT` = 11 (1,367). We keep this coding so that both sources give
models identical feature representations; one-hot encoding these variables
would be a reasonable alternative but a separate change.

**Scaling and imputation.** The split is made first (below). Then MinMax
scaling and `sklearn.impute.KNNImputer` (k = 5, uniform weights,
nan-Euclidean distance) are fitted on the training rows only and applied to
all rows. The imputer uses the 58 model inputs (56 features, `sexbin`,
`racebin`), never the label. The authors' exact k-NN configuration is not
public (k = 5 on scaled data reproduces only about 18% of their imputed
values), so k = 5 (scikit-learn's default) is our own choice.

**Split.** `split_row_ids()` splits by row id before anything is fitted.
Students who are also in the authors' table get exactly the assignment that
`load_dataset("hsls")` gives them; the others go to the test set when a
seeded per-row random key is below `test_size` (with the defaults: 11,874
training and 2,969 test rows, of which 267 and 67 are not in the authors'
table).

**Diagnostic option.** `fit_mode="full"` fits the threshold, scaler and
imputer on all rows, the authors' order of operations. Test rows then
influence training features, so use it only for controlled comparisons.

**Limitation for unlearning.** The scaler, imputer and threshold are fitted
once per split and then held fixed, so unlearning and retraining operate on
the same prepared features. A deleted student may still have influenced
those fitted quantities (for example as an imputation neighbour). Deletion
therefore removes students from model training only: retraining is exact
with respect to the fixed features, Fair Unlearning approximates that
retraining (with the (ε, δ) certificate), and neither undoes the deleted
students' influence on preprocessing. The same holds for every dataset in
the existing experiments: `experiments/run_deletion_sweep.py` deletes rows
from data already scaled with the full training set (the largest
training-row norm, and Adult's standardization), and the authors' HSLS
table was imputed and scaled before any split.

## HSLS file and attribution

`hsls_df_knn_impute_past_v2.pkl` is the preprocessed High School
Longitudinal Study table used by Oesterling et al. (2024). It derives from
the NCES HSLS:09 public-use data. Its 56-feature, 14,509-row layout matches
the HSLS table of the Fair Projection authors (Alghamdi et al., "Beyond Adult
and COMPAS: Fair Multi-Class Prediction via Information Projection", NeurIPS
2022, Appendix A.2.1.1); Jeong, Wang and Calmon (AAAI 2022) use a different,
smaller HSLS feature set.

It is copied unmodified from the official code release:

> Oesterling, Ma, Calmon, Lakkaraju. *Fair Machine Unlearning: Data Removal
> while Mitigating Disparities.* AISTATS 2024.
> https://github.com/AI4LIFE-GROUP/fair-unlearning
> Licensed under the Apache License, Version 2.0.

The preprocessing in `src/data/datasets.py` reimplements the official
`prepare_adult.py`, `prepare_compas.py`, `prepare_hsls.py` and
`Dataloader.py` from that repository.
