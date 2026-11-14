"""
HSLS built independently from the NCES public-use file.

The existing HSLS pipeline (datasets.prepare_hsls) starts from the authors'
already-imputed pickle. This module rebuilds the table from
hsls_16_student_v1_0.csv (HSLS:09 2016 public-use student file, NCES), using
only documented NCES variables and choices that can be justified without the
authors' code. See data/README.md for the full description and evidence.

Rows (row_set)
    "principled" (default, 14,843 rows): X1SEX and X1RACE present, base-year
        math score X1TXMSCR and first-follow-up score X2TXMSCR present, and
        both base-year questionnaires completed (X1SQSTAT != 8 and
        X1PQSTAT != 8, i.e. no unit non-response).
    "s1m8grade"  (diagnostic, 14,584): additionally requires S1M8GRADE
        (8th-grade math grade). This rule was found by matching the authors'
        rows, not derived independently.
    "authors_ids" (diagnostic control, 14,509): exactly the authors' rows,
        taken from the index of their pickle. Not a filtering rule.

Features: the same 56 base-year variables (X1*, S1*, P1*) as the authors'
    table, so models see the same inputs; plus sexbin and racebin.
Label: gradebin = 1[X1TXMTH > median], X1TXMTH = base-year (fall 2009, 9th
    grade) mathematics IRT theta score. The median is taken over the fitting
    rows (training rows by default).
Groups: racebin = 1 for X1RACE in {2 Asian, 8 White}, else 0.
        sexbin  = 1 for X1SEX == 2 (female), matching the authors' coding.
Missing values: only the NCES reserve codes -9..-3 (missing, unit
    non-response, legitimate skip, not applicable, suppressed, not
    administered, carry-through) become NaN. X1MTHID, X1MTHEFF and X1MTHINT
    are continuous scales whose negative values are real data and are kept.
    (The authors' table treats those negative scale scores as missing.)
Scaling and imputation: MinMax scaling, then KNNImputer(k=5, uniform
    weights, nan-Euclidean distance) on the 58 model inputs (56 features +
    sexbin + racebin; never the label). Both are fitted on the training rows
    only and then applied to all rows, so no test information leaks into
    training. fit_mode="full" fits on all rows instead (the authors' order of
    operations); it is a diagnostic only.

Row identifiers: the 0-based row position in the NCES student file, which is
sorted by STU_ID. The authors' pickle is indexed the same way, so the two
tables align row for row.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data.datasets import RAW_DIR, to_model_inputs

NCES_DIR = RAW_DIR / "hsls" / "nces_2016_public_use"
STUDENT_CSV = NCES_DIR / "hsls_16_student_v1_0.csv"
AUTHORS_PICKLE = RAW_DIR / "hsls" / "hsls_df_knn_impute_past_v2.pkl"
NCES_NUM_ROWS = 23_503          # students in the HSLS:09 student file

# The 56 base-year features of the authors' table, in the same order.
FEATURES = [
    "X1HHNUMBER", "X1FAMINCOME", "X1MTHID", "X1MTHEFF", "X1MTHINT",
    "S1LANG1ST", "S1TEPOPULAR", "S1TEMAKEFUN", "S1MTHCOMP", "S1SCICOMP",
    "S1MTCHRESPCT", "S1MTCHFAIR", "S1MTCHINTRST", "S1MTCHEASY",
    "S1MTCHVALUES", "S1MTCHCONF", "S1APCALC", "S1IBCALC", "P1REPEATGRD",
    "P1ELLEVER", "P1MARSTAT", "P1YRBORN1", "P1YRBORN2", "X1PAR1RACE",
    "X1PAR2RACE", "P1USBORN1", "P1USBORN2", "X1PAR1EDU", "X1PAR2EDU",
    "X1PAR1OCC2", "X1PAR2OCC2", "P1JOBNOW1", "P1JOBONET1_STEM1",
    "P1JOBONET2_STEM1", "X1PARRESP", "X1P1RELATION", "P1HHTIME",
    "P1EDUASPIRE", "P1EDUEXPECT", "P1MTHHWEFF", "P1SCIHWEFF", "P1ENGHWEFF",
    "P1MTHCOMP", "P1SCICOMP", "P1ENGCOMP", "P1MUSEUM", "P1COMPUTER",
    "P1FIXED", "P1SCIFAIR", "P1SCIPROJ", "P1STEMDISC", "P1CAMPMS",
    "P1CAMPOTH", "P1NOOUTSCH", "P1NOACT", "S1M8GRADE",
]
# Continuous scales: negative values are valid scores, not missing codes.
SCALE_FEATURES = ["X1MTHID", "X1MTHEFF", "X1MTHINT"]
RESERVE_CODES = (-9, -8, -7, -6, -5, -4, -3)
LABEL_SOURCE = "X1TXMTH"
SELECTION_COLUMNS = ["X1SEX", "X1RACE", "X1TXMSCR", "X2TXMSCR",
                     "X1SQSTAT", "X1PQSTAT"]
OUTPUT_COLUMNS = FEATURES + ["gradebin", "racebin", "sexbin"]
ROW_SETS = ("principled", "s1m8grade", "authors_ids")
KNN_NEIGHBORS = 5
SPLIT_KEY_SEED = 20_261          # mixes with random_state for per-row keys


# ----------------------------------------------------------------------
# Reading and row selection
# ----------------------------------------------------------------------

@lru_cache(maxsize=2)
def _read_student_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the HSLS:09 2016 public-use student "
            "file from NCES into data/raw/hsls/nces_2016_public_use/ "
            "(see data/README.md). It is gitignored and never committed."
        )
    columns = ["STU_ID"] + SELECTION_COLUMNS + [LABEL_SOURCE] + FEATURES
    raw = pd.read_csv(path, usecols=columns)
    if len(raw) != NCES_NUM_ROWS or not raw["STU_ID"].is_monotonic_increasing:
        raise ValueError(
            f"{path} does not look like the HSLS:09 student file "
            f"({len(raw)} rows; expected {NCES_NUM_ROWS} sorted by STU_ID)."
        )
    raw.index.name = "nces_row"
    return raw[["STU_ID"] + SELECTION_COLUMNS + [LABEL_SOURCE] + FEATURES]


def read_nces_student(path=STUDENT_CSV):
    """The needed columns of the NCES student file (raw codes), cached."""
    return _read_student_file(str(path)).copy()


def authors_row_ids(pickle_path=AUTHORS_PICKLE):
    """NCES row ids of the authors' 14,509 students (their pickle's index)."""
    return pd.Index(pd.read_pickle(pickle_path).index, name="nces_row")


def _is_reserve(values):
    return values.isin(RESERVE_CODES)


def _is_present(values):
    """Observed: neither an NCES reserve code nor an actual NaN."""
    return values.notna() & ~_is_reserve(values)


def select_rows(raw, row_set="principled"):
    """NCES row ids kept by one of the row selections (see module docstring)."""
    if row_set not in ROW_SETS:
        raise ValueError(f"row_set must be one of {ROW_SETS}, got {row_set!r}")
    if row_set == "authors_ids":
        ids = authors_row_ids()
        return ids[ids.isin(raw.index)]

    # Every required field must be observed: a NaN never counts as present
    # (note that NaN != 8 is True, so the status checks need notna too).
    keep = (
        raw["X1SEX"].isin([1, 2])
        & raw["X1RACE"].between(1, 8)
        & _is_present(raw["X1TXMSCR"])
        & _is_present(raw["X2TXMSCR"])
        & _is_present(raw[LABEL_SOURCE])
        & raw["X1SQSTAT"].notna() & (raw["X1SQSTAT"] != 8)
        & raw["X1PQSTAT"].notna() & (raw["X1PQSTAT"] != 8)
    )
    if row_set == "s1m8grade":
        keep &= _is_present(raw["S1M8GRADE"])
    return raw.index[keep]


# ----------------------------------------------------------------------
# Train/test split by row id
# ----------------------------------------------------------------------

def _row_keys(random_state):
    """One uniform number per NCES row, the same for every row selection."""
    rng = np.random.default_rng([SPLIT_KEY_SEED, random_state])
    return rng.random(NCES_NUM_ROWS)


def split_row_ids(row_ids, test_size=0.2, random_state=0,
                  align_with_authors=True):
    """
    Split NCES row ids into train and test before any fitting.

    align_with_authors=True: students in the authors' table get exactly the
    assignment that load_dataset("hsls", test_size, random_state) gives them;
    every other student is in the test set when its per-row key is below
    test_size. Shared students therefore have the same assignment in both
    sources. With align_with_authors=False every student uses its key.

    Returned ids are ordered by the per-row key: a fixed random order that is
    the same for every row selection.
    """
    row_ids = pd.Index(row_ids)
    keys = _row_keys(random_state)
    in_test = keys[row_ids.to_numpy()] < test_size

    if align_with_authors:
        authors = authors_row_ids()
        _, test_pos = train_test_split(np.arange(len(authors)),
                                       test_size=test_size,
                                       random_state=random_state)
        authors_test = set(authors[test_pos].tolist())
        shared = row_ids.isin(authors)
        in_test[shared] = row_ids[shared].isin(authors_test)

    order = np.argsort(keys[row_ids.to_numpy()], kind="stable")
    ordered, in_test = row_ids[order], in_test[order]
    return ordered[~in_test].to_numpy(), ordered[in_test].to_numpy()


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------

def prepare_hsls_nces(row_set="principled", train_ids=None, fit_mode="train",
                      label_threshold=None, n_neighbors=KNN_NEIGHBORS,
                      raw=None):
    """
    Build the HSLS table from NCES data.

    train_ids: NCES row ids of the training rows; required for
        fit_mode="train" (the default), where the label median, MinMax scaler
        and KNN imputer are fitted on these rows only.
    fit_mode="full": fit on all selected rows (diagnostic; leaks test data).
    label_threshold: fixed X1TXMTH threshold instead of the fitted median,
        for controlled comparisons (the authors' table uses 0.1540).
    raw: optional pre-read raw table (defaults to read_nces_student()).

    Returns (table, info). table has OUTPUT_COLUMNS (same names and order as
    datasets.prepare_hsls()) and is indexed by NCES row id; info records the
    fitted quantities.
    """
    if fit_mode not in ("train", "full"):
        raise ValueError(f"fit_mode must be 'train' or 'full', got {fit_mode!r}")
    raw = read_nces_student() if raw is None else raw
    ids = select_rows(raw, row_set)
    data = raw.loc[ids]

    if fit_mode == "train":
        if train_ids is None:
            raise ValueError("fit_mode='train' needs train_ids; split first "
                             "(split_row_ids) so nothing is fitted on test rows.")
        fit_ids = pd.Index(train_ids)
        if not fit_ids.isin(ids).all():
            raise ValueError("train_ids must be a subset of the selected rows.")
    else:
        fit_ids = ids

    # Label: base-year math score above the threshold.
    score = data[LABEL_SOURCE].astype(float)
    if not _is_present(score).all():    # a NaN would silently become class 0
        raise ValueError(f"selected rows contain missing {LABEL_SOURCE}")
    threshold = (float(score.loc[fit_ids].median()) if label_threshold is None
                 else float(label_threshold))
    gradebin = (score > threshold).astype(int)

    racebin = data["X1RACE"].isin([2, 8]).astype(int)   # Asian, White
    sexbin = (data["X1SEX"] == 2).astype(int)           # female

    # Reserve codes -> NaN; everything else (incl. negative scale scores) is data.
    features = data[FEATURES].astype(float)
    features = features.mask(_is_reserve(features))
    other = [c for c in FEATURES if c not in SCALE_FEATURES]
    if (features[other] < 0).any().any():
        bad = features[other].columns[(features[other] < 0).any()].tolist()
        raise ValueError(f"unexpected negative codes in {bad}")
    n_missing = int(features.isna().sum().sum())

    inputs = pd.concat([features, sexbin.rename("sexbin"),
                        racebin.rename("racebin")], axis=1)
    scaler = MinMaxScaler().fit(inputs.loc[fit_ids])          # NaNs ignored
    scaled = pd.DataFrame(scaler.transform(inputs), index=inputs.index,
                          columns=inputs.columns)
    if scaled.loc[fit_ids].isna().all().any():
        raise ValueError("a feature has no observed values in the fitting rows")
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    imputer.fit(scaled.loc[fit_ids])
    imputed = pd.DataFrame(imputer.transform(scaled), index=scaled.index,
                           columns=scaled.columns)

    table = imputed[FEATURES].copy()
    table["gradebin"] = gradebin
    table["racebin"] = racebin
    table["sexbin"] = sexbin
    table.index.name = "nces_row"

    info = {
        "row_set": row_set,
        "fit_mode": fit_mode,
        "num_rows": len(table),
        "num_fit_rows": len(fit_ids),
        "label_threshold": threshold,
        "label_threshold_fitted": label_threshold is None,
        "n_neighbors": n_neighbors,
        "num_imputed_cells": n_missing,
        "scaler_min": pd.Series(scaler.data_min_, index=inputs.columns),
        "scaler_max": pd.Series(scaler.data_max_, index=inputs.columns),
    }
    return table[OUTPUT_COLUMNS], info


def load_hsls_nces(row_set="principled", test_size=0.2, random_state=0,
                   fit_mode="train", label_threshold=None,
                   n_neighbors=KNN_NEIGHBORS, align_split=True, raw=None):
    """
    NCES-built HSLS in the same format as datasets.load_dataset("hsls").

    Order of operations: select rows -> split row ids -> fit label threshold,
    scaler and imputer on the training rows -> transform all rows ->
    to_model_inputs (divide by the largest training-row norm, add intercept).

    Extra keys: train_row_ids, test_row_ids (NCES row ids, in the same order
    as the rows of X_train / X_test) and preprocessing (see
    prepare_hsls_nces).
    """
    raw = read_nces_student() if raw is None else raw
    ids = select_rows(raw, row_set)
    train_ids, test_ids = split_row_ids(ids, test_size, random_state,
                                        align_with_authors=align_split)
    table, info = prepare_hsls_nces(
        row_set=row_set, train_ids=train_ids, fit_mode=fit_mode,
        label_threshold=label_threshold, n_neighbors=n_neighbors, raw=raw,
    )

    features = table.drop(columns=["gradebin"])    # racebin stays an input
    train, test = table.loc[train_ids], table.loc[test_ids]
    data = to_model_inputs(
        features.loc[train_ids].to_numpy(dtype=np.float64),
        features.loc[test_ids].to_numpy(dtype=np.float64),
        train["gradebin"].to_numpy(dtype=np.float64),
        test["gradebin"].to_numpy(dtype=np.float64),
        train["racebin"].to_numpy().astype(np.int64),
        test["racebin"].to_numpy().astype(np.int64),
        list(features.columns),
    )
    data["train_row_ids"] = np.asarray(train_ids)
    data["test_row_ids"] = np.asarray(test_ids)
    data["preprocessing"] = info
    return data
