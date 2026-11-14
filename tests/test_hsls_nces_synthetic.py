"""
Synthetic-data tests for src/data/hsls_nces.py.

They build a small fake NCES table in memory, so they run everywhere (CI
included) without the NCES student file or the authors' pickle. The
real-data integration tests are in tests/test_hsls_nces.py.
"""

import numpy as np
import pandas as pd
import pytest

import src.data.hsls_nces as hn
from src.data.hsls_nces import (
    FEATURES, LABEL_SOURCE, SCALE_FEATURES, SELECTION_COLUMNS,
    prepare_hsls_nces, select_rows, split_row_ids,
)

N = 150
BAD_ROWS = {          # row -> (column, value): each must be excluded
    0: ("X1SEX", np.nan),
    1: ("X1RACE", -9),
    2: ("X1TXMSCR", np.nan),
    3: ("X2TXMSCR", -8),
    4: ("X1SQSTAT", np.nan),
    5: ("X1PQSTAT", 8),
    6: (LABEL_SOURCE, np.nan),
    7: (LABEL_SOURCE, -9),
    8: ("X1SQSTAT", 8),
    9: ("X2TXMSCR", np.nan),
}
NO_S1M8GRADE = {10: np.nan, 11: -9, 12: -7}


def make_raw(seed=0):
    """Fake NCES student table: raw codes, indexed by row position."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(index=pd.RangeIndex(N, name="nces_row"))
    raw["STU_ID"] = 10_001 + np.arange(N)
    raw["X1SEX"] = rng.integers(1, 3, N).astype(float)
    raw["X1RACE"] = rng.integers(1, 9, N).astype(float)
    raw["X1TXMSCR"] = rng.uniform(20, 80, N)
    raw["X2TXMSCR"] = rng.uniform(20, 80, N)
    raw["X1SQSTAT"] = 1.0
    raw["X1PQSTAT"] = 1.0
    raw[LABEL_SOURCE] = rng.normal(0, 1, N).round(4)
    for col in FEATURES:
        raw[col] = rng.integers(1, 6, N).astype(float)
    for col in SCALE_FEATURES:              # continuous, incl. negatives
        raw[col] = rng.uniform(-2.5, 2.0, N).round(2)

    for row, (col, value) in BAD_ROWS.items():
        raw.loc[row, col] = value
    for row, value in NO_S1M8GRADE.items():
        raw.loc[row, "S1M8GRADE"] = value
    # Missing feature values in otherwise valid rows: reserve codes and NaN.
    raw.loc[20, "P1MARSTAT"] = -9
    raw.loc[21, "S1MTHCOMP"] = -7
    raw.loc[22, "X1PAR2EDU"] = np.nan
    raw.loc[23, "X1MTHEFF"] = -9            # reserve code in a scale
    raw.loc[24, "X1MTHID"] = np.nan
    return raw[["STU_ID"] + SELECTION_COLUMNS + [LABEL_SOURCE] + FEATURES]


@pytest.fixture
def raw():
    return make_raw()


# ----------------------------------------------------------------------
# Row selection and missing labels
# ----------------------------------------------------------------------

def test_select_rows_excludes_nan_and_reserve_codes(raw):
    principled = set(select_rows(raw, "principled"))
    assert principled == set(range(N)) - set(BAD_ROWS)
    s1m8 = set(select_rows(raw, "s1m8grade"))
    assert s1m8 == principled - set(NO_S1M8GRADE)


def test_missing_label_is_rejected_not_class_zero(raw, monkeypatch):
    """A NaN or reserve-coded label raises instead of becoming gradebin 0."""
    for row in (6, 7):
        monkeypatch.setattr(hn, "authors_row_ids",
                            lambda row=row: pd.Index([row, 13, 14, 15]))
        with pytest.raises(ValueError, match=LABEL_SOURCE):
            prepare_hsls_nces(row_set="authors_ids", fit_mode="full", raw=raw)


# ----------------------------------------------------------------------
# Missing feature values and valid negative scale scores
# ----------------------------------------------------------------------

def test_reserve_codes_and_nan_features_are_imputed(raw):
    table, info = prepare_hsls_nces(fit_mode="full", raw=raw)
    # Rows 20-24, plus rows 10-12: the principled rows keep students without
    # S1M8GRADE, whose missing grade is then imputed like any feature.
    assert info["num_imputed_cells"] == 5 + len(NO_S1M8GRADE)
    assert not table.isna().any().any()

    # Observed values pass through unchanged (only MinMax-scaled).
    lo, hi = info["scaler_min"], info["scaler_max"]
    data = raw.loc[table.index, FEATURES]
    expected = (data - lo[FEATURES]) / (hi[FEATURES] - lo[FEATURES])
    observed = (data.notna() & ~data.isin(hn.RESERVE_CODES)).to_numpy()
    diff = np.abs(table[FEATURES].to_numpy() - expected.to_numpy())
    assert diff[observed].max() < 1e-12


def test_valid_negative_scale_scores_are_kept(raw):
    table, info = prepare_hsls_nces(fit_mode="full", raw=raw)
    for col in SCALE_FEATURES:
        values = raw.loc[table.index, col]
        negative = values.lt(0) & ~values.isin(hn.RESERVE_CODES)
        assert negative.sum() > 10
        assert info["scaler_min"][col] < 0          # negatives set the scale
        assert table.loc[negative[negative].index, col].max() < \
            table.loc[values[values >= 0].index, col].min()


def test_negative_non_reserve_code_in_categorical_feature_raises(raw):
    raw.loc[30, "S1APCALC"] = -1          # not a reserve code, not a scale
    with pytest.raises(ValueError, match="S1APCALC"):
        prepare_hsls_nces(fit_mode="full", raw=raw)


# ----------------------------------------------------------------------
# Labels and group columns
# ----------------------------------------------------------------------

def test_label_and_group_construction(raw):
    ids = select_rows(raw, "principled")
    train_ids, _ = split_row_ids(ids, align_with_authors=False)
    table, info = prepare_hsls_nces(train_ids=train_ids, raw=raw)

    score = raw.loc[table.index, LABEL_SOURCE]
    threshold = raw.loc[train_ids, LABEL_SOURCE].median()
    assert info["label_threshold"] == threshold
    assert info["label_threshold_fitted"] is True
    assert (table["gradebin"] == (score > threshold).astype(int)).all()

    race = raw.loc[table.index, "X1RACE"]
    assert (table["racebin"] == race.isin([2, 8]).astype(int)).all()
    sex = raw.loc[table.index, "X1SEX"]
    assert (table["sexbin"] == (sex == 2).astype(int)).all()

    # A fixed threshold is strict: a score equal to it is class 0.
    tie_row = int(ids[0])
    fixed = float(raw.loc[tie_row, LABEL_SOURCE])
    table2, info2 = prepare_hsls_nces(train_ids=train_ids,
                                      label_threshold=fixed, raw=raw)
    assert info2["label_threshold_fitted"] is False
    assert table2.loc[tie_row, "gradebin"] == 0
    assert (table2["gradebin"] == (score > fixed).astype(int)).all()


# ----------------------------------------------------------------------
# Training-only preprocessing
# ----------------------------------------------------------------------

def test_training_only_preprocessing(raw):
    ids = select_rows(raw, "principled")
    train_ids, test_ids = split_row_ids(ids, align_with_authors=False)
    assert len(test_ids) > 5
    # Give one test row a missing value, so its imputation can be checked.
    probe = int(test_ids[0])
    raw.loc[probe, "P1MUSEUM"] = -9
    table, info = prepare_hsls_nces(train_ids=train_ids, raw=raw)

    # Change every other test row drastically, labels included.
    others = [i for i in test_ids if i != probe]
    perturbed = raw.copy()
    perturbed.loc[others, FEATURES] = 99.0
    perturbed.loc[others, LABEL_SOURCE] = 50.0
    table2, info2 = prepare_hsls_nces(train_ids=train_ids, raw=perturbed)

    # Nothing fitted on training rows moves...
    assert info2["label_threshold"] == info["label_threshold"]
    pd.testing.assert_series_equal(info2["scaler_min"], info["scaler_min"])
    pd.testing.assert_series_equal(info2["scaler_max"], info["scaler_max"])
    pd.testing.assert_frame_equal(table2.loc[train_ids], table.loc[train_ids])
    # ...and test rows are never imputation donors for each other.
    pd.testing.assert_series_equal(table2.loc[probe], table.loc[probe])

    # Sanity check: fitting on the full table would have picked these up.
    _, info_full = prepare_hsls_nces(fit_mode="full", raw=perturbed)
    assert info_full["scaler_max"]["P1MUSEUM"] != info["scaler_max"]["P1MUSEUM"]


def test_train_ids_must_be_selected_rows(raw):
    with pytest.raises(ValueError, match="train_ids"):
        prepare_hsls_nces(raw=raw)
    with pytest.raises(ValueError, match="subset"):
        prepare_hsls_nces(train_ids=[0, 50], raw=raw)   # row 0 is excluded
