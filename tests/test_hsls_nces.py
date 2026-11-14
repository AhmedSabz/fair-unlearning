"""
Tests for the NCES-built HSLS table (src/data/hsls_nces.py).

They need the NCES public-use student file, which is gitignored, so they are
skipped when it is absent (for example in CI). Run with:
    python -m pytest tests/test_hsls_nces.py
"""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from src.data.datasets import load_dataset, prepare_hsls
from src.data.hsls_nces import (
    FEATURES, LABEL_SOURCE, OUTPUT_COLUMNS, RESERVE_CODES, SCALE_FEATURES,
    STUDENT_CSV, authors_row_ids, load_hsls_nces, prepare_hsls_nces,
    read_nces_student, select_rows, split_row_ids,
)

pytestmark = pytest.mark.skipif(
    not STUDENT_CSV.exists(),
    reason="NCES HSLS student file not present (see data/README.md)",
)


@pytest.fixture(scope="module")
def raw():
    return read_nces_student()


@pytest.fixture(scope="module")
def split(raw):
    return split_row_ids(select_rows(raw, "principled"))


@pytest.fixture(scope="module")
def prepared(raw, split):
    return prepare_hsls_nces(train_ids=split[0], raw=raw)


@pytest.fixture(scope="module")
def authors_table():
    return prepare_hsls()


@pytest.fixture(scope="module")
def authors_aligned(raw):
    """Diagnostic control: the authors' rows, fitted on the full table."""
    return prepare_hsls_nces(row_set="authors_ids", fit_mode="full", raw=raw)


# ----------------------------------------------------------------------
# Row selection
# ----------------------------------------------------------------------

def test_row_selection_counts_and_nesting(raw):
    principled = select_rows(raw, "principled")
    s1m8 = select_rows(raw, "s1m8grade")
    authors = select_rows(raw, "authors_ids")
    assert len(principled) == 14_843
    assert len(s1m8) == 14_584
    assert len(authors) == 14_509
    assert s1m8.isin(principled).all()
    assert authors.isin(s1m8).all()
    # 259 students lack S1M8GRADE; 75 more are kept by every rule we can
    # justify but were excluded by the authors for an undocumented reason.
    assert (~principled.isin(s1m8)).sum() == 259
    assert (~s1m8.isin(authors)).sum() == 75


def test_row_ids_are_nces_positions(raw):
    assert raw.index.name == "nces_row"
    assert (raw.index == np.arange(len(raw))).all()
    assert raw["STU_ID"].is_monotonic_increasing


# ----------------------------------------------------------------------
# Missing values, labels, sensitive attribute
# ----------------------------------------------------------------------

def test_only_reserve_codes_are_imputed(raw, prepared):
    """Observed values (incl. negative scale scores) pass through unchanged."""
    table, info = prepared
    data = raw.loc[table.index, FEATURES].astype(float)
    reserve = data.isin(RESERVE_CODES)
    assert info["num_imputed_cells"] == int(reserve.to_numpy().sum())

    lo = info["scaler_min"][FEATURES]
    hi = info["scaler_max"][FEATURES]
    expected = (data - lo) / (hi - lo)
    observed = ~reserve.to_numpy()
    diff = np.abs(table[FEATURES].to_numpy() - expected.to_numpy())[observed]
    assert diff.max() < 1e-12
    assert not table.isna().any().any()

    for col in SCALE_FEATURES:       # valid negative scores are kept as data
        assert ((data[col] < 0) & ~reserve[col]).sum() > 1000


def test_labels_and_groups_match_authors_table(raw, authors_table,
                                              authors_aligned):
    """Same rows + full-table fitting reproduces the authors' columns exactly."""
    table, info = authors_aligned
    assert info["label_threshold"] == pytest.approx(0.1540)
    assert (table.index == authors_table.index).all()
    assert list(table.columns) == list(authors_table.columns)
    for col in ("gradebin", "racebin", "sexbin"):
        assert (table[col] == authors_table[col]).all(), col

    # Where a non-scale feature was observed, the scaled values agree too.
    other = [c for c in FEATURES if c not in SCALE_FEATURES]
    observed = ~raw.loc[table.index, other].isin(RESERVE_CODES).to_numpy()
    diff = np.abs(table[other].to_numpy() - authors_table[other].to_numpy())
    assert diff[observed].max() < 1e-12


def test_authors_table_drops_negative_scale_scores(raw, authors_table,
                                                   authors_aligned):
    """
    Documents a difference: the authors' table treats valid negative values
    of the three math scales as missing (and imputes them). Ours keeps them.
    """
    ours, _ = authors_aligned
    for col in SCALE_FEATURES:
        values = raw.loc[authors_table.index, col].astype(float)
        negative = (values < 0) & ~values.isin(RESERVE_CODES)
        nonneg = values >= 0
        # Ours: every negative score maps below every non-negative one.
        assert ours.loc[negative, col].max() < ours.loc[nonneg, col].min()
        # Authors: the scale starts at 0 for the smallest non-negative score,
        # so the negative scores cannot have kept their values.
        assert authors_table.loc[nonneg, col].min() == 0.0
        assert (authors_table.loc[negative, col] > 0).mean() > 0.9


def test_label_uses_training_median_by_default(raw, prepared, split):
    table, info = prepared
    score = raw.loc[split[0], LABEL_SOURCE]
    assert info["label_threshold"] == pytest.approx(float(score.median()))
    expected = (raw.loc[table.index, LABEL_SOURCE]
                > info["label_threshold"]).astype(int)
    assert (table["gradebin"] == expected).all()


# ----------------------------------------------------------------------
# No train/test leakage
# ----------------------------------------------------------------------

def test_train_only_fit_ignores_test_rows(raw, prepared, split):
    """Changing every test row leaves all training outputs unchanged."""
    train_ids, test_ids = split
    table, info = prepared

    perturbed = raw.copy()
    for col in FEATURES:
        perturbed.loc[test_ids, col] = raw[col].max() + 100.0
    perturbed.loc[test_ids, LABEL_SOURCE] = 10.0
    table2, info2 = prepare_hsls_nces(train_ids=train_ids, raw=perturbed)

    assert info2["label_threshold"] == info["label_threshold"]
    pd.testing.assert_series_equal(info2["scaler_min"], info["scaler_min"])
    pd.testing.assert_series_equal(info2["scaler_max"], info["scaler_max"])
    pd.testing.assert_frame_equal(table2.loc[train_ids], table.loc[train_ids])
    assert not table2.loc[test_ids].equals(table.loc[test_ids])


def test_train_mode_requires_train_ids(raw):
    with pytest.raises(ValueError, match="train_ids"):
        prepare_hsls_nces(raw=raw)
    with pytest.raises(ValueError, match="row_set"):
        select_rows(raw, "everyone")
    with pytest.raises(ValueError, match="subset"):
        prepare_hsls_nces(row_set="s1m8grade", train_ids=[-1], raw=raw)


def test_diagnostic_options(raw):
    """Fixed threshold, the S1M8GRADE row set and an unaligned split."""
    ids = select_rows(raw, "s1m8grade")
    train_ids, test_ids = split_row_ids(ids, align_with_authors=False)
    assert set(train_ids) | set(test_ids) == set(ids)
    assert 0.18 < len(test_ids) / len(ids) < 0.22

    table, info = prepare_hsls_nces(row_set="s1m8grade", train_ids=train_ids,
                                    label_threshold=0.1540, raw=raw)
    assert len(table) == 14_584
    assert info["label_threshold"] == 0.1540
    assert info["label_threshold_fitted"] is False
    expected = (raw.loc[table.index, LABEL_SOURCE] > 0.1540).astype(int)
    assert (table["gradebin"] == expected).all()
    assert not table.isna().any().any()


# ----------------------------------------------------------------------
# Split and backward compatibility
# ----------------------------------------------------------------------

def test_split_matches_existing_hsls_split(raw, authors_table):
    """Shared students keep exactly the split load_dataset("hsls") uses."""
    authors = authors_row_ids()
    train_pos, test_pos = train_test_split(np.arange(len(authors)),
                                           test_size=0.2, random_state=0)

    # The positional split is the one load_dataset("hsls") applies.
    ref = load_dataset("hsls")
    features = authors_table.drop(columns=["gradebin"]).to_numpy()
    expected = features[train_pos] / ref["max_train_row_norm"]
    assert torch.equal(ref["X_train"][:, :-1],
                       torch.tensor(expected, dtype=torch.float64))

    ids = select_rows(raw, "principled")
    train_ids, test_ids = split_row_ids(ids)
    assert set(train_ids) & set(authors) == set(authors[train_pos])
    assert set(test_ids) & set(authors) == set(authors[test_pos])
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(ids)

    extra = [i for i in ids if i not in set(authors)]
    extra_test = len(set(extra) & set(test_ids)) / len(extra)
    assert 0.1 < extra_test < 0.3
    again = split_row_ids(ids)
    assert np.array_equal(again[0], train_ids)
    assert np.array_equal(again[1], test_ids)


def test_load_hsls_nces_has_load_dataset_format(raw, prepared, split):
    data = load_hsls_nces(raw=raw)
    ref = load_dataset("hsls")
    assert set(ref) <= set(data)
    assert list(data["feature_names"]) == list(ref["feature_names"])
    assert len(data["X_train"]) + len(data["X_test"]) == 14_843
    assert np.array_equal(data["train_row_ids"], split[0])
    assert np.array_equal(data["test_row_ids"], split[1])
    for key in ("X_train", "X_test"):
        assert not torch.isnan(data[key]).any()
        assert torch.all(data[key][:, -1] == 1)
    assert data["X_train"][:, :-1].norm(dim=1).max() <= 1 + 1e-12

    # Same values as the prepared table, scaled by the training max norm.
    table, _ = prepared
    features = table.drop(columns=["gradebin"]).loc[split[0]].to_numpy()
    expected = features / data["max_train_row_norm"]
    assert torch.allclose(data["X_train"][:, :-1],
                          torch.tensor(expected, dtype=torch.float64),
                          rtol=0, atol=1e-12)
    assert list(table.columns) == OUTPUT_COLUMNS
