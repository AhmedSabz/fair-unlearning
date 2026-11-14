"""
Tests for the HSLS two-source comparison (experiments/hsls_sources.py).

The baseline-anchor and sampler tests need only files in the repository; the
two-source tests need the NCES student file and are skipped without it.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.hsls_sources import (
    AUTHORS_THRESHOLD, authors_arm, check_identity, choose_deletion_ids,
    deletion_plan, evaluate, evaluation_sets, full_training, load_sources,
    manifest, paired_differences, unlearn_and_retrain,
)
from src.data.datasets import prepare_hsls
from experiments.run_deletion_sweep import (
    NOISE_SEED_OFFSET, PAPER_HYPERPARAMS, SIGMA, choose_deletions, group_roles,
)
from src.data.hsls_nces import (
    FEATURES, SCALE_FEATURES, STUDENT_CSV, read_nces_student,
)

HP = PAPER_HYPERPARAMS["hsls"]
needs_nces = pytest.mark.skipif(not STUDENT_CSV.exists(),
                                reason="NCES HSLS student file not present")


def noise_vector(d=59, split_seed=0):
    return SIGMA * torch.randn(
        d, generator=torch.Generator().manual_seed(NOISE_SEED_OFFSET + split_seed),
        dtype=torch.float64)


# ----------------------------------------------------------------------
# Baseline anchor: the driver reproduces the committed authors' results
# ----------------------------------------------------------------------

def test_driver_reproduces_committed_authors_baseline():
    """Same deletions as run_deletion_sweep -> same metrics as committed."""
    committed = pd.read_csv("experiments/results/deletion_sweep_raw_hsls.csv")
    d = authors_arm()
    sets = {"test": (d["X_test"], d["y_test"], d["sensitive_test"])}
    noise = noise_vector()
    originals = full_training(d, HP, noise)
    metrics = ["accuracy", "aeod", "dp", "eopp", "subgroup_acc"]

    for loss, method in (("bce", "Full Training (BCE)"),
                         ("fair", "Full Training (Fair)")):
        ours = evaluate(originals[loss], sets)["test"]
        ref = committed[committed.method == method].iloc[0]
        for m in metrics:
            assert ours[m] == ref[m], (method, m)

    s = d["sensitive_train"]
    deleted = choose_deletions("random", 0.05, s, group_roles(s), 0)
    unl, ret, _, _, _ = unlearn_and_retrain(d, originals["fair"], deleted, HP,
                                            noise)
    for method, theta in (("Fair Unlearning", unl), ("Retraining (Fair)", ret)):
        ref = committed[(committed.method == method)
                        & (committed.deletion_type == "random")
                        & np.isclose(committed.deletion_rate, 0.05)
                        & (committed.seed == 0)].iloc[0]
        ours = evaluate(theta, sets)["test"]
        for m in metrics:
            assert ours[m] == ref[m], (method, m)


def test_authors_arm_has_consistent_row_ids():
    d = authors_arm()            # raises if rows and ids do not line up
    ids = np.concatenate([d["train_row_ids"], d["test_row_ids"]])
    assert len(ids) == len(set(ids)) == 14_509
    assert len(d["train_row_ids"]) == len(d["X_train"])
    assert len(d["test_row_ids"]) == len(d["X_test"])


# ----------------------------------------------------------------------
# Deletion sampler (synthetic ids)
# ----------------------------------------------------------------------

def test_choose_deletion_ids_is_deterministic_and_nested():
    pool = np.arange(100, 5100, 3)
    a = choose_deletion_ids(pool, 50, seed=0)
    assert np.array_equal(a, choose_deletion_ids(pool[::-1], 50, seed=0))
    assert len(set(a)) == 50 and set(a) <= set(pool)
    assert set(a) <= set(choose_deletion_ids(pool, 200, seed=0))   # nested
    assert set(a) != set(choose_deletion_ids(pool, 50, seed=1))
    with pytest.raises(ValueError):
        choose_deletion_ids(pool, len(pool), seed=0)
    with pytest.raises(ValueError):
        choose_deletion_ids(pool, 0, seed=0)


# ----------------------------------------------------------------------
# Two sources (needs the NCES file)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def sources():
    data, raw, authors_table = load_sources()
    sets, shared_test = evaluation_sets(data, authors_table)
    return data, raw, authors_table, sets, shared_test


@needs_nces
def test_identity_check_rejects_misaligned_rows():
    raw = read_nces_student()
    check_identity(raw)                                    # aligned: passes
    shuffled = raw.sample(frac=1.0, random_state=0)
    shuffled.index = raw.index
    with pytest.raises(RuntimeError, match="do not match"):
        check_identity(shuffled)


@needs_nces
def test_identity_check_uses_observed_raw_features():
    raw = read_nces_student()
    ids = prepare_hsls().index
    other = [c for c in FEATURES if c not in SCALE_FEATURES]

    # Rows shuffled within identical (sex, race, label) groups: the first
    # three checks still pass, so only the raw-feature check can catch it.
    groups = raw.loc[ids].groupby([raw.loc[ids, "X1SEX"],
                                   raw.loc[ids, "X1RACE"].isin([2, 8]),
                                   raw.loc[ids, "X1TXMTH"] > AUTHORS_THRESHOLD]).groups
    rng = np.random.default_rng(0)
    src = np.concatenate([np.asarray(g) for g in groups.values()])
    dst = np.concatenate([rng.permutation(np.asarray(g)) for g in groups.values()])
    shuffled = raw.copy()
    shuffled.loc[src, other] = raw.loc[dst, other].to_numpy()
    with pytest.raises(RuntimeError, match="observed feature values"):
        check_identity(shuffled)

    # One observed raw value changed to another valid code: caught.
    values = raw.loc[ids, "P1MARSTAT"]
    row = values[values == 1].index[0]
    one_cell = raw.copy()
    one_cell.loc[row, "P1MARSTAT"] = 2
    with pytest.raises(RuntimeError, match="observed feature values"):
        check_identity(one_cell)

    # A missing (reserve-coded) cell is not compared: the authors imputed it.
    missing = raw.loc[ids, "P1MARSTAT"]
    row = missing[missing == -9].index[0]
    recoded = raw.copy()
    recoded.loc[row, "P1MARSTAT"] = -7
    check_identity(recoded)


@needs_nces
def test_manifest_records_driver_source_hash(sources):
    data, raw, authors_table, _, shared_test = sources
    meta = manifest(data, raw, authors_table, shared_test, torch.zeros(59),
                    (0.05,), (0,), 0, HP)
    expected = hashlib.sha256(Path("experiments/hsls_sources.py").read_bytes()).hexdigest()
    assert meta["driver_sha256"] == expected


@needs_nces
def test_split_alignment_and_cohorts(sources):
    data, _, authors_table, _, shared_test = sources
    a, b = data["A"], data["B"]
    assert set(a["train_row_ids"]) <= set(b["train_row_ids"])
    assert set(a["test_row_ids"]) <= set(b["test_row_ids"])
    assert not set(a["train_row_ids"]) & set(b["test_row_ids"])
    assert set(authors_table.index) <= set(b["train_row_ids"]) | set(b["test_row_ids"])
    assert len(shared_test) == 2_902
    assert list(a["feature_names"]) == list(b["feature_names"])


@needs_nces
@pytest.mark.parametrize("rate", [0.05, 0.20])
def test_deletion_plan_pairs_students(sources, rate):
    data = sources[0]
    plan = deletion_plan(data, rate, seed=3)
    paired = plan[("A", "paired")]
    assert np.array_equal(paired, plan[("B", "paired")])
    assert len(paired) == round(rate * len(data["A"]["train_row_ids"]))
    assert set(paired) <= set(data["A"]["train_row_ids"])
    same = plan[("B", "same_rate")]
    assert len(same) == round(rate * len(data["B"]["train_row_ids"]))
    assert set(same) <= set(data["B"]["train_row_ids"])


@needs_nces
def test_evaluation_sets_align_students_and_labels(sources):
    data, raw, authors_table, sets, shared_test = sources
    for name in ("shared_native", "shared_common_label"):
        assert len(sets["A"][name][1]) == len(sets["B"][name][1]) == len(shared_test)
    # Same students: identical sensitive attribute in the same order.
    assert torch.equal(sets["A"]["shared_native"][2], sets["B"]["shared_native"][2])
    # Common label = the authors' label, identical for both sources.
    common = authors_table.loc[shared_test, "gradebin"].to_numpy()
    for src in ("A", "B"):
        assert np.array_equal(sets[src]["shared_common_label"][1].numpy(), common)
    # B's own labels differ only where the two thresholds disagree.
    score = raw.loc[shared_test, "X1TXMTH"]
    thr_b = data["B"]["preprocessing"]["label_threshold"]
    expected = int(((score > AUTHORS_THRESHOLD) != (score > thr_b)).sum())
    differ = int((sets["B"]["shared_native"][1] != sets["B"]["shared_common_label"][1]).sum())
    assert differ == expected > 0


def test_paired_differences_flags_exact_pairing():
    rows = []
    for src, mode in (("A", "paired"), ("B", "paired"), ("B", "same_rate")):
        for seed in (0, 1):
            for ev in ("native", "shared_native"):
                rows.append({"source": src, "mode": mode, "rate": 0.05,
                             "seed": seed, "method": "Fair Unlearning",
                             "eval_set": ev, "accuracy": 0.7, "aeod": 0.03,
                             "dp": 0.1, "eopp": 0.1, "subgroup_acc": 0.01,
                             "param_distance": 1e-6})
    out = pd.DataFrame(paired_differences(rows))
    exact = out.groupby(["mode", "eval_set"])["exactly_paired"].first().to_dict()
    assert exact == {("paired", "native"): False,
                     ("paired", "shared_native"): True,
                     ("same_rate", "native"): False,
                     ("same_rate", "shared_native"): False}
