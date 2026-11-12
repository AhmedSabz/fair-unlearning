"""
Automated tests for the paper reproduction (run with: python -m pytest tests).

Covers the data setup, the fair objective, the unlearning methods, the
baselines, the certificate and the metrics. Each test is small and fast; the
dataset tests load the real files in data/raw/.

The older tests/test_*.py files are scripts with a main() function (run with
python -m tests.<name>); pytest imports them but finds no tests in them.
"""

import itertools
import math

import numpy as np
import pytest
import torch

from src.data.datasets import load_dataset, prepare_adult, prepare_compas, prepare_hsls
from src.evaluation.metrics import compute_metrics
from src.objectives.derivatives import compute_gradient
from src.objectives.fair_loss import fair_logistic_loss
from src.objectives.train import train_model
from src.unlearning.baselines import (
    pru_precompute, pru_unlearn, sisa_predict, sisa_shards, sisa_train,
    sisa_unlearn,
)
from src.unlearning.certificate import (
    epsilon_from_eps_prime, theorem_eps_prime,
)
from src.unlearning.newton import newton_unlearn


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def toy():
    """Small random problem: 60 rows, 4 features + intercept, two groups."""
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(60, 4, generator=gen, dtype=torch.float64)
    X = X / X.norm(dim=1).max()
    X = torch.cat([X, torch.ones(60, 1, dtype=torch.float64)], dim=1)
    s = (torch.rand(60, generator=gen) < 0.4).long()
    logits = X @ torch.tensor([2.0, -1.0, 0.5, 1.0, 0.0], dtype=torch.float64)
    y = (torch.rand(60, generator=gen, dtype=torch.float64)
         < torch.sigmoid(logits + 0.8 * s)).double()
    return X, y, s


# ----------------------------------------------------------------------
# Data: sizes from the paper / official cleaned files
# ----------------------------------------------------------------------

@pytest.mark.parametrize("prepare, rows, columns", [
    (prepare_adult, 46447, 21),
    (prepare_compas, 5278, 7),
    (prepare_hsls, 14509, 59),
])
def test_cleaned_tables_match_official_sizes(prepare, rows, columns):
    df = prepare()
    assert df.shape == (rows, columns)
    assert not df.isna().any().any()


@pytest.mark.parametrize("name, features", [
    ("adult", 21), ("compas", 7), ("hsls", 59),
])
def test_loaded_split_matches_paper_setup(name, features):
    data = load_dataset(name)
    X = data["X_train"]
    assert X.shape[1] == features                  # incl. intercept
    assert torch.all(X[:, -1] == 1)                # intercept column
    norms = X[:, :-1].norm(dim=1)
    assert norms.max().item() == pytest.approx(1.0)   # ||x|| <= 1
    assert len(X) + len(data["X_test"]) > 0
    n = len(X) + len(data["X_test"])
    assert len(data["X_test"]) == pytest.approx(0.2 * n, abs=1)
    assert set(data["sensitive_train"].unique().tolist()) == {0, 1}


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------

def test_fairness_penalty_equals_pairwise_definition(toy):
    """Paper Eq. (3), computed by brute force over all cross-group pairs."""
    X, y, s = toy
    theta = torch.randn(X.shape[1], dtype=torch.float64,
                        generator=torch.Generator().manual_seed(1))
    a = [i for i in range(len(y)) if s[i] == 0]
    b = [j for j in range(len(y)) if s[j] == 1]
    total = sum(float(X[i] @ theta - X[j] @ theta)
                for i, j in itertools.product(a, b) if y[i] == y[j])
    expected = (total / (len(a) * len(b))) ** 2
    _, parts = fair_logistic_loss(theta, X, y, s, lambda_reg=0.0, gamma=1.0)
    assert parts["fairness"].item() == pytest.approx(expected, rel=1e-10)


def test_gamma_zero_is_plain_bce_and_allows_one_group(toy):
    X, y, _ = toy
    one_group = torch.zeros(len(y), dtype=torch.long)
    theta = torch.zeros(X.shape[1], dtype=torch.float64)
    loss, parts = fair_logistic_loss(theta, X, y, one_group, lambda_reg=0.0,
                                     gamma=0.0)
    assert parts["fairness"].item() == 0.0
    assert loss.item() == pytest.approx(math.log(2))


def test_noise_term_is_b_dot_theta_over_n(toy):
    X, y, s = toy
    theta = torch.ones(X.shape[1], dtype=torch.float64)
    b = torch.arange(X.shape[1], dtype=torch.float64)
    with_noise, _ = fair_logistic_loss(theta, X, y, s, gamma=1.0, noise=b)
    without, _ = fair_logistic_loss(theta, X, y, s, gamma=1.0)
    assert (with_noise - without).item() == pytest.approx(
        (b @ theta).item() / len(y))


@pytest.mark.parametrize("gamma", [0.0, 10.0])
def test_training_reaches_stationary_point(toy, gamma):
    X, y, s = toy
    b = torch.randn(X.shape[1], dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2))
    theta = train_model(X, y, s, lambda_reg=1e-2, gamma=gamma, noise=b)
    grad = compute_gradient(theta, X, y, s, lambda_reg=1e-2, gamma=gamma,
                            noise=b)
    assert grad.norm().item() < 1e-7


# ----------------------------------------------------------------------
# Unlearning
# ----------------------------------------------------------------------

@pytest.mark.parametrize("gamma", [0.0, 10.0])
def test_newton_step_is_zero_when_nothing_is_deleted(toy, gamma):
    X, y, s = toy
    theta = train_model(X, y, s, lambda_reg=1e-2, gamma=gamma)
    unlearned, _ = newton_unlearn(theta, X, y, s, lambda_reg=1e-2, gamma=gamma)
    assert (unlearned - theta).norm().item() < 1e-6


@pytest.mark.parametrize("gamma", [0.0, 10.0])
def test_newton_step_approximates_retraining(toy, gamma):
    """One Newton step lands much closer to retraining than not updating."""
    X, y, s = toy
    b = torch.randn(X.shape[1], dtype=torch.float64,
                    generator=torch.Generator().manual_seed(3))
    theta = train_model(X, y, s, lambda_reg=1e-2, gamma=gamma, noise=b)
    keep = torch.ones(len(y), dtype=torch.bool)
    keep[:3] = False
    retrained = train_model(X[keep], y[keep], s[keep], lambda_reg=1e-2,
                            gamma=gamma, noise=b)
    unlearned, _ = newton_unlearn(theta, X[keep], y[keep], s[keep],
                                  lambda_reg=1e-2, gamma=gamma, noise=b)
    no_update = (theta - retrained).norm().item()
    assert (unlearned - retrained).norm().item() < 0.1 * no_update


def test_sisa_only_retrains_affected_shards(toy):
    X, y, s = toy
    shards = sisa_shards(len(y))
    thetas = sisa_train(X, y, s, shards, lambda_reg=1e-2)
    new = sisa_unlearn(X, y, s, shards, thetas, np.array([0, 1]),
                       lambda_reg=1e-2)
    assert not torch.equal(new[0], thetas[0])          # shard 0 had rows 0, 1
    assert all(new[i] is thetas[i] for i in range(1, len(shards)))
    votes = sisa_predict(new, X)
    assert set(votes.unique().tolist()) <= {0, 1}


def test_pru_returns_finite_update(toy):
    X, y, s = toy
    theta = train_model(X, y, s, lambda_reg=1e-2, gamma=0.0)
    pre = pru_precompute(X.numpy(), y.numpy(), theta.numpy(), 1e-2)
    updated = pru_unlearn(pre, np.array([0, 5, 9]))
    assert updated.shape == theta.shape
    assert torch.isfinite(updated).all()


# ----------------------------------------------------------------------
# Certificate and metrics
# ----------------------------------------------------------------------

def test_epsilon_formula_and_scaling():
    # Lemma 2: epsilon = sqrt(2 ln(1.5 / delta)) * eps_prime / sigma
    assert epsilon_from_eps_prime(1.0, 1.0, 1e-4) == pytest.approx(
        math.sqrt(2 * math.log(1.5e4)))
    assert epsilon_from_eps_prime(1.0, 10.0) == pytest.approx(
        epsilon_from_eps_prime(1.0, 1.0) / 10)
    assert epsilon_from_eps_prime(1.0, 0.0) == math.inf


def test_theorem_bound_has_no_fairness_term_for_bce():
    s = torch.tensor([0] * 40 + [1] * 60)
    theta = torch.ones(3, dtype=torch.float64)
    bce = theorem_eps_prime(theta, s, torch.tensor([0]), 1e-2, fair=False)
    # psi / (lambda^2 (n - m)) * (2 m g)^2 with psi = 1/4, g = 1, m = 1
    assert bce == pytest.approx(0.25 / (1e-4 * 99) * 4)
    fair = theorem_eps_prime(theta, s, torch.tensor([0]), 1e-2, fair=True)
    assert fair > bce


def test_metrics_on_hand_computed_example():
    # Group a: TPR 0.5, FPR 0.0; group b: TPR 1.0, FPR 0.5
    y = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
    s = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    pred = torch.tensor([1, 0, 0, 0, 1, 1, 1, 0])
    m = compute_metrics(pred, y, s)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["aeod"] == pytest.approx(0.5 * (0.5 + 0.5))
    assert m["eopp"] == pytest.approx(0.5)
    assert m["dp"] == pytest.approx(abs(1 / 4 - 3 / 4))
    assert m["subgroup_acc"] == pytest.approx(abs(3 / 4 - 3 / 4))
