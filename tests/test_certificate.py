"""
Tests for certificate-bound selection (src/unlearning/certificate.py).

Synthetic data only, so they run everywhere. These check how bounds are
computed and selected; they do not prove an unlearning guarantee.
"""

import math

import numpy as np
import pytest
import torch

from src.objectives.derivatives import compute_gradient
from src.objectives.train import train_model
from src.unlearning.certificate import (
    DATA_BOUND_MAX_ROW_NORM, LOGISTIC_L3, PSI, certificate,
    data_dependent_eps_prime, epsilon_from_eps_prime, theorem_eps_prime,
)
from src.unlearning.newton import newton_unlearn


def rows_with_norm(norms, d=4, seed=0):
    """Rows with the given Euclidean norms, last column an intercept of 1."""
    gen = torch.Generator().manual_seed(seed)
    feats = torch.randn(len(norms), d - 1, generator=gen, dtype=torch.float64)
    feats = feats / feats.norm(dim=1, keepdim=True)
    out = []
    for f, r in zip(feats, norms):
        out.append(torch.cat([f * math.sqrt(r**2 - 1), torch.ones(1,
                              dtype=torch.float64)]))
    return torch.stack(out)


def manual_data_bound(X, theta0, theta1):
    step = theta1 - theta0
    return float(PSI * torch.linalg.matrix_norm(X, ord=2) * step.norm()
                 * (X @ step).norm())


def test_data_bound_threshold_constant():
    assert LOGISTIC_L3 == pytest.approx(1 / (6 * math.sqrt(3)))
    assert DATA_BOUND_MAX_ROW_NORM == pytest.approx(1.5 * math.sqrt(3))
    assert DATA_BOUND_MAX_ROW_NORM > math.sqrt(2)     # our rows are covered


@pytest.mark.parametrize("max_norm", [1.0, math.sqrt(2), 2.5])
def test_data_bound_eligible_norms(max_norm):
    X = rows_with_norm([1.0, 1.2, max_norm, 1.1])
    theta0 = torch.zeros(4, dtype=torch.float64)
    theta1 = torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch.float64)
    value = data_dependent_eps_prime(X, theta0, theta1)
    assert math.isfinite(value)
    assert value == pytest.approx(manual_data_bound(X, theta0, theta1),
                                  rel=1e-12)


@pytest.mark.parametrize("max_norm", [2.7, 5.0])
def test_data_bound_ineligible_norms(max_norm):
    X = rows_with_norm([1.0, max_norm, 1.1])
    theta0 = torch.zeros(4, dtype=torch.float64)
    theta1 = torch.full((4,), 0.1, dtype=torch.float64)
    assert data_dependent_eps_prime(X, theta0, theta1) == math.inf
    s = torch.tensor([0, 1, 0])
    cert = certificate(X, s, torch.tensor([0]), theta0, theta1, 1e-2, 1.0)
    assert cert["epsilon"] == math.inf
    assert cert["epsilon_official_code"] == math.inf


def test_theorem_bound_is_never_selected():
    """Even when the reference theorem value is smaller, epsilon uses A.2."""
    X = rows_with_norm([1.0, 1.3, 1.4, 1.2, 1.1, 1.05])
    s = torch.tensor([0, 1, 0, 1, 0, 1])
    deleted = torch.tensor([0])
    theta0 = torch.zeros(4, dtype=torch.float64)
    theta1 = torch.full((4,), 5.0, dtype=torch.float64)   # large step
    lam, sigma, delta = 100.0, 2.0, 1e-3                  # tiny theorem value
    cert = certificate(X[1:], s, deleted, theta0, theta1, lam, sigma, delta)
    assert cert["eps_prime_theorem"] < cert["eps_prime_data"]
    assert cert["eps_prime_theorem"] == theorem_eps_prime(theta0, s, deleted,
                                                          lam)
    c = math.sqrt(2 * math.log(1.5 / delta))
    assert cert["epsilon"] == pytest.approx(c * cert["eps_prime_data"] / sigma)
    assert cert["epsilon_official_code"] == pytest.approx(
        c * cert["eps_prime_data"] / sigma**2)


def test_epsilon_conversion():
    for eps_prime, sigma, delta in [(1.0, 1.0, 1e-4), (0.3, 10.0, 1e-2)]:
        expected = math.sqrt(2 * math.log(1.5 / delta)) * eps_prime / sigma
        assert epsilon_from_eps_prime(eps_prime, sigma, delta) == \
            pytest.approx(expected)
    assert epsilon_from_eps_prime(1.0, 0.0) == math.inf
    assert epsilon_from_eps_prime(math.inf, 1.0) == math.inf


@pytest.mark.parametrize("gamma", [0.0, 10.0])
def test_data_bound_on_real_unlearning_step(gamma):
    """
    End-to-end on a small problem with intercept rows of norm up to sqrt(2):
    the bound is finite and at least the actual residual. A regression check
    of the computation, not a proof (the proof is the derivation).
    """
    gen = torch.Generator().manual_seed(1)
    feats = torch.randn(80, 4, generator=gen, dtype=torch.float64)
    feats = feats / feats.norm(dim=1).max()
    X = torch.cat([feats, torch.ones(80, 1, dtype=torch.float64)], dim=1)
    s = (torch.rand(80, generator=gen) < 0.4).long()
    y = (torch.rand(80, generator=gen, dtype=torch.float64)
         < torch.sigmoid(X @ torch.tensor([2.0, -1.0, 0.5, 1.0, 0.0],
                                          dtype=torch.float64))).double()
    b = torch.randn(5, generator=gen, dtype=torch.float64)
    lam = 1e-2
    theta0 = train_model(X, y, s, lambda_reg=lam, gamma=gamma, noise=b)
    keep = torch.ones(80, dtype=torch.bool); keep[:8] = False
    theta1, _ = newton_unlearn(theta0, X[keep], y[keep], s[keep],
                               lambda_reg=lam, gamma=gamma, noise=b)
    assert X.norm(dim=1).max() <= math.sqrt(2) + 1e-12
    bound = data_dependent_eps_prime(X[keep], theta0, theta1)
    # Summed-form residual: mean-form gradient times the number of rows.
    residual = compute_gradient(theta1, X[keep], y[keep], s[keep],
                                lambda_reg=lam, gamma=gamma,
                                noise=b).norm().item() * int(keep.sum())
    assert math.isfinite(bound)
    assert residual <= bound
    cert = certificate(X[keep], s, torch.arange(8), theta0, theta1, lam, 1.0,
                       fair=gamma > 0)
    assert cert["eps_prime_data"] == bound
    assert np.isfinite(cert["epsilon"])


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

def _valid_inputs():
    X = rows_with_norm([1.0, 1.2, 1.1])
    theta0 = torch.zeros(4, dtype=torch.float64)
    theta1 = torch.full((4,), 0.1, dtype=torch.float64)
    return X, theta0, theta1


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_inputs_raise(bad):
    X, theta0, theta1 = _valid_inputs()
    for args in ((X.clone().index_fill_(1, torch.tensor([0]), bad), theta0, theta1),
                 (X, theta0.clone().fill_(bad), theta1),
                 (X, theta0, theta1.clone().fill_(bad))):
        with pytest.raises(ValueError, match="non-finite"):
            data_dependent_eps_prime(*args)
    with pytest.raises(ValueError, match="non-finite"):
        theorem_eps_prime(theta0.clone().fill_(bad), torch.tensor([0, 1, 0]),
                          torch.tensor([0]), 1e-2)


def test_empty_and_mismatched_inputs_raise():
    X, theta0, theta1 = _valid_inputs()
    with pytest.raises(ValueError, match="empty"):
        data_dependent_eps_prime(X[:0], theta0, theta1)
    with pytest.raises(ValueError, match="2-dimensional"):
        data_dependent_eps_prime(X[0], theta0, theta1)
    with pytest.raises(ValueError, match="columns"):
        data_dependent_eps_prime(X, theta0[:3], theta1[:3])
    with pytest.raises(ValueError, match="torch.Tensor"):
        data_dependent_eps_prime(X.numpy(), theta0, theta1)
    s = torch.tensor([0, 1, 0])
    with pytest.raises(ValueError, match="no data would remain"):
        theorem_eps_prime(theta0, s, torch.tensor([0, 1, 2]), 1e-2)
    with pytest.raises(ValueError, match="both groups"):
        theorem_eps_prime(theta0, torch.tensor([0, 0, 0]), torch.tensor([0]),
                          1e-2)
    with pytest.raises(ValueError, match="lambda_reg"):
        theorem_eps_prime(theta0, s, torch.tensor([0]), 0.0)


@pytest.mark.parametrize("sigma, delta", [
    (-1.0, 1e-4), (math.nan, 1e-4), (math.inf, 1e-4),
    (1.0, 0.0), (1.0, 1.0), (1.0, 1.5), (1.0, math.nan),
])
def test_invalid_sigma_or_delta_raise(sigma, delta):
    X, theta0, theta1 = _valid_inputs()
    with pytest.raises(ValueError, match="sigma|delta"):
        epsilon_from_eps_prime(1.0, sigma, delta)
    with pytest.raises(ValueError, match="sigma|delta"):
        certificate(X, torch.tensor([0, 1, 0, 1]), torch.tensor([3]), theta0,
                    theta1, 1e-2, sigma, delta)


def test_invalid_eps_prime_raises_and_zero_sigma_means_no_certificate():
    for bad in (-1.0, math.nan):
        with pytest.raises(ValueError, match="eps_prime"):
            epsilon_from_eps_prime(bad, 1.0)
    X, theta0, theta1 = _valid_inputs()
    cert = certificate(X, torch.tensor([0, 1, 0, 1]), torch.tensor([3]),
                       theta0, theta1, 1e-2, 0.0)
    assert cert["epsilon"] == math.inf
    assert cert["epsilon_official_code"] == math.inf
    assert math.isfinite(cert["eps_prime_data"])


def test_validation_does_not_change_valid_values():
    """Same numbers as the plain formulas for valid inputs."""
    X, theta0, theta1 = _valid_inputs()
    s, deleted, lam, sigma, delta = (torch.tensor([0, 1, 0, 1]),
                                     torch.tensor([3]), 1e-2, 1.7, 1e-3)
    cert = certificate(X, s, deleted, theta0, theta1, lam, sigma, delta)
    data = manual_data_bound(X, theta0, theta1)
    c = math.sqrt(2 * math.log(1.5 / delta))
    assert cert["eps_prime_data"] == data
    assert cert["epsilon"] == c * data / sigma
    assert cert["epsilon_official_code"] == c * data / sigma**2
    n, m, n_a, n_b, m_a, m_b = 4, 1, 2, 2, 0, 1
    F = 8 * (m_a**2 * n_b**2 + m_b**2 * n_a**2 - m_a**2 * m_b**2) * (n - m) \
        / (n_a**2 * n_b**2)
    assert cert["eps_prime_theorem"] == PSI / (lam**2 * (n - m)) * (
        2 * m * 1.0 + float(theta0.norm()) * F) ** 2
