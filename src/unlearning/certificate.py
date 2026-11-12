"""
(epsilon, delta) unlearning certificate, Oesterling et al. (AISTATS 2024).

Both training and retraining minimize the noisy loss L(theta) + b^T theta with
b ~ N(0, sigma^2 I). If the unlearned model's gradient on the remaining data
is at most eps_prime, the unlearned and retrained models are
(epsilon, delta)-indistinguishable with (Lemma 2, from Guo et al. 2020)

    delta = 1.5 exp(-c^2 / 2),   sigma = c * eps_prime / epsilon
    =>  epsilon = sqrt(2 ln(1.5 / delta)) * eps_prime / sigma.

The paper bounds eps_prime two ways and reports the smaller epsilon:

    theorem bound      Lemma 1 / Eq. (13), worst case over data:
        psi / (lambda^2 (n - m)) * (2 m g + ||theta*|| * F)^2,
        F = 8 (m_a^2 n_b^2 + m_b^2 n_a^2 - m_a^2 m_b^2)(n - m) / (n_a^2 n_b^2)
        with psi = 1/4 and g = 1 for logistic loss. F = 0 for plain BCE
        (no fairness term), which recovers Guo et al.'s bound.
    data-dependent     Appendix A.2 (Guo et al., Corollary 1):
        psi * ||X'||_2 * ||step||_2 * ||X' step||_2,
        X' = remaining data, step = theta_unlearned - theta_original.

Both are for the summed (not averaged) loss, as in the paper's proofs.

Note on the official code: its eps_delta_experiments.py divides by sigma^2
instead of sigma. The two agree at sigma = 1 (all main results). We report
the Lemma 2 formula as `epsilon` and the official-code value separately as
`epsilon_official_code`, so the paper's Figure 2 can be compared directly.
"""

import math

import torch

PSI = 0.25   # logistic loss: second derivative is 1/4-Lipschitz
G = 1.0      # logistic loss: ||l'|| <= 1 for ||x|| <= 1


def theorem_eps_prime(theta_original, sensitive_train, deleted, lambda_reg,
                      fair=True):
    """Gradient-norm bound from Lemma 1 (Eq. 13)."""
    s = sensitive_train
    n = len(s)
    m = len(deleted)
    n_a = int((s == 0).sum())
    n_b = int((s == 1).sum())
    m_a = int((s[deleted] == 0).sum())
    m_b = int((s[deleted] == 1).sum())

    if fair:
        fairness_term = (
            8 * (m_a**2 * n_b**2 + m_b**2 * n_a**2 - m_a**2 * m_b**2)
            * (n - m) / (n_a**2 * n_b**2)
        )
    else:
        fairness_term = 0.0

    theta_norm = float(torch.linalg.norm(theta_original))
    return (PSI / (lambda_reg**2 * (n - m))
            * (2 * m * G + theta_norm * fairness_term) ** 2)


def data_dependent_eps_prime(X_remaining, theta_original, theta_unlearned):
    """Tighter, data-dependent bound (Appendix A.2)."""
    step = theta_unlearned - theta_original
    return float(
        PSI
        * torch.linalg.matrix_norm(X_remaining, ord=2)
        * torch.linalg.norm(step)
        * torch.linalg.norm(X_remaining @ step)
    )


def epsilon_from_eps_prime(eps_prime, sigma, delta=1e-4):
    """Lemma 2: epsilon for noise level sigma and failure probability delta."""
    if sigma <= 0:
        return math.inf
    return math.sqrt(2 * math.log(1.5 / delta)) * eps_prime / sigma


def certificate(X_remaining, sensitive_train, deleted, theta_original,
                theta_unlearned, lambda_reg, sigma, delta=1e-4, fair=True):
    """
    All certificate quantities for one unlearning request.

    Returns eps_prime from each bound, the epsilon the paper reports
    (smaller of the two), and the official-code variant (divides by sigma^2).
    """
    eps_theorem = theorem_eps_prime(
        theta_original, sensitive_train, deleted, lambda_reg, fair=fair
    )
    eps_data = data_dependent_eps_prime(
        X_remaining, theta_original, theta_unlearned
    )
    eps_prime = min(eps_theorem, eps_data)
    epsilon = epsilon_from_eps_prime(eps_prime, sigma, delta)
    epsilon_code = (
        math.sqrt(2 * math.log(1.5 / delta)) * eps_prime / sigma**2
        if sigma > 0 else math.inf
    )
    return {
        "eps_prime_theorem": eps_theorem,
        "eps_prime_data": eps_data,
        "epsilon": epsilon,
        "epsilon_official_code": epsilon_code,
        "delta": delta,
        "sigma": sigma,
    }
