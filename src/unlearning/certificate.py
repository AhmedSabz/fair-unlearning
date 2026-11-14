"""
(epsilon, delta) unlearning certificate, Oesterling et al. (AISTATS 2024).

Both training and retraining minimize the noisy loss L(theta) + b^T theta with
b ~ N(0, sigma^2 I). If the unlearned model's gradient on the remaining data
is at most eps_prime, the unlearned and retrained models are
(epsilon, delta)-indistinguishable with (Lemma 2, from Guo et al. 2020)

    delta = 1.5 exp(-c^2 / 2),   sigma = c * eps_prime / epsilon
    =>  epsilon = sqrt(2 ln(1.5 / delta)) * eps_prime / sigma.

The paper bounds eps_prime two ways and reports the smaller epsilon. We
compute both but use only the data-dependent bound:

    data-dependent     Appendix A.2 (Guo et al., Corollary 1), USED:
        psi * ||X'||_2 * ||step||_2 * ||X' step||_2,
        X' = remaining training rows (incl. the intercept column),
        step = theta_unlearned - theta_original (an undamped Newton step).
        Only the logistic term's Hessian varies with theta (the L2 and
        fairness terms are quadratic, the noise term linear), so
            G(theta-) = int_0^1 X'^T D_t X' step dt,
            |D_t,ii| <= c t |x_i^T step|,  c = max|l'''| = 1/(6 sqrt 3),
        giving ||G(theta-)|| <= (c / 2) R ||X'|| ||step|| ||X' step||, with
        R = max_i ||x_i||. psi = 1/4 covers this whenever c R <= psi
        (R <= 2.6; this also covers the paper's mean-value form, which drops
        the 1/2). Our training rows have R <= sqrt(2). For larger rows the
        bound is not established and is returned as inf.
    theorem bound      Lemma 1 / Eq. (13), REFERENCE ONLY (never selected):
        psi / (lambda^2 (n - m)) * (2 m g + ||theta*|| * F)^2,
        F = 8 (m_a^2 n_b^2 + m_b^2 n_a^2 - m_a^2 m_b^2)(n - m) / (n_a^2 n_b^2)
        with psi = 1/4 and g = 1. Its derivation does not hold here: it
        assumes ||x_i|| <= 1 (our intercept gives up to sqrt(2)); the step
        from (28) to (29) drops gamma from the fairness term; (25)/(27) bound
        ||theta*|| from the noise-free stationarity condition and divide by a
        matrix. Kept as a column for comparison with the paper only.

Both are for the summed (not averaged) loss, as in the paper's proofs.

The epsilon values computed here are POST-HOC ESTIMATES based on a
residual-gradient bound, not independently established formal unlearning
guarantees. A valid eps_prime bounds the gradient residual only. Lemma 2
needs eps_prime fixed before the noise is calibrated, but we compute it
after the noise-dependent update; it also assumes exact optima (ours stop at
a gradient norm of about 1e-8), and preprocessing fitted before deletion is
not unlearned (see README, "Noise and certificate").

The data-dependent bound assumes theta_unlearned is an UNDAMPED Newton step
from theta_original on the remaining-data objective (same lambda, gamma and
noise b), i.e. newton_unlearn(..., damping=0) on X_remaining. It is not
valid for arbitrary parameter updates; the function cannot check this.

Invalid inputs (non-finite values, empty or mismatched arrays, negative or
non-finite sigma, delta outside (0, 1)) raise ValueError. sigma = 0 (no
noise) gives epsilon = inf: no certificate.

Note on the official code: its eps_delta_experiments.py divides by sigma^2
instead of sigma. The two agree at sigma = 1 (all main results). We report
the Lemma 2 formula as `epsilon` and the official-code value separately as
`epsilon_official_code`, so the paper's Figure 2 can be compared directly.
"""

import math

import torch

PSI = 0.25   # constant used by both bounds (l'' <= 1/4; a loose Lipschitz
             # constant for l'', whose exact one is LOGISTIC_L3 below)
G = 1.0      # theorem bound: ||l'|| <= 1, which requires ||x|| <= 1
LOGISTIC_L3 = 1 / (6 * math.sqrt(3))   # max |l'''| of the logistic loss
# Largest row norm for which PSI covers the data-dependent bound.
DATA_BOUND_MAX_ROW_NORM = PSI / LOGISTIC_L3   # = 1.5 sqrt(3) ~ 2.598


# ----------------------------------------------------------------------
# Input validation (no effect on the values computed for valid inputs)
# ----------------------------------------------------------------------

def _finite_tensor(name, value, ndim):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, "
                         f"got shape {tuple(value.shape)}")
    if value.numel() == 0:
        raise ValueError(f"{name} is empty")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _check_noise(sigma, delta):
    if not (isinstance(sigma, (int, float)) and math.isfinite(sigma)
            and sigma >= 0):
        raise ValueError(f"sigma must be a finite number >= 0, got {sigma!r}")
    if not (isinstance(delta, (int, float)) and 0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")


def theorem_eps_prime(theta_original, sensitive_train, deleted, lambda_reg,
                      fair=True):
    """
    Eq. (13) as published, for reference only: certificate() never selects
    it, because its derivation does not hold for our inputs (module
    docstring).
    """
    _finite_tensor("theta_original", theta_original, 1)
    _finite_tensor("sensitive_train", sensitive_train, 1)
    if not (isinstance(lambda_reg, (int, float)) and math.isfinite(lambda_reg)
            and lambda_reg > 0):
        raise ValueError(f"lambda_reg must be a finite number > 0, "
                         f"got {lambda_reg!r}")
    s = sensitive_train
    n = len(s)
    m = len(deleted)
    if m >= n:
        raise ValueError(f"cannot delete {m} of {n} rows: no data would remain")
    n_a = int((s == 0).sum())
    n_b = int((s == 1).sum())
    m_a = int((s[deleted] == 0).sum())
    m_b = int((s[deleted] == 1).sum())

    if fair:
        if n_a == 0 or n_b == 0:
            raise ValueError("the fairness term needs both groups present")
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
    """
    Data-dependent bound (Appendix A.2) on the gradient residual of the
    summed noisy loss at theta_unlearned. Valid only when theta_unlearned is
    an undamped Newton step from theta_original on the objective over
    X_remaining (module docstring); not for arbitrary updates. Returns inf
    when a row of X_remaining is longer than DATA_BOUND_MAX_ROW_NORM, where
    PSI no longer covers the bound.
    """
    _finite_tensor("X_remaining", X_remaining, 2)
    _finite_tensor("theta_original", theta_original, 1)
    _finite_tensor("theta_unlearned", theta_unlearned, 1)
    d = X_remaining.shape[1]
    if theta_original.shape[0] != d or theta_unlearned.shape[0] != d:
        raise ValueError(
            f"X_remaining has {d} columns but theta_original/theta_unlearned "
            f"have {theta_original.shape[0]}/{theta_unlearned.shape[0]} entries")
    if torch.linalg.norm(X_remaining, dim=1).max() > DATA_BOUND_MAX_ROW_NORM:
        return math.inf
    step = theta_unlearned - theta_original
    return float(
        PSI
        * torch.linalg.matrix_norm(X_remaining, ord=2)
        * torch.linalg.norm(step)
        * torch.linalg.norm(X_remaining @ step)
    )


def epsilon_from_eps_prime(eps_prime, sigma, delta=1e-4):
    """
    Lemma 2: epsilon for noise level sigma and failure probability delta.
    sigma = 0 (no noise) or eps_prime = inf gives inf (no certificate).
    """
    _check_noise(sigma, delta)
    if not (isinstance(eps_prime, (int, float)) and eps_prime >= 0):
        raise ValueError(f"eps_prime must be a number >= 0, got {eps_prime!r}")
    if sigma == 0:
        return math.inf
    return math.sqrt(2 * math.log(1.5 / delta)) * eps_prime / sigma


def certificate(X_remaining, sensitive_train, deleted, theta_original,
                theta_unlearned, lambda_reg, sigma, delta=1e-4, fair=True):
    """
    All certificate quantities for one unlearning request.

    Returns eps_prime from each bound, epsilon from the data-dependent bound
    only (inf if that bound does not apply), and the official-code variant
    (divides by sigma^2). eps_prime_theorem is a reference value, never used.
    epsilon is a post-hoc estimate, not a formal guarantee (module docstring).
    """
    _check_noise(sigma, delta)
    eps_theorem = theorem_eps_prime(
        theta_original, sensitive_train, deleted, lambda_reg, fair=fair
    )
    eps_data = data_dependent_eps_prime(
        X_remaining, theta_original, theta_unlearned
    )
    eps_prime = eps_data          # the theorem bound is never selected
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
