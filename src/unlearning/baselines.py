"""
Unlearning baselines from the paper's Section 5.1, for BCE-trained models.

    SISA (Bourtoule et al., 2020): split the training set into shards, train one
        model per shard, predict by majority vote, unlearn by retraining only
        the shards that held deleted rows. Exact unlearning.
    PRU  (Izzo et al., 2021): projective residual update. Approximates the
        retrained model with leave-k-out predictions and a projected update,
        avoiding a full Hessian solve over the remaining data.

Both follow the authors' official code (fair_retraining.py, adapted there from
Izzo et al.'s code): 5 SISA shards with a vote of at least 3, and PRU applied
to logistic regression through the working response
    Z = X theta + W^{-1} (y - sigmoid(X theta)),  W = diag(p (1 - p)) + 1e-6 I.

One deliberate difference: the official lin_res builds the PRU gradient from
X[i] (rows 0..k-1 of the training set) instead of X[ind[i]] (the deleted rows).
Izzo et al.'s Algorithm uses the deleted rows; we follow the algorithm.
"""

import numpy as np
import torch

from src.objectives.train import train_model


# ----------------------------------------------------------------------
# SISA
# ----------------------------------------------------------------------

NUM_SHARDS = 5


def sisa_shards(num_rows, num_shards=NUM_SHARDS):
    """Contiguous index blocks, like np.split in the official code."""
    return np.array_split(np.arange(num_rows), num_shards)


def sisa_train(X, y, sensitive, shards, lambda_reg, max_iter=500):
    """Train one BCE model per shard (no fairness term, no noise)."""
    return [
        train_model(X[idx], y[idx], sensitive[idx],
                    lambda_reg=lambda_reg, gamma=0.0, max_iter=max_iter)
        for idx in shards
    ]


def sisa_unlearn(X, y, sensitive, shards, shard_thetas, deleted,
                 lambda_reg, max_iter=500):
    """Retrain only the shards that contain deleted rows."""
    deleted = set(np.asarray(deleted).tolist())
    new_thetas = []
    for idx, theta in zip(shards, shard_thetas):
        if not deleted.intersection(idx.tolist()):
            new_thetas.append(theta)
            continue
        keep = np.array([i for i in idx if i not in deleted])
        new_thetas.append(
            train_model(X[keep], y[keep], sensitive[keep],
                        lambda_reg=lambda_reg, gamma=0.0, max_iter=max_iter)
        )
    return new_thetas


def sisa_predict(shard_thetas, X):
    """Majority vote: predict 1 when at least 3 of the 5 shard models say 1."""
    votes = torch.stack([(X @ theta >= 0).long() for theta in shard_thetas])
    return (votes.sum(dim=0) >= (len(shard_thetas) // 2 + 1)).long()


# ----------------------------------------------------------------------
# PRU
# ----------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def pru_precompute(X, y, theta, lambda_reg):
    """
    Quantities that depend only on the full training set and theta_original.
    Done once, before any deletion request (the official code does the same).
    X already contains the intercept column.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    logits = X @ theta
    p = _sigmoid(logits)
    w = p * (1 - p) + 1e-6            # diag of W_full, nonsingular
    Z = logits + (y - p) / w          # working response

    d = X.shape[1]
    A = np.linalg.inv(X.T @ (w[:, None] * X) + lambda_reg * np.eye(d))
    AXtWZ = A @ (X.T @ (w * Z))       # so that (H Z)_i = x_i^T AXtWZ
    return {"X": X, "Z": Z, "w": w, "A": A, "AXtWZ": AXtWZ, "theta": theta}


def _lko_predictions(pre, ind):
    """
    Leave-k-out predictions on the deleted rows (Izzo et al., LKO_pred).
    Uses only the k rows of the hat matrix H = X A X^T W that are needed.
    """
    X, Z, w, A = pre["X"], pre["Z"], pre["w"], pre["A"]
    X_ind = X[ind]
    H_ind_Z = X_ind @ pre["AXtWZ"]                  # (H Z)[ind]
    H_ind_ind = (X_ind @ A @ X_ind.T) * w[ind][None, :]
    h_diag = np.diag(H_ind_ind)

    loo = (Z[ind] - H_ind_Z) / (1 - h_diag)         # leave-one-out residuals
    S = -H_ind_ind / (1 - h_diag)[:, None]
    np.fill_diagonal(S, 1.0)
    lko_residual = np.linalg.solve(S, loo)
    return Z[ind] - lko_residual


def pru_unlearn(pre, deleted):
    """Projective residual update (Izzo et al., Algorithm 3 for logistic)."""
    ind = np.asarray(deleted)
    X, theta = pre["X"], pre["theta"]
    X_ind = X[ind]
    k, d = X_ind.shape

    lko = _lko_predictions(pre, ind)

    # Orthonormal basis of the deleted rows (Gram-Schmidt via QR).
    if k <= d:
        q, r = np.linalg.qr(X_ind.T)
    else:
        q, r = np.linalg.qr(X_ind.T, mode="complete")
    U, C = q.T, r.T
    eigenvalues, a = np.linalg.eigh(C.T @ C)
    V = a.T @ U

    grad = X_ind.T @ (X_ind @ theta - lko)

    step = np.zeros(d)
    for i in range(min(d, k)):
        if eigenvalues[i] > 1e-10:
            step += (V[i] @ grad) / eigenvalues[i] * V[i]

    return torch.tensor(theta - step, dtype=torch.float64)
