
import torch
import torch.nn.functional as F


def fair_logistic_loss(
    theta,
    X,
    y,
    sensitive,
    lambda_reg=1e-4,
    gamma=1.0,
    noise=None,
):
    """
    Fair logistic regression objective.

    Args:
        theta: Model parameters, shape (num_features,)
        X: Feature matrix, shape (num_samples, num_features)
        y: Binary labels, shape (num_samples,)
        sensitive: Binary sensitive attribute, shape (num_samples,)
                   Group 0 and Group 1
        lambda_reg: L2 regularization strength
        gamma: Fairness penalty strength
        noise: Optional objective-perturbation vector b, shape
               (num_features,). Adds b^T theta / n, the paper's noisy loss
               L^b = L + b^T theta written per sample (the official code
               divides by the number of training rows the same way).
               Use the SAME b for original training and retraining.

    Returns:
        total_loss: Combined objective
        components: Dictionary containing individual losses
    """

    # Make sure inputs use compatible floating-point types.
    X = X.to(dtype=theta.dtype)
    y = y.to(dtype=theta.dtype)
    sensitive = sensitive.to(device=theta.device)

    # --------------------------------------------------
    # 1. Binary Cross Entropy
    # --------------------------------------------------

    logits = X @ theta

    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        y
    )

    # --------------------------------------------------
    # 2. Fairness Penalty
    # --------------------------------------------------

    if gamma == 0:
        # Plain BCE: no fairness term, so groups are not needed (e.g. a SISA
        # shard may contain only one group after minority deletions).
        fairness_loss = torch.zeros((), dtype=theta.dtype, device=theta.device)
    else:
        group_a = sensitive == 0
        group_b = sensitive == 1

        Xa = X[group_a]
        Xb = X[group_b]

        na = Xa.shape[0]
        nb = Xb.shape[0]

        if na == 0 or nb == 0:
            raise ValueError("Both sensitive groups must have samples.")

        fairness_vector = torch.zeros_like(theta)

        # Compare examples with the same true label.
        for label in (0, 1):

            Xa_y = Xa[y[group_a] == label]
            Xb_y = Xb[y[group_b] == label]

            count_a = Xa_y.shape[0]
            count_b = Xb_y.shape[0]

            fairness_vector += (
                count_b * Xa_y.sum(dim=0)
                - count_a * Xb_y.sum(dim=0)
            ) / (na * nb)

        fairness_difference = torch.dot(fairness_vector, theta)

        fairness_loss = fairness_difference ** 2

    # --------------------------------------------------
    # 3. L2 Regularization
    # --------------------------------------------------

    regularization_loss = (
        lambda_reg / 2
    ) * torch.sum(theta ** 2)

    # --------------------------------------------------
    # 4. Objective perturbation (for the unlearning certificate)
    # --------------------------------------------------

    if noise is not None:
        noise_loss = torch.dot(noise.to(dtype=theta.dtype), theta) / X.shape[0]
    else:
        noise_loss = torch.zeros((), dtype=theta.dtype, device=theta.device)

    # --------------------------------------------------
    # 5. Combined Objective
    # --------------------------------------------------

    total_loss = (
        bce_loss
        + gamma * fairness_loss
        + regularization_loss
        + noise_loss
    )

    components = {
        "bce": bce_loss,
        "fairness": fairness_loss,
        "regularization": regularization_loss,
        "noise": noise_loss,
    }

    return total_loss, components
