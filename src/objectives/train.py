
import torch

from src.objectives.fair_loss import fair_logistic_loss


def train_model(
    X,
    y,
    sensitive,
    lambda_reg=1e-4,
    gamma=1.0,
    max_iter=500,
    noise=None,
):
    """
    Minimize the fair logistic loss with L-BFGS (strong Wolfe line search).
    noise: optional perturbation vector b (see fair_logistic_loss).
    """
    theta = torch.zeros(
        X.shape[1],
        dtype=X.dtype,
        device=X.device,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        loss, _ = fair_logistic_loss(
            theta,
            X,
            y,
            sensitive,
            lambda_reg=lambda_reg,
            gamma=gamma,
            noise=noise,
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    return theta.detach()
