import torch

from src.objectives.fair_loss import fair_logistic_loss


def compute_gradient(theta, X, y, sensitive,
                     lambda_reg=1e-4, gamma=1.0, noise=None):
    """
    Compute the gradient of the fair logistic regression objective.
    noise: optional perturbation vector b (see fair_logistic_loss).
    """

    theta = theta.detach().clone().requires_grad_(True)

    loss, _ = fair_logistic_loss(
        theta, X, y, sensitive,
        lambda_reg=lambda_reg,
        gamma=gamma,
        noise=noise,
    )

    gradient = torch.autograd.grad(loss, theta)[0]

    return gradient.detach()


def compute_hessian(theta, X, y, sensitive,
                    lambda_reg=1e-4, gamma=1.0, noise=None):
    """
    Compute the Hessian matrix of the fair logistic regression objective.
    The noise term b^T theta / n is linear, so it does not change the Hessian;
    the argument is accepted so callers can pass the same settings everywhere.
    """

    theta = theta.detach().clone().requires_grad_(True)

    def objective(params):
        loss, _ = fair_logistic_loss(
            params, X, y, sensitive,
            lambda_reg=lambda_reg,
            gamma=gamma,
            noise=noise,
        )
        return loss

    hessian = torch.autograd.functional.hessian(
        objective,
        theta
    )

    return hessian.detach()
