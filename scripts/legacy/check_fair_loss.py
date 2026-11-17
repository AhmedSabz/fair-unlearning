"""
Fair objective on a toy problem: finite loss and gradient.

Legacy script from the project's first iteration (the pytest suite in
tests/test_reproduction.py covers the same ground). Run from the repo root:
    python -m scripts.legacy.check_fair_loss
"""

import torch

from src.objectives.fair_loss import fair_logistic_loss


def main():

    # Six samples, two features.
    X = torch.tensor([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 2.5],
        [3.0, 1.0],
        [2.5, 2.0],
        [1.0, 3.0],
    ], dtype=torch.float64)

    # Binary labels.
    y = torch.tensor(
        [0, 1, 0, 1, 1, 0],
        dtype=torch.float64
    )

    # Sensitive groups: 0 and 1.
    sensitive = torch.tensor([0, 0, 0, 1, 1, 1])

    # Model parameters.
    theta = torch.tensor(
        [0.1, -0.2],
        dtype=torch.float64,
        requires_grad=True
    )

    loss, components = fair_logistic_loss(
        theta,
        X,
        y,
        sensitive
    )

    # Calculate gradient.
    gradient = torch.autograd.grad(loss, theta)[0]

    print("Total loss:", loss.item())
    print("BCE:", components["bce"].item())
    print("Fairness:", components["fairness"].item())
    print("Regularization:", components["regularization"].item())
    print("Gradient:", gradient)

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()

    print("\nFair loss test passed!")


if __name__ == "__main__":
    main()
