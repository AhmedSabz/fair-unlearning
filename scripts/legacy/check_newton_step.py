"""
One Newton unlearning step on a toy problem.

Legacy script from the project's first iteration (the pytest suite in
tests/test_reproduction.py covers the same ground). Run from the repo root:
    python -m scripts.legacy.check_newton_step
"""

import torch

from src.unlearning.newton import newton_unlearn


def main():
    X = torch.tensor([
        [1.0, 0.5],
        [1.0, 1.0],
        [1.0, 1.5],
        [1.0, 2.0],
        [1.0, 2.5],
        [1.0, 3.0],
    ], dtype=torch.float64)

    y = torch.tensor(
        [0, 1, 0, 1, 1, 0],
        dtype=torch.float64
    )

    sensitive = torch.tensor([0, 0, 0, 1, 1, 1])

    theta_original = torch.tensor(
        [0.1, -0.2],
        dtype=torch.float64
    )

    # Simulate deleting the first sample.
    X_remaining = X[1:]
    y_remaining = y[1:]
    sensitive_remaining = sensitive[1:]

    theta_unlearned, info = newton_unlearn(
        theta_original,
        X_remaining,
        y_remaining,
        sensitive_remaining,
    )

    print("Original parameters:")
    print(theta_original)

    print("\nUnlearned parameters:")
    print(theta_unlearned)

    print("\nNewton step:")
    print(info["step"])

    assert theta_unlearned.shape == theta_original.shape
    assert torch.isfinite(theta_unlearned).all()
    assert torch.isfinite(info["hessian"]).all()

    print("\nUnlearning test passed!")


if __name__ == "__main__":
    main()
