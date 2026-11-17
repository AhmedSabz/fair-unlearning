"""
Gradient and Hessian on a toy problem: shapes, finiteness, symmetry.

Legacy script from the project's first iteration (the pytest suite in
tests/test_reproduction.py covers the same ground). Run from the repo root:
    python -m scripts.legacy.check_derivatives
"""

import torch

from src.objectives.derivatives import (
    compute_gradient,
    compute_hessian
)


def main():

    X = torch.tensor([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 2.5],
        [3.0, 1.0],
        [2.5, 2.0],
        [1.0, 3.0],
    ], dtype=torch.float64)

    y = torch.tensor(
        [0, 1, 0, 1, 1, 0],
        dtype=torch.float64
    )

    sensitive = torch.tensor([0, 0, 0, 1, 1, 1])

    theta = torch.tensor(
        [0.1, -0.2],
        dtype=torch.float64
    )

    gradient = compute_gradient(
        theta, X, y, sensitive
    )

    hessian = compute_hessian(
        theta, X, y, sensitive
    )

    print("Gradient:")
    print(gradient)

    print("\nHessian:")
    print(hessian)

    print("\nHessian shape:", hessian.shape)

    # Verify dimensions.
    assert gradient.shape == (2,)
    assert hessian.shape == (2, 2)

    # Verify numerical validity.
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian).all()

    # Hessian should be symmetric.
    assert torch.allclose(
        hessian,
        hessian.T,
        atol=1e-8
    )

    print("\nDerivative tests passed!")


if __name__ == "__main__":
    main()