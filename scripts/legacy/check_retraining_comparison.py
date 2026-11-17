"""
Newton step vs. retraining on a toy problem.

Legacy script from the project's first iteration (the pytest suite in
tests/test_reproduction.py covers the same ground). Run from the repo root:
    python -m scripts.legacy.check_retraining_comparison
"""

import torch

from src.objectives.train import train_model
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
        dtype=torch.float64,
    )

    sensitive = torch.tensor([0, 0, 0, 1, 1, 1])

    # Delete the first sample.
    X_remaining = X[1:]
    y_remaining = y[1:]
    sensitive_remaining = sensitive[1:]

    # 1. Train on the original dataset.
    theta_original = train_model(X, y, sensitive)

    # 2. Apply Newton unlearning.
    theta_unlearned, _ = newton_unlearn(
        theta_original,
        X_remaining,
        y_remaining,
        sensitive_remaining,
    )

    # 3. Retrain from scratch on the remaining dataset.
    theta_retrained = train_model(
        X_remaining,
        y_remaining,
        sensitive_remaining,
    )

    # 4. Compare the two parameter vectors.
    parameter_distance = torch.norm(
        theta_unlearned - theta_retrained
    ).item()

    print("Original trained parameters:")
    print(theta_original)

    print("\nNewton-unlearned parameters:")
    print(theta_unlearned)

    print("\nRetrained parameters:")
    print(theta_retrained)

    print("\nParameter distance:")
    print(parameter_distance)


if __name__ == "__main__":
    main()
