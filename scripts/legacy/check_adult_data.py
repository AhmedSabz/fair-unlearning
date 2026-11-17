"""
Sanity checks on the paper-matched Adult loader.

Legacy script from the project's first iteration (the pytest suite in
tests/test_reproduction.py covers the same ground). Run from the repo root:
    python -m scripts.legacy.check_adult_data
"""

import torch

from src.data.adult import load_adult


def main():
    data = load_adult()

    print("ADULT DATASET SUMMARY")
    print("---------------------")

    print("Training features:", data["X_train"].shape)
    print("Training labels:", data["y_train"].shape)
    print("Test features:", data["X_test"].shape)
    print("Test labels:", data["y_test"].shape)

    print("\nTraining income distribution:")
    print(data["y_train"].to(torch.long).bincount())

    print("\nTest income distribution:")
    print(data["y_test"].to(torch.long).bincount())

    print("\nTraining sensitive group distribution:")
    print(data["sensitive_train"].bincount())

    print("\nTest sensitive group distribution:")
    print(data["sensitive_test"].bincount())

    # Validate dimensions.
    assert data["X_train"].shape[0] == data["y_train"].shape[0]
    assert data["X_test"].shape[0] == data["y_test"].shape[0]
    assert data["X_train"].shape[1] == data["X_test"].shape[1]

    # Validate numerical values.
    assert data["X_train"].isfinite().all()
    assert data["X_test"].isfinite().all()

    # Validate labels.
    assert set(data["y_train"].unique().tolist()).issubset({0.0, 1.0})
    assert set(data["y_test"].unique().tolist()).issubset({0.0, 1.0})

    # Validate sensitive groups.
    assert data["sensitive_train"].unique().numel() == 2
    assert data["sensitive_test"].unique().numel() == 2

    print("\nAdult dataset loading test passed!")


if __name__ == "__main__":
    main()
