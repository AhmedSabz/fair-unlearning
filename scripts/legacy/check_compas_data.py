"""
Sanity checks on the paper-matched COMPAS loader.

Legacy script from the project's first iteration (the pytest suite in
tests/test_reproduction.py covers the same ground). Run from the repo root:
    python -m scripts.legacy.check_compas_data
"""

from src.data.compas import load_compas


def main():
    data = load_compas()

    print("COMPAS DATASET SUMMARY")
    print("----------------------")

    print("Training features:", data["X_train"].shape)
    print("Training labels:", data["y_train"].shape)

    print("Test features:", data["X_test"].shape)
    print("Test labels:", data["y_test"].shape)

    print("\nTraining recidivism distribution:")
    print(data["y_train"].long().bincount())

    print("\nTraining protected-group distribution:")
    print(data["sensitive_train"].bincount())

    print("\nTest protected-group distribution:")
    print(data["sensitive_test"].bincount())

    assert data["X_train"].shape[0] == len(data["y_train"])
    assert data["X_test"].shape[0] == len(data["y_test"])

    assert data["X_train"].shape[1] == data["X_test"].shape[1]

    assert data["X_train"].isfinite().all()
    assert data["X_test"].isfinite().all()

    assert data["sensitive_train"].unique().numel() == 2
    assert data["sensitive_test"].unique().numel() == 2

    print("\nCOMPAS dataset loading test passed!")


if __name__ == "__main__":
    main()
