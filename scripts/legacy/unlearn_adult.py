"""
Fair Unlearning vs. retraining on Adult for one 1% random deletion (no noise).

Needs the model written by `python -m scripts.legacy.train_adult`.
Legacy script from the project's first iteration, run from the repo root:
    python -m scripts.legacy.unlearn_adult
The reported experiments are in experiments/run_deletion_sweep.py.
"""

import time

import torch

from scripts.legacy._common import MODEL_DIR, PAPER_HYPERPARAMS
from src.data.datasets import load_dataset
from src.evaluation.metrics import calculate_metrics
from src.objectives.train import train_model
from src.unlearning.newton import newton_unlearn


def main():
    hp = PAPER_HYPERPARAMS["adult"]
    data = load_dataset("adult")
    X_train, y_train, s_train = (data["X_train"], data["y_train"],
                                 data["sensitive_train"])
    X_test, y_test, s_test = data["X_test"], data["y_test"], data["sensitive_test"]

    path = MODEL_DIR / "adult_model_original.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run python -m scripts.legacy.train_adult first")
    theta_original = torch.load(path, map_location="cpu", weights_only=True)

    # Randomly delete 1% of the training records.
    num_delete = int(0.01 * len(X_train))
    generator = torch.Generator().manual_seed(42)
    deleted = torch.randperm(len(X_train), generator=generator)[:num_delete]
    keep = torch.ones(len(X_train), dtype=torch.bool)
    keep[deleted] = False
    X_rem, y_rem, s_rem = X_train[keep], y_train[keep], s_train[keep]
    print(f"Training rows {len(X_train)}, deleted {num_delete}, "
          f"remaining {len(X_rem)}")

    start = time.perf_counter()
    theta_unlearned, _ = newton_unlearn(
        theta_original, X_rem, y_rem, s_rem,
        lambda_reg=hp["lambda_reg"], gamma=hp["gamma"],   # undamped, as in the paper
    )
    unlearning_time = time.perf_counter() - start

    start = time.perf_counter()
    theta_retrained = train_model(X_rem, y_rem, s_rem,
                                  lambda_reg=hp["lambda_reg"],
                                  gamma=hp["gamma"], max_iter=500)
    retraining_time = time.perf_counter() - start

    print("\n========== ADULT UNLEARNING RESULTS ==========")
    for label, theta in (("Original", theta_original),
                         ("Fair Unlearning", theta_unlearned),
                         ("Retrained", theta_retrained)):
        m = calculate_metrics(theta, X_test, y_test, s_test)
        print(f"{label:16s} accuracy {m['accuracy']:.4f}  AEOD {m['aeod']:.4f}")
    distance = torch.norm(theta_unlearned - theta_retrained).item()
    print(f"\nParameter distance (unlearned vs retrained): {distance:.3e}")
    print(f"Runtime: unlearning {unlearning_time:.4f}s, "
          f"retraining {retraining_time:.4f}s")


if __name__ == "__main__":
    main()
