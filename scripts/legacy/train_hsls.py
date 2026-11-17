"""
Train the fair logistic regression on HSLS with the paper's lambda and gamma
(no noise) and save it to scripts/legacy/models/hsls_model_original.pth.

Legacy script from the project's first iteration, run from the repo root:
    python -m scripts.legacy.train_hsls
The reported experiments are in experiments/run_deletion_sweep.py.
"""

import torch

from scripts.legacy._common import GROUP_NAMES, MODEL_DIR, PAPER_HYPERPARAMS
from src.data.datasets import load_dataset
from src.evaluation.metrics import calculate_metrics
from src.objectives.train import train_model

DATASET = "hsls"


def main():
    hp = PAPER_HYPERPARAMS[DATASET]
    print(f"Loading {DATASET} dataset...")
    data = load_dataset(DATASET)

    print(f"\nTraining fair logistic regression "
          f"(lambda = {hp['lambda_reg']}, gamma = {hp['gamma']})...")
    theta = train_model(
        data["X_train"], data["y_train"], data["sensitive_train"],
        lambda_reg=hp["lambda_reg"], gamma=hp["gamma"], max_iter=500,
    )
    print("Model parameter shape:", tuple(theta.shape))

    m = calculate_metrics(theta, data["X_test"], data["y_test"],
                          data["sensitive_test"])
    print(f"\n{DATASET.upper()} TEST RESULTS")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"AEOD:     {m['aeod']:.4f}")
    print(f"Groups:   {GROUP_NAMES[DATASET][0]}, {GROUP_NAMES[DATASET][1]}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{DATASET}_model_original.pth"
    torch.save(theta, path)
    print(f"\nSaved model to {path}")


if __name__ == "__main__":
    main()
