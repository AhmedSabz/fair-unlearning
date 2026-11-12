"""
Noise vs. unlearning certificate: reproduces the paper's Figure 2.

Setup, as in the official eps_delta_experiments.py:
    - one train/test split, one random set of k = 100 training rows to unlearn
    - sigma (noise std) in 1e-4, 1e-3, ..., 1e4
    - delta in 1e-4, 1e-3, 1e-2, 1e-1
    - 5 repeats per sigma, each with a new noise vector b
    - fair loss with the paper's gamma and lambda; Fair Unlearning step

For each (sigma, repeat) the script trains the noisy original model, unlearns
the 100 rows with Fair Unlearning, retrains for reference, and records the
unlearned model's test accuracy and AEOD plus its (epsilon, delta) certificate
(src/unlearning/certificate.py). More noise gives a smaller epsilon (stronger
guarantee) but costs accuracy; that tradeoff is Figure 2.

Usage (from the repo root):
    python -m experiments.run_noise_sweep
    python -m experiments.run_noise_sweep --datasets compas --repeats 2
    python -m experiments.run_noise_sweep --plot-only

Outputs (experiments/results/noise_sweep/):
    noise_sweep_raw_<dataset>.csv     every sigma, repeat and delta
    noise_sweep_summary.csv           mean +/- std over repeats
    plots/<dataset>_accuracy_vs_sigma.png
    plots/<dataset>_epsilon_vs_sigma.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from experiments.run_deletion_sweep import (
    MAX_ITER, PAPER_HYPERPARAMS, read_csv, write_csv,
)
from src.data.datasets import load_dataset
from src.evaluation.metrics import calculate_metrics
from src.objectives.train import train_model
from src.unlearning.certificate import certificate
from src.unlearning.newton import newton_unlearn


SIGMAS = [10.0 ** p for p in range(-4, 5)]
DELTAS = [1e-4, 1e-3, 1e-2, 1e-1]
NUM_REMOVED = 100
OUT_DIR = Path("experiments/results/noise_sweep")
RAW_PATTERN = "noise_sweep_raw_{}.csv"


def run_dataset(name, repeats):
    hp = PAPER_HYPERPARAMS[name]
    data = load_dataset(name, random_state=0)
    X, y, s = data["X_train"], data["y_train"], data["sensitive_train"]
    X_test, y_test, s_test = data["X_test"], data["y_test"], data["sensitive_test"]

    generator = torch.Generator().manual_seed(0)
    deleted = torch.randperm(len(X), generator=generator)[:NUM_REMOVED]
    keep = torch.ones(len(X), dtype=torch.bool)
    keep[deleted] = False
    X_rem, y_rem, s_rem = X[keep], y[keep], s[keep]
    print(f"\n{name}: n = {len(X)}, unlearning {NUM_REMOVED} rows, "
          f"gamma = {hp['gamma']}, lambda = {hp['lambda_reg']}")

    rows = []
    for sigma in SIGMAS:
        for repeat in range(repeats):
            noise_gen = torch.Generator().manual_seed(1000 * repeat + 7)
            noise = sigma * torch.randn(X.shape[1], generator=noise_gen,
                                        dtype=torch.float64)
            original = train_model(X, y, s, lambda_reg=hp["lambda_reg"],
                                   gamma=hp["gamma"], max_iter=MAX_ITER,
                                   noise=noise)
            unlearned, _ = newton_unlearn(
                original, X_rem, y_rem, s_rem, lambda_reg=hp["lambda_reg"],
                gamma=hp["gamma"], noise=noise,
            )
            retrained = train_model(X_rem, y_rem, s_rem,
                                    lambda_reg=hp["lambda_reg"],
                                    gamma=hp["gamma"], max_iter=MAX_ITER,
                                    noise=noise)
            metrics = calculate_metrics(unlearned, X_test, y_test, s_test)
            retrain_metrics = calculate_metrics(retrained, X_test, y_test,
                                                s_test)
            distance = torch.norm(unlearned - retrained).item()

            for delta in DELTAS:
                cert = certificate(X_rem, s, deleted, original, unlearned,
                                   hp["lambda_reg"], sigma, delta, fair=True)
                rows.append({
                    "dataset": name, "sigma": sigma, "delta": delta,
                    "repeat": repeat, "num_removed": NUM_REMOVED,
                    "accuracy": metrics["accuracy"], "aeod": metrics["aeod"],
                    "retrain_accuracy": retrain_metrics["accuracy"],
                    "retrain_aeod": retrain_metrics["aeod"],
                    "param_distance": distance,
                    "epsilon": cert["epsilon"],
                    "epsilon_official_code": cert["epsilon_official_code"],
                    "eps_prime_data": cert["eps_prime_data"],
                    "eps_prime_theorem": cert["eps_prime_theorem"],
                })
            print(f"  sigma {sigma:8.0e} repeat {repeat} | "
                  f"acc {metrics['accuracy']:.4f} AEOD {metrics['aeod']:.4f} "
                  f"| eps(delta=1e-4) {rows[-len(DELTAS)]['epsilon']:.3e}")
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["dataset"], float(r["sigma"]), float(r["delta"]))
        groups.setdefault(key, []).append(r)
    summary = []
    for (dataset, sigma, delta), items in sorted(groups.items()):
        out = {"dataset": dataset, "sigma": sigma, "delta": delta,
               "repeats": len(items)}
        for field in ("accuracy", "aeod", "retrain_accuracy", "retrain_aeod",
                      "param_distance", "epsilon", "epsilon_official_code"):
            values = np.array([float(r[field]) for r in items])
            out[f"{field}_mean"] = float(values.mean())
            out[f"{field}_std"] = (float(values.std(ddof=1))
                                   if len(values) > 1 else 0.0)
        summary.append(out)
    return summary


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({r["dataset"] for r in summary}):
        rows = [r for r in summary if r["dataset"] == dataset]

        # Accuracy vs sigma, labelled with epsilon at delta = 1e-4.
        base = sorted((r for r in rows if float(r["delta"]) == 1e-4),
                      key=lambda r: float(r["sigma"]))
        sig = [float(r["sigma"]) for r in base]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.errorbar(sig, [float(r["accuracy_mean"]) for r in base],
                    yerr=[float(r["accuracy_std"]) for r in base],
                    marker="o", capsize=3, color="tab:green",
                    label="Fair Unlearning")
        ax.plot(sig, [float(r["retrain_accuracy_mean"]) for r in base],
                linestyle="--", color="gray", label="Retraining (Fair)")
        for r in base:
            eps = float(r["epsilon_mean"])
            ax.annotate(f"ε={eps:.1e}" if math.isfinite(eps) else "",
                        (float(r["sigma"]), float(r["accuracy_mean"])),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=5, ha="center", rotation=45)
        ax.set_xscale("log")
        ax.set_xlabel("noise σ")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{dataset.upper()}: unlearning {NUM_REMOVED} rows "
                     f"(δ = 1e-4)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_accuracy_vs_sigma.png", dpi=150)
        plt.close(fig)

        # Epsilon vs sigma for each delta.
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for delta in DELTAS:
            pts = sorted((float(r["sigma"]), float(r["epsilon_mean"]))
                         for r in rows if float(r["delta"]) == delta)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    label=f"δ = {delta:g}")
        # Same certificate as the official code computes it (divides by
        # sigma^2 instead of sigma; identical at sigma = 1).
        pts = sorted((float(r["sigma"]), float(r["epsilon_official_code_mean"]))
                     for r in rows if float(r["delta"]) == 1e-4)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=":",
                color="black", label="δ = 0.0001, official code (÷σ²)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("noise σ")
        ax.set_ylabel("ε (smaller = stronger guarantee)")
        ax.set_title(f"{dataset.upper()}: certificate vs noise")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_epsilon_vs_sigma.png", dpi=150)
        plt.close(fig)
    print(f"Plots: {plot_dir}")


def build_summary():
    rows = []
    for path in sorted(OUT_DIR.glob(RAW_PATTERN.format("*"))):
        rows.extend(read_csv(path))
    summary = summarize(rows)
    write_csv(OUT_DIR / "noise_sweep_summary.csv", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--datasets", nargs="+",
                        default=["adult", "compas", "hsls"],
                        choices=["adult", "compas", "hsls"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        for name in args.datasets:
            rows = run_dataset(name, args.repeats)
            write_csv(OUT_DIR / RAW_PATTERN.format(name), rows)
    plot(build_summary())


if __name__ == "__main__":
    main()
