"""
Fairness-accuracy tradeoff over the fairness weight gamma (paper App. A.9).

Setup, following the official fairness_tradeoff_experiments.py:
    - one train/test split (seed 0), the paper's lambda per dataset (Table 1)
    - noise sigma = 1; a new noise vector b for every (gamma, repeat), as in
      the official script
    - for every gamma, train the fair model, then unlearn 1% and 5% of the
      training set at random with Fair Unlearning (and retrain for reference)
    - 5 repeats; each repeat re-draws the deleted rows and the noise

Gamma grid: the official script sweeps 1e-5 ... 1e-1. We add 0 (plain BCE)
and 1, 10, 100, 1000 so the curve also covers the paper's chosen values
(Table 1: Adult 1, COMPAS and HSLS 10) and the strongly fair end.

The official script reports the metrics of the trained (full-data) model at
each gamma; we record those ("Full Training (Fair)") and also the Fair
Unlearning and Retraining (Fair) models after each deletion, so the figure
shows that unlearning keeps the model on the same tradeoff curve.

Usage (from the repo root):
    python -m experiments.run_fairness_tradeoff
    python -m experiments.run_fairness_tradeoff --datasets compas --repeats 2
    python -m experiments.run_fairness_tradeoff --plot-only

Outputs (experiments/results/fairness_tradeoff/):
    tradeoff_raw_<dataset>.csv     every gamma, repeat, deletion rate, method
    tradeoff_summary.csv           mean +/- std over repeats
    plots/<dataset>_tradeoff.png   AEOD vs accuracy, one point per gamma
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from experiments.run_deletion_sweep import (
    MAX_ITER, METRICS, PAPER_HYPERPARAMS, SIGMA, choose_deletions,
    group_roles, read_csv, write_csv,
)
from src.data.datasets import load_dataset
from src.evaluation.metrics import calculate_metrics
from src.objectives.train import train_model
from src.unlearning.newton import newton_unlearn


OFFICIAL_GAMMAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
GAMMAS = [0.0] + OFFICIAL_GAMMAS + [1.0, 10.0, 100.0, 1000.0]
DELETION_RATES = [0.01, 0.05]     # official script: 1% and 5%, random
NOISE_SEED_OFFSET = 20_000        # independent of the deletion-sweep draws
OUT_DIR = Path("experiments/results/fairness_tradeoff")
RAW_PATTERN = "tradeoff_raw_{}.csv"
SUMMARY_NAME = "tradeoff_summary.csv"


def noise_vector(dim, gamma_index, repeat, sigma):
    """New b for every (gamma, repeat), reproducible from the indices."""
    if sigma == 0:
        return None
    gen = torch.Generator().manual_seed(
        NOISE_SEED_OFFSET + 100 * repeat + gamma_index)
    return sigma * torch.randn(dim, generator=gen, dtype=torch.float64)


def run_dataset(name, repeats, sigma=SIGMA):
    lam = PAPER_HYPERPARAMS[name]["lambda_reg"]
    data = load_dataset(name, random_state=0)
    X, y, s = data["X_train"], data["y_train"], data["sensitive_train"]
    test = (data["X_test"], data["y_test"], data["sensitive_test"])
    roles = group_roles(s)
    print(f"\n{name}: n = {len(X)}, lambda = {lam}, sigma = {sigma}")

    rows = []
    for gi, gamma in enumerate(GAMMAS):
        for repeat in range(repeats):
            noise = noise_vector(X.shape[1], gi, repeat, sigma)
            original = train_model(X, y, s, lambda_reg=lam, gamma=gamma,
                                   max_iter=MAX_ITER, noise=noise)
            base = {"dataset": name, "gamma": gamma, "repeat": repeat,
                    "lambda_reg": lam, "sigma": sigma}
            full = calculate_metrics(original, *test)
            rows.append({**base, "deletion_rate": 0.0,
                         "method": "Full Training (Fair)", **full})

            for rate in DELETION_RATES:
                deleted = choose_deletions("random", rate, s, roles, repeat)
                keep = torch.ones(len(X), dtype=torch.bool)
                keep[deleted] = False
                unlearned, _ = newton_unlearn(
                    original, X[keep], y[keep], s[keep], lambda_reg=lam,
                    gamma=gamma, noise=noise)
                retrained = train_model(X[keep], y[keep], s[keep],
                                        lambda_reg=lam, gamma=gamma,
                                        max_iter=MAX_ITER, noise=noise)
                distance = torch.norm(unlearned - retrained).item()
                for method, theta in (("Fair Unlearning", unlearned),
                                      ("Retraining (Fair)", retrained)):
                    rows.append({
                        **base, "deletion_rate": rate, "method": method,
                        **calculate_metrics(theta, *test),
                        "param_distance": distance if method == "Fair Unlearning" else 0.0,
                    })
            print(f"  gamma {gamma:8.0e} repeat {repeat} | full training "
                  f"acc {full['accuracy']:.4f} AEOD {full['aeod']:.4f}")
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["dataset"], float(r["gamma"]), float(r["deletion_rate"]),
               r["method"])
        groups.setdefault(key, []).append(r)
    summary = []
    for (dataset, gamma, rate, method), items in sorted(groups.items()):
        out = {"dataset": dataset, "gamma": gamma, "deletion_rate": rate,
               "method": method, "repeats": len(items)}
        for field in METRICS + ["param_distance"]:
            values = np.array([float(r[field]) for r in items
                               if r.get(field, "") != ""])
            if len(values) == 0:
                continue
            out[f"{field}_mean"] = float(values.mean())
            out[f"{field}_std"] = (float(values.std(ddof=1))
                                   if len(values) > 1 else 0.0)
        summary.append(out)
    return summary


def build_summary():
    rows = []
    for path in sorted(OUT_DIR.glob(RAW_PATTERN.format("*"))):
        rows.extend(read_csv(path))
    summary = summarize(rows)
    write_csv(OUT_DIR / SUMMARY_NAME, summary)
    return summary


def gamma_label(gamma):
    return "0" if gamma == 0 else f"{gamma:g}"


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({r["dataset"] for r in summary}):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for method, rate, style in (
            ("Full Training (Fair)", 0.0, dict(color="black", marker="o")),
            ("Fair Unlearning", 0.05, dict(color="tab:green", marker="x",
                                           linestyle="--")),
        ):
            pts = sorted(
                (float(r["gamma"]), float(r["aeod_mean"]),
                 float(r["accuracy_mean"]))
                for r in summary if r["dataset"] == dataset
                and r["method"] == method
                and math.isclose(float(r["deletion_rate"]), rate))
            if not pts:
                continue
            label = method if rate == 0 else f"{method} ({rate:.0%} deleted)"
            ax.plot([p[1] for p in pts], [p[2] for p in pts], label=label,
                    markersize=4, linewidth=1, **style)
            if rate == 0:
                for g, x, acc in pts:
                    ax.annotate(f"γ={gamma_label(g)}", (x, acc), fontsize=5,
                                textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel("AEOD (lower is fairer)")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{dataset.upper()}: fairness-accuracy tradeoff over γ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_tradeoff.png", dpi=150)
        plt.close(fig)
    print(f"Plots: {plot_dir}")


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
            write_csv(OUT_DIR / RAW_PATTERN.format(name),
                      run_dataset(name, args.repeats))
    plot(build_summary())


if __name__ == "__main__":
    main()
