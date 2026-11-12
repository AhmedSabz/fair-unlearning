"""
Deletion sweep: reproduces the paper's Figures 1, 3, 5-7 and Tables 4-6.

For every (dataset, deletion type, deletion rate, seed), this script evaluates
the same methods as the paper (Section 5.1):

    Fair loss (solid lines in the paper)
        Full Training (Fair)  original fair model, nothing removed (reference)
        Fair Unlearning       the paper's method: Newton step on the fair loss
        Retraining (Fair)     retrain from scratch on the fair loss
    BCE loss (dashed lines in the paper)
        Full Training (BCE)   original BCE model, nothing removed (reference)
        Newton (BCE)          Guo et al. (2020): Newton step on BCE
        SISA (BCE)            Bourtoule et al. (2020): 5 shards, majority vote
        PRU (BCE)             Izzo et al. (2021): projective residual update
        Retraining (BCE)      retrain from scratch on BCE

Deletion settings follow the paper (Section 5.2), as % of the WHOLE training set:
    random   : 1, 5, 10, 15, 20 %, from anyone
    majority : 1, 5, 10, 15, 20 %, all from the larger race group
    minority : 1, 5, 10 %,         all from the smaller race group

Noise: training and retraining minimize L + b^T theta with b ~ N(0, sigma^2 I),
sigma = 1 as in the paper (App. A.6). The same b is used for a split's original
models and every retraining on it. SISA shards are trained without noise, as
in the official code. For Fair Unlearning and Newton (BCE) each request also
gets its (epsilon, delta) certificate at delta = 1e-4 (src/unlearning/certificate.py).

Repeats: each seed re-draws the deleted rows on one fixed train/test split
(paper: 5 repeats, mean +/- std). --vary-split also draws a new split, noise
vector and original models per seed, to measure split-to-split variation.

Data preprocessing matches the paper's official code (src/data/datasets.py).

Usage (from the repo root):
    python -m experiments.run_deletion_sweep
    python -m experiments.run_deletion_sweep --datasets compas adult --seeds 5
    python -m experiments.run_deletion_sweep --quick          # 1 seed, smoke test
    python -m experiments.run_deletion_sweep --plot-only      # redo summary + plots
    python -m experiments.run_deletion_sweep --vary-split     # new split per seed
    python -m experiments.run_deletion_sweep --sigma 0        # no noise

Outputs (experiments/results/, or experiments/results/vary_split/):
    deletion_sweep_raw_<dataset>.csv   every seed, every method
    deletion_sweep_summary.csv         mean +/- std over seeds, all datasets
    plots/deletion_sweep/<dataset>_<deletion>_<metric>.png
        metric: aeod, accuracy, dp, eopp, subgroup_acc
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import load_dataset
from src.evaluation.metrics import calculate_metrics, compute_metrics
from src.objectives.train import train_model
from src.unlearning.baselines import (
    pru_precompute, pru_unlearn, sisa_predict, sisa_shards, sisa_train,
    sisa_unlearn,
)
from src.unlearning.certificate import certificate
from src.unlearning.newton import newton_unlearn


# Paper Table 1 (Appendix A.6). Override with --gamma / --lambda-reg.
PAPER_HYPERPARAMS = {
    "adult": {"lambda_reg": 1e-4, "gamma": 1.0},
    "compas": {"lambda_reg": 1e-4, "gamma": 10.0},
    "hsls": {"lambda_reg": 1e-2, "gamma": 10.0},
}

DELETION_RATES = {
    "random": [0.01, 0.05, 0.10, 0.15, 0.20],
    "majority": [0.01, 0.05, 0.10, 0.15, 0.20],
    "minority": [0.01, 0.05, 0.10],
}

SIGMA = 1.0      # objective-perturbation noise (paper App. A.6)
DELTA = 1e-4     # certificate failure probability (paper App. A.6)
MAX_ITER = 500
NOISE_SEED_OFFSET = 10_000   # keeps noise draws independent of deletion draws

METHODS = [
    # (name, loss type, kind)
    ("Full Training (Fair)", "fair", "full"),
    ("Fair Unlearning", "fair", "newton"),
    ("Retraining (Fair)", "fair", "retrain"),
    ("Full Training (BCE)", "bce", "full"),
    ("Newton (BCE)", "bce", "newton"),
    ("SISA (BCE)", "bce", "sisa"),
    ("PRU (BCE)", "bce", "pru"),
    ("Retraining (BCE)", "bce", "retrain"),
]
RETRAIN_METHOD = {"fair": "Retraining (Fair)", "bce": "Retraining (BCE)"}
METRICS = ["accuracy", "aeod", "dp", "eopp", "subgroup_acc"]
SUMMARY_FIELDS = METRICS + [
    "param_distance", "runtime_s", "epsilon", "epsilon_official_code",
    "eps_prime_data", "eps_prime_theorem",
]

BASE_RESULTS_DIR = Path("experiments/results")
RAW_CSV_PATTERN = "deletion_sweep_raw_{}.csv"  # one file per dataset
SUMMARY_NAME = "deletion_sweep_summary.csv"
PLOT_SUBDIR = Path("plots") / "deletion_sweep"


def results_dir(vary_split=False):
    """Paper protocol -> experiments/results/; --vary-split -> .../vary_split/."""
    return BASE_RESULTS_DIR / "vary_split" if vary_split else BASE_RESULTS_DIR


# ----------------------------------------------------------------------
# Deletion sets
# ----------------------------------------------------------------------

def group_roles(sensitive_train):
    """Return {'minority': group_id, 'majority': group_id} by training counts."""
    count_0 = (sensitive_train == 0).sum().item()
    count_1 = (sensitive_train == 1).sum().item()
    if count_0 <= count_1:
        return {"minority": 0, "majority": 1}
    return {"minority": 1, "majority": 0}


def choose_deletions(deletion_type, rate, sensitive_train, roles, seed):
    """
    Pick indices to delete. num_delete is a fraction of the WHOLE training set.
    Returns None if the request would empty a group (the fair loss needs both).
    """
    n = len(sensitive_train)
    num_delete = int(round(rate * n))  # paper code: np.round(n / 100 * pct)
    generator = torch.Generator().manual_seed(seed)

    if deletion_type == "random":
        pool = torch.arange(n)
    else:
        group_id = roles[deletion_type]
        pool = torch.where(sensitive_train == group_id)[0]

    if num_delete >= len(pool):
        return None

    shuffled = pool[torch.randperm(len(pool), generator=generator)]
    return shuffled[:num_delete]


# ----------------------------------------------------------------------
# One train/test split: data, noise, original models, baseline setup
# ----------------------------------------------------------------------

def prepare_split(name, hp, split_seed, sigma):
    data = load_dataset(name, random_state=split_seed)
    X, y, s = data["X_train"], data["y_train"], data["sensitive_train"]
    roles = group_roles(s)
    print(
        f"Split {split_seed} | train size {len(s)} | "
        f"minority = group {roles['minority']} "
        f"({(s == roles['minority']).sum().item()} rows) | "
        f"gamma = {hp['gamma']}, lambda = {hp['lambda_reg']}, sigma = {sigma}"
    )

    generator = torch.Generator().manual_seed(NOISE_SEED_OFFSET + split_seed)
    noise = (
        sigma * torch.randn(X.shape[1], generator=generator, dtype=torch.float64)
        if sigma > 0 else None
    )

    originals = {}
    for loss_type in ("fair", "bce"):
        gamma = hp["gamma"] if loss_type == "fair" else 0.0
        originals[loss_type] = train_model(
            X, y, s, lambda_reg=hp["lambda_reg"], gamma=gamma,
            max_iter=MAX_ITER, noise=noise,
        )
        m = calculate_metrics(originals[loss_type], data["X_test"],
                              data["y_test"], data["sensitive_test"])
        print(f"  Full Training ({loss_type.upper()}): "
              f"acc {m['accuracy']:.4f}, AEOD {m['aeod']:.4f}")

    start = time.perf_counter()
    shards = sisa_shards(len(X))
    shard_thetas = sisa_train(X, y, s, shards, hp["lambda_reg"], MAX_ITER)
    sisa_time = time.perf_counter() - start

    start = time.perf_counter()
    pru_pre = pru_precompute(X.numpy(), y.numpy(), originals["bce"].numpy(),
                             hp["lambda_reg"])
    pru_time = time.perf_counter() - start
    print(f"  Precompute: SISA shards {sisa_time:.2f}s, PRU {pru_time:.2f}s")

    return {
        "data": data, "roles": roles, "noise": noise, "originals": originals,
        "shards": shards, "shard_thetas": shard_thetas, "pru": pru_pre,
    }


# ----------------------------------------------------------------------
# One scenario: every method on one deletion set
# ----------------------------------------------------------------------

def run_scenario(split, deleted, hp, sigma):
    data, noise, originals = split["data"], split["noise"], split["originals"]
    X, y, s = data["X_train"], data["y_train"], data["sensitive_train"]
    X_test, y_test, s_test = data["X_test"], data["y_test"], data["sensitive_test"]

    keep = torch.ones(len(X), dtype=torch.bool)
    keep[deleted] = False
    X_rem, y_rem, s_rem = X[keep], y[keep], s[keep]

    rows, thetas = [], {}
    for method, loss_type, kind in METHODS:
        gamma = hp["gamma"] if loss_type == "fair" else 0.0
        theta, predictions, runtime = None, None, float("nan")
        start = time.perf_counter()

        if kind == "full":
            theta = originals[loss_type]
        elif kind == "newton":
            theta, _ = newton_unlearn(
                originals[loss_type], X_rem, y_rem, s_rem,
                lambda_reg=hp["lambda_reg"], gamma=gamma, noise=noise,
            )
            runtime = time.perf_counter() - start
        elif kind == "retrain":
            theta = train_model(
                X_rem, y_rem, s_rem, lambda_reg=hp["lambda_reg"],
                gamma=gamma, max_iter=MAX_ITER, noise=noise,
            )
            runtime = time.perf_counter() - start
        elif kind == "sisa":
            new_thetas = sisa_unlearn(
                X, y, s, split["shards"], split["shard_thetas"],
                deleted.numpy(), hp["lambda_reg"], MAX_ITER,
            )
            runtime = time.perf_counter() - start
            predictions = sisa_predict(new_thetas, X_test)
        elif kind == "pru":
            theta = pru_unlearn(split["pru"], deleted.numpy())
            runtime = time.perf_counter() - start

        if predictions is None:
            predictions = (X_test @ theta >= 0).long()
        row = {"method": method, "loss_type": loss_type, "kind": kind,
               "runtime_s": runtime}
        row.update(compute_metrics(predictions, y_test, s_test))
        rows.append(row)
        thetas[method] = theta

    # Distance to the retrained model of the same loss; certificate for the
    # two Newton-type methods.
    for row in rows:
        theta = thetas[row["method"]]
        target = thetas[RETRAIN_METHOD[row["loss_type"]]]
        row["param_distance"] = (
            torch.norm(theta - target).item() if theta is not None
            else float("nan")
        )
        cert = {}
        if row["kind"] == "newton":
            cert = certificate(
                X_rem, s, deleted, originals[row["loss_type"]], theta,
                hp["lambda_reg"], sigma, DELTA,
                fair=row["loss_type"] == "fair",
            )
        for key in ("epsilon", "epsilon_official_code", "eps_prime_data",
                    "eps_prime_theorem"):
            row[key] = cert.get(key, float("nan"))
    return rows


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------

def sweep(datasets, num_seeds, gamma_override, lambda_override,
          vary_split=False, sigma=SIGMA):
    out_dir = results_dir(vary_split)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        all_rows = []
        print(f"\n{'=' * 60}\nDataset: {name}\n{'=' * 60}")

        hp = dict(PAPER_HYPERPARAMS[name])
        if gamma_override is not None:
            hp["gamma"] = gamma_override
        if lambda_override is not None:
            hp["lambda_reg"] = lambda_override

        split_seeds = list(range(num_seeds)) if vary_split else [0]
        splits = {seed: prepare_split(name, hp, seed, sigma)
                  for seed in split_seeds}

        for deletion_type, rates in DELETION_RATES.items():
            for rate in rates:
                for seed in range(num_seeds):
                    split_seed = seed if vary_split else 0
                    split = splits[split_seed]
                    deleted = choose_deletions(
                        deletion_type, rate, split["data"]["sensitive_train"],
                        split["roles"], seed,
                    )
                    if deleted is None:
                        print(f"  skip {deletion_type} {rate:.0%}: "
                              f"would empty the group")
                        break

                    rows = run_scenario(split, deleted, hp, sigma)
                    for row in rows:
                        row.update({
                            "dataset": name,
                            "deletion_type": deletion_type,
                            "deletion_rate": rate,
                            "seed": seed,
                            "split_seed": split_seed,
                            "num_deleted": len(deleted),
                            "gamma": hp["gamma"],
                            "lambda_reg": hp["lambda_reg"],
                            "sigma": sigma,
                        })
                    all_rows.extend(rows)

                    by = {r["method"]: r for r in rows}
                    print(
                        f"  {deletion_type:8s} {rate:4.0%} seed {seed} | AEOD "
                        f"FU/RF {by['Fair Unlearning']['aeod']:.3f}/"
                        f"{by['Retraining (Fair)']['aeod']:.3f} "
                        f"N/RB {by['Newton (BCE)']['aeod']:.3f}/"
                        f"{by['Retraining (BCE)']['aeod']:.3f} "
                        f"SISA {by['SISA (BCE)']['aeod']:.3f} "
                        f"PRU {by['PRU (BCE)']['aeod']:.3f} | "
                        f"eps {by['Fair Unlearning']['epsilon']:.2e}"
                    )

        raw_path = out_dir / RAW_CSV_PATTERN.format(name)
        write_csv(raw_path, all_rows)
        print(f"Raw results: {raw_path}")

    return build_summary(out_dir)


def build_summary(out_dir):
    """Combine every per-dataset raw file into one summary CSV."""
    rows = []
    for path in sorted(out_dir.glob(RAW_CSV_PATTERN.format("*"))):
        rows.extend(read_csv(path))
    summary = summarize(rows)
    summary_path = out_dir / SUMMARY_NAME
    write_csv(summary_path, summary)
    print(f"Summary results: {summary_path}")
    return summary


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _mean_std(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def summarize(rows):
    """Mean and std over seeds for each (dataset, deletion, rate, method)."""
    groups = {}
    for row in rows:
        key = (row["dataset"], row["deletion_type"],
               float(row["deletion_rate"]), row["method"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, deletion_type, rate, method), items in groups.items():
        out = {"dataset": dataset, "deletion_type": deletion_type,
               "deletion_rate": rate, "method": method,
               "loss_type": items[0]["loss_type"], "num_seeds": len(items)}
        for field in SUMMARY_FIELDS:
            values = [float(r[field]) for r in items if r.get(field, "") != ""]
            if not values:
                continue
            mean, std = _mean_std(values)
            out[f"{field}_mean"], out[f"{field}_std"] = mean, std
        summary.append(out)
    return summary


STYLES = {
    # Fair loss: solid. BCE loss: dashed. References: dotted.
    "Full Training (Fair)": dict(color="black", linestyle=":", marker=""),
    "Fair Unlearning": dict(color="tab:green", linestyle="-", marker="o"),
    "Retraining (Fair)": dict(color="tab:olive", linestyle="-", marker="x"),
    "Full Training (BCE)": dict(color="gray", linestyle=":", marker=""),
    "Newton (BCE)": dict(color="tab:blue", linestyle="--", marker="o"),
    "SISA (BCE)": dict(color="tab:purple", linestyle="--", marker="s"),
    "PRU (BCE)": dict(color="tab:red", linestyle="--", marker="^"),
    "Retraining (BCE)": dict(color="tab:orange", linestyle="--", marker="x"),
}
METRIC_LABELS = {
    "aeod": "Absolute equalized odds difference",
    "accuracy": "Test accuracy",
    "dp": "Absolute demographic parity difference",
    "eopp": "Absolute equality of opportunity difference",
    "subgroup_acc": "Absolute subgroup accuracy difference",
}


def plot(summary, out_dir):
    """One plot per (dataset, deletion type, metric)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / PLOT_SUBDIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    combos = sorted({(r["dataset"], r["deletion_type"]) for r in summary})
    for dataset, deletion_type in combos:
        for metric, label in METRIC_LABELS.items():
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for method, style in STYLES.items():
                # Rows from older result files may lack newer metrics.
                pts = sorted(
                    (float(r["deletion_rate"]), float(r[f"{metric}_mean"]),
                     float(r[f"{metric}_std"]))
                    for r in summary
                    if r["dataset"] == dataset
                    and r["deletion_type"] == deletion_type
                    and r["method"] == method
                    and r.get(f"{metric}_mean", "") != ""
                )
                if not pts:
                    continue
                ax.errorbar([p[0] * 100 for p in pts], [p[1] for p in pts],
                            yerr=[p[2] for p in pts], label=method,
                            capsize=3, **style)
            ax.set_title(f"{dataset.upper()}: {deletion_type} deletion")
            ax.set_xlabel("% of training set deleted")
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{dataset}_{deletion_type}_{metric}.png",
                        dpi=150)
            plt.close(fig)
    print(f"Plots: {plot_dir}")


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--datasets", nargs="+", default=["adult", "compas", "hsls"],
        choices=["adult", "compas", "hsls"],
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override the paper's per-dataset gamma.")
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="Override the paper's per-dataset lambda.")
    parser.add_argument("--sigma", type=float, default=SIGMA,
                        help="Noise std for b (paper: 1). 0 disables noise.")
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, for a fast smoke test.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from the existing raw CSVs.")
    parser.add_argument("--vary-split", action="store_true",
                        help="New train/test split (and new noise and "
                             "original models) for every seed. The paper "
                             "keeps one split.")
    args = parser.parse_args()
    out_dir = results_dir(args.vary_split)

    if args.plot_only:
        plot(build_summary(out_dir), out_dir)
        return

    seeds = 1 if args.quick else args.seeds
    summary = sweep(args.datasets, seeds, args.gamma, args.lambda_reg,
                    args.vary_split, args.sigma)
    plot(summary, out_dir)


if __name__ == "__main__":
    main()
