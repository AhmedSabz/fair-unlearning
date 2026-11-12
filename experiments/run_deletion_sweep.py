"""
Task 2: Newton unlearning vs. full retraining, matching the paper's setup.

For every (dataset, deletion type, deletion rate, seed), this script runs
four methods, the same four lines as the paper's Figures 1 and 3:

    Fair Unlearning   = Newton step on the fair loss (gamma > 0)
    Retraining (Fair) = retrain from scratch on the fair loss
    Newton (BCE)      = Newton step on plain BCE (gamma = 0), Guo et al. 2020
    Retraining (BCE)  = retrain from scratch on plain BCE

Deletion settings follow the paper (Section 5.2):
    random   : 1, 5, 10, 15, 20 % of the training set, from anyone
    majority : 1, 5, 10, 15, 20 % of the training set, all from the larger group
    minority : 1, 5, 10 %       of the training set, all from the smaller group

Percentages are of the WHOLE training set, not of the group.
Each seed draws a different random deletion set (paper: 5 repeats, mean +/- std).

Not included: the objective-perturbation noise (b^T theta) used for the
(epsilon, delta) certificate. This script measures how close Newton gets to
retraining, which is what the paper's accuracy/AEOD figures report.

Usage (from the repo root):
    python -m experiments.run_deletion_sweep
    python -m experiments.run_deletion_sweep --datasets compas adult --seeds 5
    python -m experiments.run_deletion_sweep --quick          # 1 seed, smoke test
    python -m experiments.run_deletion_sweep --plot-only      # redo summary + plots

Outputs (experiments/results/):
    deletion_sweep_raw_<dataset>.csv   every seed, every method
    deletion_sweep_summary.csv         mean +/- std over seeds, all datasets
    plots/deletion_sweep/<dataset>_<deletion>_<aeod|accuracy>.png
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from src.objectives.train import train_model
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

DAMPING = 1e-4
MAX_ITER = 500
RESULTS_DIR = Path("experiments/results")
RAW_CSV_PATTERN = "deletion_sweep_raw_{}.csv"  # one file per dataset
SUMMARY_CSV = RESULTS_DIR / "deletion_sweep_summary.csv"
PLOT_DIR = RESULTS_DIR / "plots" / "deletion_sweep"

METHODS = [
    # (method name, loss type, kind)
    ("Fair Unlearning", "fair", "newton"),
    ("Retraining (Fair)", "fair", "retrain"),
    ("Newton (BCE)", "bce", "newton"),
    ("Retraining (BCE)", "bce", "retrain"),
]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_dataset(name):
    """Import lazily so a missing HSLS csv doesn't block Adult/COMPAS."""
    if name == "adult":
        from src.data.adult import load_adult
        return load_adult()
    if name == "compas":
        from src.data.compas import load_compas
        return load_compas()
    if name == "hsls":
        from src.data.hsls import load_hsls
        return load_hsls()
    raise ValueError(f"Unknown dataset: {name}")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def calculate_metrics(theta, X, y, sensitive):
    """Test accuracy and AEOD = 0.5 * (|TPR_a - TPR_b| + |FPR_a - FPR_b|)."""
    with torch.no_grad():
        predictions = (X @ theta >= 0).long()  # same as sigmoid >= 0.5
        labels = y.long()

        accuracy = (predictions == labels).double().mean().item()

        rates = {}
        for group in (0, 1):
            mask = sensitive == group
            pred = predictions[mask]
            actual = labels[mask]

            tp = ((pred == 1) & (actual == 1)).sum().item()
            fn = ((pred == 0) & (actual == 1)).sum().item()
            fp = ((pred == 1) & (actual == 0)).sum().item()
            tn = ((pred == 0) & (actual == 0)).sum().item()

            rates[group] = {
                "TPR": tp / max(tp + fn, 1),
                "FPR": fp / max(fp + tn, 1),
            }

        aeod = 0.5 * (
            abs(rates[0]["TPR"] - rates[1]["TPR"])
            + abs(rates[0]["FPR"] - rates[1]["FPR"])
        )

    return accuracy, aeod


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
    num_delete = int(rate * n)
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
# One scenario: 4 methods on one deletion set
# ----------------------------------------------------------------------

def run_scenario(data, originals, deleted, hp):
    X_train, y_train, s_train = (
        data["X_train"], data["y_train"], data["sensitive_train"]
    )
    X_test, y_test, s_test = (
        data["X_test"], data["y_test"], data["sensitive_test"]
    )

    keep = torch.ones(len(X_train), dtype=torch.bool)
    keep[deleted] = False
    X_rem, y_rem, s_rem = X_train[keep], y_train[keep], s_train[keep]

    rows = []
    thetas = {}

    for method, loss_type, kind in METHODS:
        gamma = hp["gamma"] if loss_type == "fair" else 0.0
        start = time.perf_counter()

        if kind == "newton":
            theta, _ = newton_unlearn(
                originals[loss_type], X_rem, y_rem, s_rem,
                lambda_reg=hp["lambda_reg"], gamma=gamma, damping=DAMPING,
            )
        else:
            theta = train_model(
                X_rem, y_rem, s_rem,
                lambda_reg=hp["lambda_reg"], gamma=gamma, max_iter=MAX_ITER,
            )

        runtime = time.perf_counter() - start
        thetas[(loss_type, kind)] = theta
        accuracy, aeod = calculate_metrics(theta, X_test, y_test, s_test)

        rows.append({
            "method": method,
            "loss_type": loss_type,
            "kind": kind,
            "accuracy": accuracy,
            "aeod": aeod,
            "runtime_s": runtime,
        })

    # How far each Newton jump lands from its own retrained optimum.
    for row in rows:
        lt = row["loss_type"]
        row["param_distance"] = torch.norm(
            thetas[(lt, "newton")] - thetas[(lt, "retrain")]
        ).item()

    return rows


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------

def sweep(datasets, num_seeds, gamma_override, lambda_override):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in datasets:
        all_rows = []
        print(f"\n{'=' * 60}\nDataset: {name}\n{'=' * 60}")
        try:
            data = load_dataset(name)
        except FileNotFoundError as error:
            print(f"Skipping {name}: {error}")
            continue

        hp = dict(PAPER_HYPERPARAMS[name])
        if gamma_override is not None:
            hp["gamma"] = gamma_override
        if lambda_override is not None:
            hp["lambda_reg"] = lambda_override

        s_train = data["sensitive_train"]
        roles = group_roles(s_train)
        print(
            f"Train size {len(s_train)} | "
            f"minority = group {roles['minority']} "
            f"({(s_train == roles['minority']).sum().item()} rows) | "
            f"gamma = {hp['gamma']}, lambda = {hp['lambda_reg']}"
        )

        # Original models on the full training set, one per loss type.
        originals = {}
        reference = {}
        for loss_type in ("fair", "bce"):
            gamma = hp["gamma"] if loss_type == "fair" else 0.0
            originals[loss_type] = train_model(
                data["X_train"], data["y_train"], s_train,
                lambda_reg=hp["lambda_reg"], gamma=gamma, max_iter=MAX_ITER,
            )
            reference[loss_type] = calculate_metrics(
                originals[loss_type],
                data["X_test"], data["y_test"], data["sensitive_test"],
            )
            print(
                f"Full Training ({loss_type.upper()}): "
                f"acc {reference[loss_type][0]:.4f}, "
                f"AEOD {reference[loss_type][1]:.4f}"
            )

        for deletion_type, rates in DELETION_RATES.items():
            for rate in rates:
                for seed in range(num_seeds):
                    deleted = choose_deletions(
                        deletion_type, rate, s_train, roles, seed
                    )
                    if deleted is None:
                        print(
                            f"  skip {deletion_type} {rate:.0%}: "
                            f"would empty the group"
                        )
                        break

                    rows = run_scenario(data, originals, deleted, hp)
                    for row in rows:
                        row.update({
                            "dataset": name,
                            "deletion_type": deletion_type,
                            "deletion_rate": rate,
                            "seed": seed,
                            "num_deleted": len(deleted),
                            "gamma": hp["gamma"],
                            "lambda_reg": hp["lambda_reg"],
                            "full_fair_accuracy": reference["fair"][0],
                            "full_fair_aeod": reference["fair"][1],
                            "full_bce_accuracy": reference["bce"][0],
                            "full_bce_aeod": reference["bce"][1],
                        })
                    all_rows.extend(rows)

                    fair_n, fair_r = rows[0], rows[1]
                    print(
                        f"  {deletion_type:8s} {rate:4.0%} seed {seed} | "
                        f"AEOD fair N/R {fair_n['aeod']:.4f}/{fair_r['aeod']:.4f} "
                        f"| BCE N/R {rows[2]['aeod']:.4f}/{rows[3]['aeod']:.4f} "
                        f"| dist fair {fair_n['param_distance']:.2e}"
                    )

        # One raw file per dataset, so datasets can be run separately.
        raw_path = RESULTS_DIR / RAW_CSV_PATTERN.format(name)
        write_csv(raw_path, all_rows)
        print(f"Raw results: {raw_path}")

    return build_summary()


def build_summary():
    """Combine every per-dataset raw file into one summary CSV."""
    rows = []
    for path in sorted(RESULTS_DIR.glob(RAW_CSV_PATTERN.format("*"))):
        rows.extend(read_csv(path))
    summary = summarize(rows)
    write_csv(SUMMARY_CSV, summary)
    print(f"Summary results: {SUMMARY_CSV}")
    return summary


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    """Mean and std over seeds for each (dataset, deletion, rate, method)."""
    groups = {}
    for row in rows:
        key = (
            row["dataset"], row["deletion_type"],
            float(row["deletion_rate"]), row["method"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, deletion_type, rate, method), items in groups.items():
        out = {
            "dataset": dataset,
            "deletion_type": deletion_type,
            "deletion_rate": rate,
            "method": method,
            "num_seeds": len(items),
        }
        for metric in ("accuracy", "aeod", "param_distance", "runtime_s"):
            values = torch.tensor([float(r[metric]) for r in items])
            out[f"{metric}_mean"] = values.mean().item()
            out[f"{metric}_std"] = (
                values.std().item() if len(values) > 1 else 0.0
            )
        summary.append(out)
    return summary


def plot(summary):
    """One AEOD plot and one accuracy plot per (dataset, deletion type)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    styles = {
        "Fair Unlearning": dict(color="tab:green", linestyle="-", marker="o"),
        "Retraining (Fair)": dict(color="black", linestyle="-", marker="x"),
        "Newton (BCE)": dict(color="tab:blue", linestyle="--", marker="o"),
        "Retraining (BCE)": dict(color="tab:gray", linestyle="--", marker="x"),
    }

    combos = sorted({(r["dataset"], r["deletion_type"]) for r in summary})
    for dataset, deletion_type in combos:
        for metric, label in (("aeod", "AEOD (lower is fairer)"),
                              ("accuracy", "Test accuracy")):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for method, style in styles.items():
                pts = sorted(
                    (float(r["deletion_rate"]),
                     float(r[f"{metric}_mean"]),
                     float(r[f"{metric}_std"]))
                    for r in summary
                    if r["dataset"] == dataset
                    and r["deletion_type"] == deletion_type
                    and r["method"] == method
                )
                if not pts:
                    continue
                xs = [p[0] * 100 for p in pts]
                ax.errorbar(
                    xs, [p[1] for p in pts], yerr=[p[2] for p in pts],
                    label=method, capsize=3, **style,
                )
            ax.set_title(f"{dataset.upper()}: {deletion_type} deletion")
            ax.set_xlabel("% of training set deleted")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            out = PLOT_DIR / f"{dataset}_{deletion_type}_{metric}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
    print(f"Plots: {PLOT_DIR}")


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
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, for a fast smoke test.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from the existing summary CSV.")
    args = parser.parse_args()

    if args.plot_only:
        plot(build_summary())
        return

    seeds = 1 if args.quick else args.seeds
    summary = sweep(args.datasets, seeds, args.gamma, args.lambda_reg)
    plot(summary)


if __name__ == "__main__":
    main()
