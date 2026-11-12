"""
Report tables and figures, laid out like the paper, with the paper's numbers.

Reads the sweep outputs (run those first):
    experiments/results/deletion_sweep_summary.csv              (paper protocol)
    experiments/results/vary_split/deletion_sweep_summary.csv   (optional)
    experiments/results/noise_sweep/noise_sweep_summary.csv     (optional)

Writes experiments/results/report/:
    tables/
        accuracy_random.{md,tex}     paper Table 4: test accuracy, ours vs paper
        accuracy_minority.{md,tex}   paper Table 5
        accuracy_majority.{md,tex}   paper Table 6
        runtime.{md,tex}             paper Table 2: retraining vs Fair Unlearning
        approximation.{md,tex}       how closely each unlearning method matches
                                     its retraining target (AEOD gap, accuracy
                                     gap, parameter distance)
        certificate.{md,tex}         epsilon at delta = 1e-4, sigma = 1
    figures/ (PDF for LaTeX, PNG for slides)
        fig1_aeod_random              paper Figure 1
        fig2_noise                    paper Figure 2
        fig3_aeod_subgroups           paper Figure 3
        fig5_random_metrics           paper Figure 5 (DP, EOpp, subgroup acc.)
        fig6_minority_metrics         paper Figure 6
        fig7_majority_metrics         paper Figure 7
        split_variation               ours: paper protocol vs. new split per seed

Usage (from the repo root):
    python -m experiments.make_report
"""

import csv
import math
from pathlib import Path

RESULTS = Path("experiments/results")
OUT = RESULTS / "report"
DATASETS = ["compas", "adult", "hsls"]           # paper column order
DATASET_NAMES = {"compas": "COMPAS", "adult": "Adult", "hsls": "HSLS"}

METHOD_ORDER = [
    "Full Training (BCE)", "Retraining (BCE)", "Newton (BCE)", "SISA (BCE)",
    "PRU (BCE)", "Full Training (Fair)", "Retraining (Fair)", "Fair Unlearning",
]
PAPER_METHOD_NAMES = {
    "Newton (BCE)": "Newton (Guo et al.)",
    "SISA (BCE)": "SISA (Bourtoule et al.)",
    "PRU (BCE)": "PRU (Izzo et al.)",
    "Fair Unlearning": "Fair Unlearning (paper's method)",
}

# Rates reported in the paper's accuracy tables.
TABLE_RATES = {"random": (0.05, 0.20), "minority": (0.05, 0.10),
               "majority": (0.05, 0.20)}

# ----------------------------------------------------------------------
# Paper numbers, Appendix A.8 Tables 4-6: (mean, std) per
# method -> dataset -> rate.
# ----------------------------------------------------------------------

def _t(rows):
    """rows: method -> [compas r1, compas r2, adult r1, adult r2, hsls r1, hsls r2]."""
    table = {}
    for method, values in rows.items():
        table[method] = {}
        for i, ds in enumerate(DATASETS):
            table[method][ds] = (values[2 * i], values[2 * i + 1])
    return table


PAPER_ACCURACY = {
    "random": _t({
        "Full Training (BCE)": [(0.652, 0.000), (0.652, 0.000), (0.817, 0.000), (0.816, 0.000), (0.724, 0.000), (0.724, 0.000)],
        "Retraining (BCE)": [(0.652, 0.001), (0.654, 0.003), (0.817, 0.000), (0.817, 0.001), (0.724, 0.002), (0.724, 0.002)],
        "Newton (BCE)": [(0.652, 0.001), (0.654, 0.003), (0.817, 0.000), (0.817, 0.001), (0.724, 0.002), (0.724, 0.002)],
        "SISA (BCE)": [(0.653, 0.001), (0.656, 0.004), (0.817, 0.000), (0.817, 0.001), (0.725, 0.001), (0.724, 0.003)],
        "PRU (BCE)": [(0.688, 0.005), (0.691, 0.001), (0.810, 0.003), (0.810, 0.001), (0.726, 0.004), (0.729, 0.000)],
        "Full Training (Fair)": [(0.647, 0.000), (0.647, 0.000), (0.819, 0.000), (0.819, 0.000), (0.713, 0.000), (0.713, 0.000)],
        "Retraining (Fair)": [(0.646, 0.002), (0.642, 0.003), (0.819, 0.000), (0.819, 0.001), (0.713, 0.002), (0.713, 0.002)],
        "Fair Unlearning": [(0.646, 0.002), (0.642, 0.003), (0.819, 0.000), (0.819, 0.001), (0.713, 0.002), (0.713, 0.002)],
    }),
    "minority": _t({
        "Full Training (BCE)": [(0.652, 0.000), (0.652, 0.000), (0.821, 0.000), (0.821, 0.000), (0.729, 0.000), (0.729, 0.000)],
        "Retraining (BCE)": [(0.651, 0.002), (0.652, 0.001), (0.821, 0.000), (0.822, 0.000), (0.729, 0.001), (0.730, 0.001)],
        "Newton (BCE)": [(0.651, 0.002), (0.652, 0.001), (0.821, 0.000), (0.823, 0.000), (0.729, 0.001), (0.730, 0.002)],
        "SISA (BCE)": [(0.653, 0.001), (0.654, 0.002), (0.822, 0.000), (0.823, 0.000), (0.728, 0.001), (0.728, 0.001)],
        "PRU (BCE)": [(0.661, 0.007), (0.666, 0.007), (0.819, 0.004), (0.815, 0.035), (0.720, 0.004), (0.724, 0.003)],
        "Full Training (Fair)": [(0.653, 0.000), (0.653, 0.000), (0.812, 0.000), (0.812, 0.000), (0.723, 0.000), (0.723, 0.000)],
        "Retraining (Fair)": [(0.653, 0.001), (0.661, 0.005), (0.813, 0.000), (0.816, 0.001), (0.713, 0.002), (0.713, 0.002)],
        "Fair Unlearning": [(0.652, 0.001), (0.661, 0.004), (0.814, 0.000), (0.816, 0.002), (0.724, 0.001), (0.721, 0.001)],
    }),
    "majority": _t({
        "Full Training (BCE)": [(0.654, 0.000), (0.654, 0.000), (0.814, 0.000), (0.814, 0.000), (0.738, 0.000), (0.738, 0.000)],
        "Retraining (BCE)": [(0.654, 0.001), (0.652, 0.002), (0.814, 0.000), (0.814, 0.000), (0.736, 0.000), (0.736, 0.002)],
        "Newton (BCE)": [(0.654, 0.001), (0.652, 0.002), (0.814, 0.000), (0.814, 0.001), (0.736, 0.000), (0.735, 0.002)],
        "SISA (BCE)": [(0.654, 0.002), (0.653, 0.004), (0.814, 0.000), (0.815, 0.001), (0.734, 0.001), (0.733, 0.002)],
        "PRU (BCE)": [(0.696, 0.006), (0.688, 0.003), (0.812, 0.001), (0.811, 0.002), (0.723, 0.012), (0.731, 0.004)],
        "Full Training (Fair)": [(0.666, 0.000), (0.666, 0.000), (0.815, 0.000), (0.815, 0.000), (0.725, 0.000), (0.725, 0.000)],
        "Retraining (Fair)": [(0.663, 0.004), (0.653, 0.002), (0.815, 0.000), (0.813, 0.000), (0.724, 0.001), (0.725, 0.002)],
        "Fair Unlearning": [(0.663, 0.004), (0.653, 0.003), (0.815, 0.000), (0.814, 0.001), (0.724, 0.001), (0.725, 0.002)],
    }),
}

# Paper Table 2 (seconds, mean, std). Cells that are not legible in the
# published text are None.
PAPER_RUNTIME = {
    ("Retraining (Fair)", "adult", 0.05): (36.796, 1.090),
    ("Fair Unlearning", "adult", 0.05): (8.800, 0.511),
    ("Retraining (Fair)", "hsls", 0.05): (14.963, 0.396),
    ("Fair Unlearning", "hsls", 0.05): (6.014, 0.249),
    ("Retraining (Fair)", "hsls", 0.20): (12.362, 0.795),
    ("Fair Unlearning", "hsls", 0.20): (5.704, 0.015),
}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_summary(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def lookup(summary, dataset, deletion_type, rate, method):
    for row in summary:
        if (row["dataset"] == dataset and row["deletion_type"] == deletion_type
                and abs(float(row["deletion_rate"]) - rate) < 1e-9
                and row["method"] == method):
            return row
    return None


def value(row, field):
    if row is None or row.get(f"{field}_mean", "") in ("", "nan"):
        return None
    return float(row[f"{field}_mean"]), float(row[f"{field}_std"] or 0)


def fmt(pair, digits=3):
    if pair is None:
        return "—"
    mean, std = pair
    if not math.isfinite(mean):
        return "—"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def write_table(name, caption, header, rows, note=""):
    """Write the same table as Markdown and as a LaTeX table* (2 columns)."""
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |",
          "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    if note:
        md += ["", note]
    (tables / f"{name}.md").write_text("\n".join(md) + "\n")

    def tex_escape(text):
        return (text.replace("±", r"$\pm$").replace("%", r"\%")
                .replace("&", r"\&").replace("—", "--").replace("ε", r"$\epsilon$")
                .replace("δ", r"$\delta$").replace("σ", r"$\sigma$")
                .replace("×", r"$\times$"))

    tex = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
           rf"\caption{{{tex_escape(caption)}}}", rf"\label{{tab:{name}}}",
           r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(tex_escape(h) for h in header) + r" \\", r"\hline"]
    tex += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in rows]
    tex += [r"\hline", r"\end{tabular}"]
    if note:
        tex += [rf"\\[2pt]\parbox{{\textwidth}}{{\footnotesize {tex_escape(note)}}}"]
    tex += [r"\end{table*}"]
    (tables / f"{name}.tex").write_text("\n".join(tex) + "\n")


def accuracy_tables(summary):
    titles = {"random": "Table 4", "minority": "Table 5", "majority": "Table 6"}
    for deletion_type, rates in TABLE_RATES.items():
        header = ["Method"]
        for ds in DATASETS:
            for rate in rates:
                header.append(f"{DATASET_NAMES[ds]} {rate:.0%}")
        rows = []
        for method in METHOD_ORDER:
            ours = [PAPER_METHOD_NAMES.get(method, method) + " (ours)"]
            paper = [PAPER_METHOD_NAMES.get(method, method) + " (paper)"]
            for ds in DATASETS:
                for i, rate in enumerate(rates):
                    row = lookup(summary, ds, deletion_type, rate, method)
                    ours.append(fmt(value(row, "accuracy")))
                    paper.append(fmt(PAPER_ACCURACY[deletion_type][method][ds][i]))
            rows += [ours, paper]
        write_table(
            f"accuracy_{deletion_type}",
            f"Test accuracy when unlearning ({deletion_type} deletion), ours "
            f"vs. paper {titles[deletion_type]}. Mean ± std over 5 repeats.",
            header, rows,
            note="% = share of the whole training set deleted. Paper values "
                 "from Oesterling et al. (2024), Appendix A.8.",
        )


def runtime_table(summary):
    header = ["Method"]
    for ds in DATASETS:
        for rate in (0.05, 0.20):
            header.append(f"{DATASET_NAMES[ds]} {rate:.0%}")
    rows = []
    for method in ("Retraining (Fair)", "Fair Unlearning"):
        ours, paper = [f"{method} (ours)"], [f"{method} (paper)"]
        for ds in DATASETS:
            for rate in (0.05, 0.20):
                row = lookup(summary, ds, "random", rate, method)
                ours.append(fmt(value(row, "runtime_s"), 4))
                paper.append(fmt(PAPER_RUNTIME.get((method, ds, rate))))
        rows += [ours, paper]
    speed = ["Speedup (ours)"]
    for ds in DATASETS:
        for rate in (0.05, 0.20):
            r = value(lookup(summary, ds, "random", rate, "Retraining (Fair)"),
                      "runtime_s")
            u = value(lookup(summary, ds, "random", rate, "Fair Unlearning"),
                      "runtime_s")
            speed.append(f"{r[0] / u[0]:.1f}×" if r and u and u[0] > 0 else "—")
    rows.append(speed)
    write_table(
        "runtime", "Runtime in seconds, retraining vs. Fair Unlearning "
        "(random deletion), ours vs. paper Table 2.", header, rows,
        note="Ours: CPU, float64; the Newton step computes the Hessian on the "
             "remaining data at unlearning time (no precomputation). Paper: "
             "32GB V100 GPU. Absolute times are not comparable across "
             "hardware; compare the speedup. — = not reported or not legible "
             "in the published table.",
    )


def approximation_table(summary):
    """How closely each unlearning method tracks its retraining target."""
    pairs = [("Fair Unlearning", "Retraining (Fair)"),
             ("Newton (BCE)", "Retraining (BCE)"),
             ("SISA (BCE)", "Retraining (BCE)"),
             ("PRU (BCE)", "Retraining (BCE)")]
    header = ["Method (vs. target)"] + [
        f"{DATASET_NAMES[ds]} {m}" for ds in DATASETS
        for m in ("AEOD gap", "acc. gap", "θ distance")]
    rows = []
    for method, target in pairs:
        row = [f"{method} vs. {target}"]
        for ds in DATASETS:
            gaps_aeod, gaps_acc, dists = [], [], []
            for deletion_type, rates in (("random", (0.01, 0.05, 0.10, 0.15, 0.20)),
                                         ("minority", (0.01, 0.05, 0.10)),
                                         ("majority", (0.01, 0.05, 0.10, 0.15, 0.20))):
                for rate in rates:
                    a = lookup(summary, ds, deletion_type, rate, method)
                    b = lookup(summary, ds, deletion_type, rate, target)
                    if a is None or b is None:
                        continue
                    gaps_aeod.append(abs(value(a, "aeod")[0] - value(b, "aeod")[0]))
                    gaps_acc.append(abs(value(a, "accuracy")[0]
                                        - value(b, "accuracy")[0]))
                    d = value(a, "param_distance")
                    if d is not None:
                        dists.append(d[0])
            row.append(f"{max(gaps_aeod):.4f}" if gaps_aeod else "—")
            row.append(f"{max(gaps_acc):.4f}" if gaps_acc else "—")
            row.append(f"{max(dists):.2e}" if dists else "—")
        rows.append(row)
    write_table(
        "approximation", "How closely each unlearning method matches its "
        "retraining target: worst case over all deletion settings (means over "
        "5 repeats).", header, rows,
        note="Gaps are absolute differences from the retraining target; "
             "θ distance is the L2 distance between parameter vectors. Lower "
             "is better. SISA is an ensemble, so it has no single parameter "
             "vector.",
    )


def certificate_table(summary):
    header = ["Deletion"] + [f"{DATASET_NAMES[ds]} {r:.0%}" for ds in DATASETS
                             for r in (0.01, 0.05, 0.20)]
    rows = []
    for deletion_type in ("random", "minority", "majority"):
        row = [deletion_type]
        for ds in DATASETS:
            for rate in (0.01, 0.05, 0.20):
                v = value(lookup(summary, ds, deletion_type, rate,
                                 "Fair Unlearning"), "epsilon")
                row.append(f"{v[0]:.2e}" if v else "—")
        rows.append(row)
    write_table(
        "certificate", "Unlearning certificate ε of Fair Unlearning at "
        "δ = 1e-4 and σ = 1 (smaller is stronger), smaller of the Theorem 1 "
        "and data-dependent bounds.", header, rows,
        note="— = setting not run (minority deletion stops at 10%).",
    )


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

STYLES = {
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
    "aeod": "AEOD", "accuracy": "Test accuracy",
    "dp": "Abs. demographic parity diff.",
    "eopp": "Abs. equality of opp. diff.",
    "subgroup_acc": "Abs. subgroup accuracy diff.",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.titlesize": 9, "legend.fontsize": 6.5})
    return plt


def _panel(ax, summary, dataset, deletion_type, metric):
    for method, style in STYLES.items():
        pts = sorted(
            (float(r["deletion_rate"]) * 100, float(r[f"{metric}_mean"]),
             float(r[f"{metric}_std"] or 0))
            for r in summary
            if r["dataset"] == dataset and r["deletion_type"] == deletion_type
            and r["method"] == method and r.get(f"{metric}_mean", "") != ""
        )
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                        yerr=[p[2] for p in pts], label=method, capsize=2,
                        markersize=3, linewidth=1, **style)
    ax.grid(alpha=0.3)


def _save(fig, name):
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{name}.png", dpi=200, bbox_inches="tight")


def grid_figure(plt, summary, name, panels, row_labels, title):
    """panels: list of rows, each a list of (dataset, deletion_type, metric)."""
    nrows, ncols = len(panels), len(panels[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 1.9 * nrows + 0.5),
                             squeeze=False)
    for i, row in enumerate(panels):
        for j, (ds, deletion_type, metric) in enumerate(row):
            ax = axes[i][j]
            _panel(ax, summary, ds, deletion_type, metric)
            if i == 0:
                ax.set_title(DATASET_NAMES[ds])
            if j == 0:
                ax.set_ylabel(row_labels[i])
            if i == nrows - 1:
                ax.set_xlabel("% of training set unlearned")
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_space = 0.42 / (1.9 * nrows + 0.5)   # ~2 legend rows, in figure units
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, legend_space, 1, 1))
    _save(fig, name)
    plt.close(fig)


def figures(summary, split_summary, noise_summary):
    plt = _plt()
    grid_figure(plt, summary, "fig1_aeod_random",
                [[(ds, "random", "aeod") for ds in DATASETS]], ["AEOD"],
                "AEOD when unlearning at random (paper Figure 1)")
    grid_figure(plt, summary, "fig3_aeod_subgroups",
                [[(ds, "minority", "aeod") for ds in DATASETS],
                 [(ds, "majority", "aeod") for ds in DATASETS]],
                ["AEOD, minority", "AEOD, majority"],
                "AEOD when unlearning from subgroups (paper Figure 3)")
    for fig_no, deletion_type in ((5, "random"), (6, "minority"),
                                  (7, "majority")):
        grid_figure(plt, summary, f"fig{fig_no}_{deletion_type}_metrics",
                    [[(ds, deletion_type, m) for ds in DATASETS]
                     for m in ("dp", "eopp", "subgroup_acc")],
                    [METRIC_LABELS[m] for m in ("dp", "eopp", "subgroup_acc")],
                    f"Other fairness metrics, {deletion_type} deletion "
                    f"(paper Figure {fig_no})")

    if split_summary:
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
        for ax, ds in zip(axes, DATASETS):
            for source, label, style in (
                (summary, "fixed split (paper)", "-"),
                (split_summary, "new split per seed", "--"),
            ):
                for method, color in (("Fair Unlearning", "tab:green"),
                                      ("Retraining (BCE)", "tab:orange")):
                    pts = sorted(
                        (float(r["deletion_rate"]) * 100, float(r["aeod_mean"]),
                         float(r["aeod_std"] or 0))
                        for r in source if r["dataset"] == ds
                        and r["deletion_type"] == "random"
                        and r["method"] == method)
                    if pts:
                        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                                    yerr=[p[2] for p in pts], capsize=2,
                                    color=color, linestyle=style, linewidth=1,
                                    marker="o", markersize=3,
                                    label=f"{method}, {label}")
            ax.set_title(DATASET_NAMES[ds])
            ax.set_xlabel("% of training set unlearned")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("AEOD (random deletion)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.12), frameon=False)
        fig.suptitle("Error bars with a fixed split vs. a new split per seed",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        _save(fig, "split_variation")
        plt.close(fig)

    if noise_summary:
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
        for ax, ds in zip(axes, DATASETS):
            rows = sorted((r for r in noise_summary if r["dataset"] == ds
                           and float(r["delta"]) == 1e-4),
                          key=lambda r: float(r["sigma"]))
            if not rows:
                continue
            sig = [float(r["sigma"]) for r in rows]
            ax.errorbar(sig, [float(r["accuracy_mean"]) for r in rows],
                        yerr=[float(r["accuracy_std"]) for r in rows],
                        color="tab:green", marker="o", markersize=3,
                        capsize=2, linewidth=1, label="Fair Unlearning")
            ax2 = ax.twinx()
            ax2.plot(sig, [float(r["epsilon_mean"]) for r in rows],
                     color="tab:blue", linestyle="--", linewidth=1,
                     label="ε (Lemma 2)")
            ax2.plot(sig, [float(r["epsilon_official_code_mean"]) for r in rows],
                     color="black", linestyle=":", linewidth=1,
                     label="ε (official code, ÷σ²)")
            ax2.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(DATASET_NAMES[ds])
            ax.set_xlabel("noise σ")
            ax.grid(alpha=0.3)
            if ds == DATASETS[0]:
                ax.set_ylabel("Test accuracy")
            if ds == DATASETS[-1]:
                ax2.set_ylabel("ε at δ = 1e-4")
        h1, l1 = axes[0].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.1), frameon=False)
        fig.suptitle("Accuracy and certificate vs. noise, unlearning 100 rows "
                     "(paper Figure 2)", fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        _save(fig, "fig2_noise")
        plt.close(fig)


# ----------------------------------------------------------------------

def main():
    summary = load_summary(RESULTS / "deletion_sweep_summary.csv")
    if not summary:
        raise SystemExit("Run python -m experiments.run_deletion_sweep first.")
    split_summary = load_summary(RESULTS / "vary_split" / "deletion_sweep_summary.csv")
    noise_summary = load_summary(RESULTS / "noise_sweep" / "noise_sweep_summary.csv")

    accuracy_tables(summary)
    runtime_table(summary)
    approximation_table(summary)
    certificate_table(summary)
    figures(summary, split_summary, noise_summary)
    print(f"Tables and figures written to {OUT}")


if __name__ == "__main__":
    main()
