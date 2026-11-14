"""
HSLS from two sources, same model and unlearning methodology.

    A  authors' cleaned HSLS (14,509 students): the validated reproduction's
       table, load_dataset("hsls"), unchanged.
    B  HSLS built independently from the NCES public-use file (14,843
       students): load_hsls_nces() with its defaults (training-only fitting,
       training-median label).

This is a supplementary comparison of the two pipelines as a whole. The
pipelines differ in cohort selection, missing-value treatment, imputation,
scaling and label threshold (data/README.md); this script does not try to
attribute differences to any one of them.

Protocol (paper hyperparameters for HSLS, split seed 0, sigma = 1):
    - Full training on BCE and fair loss for each source.
    - Random deletion of 5% and 20% of the training set, deletion seeds 0-4;
      Fair Unlearning vs. Retraining (Fair).
    - Students are matched by their row position in the original NCES
      student file (hsls_16_student_v1_0.csv, sorted by STU_ID). This is not
      a persistent identifier: the authors' table has no STU_ID column and
      is indexed by that row position, which check_identity() verifies at
      run time. Shared students have the same train/test assignment in both
      sources.
    - Deletions are drawn by student id, in two modes:
        paired     the same shared training students are deleted in both
                   sources (k = rate x authors' training size); exact pairing.
        same_rate  each source deletes rate x its own training size from its
                   own training students; NOT exactly paired. For A this is
                   identical to paired.
    - The same noise vector b in both sources (same seed, same dimension).

Evaluation sets:
    native               each source's own test set and labels
    shared_native        the shared test students, each source's own labels;
                         the label definitions differ (thresholds 0.1540 vs.
                         B's training median), so the same student can have
                         different labels in A and B
    shared_common_label  the shared test students, one common label (the
                         authors' definition, X1TXMTH > 0.1540) for both

Outputs (experiments/results/hsls_sources/): manifest.json,
full_training.csv, deletion_raw.csv, deletion_summary.csv,
paired_differences.csv, comparison.md. No student-level data is written.

Usage (from the repo root; needs the NCES file, see data/README.md):
    python -m experiments.hsls_sources --pilot   # 5%, seed 0, prints only
    python -m experiments.hsls_sources
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from experiments.run_deletion_sweep import (
    DELTA, MAX_ITER, NOISE_SEED_OFFSET, PAPER_HYPERPARAMS, SIGMA, write_csv,
)
from src.data.datasets import load_dataset, prepare_hsls
from src.data.hsls_nces import (
    AUTHORS_PICKLE, FEATURES, LABEL_SOURCE, NCES_NUM_ROWS, RESERVE_CODES,
    SCALE_FEATURES, STUDENT_CSV, authors_row_ids, load_hsls_nces,
    read_nces_student, select_rows,
)
from src.evaluation.metrics import compute_metrics, predict
from src.objectives.train import train_model
from src.unlearning.certificate import certificate
from src.unlearning.newton import newton_unlearn

OUT_DIR = Path("experiments/results/hsls_sources")
SPLIT_SEED = 0
RATES = (0.05, 0.20)
SEEDS = tuple(range(5))
DELETION_KEY_SEED = 31_337        # deletion keys, independent of split keys
AUTHORS_THRESHOLD = 0.1540        # the authors' label threshold (data/README)
METRICS = ["accuracy", "aeod", "dp", "eopp", "subgroup_acc"]
EVAL_SETS = ("native", "shared_native", "shared_common_label")


# ----------------------------------------------------------------------
# Data: both sources, with NCES row ids
# ----------------------------------------------------------------------

def authors_arm(split_seed=SPLIT_SEED):
    """load_dataset("hsls") unchanged, plus the row id of every row."""
    data = load_dataset("hsls", random_state=split_seed)
    ids = authors_row_ids().to_numpy()
    train_pos, test_pos = train_test_split(np.arange(len(ids)), test_size=0.2,
                                           random_state=split_seed)
    data["train_row_ids"], data["test_row_ids"] = ids[train_pos], ids[test_pos]
    # Guard: the rows of X_train are the authors' table at those ids.
    table = prepare_hsls().drop(columns=["gradebin"])
    expected = table.loc[data["train_row_ids"]].to_numpy() / data["max_train_row_norm"]
    if not np.array_equal(data["X_train"][:, :-1].numpy(), expected):
        raise RuntimeError("authors' rows do not line up with their row ids")
    return data


def check_identity(raw):
    """
    The authors' table is indexed by the row position in the NCES student
    file (it has no STU_ID column). Stop unless, at every matched row:
      1. sex, race and the label agree exactly; and
      2. every raw value that NCES records as observed (not a reserve code)
         in the 53 non-scale features is the value the authors' table holds
         there. Cells missing in NCES and the three math scales are skipped:
         the authors imputed those cells (and the scales' negative values),
         so they are not raw values. The authors' table stores each column
         MinMax-scaled over its observed raw values, so the stored value is
         mapped back to raw units with that column's observed raw min and
         max before comparing.
    Under a wrong alignment, check 1 holds for about half the rows and check
    2 for almost none.
    """
    table = prepare_hsls()
    rows = raw.loc[table.index]
    checks = {
        "sexbin": (rows["X1SEX"] == 2).astype(int),
        "racebin": rows["X1RACE"].isin([2, 8]).astype(int),
        "gradebin": (rows[LABEL_SOURCE] > AUTHORS_THRESHOLD).astype(int),
    }
    for col, values in checks.items():
        if not np.array_equal(values.to_numpy(), table[col].to_numpy()):
            raise RuntimeError(f"NCES rows do not match the authors' {col}; "
                               "is this the HSLS:09 2016 student file?")

    other = [c for c in FEATURES if c not in SCALE_FEATURES]
    values = rows[other].astype(float)
    observed = values.notna() & ~values.isin(RESERVE_CODES)
    lo = values.where(observed).min()
    hi = values.where(observed).max()
    back = table[other] * (hi - lo) + lo           # authors' value, raw units
    mismatch = ((back - values).abs() > 1e-6) & observed
    if mismatch.to_numpy().any():
        counts = mismatch.sum()
        worst = counts.idxmax()
        raise RuntimeError(
            f"NCES rows do not match the authors' observed feature values: "
            f"{int(counts.sum())} cells in {int((counts > 0).sum())} features "
            f"(most in {worst}); is this the HSLS:09 2016 student file?")
    return table


def load_sources(split_seed=SPLIT_SEED):
    raw = read_nces_student()
    authors_table = check_identity(raw)
    data = {"A": authors_arm(split_seed),
            "B": load_hsls_nces(random_state=split_seed, raw=raw)}
    if list(data["A"]["feature_names"]) != list(data["B"]["feature_names"]):
        raise RuntimeError("the two sources have different feature layouts")
    return data, raw, authors_table


def evaluation_sets(data, authors_table):
    """{source: {eval set: (X, y, s)}}; shared students in the same order."""
    shared = sorted(set(data["A"]["test_row_ids"]) & set(data["B"]["test_row_ids"]))
    common_y = torch.tensor(authors_table.loc[shared, "gradebin"].to_numpy(),
                            dtype=torch.float64)
    sets = {}
    for src, d in data.items():
        pos = {rid: i for i, rid in enumerate(d["test_row_ids"])}
        idx = torch.tensor([pos[rid] for rid in shared])
        X, y, s = d["X_test"], d["y_test"], d["sensitive_test"]
        sets[src] = {
            "native": (X, y, s),
            "shared_native": (X[idx], y[idx], s[idx]),
            "shared_common_label": (X[idx], common_y, s[idx]),
        }
    return sets, shared


# ----------------------------------------------------------------------
# Deletions by student id
# ----------------------------------------------------------------------

def deletion_keys(seed):
    """One uniform key per NCES row, the same for both sources."""
    return np.random.default_rng([DELETION_KEY_SEED, seed]).random(NCES_NUM_ROWS)


def choose_deletion_ids(pool_ids, k, seed):
    """The k students of pool_ids with the smallest keys."""
    pool = np.asarray(sorted(pool_ids))
    if not 0 < k < len(pool):
        raise ValueError(f"cannot delete {k} of {len(pool)} students")
    order = np.argsort(deletion_keys(seed)[pool], kind="stable")
    return pool[order[:k]]


def deletion_plan(data, rate, seed):
    """{(source, mode): deleted ids}; see the module docstring."""
    a_train, b_train = data["A"]["train_row_ids"], data["B"]["train_row_ids"]
    shared = sorted(set(a_train) & set(b_train))
    k_a = int(round(rate * len(a_train)))
    paired = choose_deletion_ids(shared, k_a, seed)
    return {
        ("A", "paired"): paired,
        ("B", "paired"): paired,
        ("B", "same_rate"): choose_deletion_ids(
            b_train, int(round(rate * len(b_train))), seed),
    }


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def evaluate(theta, sets):
    out = {}
    for name, (X, y, s) in sets.items():
        out[name] = compute_metrics(predict(theta, X), y, s)
        out[name]["n_eval"] = len(y)
    return out


def full_training(d, hp, noise):
    """Full Training (BCE) and (Fair), as in prepare_split."""
    X, y, s = d["X_train"], d["y_train"], d["sensitive_train"]
    return {loss: train_model(X, y, s, lambda_reg=hp["lambda_reg"],
                              gamma=hp["gamma"] if loss == "fair" else 0.0,
                              max_iter=MAX_ITER, noise=noise)
            for loss in ("bce", "fair")}


def unlearn_and_retrain(d, theta_fair, deleted_pos, hp, noise):
    """Fair Unlearning and Retraining (Fair) for one deletion set."""
    X, y, s = d["X_train"], d["y_train"], d["sensitive_train"]
    keep = torch.ones(len(X), dtype=torch.bool)
    keep[deleted_pos] = False
    kw = dict(lambda_reg=hp["lambda_reg"], gamma=hp["gamma"], noise=noise)
    start = time.perf_counter()
    unlearned, _ = newton_unlearn(theta_fair, X[keep], y[keep], s[keep], **kw)
    t_unlearn = time.perf_counter() - start
    start = time.perf_counter()
    retrained = train_model(X[keep], y[keep], s[keep], max_iter=MAX_ITER, **kw)
    t_retrain = time.perf_counter() - start
    cert = certificate(X[keep], s, deleted_pos, theta_fair, unlearned,
                       hp["lambda_reg"], SIGMA, DELTA, fair=True)
    return unlearned, retrained, t_unlearn, t_retrain, cert


def _sha(values):
    data = ",".join(str(int(v)) for v in sorted(values)).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------

def run(rates=RATES, seeds=SEEDS, split_seed=SPLIT_SEED, verbose=True):
    hp = PAPER_HYPERPARAMS["hsls"]
    data, raw, authors_table = load_sources(split_seed)
    sets, shared_test = evaluation_sets(data, authors_table)
    d_feat = data["A"]["X_train"].shape[1]
    noise = SIGMA * torch.randn(
        d_feat, generator=torch.Generator().manual_seed(NOISE_SEED_OFFSET + split_seed),
        dtype=torch.float64)

    full_rows, originals = [], {}
    for src, d in data.items():
        originals[src] = full_training(d, hp, noise)
        for loss, theta in originals[src].items():
            for name, m in evaluate(theta, sets[src]).items():
                full_rows.append({"source": src, "loss": loss,
                                  "method": f"Full Training ({loss.upper() if loss == 'bce' else 'Fair'})",
                                  "eval_set": name, **m})

    rows = []
    a_train = set(data["A"]["train_row_ids"])
    for rate in rates:
        for seed in seeds:
            plan = deletion_plan(data, rate, seed)
            for (src, mode), ids in plan.items():
                d = data[src]
                pos_of = {rid: i for i, rid in enumerate(d["train_row_ids"])}
                deleted_pos = torch.tensor(sorted(pos_of[r] for r in ids))
                unl, ret, t_u, t_r, cert = unlearn_and_retrain(
                    d, originals[src]["fair"], deleted_pos, hp, noise)
                base = {
                    "source": src, "mode": mode, "rate": rate, "seed": seed,
                    "num_deleted": len(ids),
                    "effective_rate": len(ids) / len(d["train_row_ids"]),
                    "num_deleted_not_in_A": int(sum(r not in a_train
                                                    for r in ids)),
                    "deleted_ids_sha": _sha(ids),
                    "param_distance": float(torch.norm(unl - ret)),
                }
                for method, theta, rt in (("Fair Unlearning", unl, t_u),
                                          ("Retraining (Fair)", ret, t_r)):
                    for name, m in evaluate(theta, sets[src]).items():
                        rows.append({**base, "method": method, "eval_set": name,
                                     "runtime_s": rt,
                                     "epsilon": cert["epsilon"]
                                     if method == "Fair Unlearning" else math.nan,
                                     **m})
                if verbose:
                    n = {r["method"]: r for r in rows[-6:]
                         if r["eval_set"] == "shared_common_label"}
                    print(f"  {src} {mode:9s} {rate:4.0%} seed {seed} | "
                          f"deleted {len(ids)} | shared-cohort AEOD FU/RF "
                          f"{n['Fair Unlearning']['aeod']:.4f}/"
                          f"{n['Retraining (Fair)']['aeod']:.4f} | "
                          f"||FU-RF|| {base['param_distance']:.2e}")
    meta = manifest(data, raw, authors_table, shared_test, noise, rates, seeds,
                    split_seed, hp)
    return full_rows, rows, meta


def manifest(data, raw, authors_table, shared_test, noise, rates, seeds,
             split_seed, hp):
    """Reproducibility metadata: configuration, counts and hashes only."""
    a, b = data["A"], data["B"]
    s1m8 = set(select_rows(raw, "s1m8grade"))
    extra = sorted(set(b["train_row_ids"]) | set(b["test_row_ids"]))
    extra = [r for r in extra if r not in set(authors_table.index)]
    shared_tr = sorted(set(a["train_row_ids"]) & set(b["train_row_ids"]))
    thr_b = b["preprocessing"]["label_threshold"]
    score = raw[LABEL_SOURCE]
    flips = {name: int(((score.loc[ids] > AUTHORS_THRESHOLD)
                        != (score.loc[ids] > thr_b)).sum())
             for name, ids in (("train", shared_tr), ("test", shared_test))}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_worktree_dirty": dirty,
        "driver_sha256": _file_sha(Path(__file__)),
        "data_sha256": {"authors_pickle": _file_sha(AUTHORS_PICKLE),
                        "nces_student_csv": _file_sha(STUDENT_CSV)},
        "config": {"split_seed": split_seed, "test_size": 0.2,
                   "deletion_rates": list(rates), "deletion_seeds": list(seeds),
                   "deletion_key_seed": DELETION_KEY_SEED,
                   "lambda_reg": hp["lambda_reg"], "gamma": hp["gamma"],
                   "sigma": SIGMA, "delta": DELTA, "max_iter": MAX_ITER,
                   "noise_seed": NOISE_SEED_OFFSET + split_seed,
                   "noise_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest()[:16],
                   "nces_options": {k: b["preprocessing"][k] for k in
                                    ("row_set", "fit_mode", "n_neighbors")}},
        "cohort": {
            "A_rows": len(a["train_row_ids"]) + len(a["test_row_ids"]),
            "B_rows": len(b["train_row_ids"]) + len(b["test_row_ids"]),
            "A_train_test": [len(a["train_row_ids"]), len(a["test_row_ids"])],
            "B_train_test": [len(b["train_row_ids"]), len(b["test_row_ids"])],
            "shared_train": len(shared_tr), "shared_test": len(shared_test),
            "only_in_A": len(set(authors_table.index)
                             - set(b["train_row_ids"]) - set(b["test_row_ids"])),
            "only_in_B": len(extra),
            "only_in_B_missing_S1M8GRADE": sum(r not in s1m8 for r in extra),
            "only_in_B_unexplained": sum(r in s1m8 for r in extra),
            "split_conflicts_shared": len(
                set(a["train_row_ids"]) & set(b["test_row_ids"]))
            + len(set(a["test_row_ids"]) & set(b["train_row_ids"])),
        },
        "labels": {"A_threshold": AUTHORS_THRESHOLD, "B_threshold": thr_b,
                   "shared_label_differences": flips,
                   "B_threshold_fitted_on": "B training rows"},
        "id_set_sha256": {"A_train": _sha(a["train_row_ids"]),
                          "B_train": _sha(b["train_row_ids"]),
                          "shared_test": _sha(shared_test)},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "torch_threads": torch.get_num_threads(),
                        "platform": platform.platform()},
    }


# ----------------------------------------------------------------------
# Summaries and report
# ----------------------------------------------------------------------

def summarize(rows):
    df = pd.DataFrame(rows)
    keys = ["source", "mode", "rate", "method", "eval_set"]
    cols = METRICS + ["param_distance", "runtime_s", "epsilon", "num_deleted",
                      "effective_rate"]
    agg = df.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def paired_differences(rows):
    """B - A per seed. Exactly paired only for mode=paired on shared sets."""
    df = pd.DataFrame(rows)
    a = df[df["source"] == "A"].drop(columns=["mode"])
    out = []
    for mode in ("paired", "same_rate"):
        b = df[(df["source"] == "B") & (df["mode"] == mode)]
        m = b.merge(a, on=["rate", "seed", "method", "eval_set"],
                    suffixes=("_B", "_A"))
        for (rate, method, ev), g in m.groupby(["rate", "method", "eval_set"]):
            exact = mode == "paired" and ev != "native"
            for metric in METRICS + ["param_distance"]:
                diff = g[f"{metric}_B"] - g[f"{metric}_A"]
                out.append({"mode": mode, "rate": rate, "method": method,
                            "eval_set": ev, "metric": metric,
                            "exactly_paired": exact, "n_seeds": len(diff),
                            "mean_A": g[f"{metric}_A"].mean(),
                            "mean_B": g[f"{metric}_B"].mean(),
                            "mean_diff_B_minus_A": diff.mean(),
                            "sd_diff": diff.std(ddof=1)})
    return out


def write_report(out_dir, full_rows, rows, meta):
    f = pd.DataFrame(full_rows)
    s = summarize(rows)
    p = pd.DataFrame(paired_differences(rows))
    c, lab = meta["cohort"], meta["labels"]
    lines = [
        "# HSLS: authors' cleaned table (A) vs. independently built NCES table (B)",
        "",
        "Supplementary comparison; same model, hyperparameters (λ = 1e-2, "
        "γ = 10, σ = 1), split seed 0 and noise vector. The two pipelines "
        "differ in several ways at once (below), so differences are not "
        "attributed to any single cause. Generated by "
        "`python -m experiments.hsls_sources`; configuration and hashes in "
        "`manifest.json`.",
        "",
        "## Cohorts and labels",
        "",
        f"- A: {c['A_rows']:,} students (train/test {c['A_train_test'][0]:,} / "
        f"{c['A_train_test'][1]:,}); B: {c['B_rows']:,} "
        f"({c['B_train_test'][0]:,} / {c['B_train_test'][1]:,}).",
        f"- Students only in A: {c['only_in_A']}. Only in B: {c['only_in_B']} "
        f"({c['only_in_B_missing_S1M8GRADE']} without S1M8GRADE, "
        f"{c['only_in_B_unexplained']} not explained by a documented rule).",
        "- Matching key: the row position in the original NCES student file "
        "(hsls_16_student_v1_0.csv, sorted by STU_ID), not a persistent "
        "STU_ID; the authors' table has no STU_ID and is indexed by that "
        "position. The run checks it against sex, race, labels and every "
        "observed raw value of 53 features.",
        f"- Shared students have the same train/test assignment (conflicts: "
        f"{c['split_conflicts_shared']}); shared test cohort: "
        f"{c['shared_test']:,}.",
        f"- Label thresholds on X1TXMTH: A {lab['A_threshold']}, B "
        f"{lab['B_threshold']:.5f} (median of B's training rows). Shared "
        f"students labelled differently: train "
        f"{lab['shared_label_differences']['train']}, test "
        f"{lab['shared_label_differences']['test']}.",
        "",
        "Known pipeline differences: rows (undocumented rule vs. "
        "questionnaire-completion rule); missing values (A treats all "
        "negative values as missing, including valid negative scores of "
        "X1MTHID/X1MTHEFF/X1MTHINT; B only NCES reserve codes); imputation "
        "(A: k-NN of unknown configuration on the full table before the "
        "split; B: KNNImputer(k=5) fitted on training rows); scaling (A: "
        "MinMax on the full table; B: MinMax fitted on training rows); label "
        "threshold (above). See data/README.md.",
        "",
        "Evaluation sets: `native` = each source's own test set and labels; "
        "`shared_native` = the shared test students, each source scored "
        "against its own label definition (so the same students can carry "
        f"different labels: {lab['shared_label_differences']['test']} here); "
        "`shared_common_label` = the shared test students with one label "
        "for both sources (the authors' definition).",
        "",
        "## Full training",
        "",
        "| Method | Eval set | A acc. | B acc. | A AEOD | B AEOD | A DP | B DP | A EOpp | B EOpp |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method in ("Full Training (BCE)", "Full Training (Fair)"):
        for ev in EVAL_SETS:
            ra = f[(f.source == "A") & (f.method == method) & (f.eval_set == ev)].iloc[0]
            rb = f[(f.source == "B") & (f.method == method) & (f.eval_set == ev)].iloc[0]
            lines.append(
                f"| {method} | {ev} | {ra.accuracy:.4f} | {rb.accuracy:.4f} | "
                f"{ra.aeod:.4f} | {rb.aeod:.4f} | {ra.dp:.4f} | {rb.dp:.4f} | "
                f"{ra.eopp:.4f} | {rb.eopp:.4f} |")
    lines += [
        "",
        "## Deletion: Fair Unlearning vs. Retraining (Fair)",
        "",
        "Mean ± std over deletion seeds 0-4. `paired`: identical deleted "
        "students in A and B. `same_rate`: B deletes the stated share of its "
        "own training set; not exactly paired with A. For A the two modes are "
        "identical, so its columns repeat the paired values.",
        "",
        "| Rate | Mode | Eval set | Method | A acc. | B acc. | A AEOD | B AEOD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rate in sorted(s["rate"].unique()):
        for mode in ("paired", "same_rate"):
            for ev in EVAL_SETS:
                for method in ("Fair Unlearning", "Retraining (Fair)"):
                    a = s[(s.source == "A") & (s["rate"] == rate)
                          & (s.method == method) & (s.eval_set == ev)].iloc[0]
                    bq = s[(s.source == "B") & (s["mode"] == mode) & (s["rate"] == rate)
                           & (s.method == method) & (s.eval_set == ev)]
                    if bq.empty:
                        continue
                    b = bq.iloc[0]
                    lines.append(
                        f"| {rate:.0%} | {mode} | {ev} | {method} | "
                        f"{a.accuracy_mean:.4f} ± {a.accuracy_std:.4f} | "
                        f"{b.accuracy_mean:.4f} ± {b.accuracy_std:.4f} | "
                        f"{a.aeod_mean:.4f} ± {a.aeod_std:.4f} | "
                        f"{b.aeod_mean:.4f} ± {b.aeod_std:.4f} |")
    lines += [
        "",
        "## Fair Unlearning vs. Retraining (Fair) within each source",
        "",
        "Mean over seeds of |FU − RF| (AEOD, accuracy; shared cohort, common "
        "label) and of the parameter distance ||θ_FU − θ_RF||.",
        "",
        "| Rate | Mode | Source | mean abs. AEOD gap | mean abs. accuracy gap | parameter distance |",
        "|---|---|---|---|---|---|",
    ]
    df = pd.DataFrame(rows)
    df = df[df.eval_set == "shared_common_label"]
    for rate in sorted(df["rate"].unique()):
        for src, mode in (("A", "paired"), ("B", "paired"), ("B", "same_rate")):
            g = df[(df.source == src) & (df["mode"] == mode) & (df["rate"] == rate)]
            fu = g[g.method == "Fair Unlearning"].set_index("seed")
            rf = g[g.method == "Retraining (Fair)"].set_index("seed")
            lines.append(
                f"| {rate:.0%} | {mode} | {src} | "
                f"{(fu.aeod - rf.aeod).abs().mean():.5f} | "
                f"{(fu.accuracy - rf.accuracy).abs().mean():.5f} | "
                f"{fu.param_distance.mean():.2e} |")
    pp = p[(p["mode"] == "paired") & (p.eval_set != "native")
           & p.metric.isin(["accuracy", "aeod"])]
    lines += [
        "",
        "## Paired differences B − A (mode = paired, shared test cohort)",
        "",
        "Per-seed differences on the same students with the same deleted "
        "students; mean ± std over 5 seeds. Seed variation only: test-set "
        "sampling variation is not included.",
        "",
        "| Rate | Eval set | Method | Metric | B − A | std |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in pp.sort_values(["rate", "eval_set", "method", "metric"]).iterrows():
        lines.append(f"| {r['rate']:.0%} | {r.eval_set} | {r.method} | {r.metric} | "
                     f"{r.mean_diff_B_minus_A:+.4f} | {r.sd_diff:.4f} |")
    lines += [
        "",
        "## Limitations",
        "",
        "- The pipelines differ in cohort, missing values, imputation, "
        "scaling and label threshold simultaneously; differences between A "
        "and B cannot be attributed to any one of them.",
        "- The 75 students without an explaining rule are kept in B; why the "
        "authors excluded them is unknown.",
        "- One train/test split (seed 0) and five deletion seeds; std values "
        "reflect deletion seeds only.",
        "- Unlearning covers the model only: preprocessing (scaling, "
        "imputation, label threshold) is fitted before deletion and kept.",
        "- ε values (in deletion_raw.csv) are post-hoc estimates, not formal "
        "guarantees (README, \"Noise and certificate\"). Runtimes are single "
        "timings.",
    ]
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--pilot", action="store_true",
                        help="5%% deletion, seed 0; print only, write nothing")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    if args.pilot:
        full_rows, rows, meta = run(rates=(0.05,), seeds=(0,))
        print(json.dumps(meta["cohort"], indent=1))
        print(json.dumps(meta["labels"], indent=1))
        for r in full_rows:
            if r["eval_set"] == "shared_common_label":
                print(f"  {r['source']} {r['method']:21s} acc {r['accuracy']:.4f} "
                      f"AEOD {r['aeod']:.4f}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_rows, rows, meta = run()
    write_csv(out_dir / "full_training.csv", full_rows)
    write_csv(out_dir / "deletion_raw.csv", rows)
    write_csv(out_dir / "deletion_summary.csv", summarize(rows).to_dict("records"))
    write_csv(out_dir / "paired_differences.csv", paired_differences(rows))
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_report(out_dir, full_rows, rows, meta)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
