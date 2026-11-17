"""Shared settings for the legacy scripts (first project iteration)."""

from pathlib import Path

from experiments.run_deletion_sweep import PAPER_HYPERPARAMS  # noqa: F401

# Model files written by train_*.py and read by unlearn_adult.py. *.pth is
# gitignored.
MODEL_DIR = Path("scripts/legacy/models")

# Sensitive attribute = race on every dataset (src/data/datasets.py).
GROUP_NAMES = {
    "adult": {0: "Black (group 0)", 1: "White (group 1)"},
    "compas": {0: "African-American (group 0)", 1: "Caucasian (group 1)"},
    "hsls": {0: "Other races (group 0)", 1: "White/Asian (group 1)"},
}
