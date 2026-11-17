"""
The team's ORIGINAL data loaders (first project iteration), kept next to the
paper-matched loaders in src/data/datasets.py for comparison. See each
module's docstring for how it differs from the paper.
"""

from src.data.original.adult import load_adult
from src.data.original.compas import load_compas
from src.data.original.hsls import load_hsls

ORIGINAL_LOADERS = {"adult": load_adult, "compas": load_compas, "hsls": load_hsls}


def load_original(name, random_state=None):
    """
    Load a dataset with the team's original preprocessing.

    random_state is passed to the COMPAS and HSLS loaders (their own default
    is 42). Adult keeps the fixed UCI train/test files, so it has no split
    seed and random_state is ignored.
    """
    if name not in ORIGINAL_LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    if name == "adult" or random_state is None:
        return ORIGINAL_LOADERS[name]()
    return ORIGINAL_LOADERS[name](random_state=random_state)
