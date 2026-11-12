"""
Adult (UCI income) loader, preprocessed as in Oesterling et al. (AISTATS 2024).

See src/data/datasets.py for every preprocessing step and its source.
"""

from src.data.datasets import load_dataset


def load_adult(test_size=0.2, random_state=0):
    """Return the adult train/test tensors used in the paper's experiments."""
    return load_dataset("adult", test_size=test_size, random_state=random_state)
