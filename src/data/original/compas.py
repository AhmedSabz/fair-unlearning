"""
ORIGINAL (team) COMPAS loader -- NOT the paper's preprocessing.

This is the project's first data setup, kept for comparison with the
paper-matched loader in src/data/datasets.py (select it with
`python -m experiments.run_deletion_sweep --loader original`). Restored
unchanged from commit 19089f1^ except for this docstring.

Differences from the paper loader:
    - target two_year_recid (paper: is_recid)
    - sensitive attribute coded African-American = 1, Caucasian = 0
      (paper: Caucasian = 1) and NOT a model input (paper: it is)
    - features age, priors_count, juvenile counts, sex, charge degree
      (paper adds length_of_stay, drops the juvenile counts)
    - stratified 80/20 split, random_state 42 by default (paper: 0)
    - standardized / one-hot features, no intercept, no row-norm scaling,
      so the (epsilon, delta) certificate does not apply (reported as inf)
"""

from pathlib import Path

import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
]

CATEGORICAL_COLUMNS = [
    "sex",
    "c_charge_degree",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_compas(
    data_path="data/raw/compas/compas-scores-two-years.csv",
    test_size=0.2,
    random_state=42,
):
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"COMPAS dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    # Keep only the two racial groups used in this experiment.
    df = df[
        df["race"].isin(["African-American", "Caucasian"])
    ].copy()

    # Apply common COMPAS screening-data eligibility filters.
    df = df[
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    # Keep only fields needed for our experiment.
    required_columns = (
        FEATURE_COLUMNS
        + ["race", "two_year_recid"]
    )

    df = df[required_columns].dropna().reset_index(drop=True)

    # Protected attribute:
    # 0 = Caucasian
    # 1 = African-American
    sensitive = (
        df["race"] == "African-American"
    ).astype("int64")

    y = df["two_year_recid"].astype("float32")

    X = df[FEATURE_COLUMNS]

    # Split before fitting preprocessing to avoid test-data leakage.
    indices = list(range(len(df)))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_df = X.iloc[train_indices]
    X_test_df = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    sensitive_train = sensitive.iloc[train_indices]
    sensitive_test = sensitive.iloc[test_indices]

    # Fit preprocessing only on training data.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float64),
        "y_train": torch.tensor(
            y_train.to_numpy(), dtype=torch.float64
        ),
        "sensitive_train": torch.tensor(
            sensitive_train.to_numpy(), dtype=torch.long
        ),
        "X_test": torch.tensor(X_test, dtype=torch.float64),
        "y_test": torch.tensor(
            y_test.to_numpy(), dtype=torch.float64
        ),
        "sensitive_test": torch.tensor(
            sensitive_test.to_numpy(), dtype=torch.long
        ),
        "feature_names": preprocessor.get_feature_names_out(),
        "preprocessor": preprocessor,
    }
