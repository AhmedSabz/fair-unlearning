"""
ORIGINAL (team) Adult loader -- NOT the paper's preprocessing.

This is the project's first data setup, kept for comparison with the
paper-matched loader in src/data/datasets.py (select it with
`python -m experiments.run_deletion_sweep --loader original`). Restored
unchanged from commit 19089f1^ except for this docstring.

Differences from the paper loader:
    - sensitive attribute: sex by default (paper: race, White vs Black);
      with sensitive_attribute="race" it is White vs all non-White
    - the sensitive attribute is NOT a model input (paper: it is)
    - rows with missing values are dropped; the UCI train/test files are
      kept as the split (paper: merged, then a random 80/20 split)
    - features: all columns except income and the sensitive one, numeric
      columns standardized, categoricals one-hot (paper drops fnlwgt,
      workclass, occupation, education, native-country)
    - no intercept column and no row-norm scaling, so ||x|| > 1 and the
      (epsilon, delta) certificate does not apply (reported as inf)
"""

from pathlib import Path

import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def load_adult(data_dir="data/raw/adult", sensitive_attribute="sex"):
    """
    Load the official Adult train/test files and preprocess them.

    The sensitive attribute is kept separate from model features.
    Returns a dictionary of PyTorch tensors and preprocessing metadata.
    """

    data_dir = Path(data_dir)

    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find adult.data and adult.test in {data_dir}"
        )

    # Read the official training data.
    train_df = pd.read_csv(
        train_path,
        names=COLUMNS,
        sep=",",
        skipinitialspace=True,
        na_values="?",
    )

    # The official test file starts with a comment line.
    test_df = pd.read_csv(
        test_path,
        names=COLUMNS,
        sep=",",
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )

    # Remove whitespace and clean income labels.
    for df in (train_df, test_df):
        for column in df.columns:
            if df[column].dtype == "object":
                df[column] = df[column].str.strip()

        df["income"] = df["income"].str.replace(".", "", regex=False)

    if sensitive_attribute not in ["sex", "race"]:
        raise ValueError("sensitive_attribute must be 'sex' or 'race'")

    # Remove rows with missing values.
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    # Separate labels and sensitive attributes.
    y_train = (train_df["income"] == ">50K").astype("float32")
    y_test = (test_df["income"] == ">50K").astype("float32")

    if sensitive_attribute == "sex":
        sensitive_train = (train_df["sex"] == "Male").astype("int64")
        sensitive_test = (test_df["sex"] == "Male").astype("int64")
    else:
        # Binary race grouping: White versus non-White.
        sensitive_train = (train_df["race"] == "White").astype("int64")
        sensitive_test = (test_df["race"] == "White").astype("int64")

    # Do not allow the sensitive attribute to become a model feature.
    feature_columns = [
        col for col in COLUMNS
        if col not in ["income", sensitive_attribute]
    ]

    categorical_columns = [
        col for col in feature_columns
        if col not in NUMERIC_COLUMNS
    ]

    # Fit preprocessing ONLY on training data.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(
        train_df[feature_columns]
    )

    X_test = preprocessor.transform(
        test_df[feature_columns]
    )

    # Convert sparse matrices to dense arrays if necessary.
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    # Convert everything to PyTorch tensors.
    result = {
        "X_train": torch.tensor(X_train, dtype=torch.float64),
        "y_train": torch.tensor(y_train.to_numpy(), dtype=torch.float64),
        "sensitive_train": torch.tensor(
            sensitive_train.to_numpy(), dtype=torch.long
        ),
        "X_test": torch.tensor(X_test, dtype=torch.float64),
        "y_test": torch.tensor(y_test.to_numpy(), dtype=torch.float64),
        "sensitive_test": torch.tensor(
            sensitive_test.to_numpy(), dtype=torch.long
        ),
        "feature_names": preprocessor.get_feature_names_out(),
        "preprocessor": preprocessor,
    }

    return result
