"""
Dataset preprocessing, matching Oesterling et al. (AISTATS 2024).

Reimplements the preprocessing in the authors' official code
(https://github.com/AI4LIFE-GROUP/fair-unlearning, Apache-2.0):
    datasets/Adult/prepare_adult.py
    datasets/COMPAS/prepare_compas.py
    datasets/HSLS/prepare_hsls.py
    datasets/Dataloader.py
The prepare_* functions below reproduce the authors' cleaned tables value for
value (Adult 46,447 rows, COMPAS 5,278, HSLS 14,509).

Setup, per the paper (Sec. 5.1, App. A.6) and the official code:
    1. Sensitive attribute = race on all three datasets.
         Adult : White (1) vs Black (0); other races dropped
         COMPAS: Caucasian (1) vs African-American (0)
         HSLS  : racebin, White/Asian (1) vs Black/Hispanic/Native/other (0)
    2. Adult : merge adult.data + adult.test; drop fnlwgt, workclass,
               occupation, education, native-country; one-hot the rest.
    3. COMPAS: ProPublica filters; features age, charge degree (+1 felony,
               -1 misdemeanor), race, sex, priors_count, length_of_stay;
               target is_recid.
    4. HSLS  : authors' k-NN-imputed survey table (from Jeong et al., AAAI
               2022); target gradebin (9th-grade math grade, binarized);
               MinMax scaled.
    5. The sensitive attribute is also a model input, as in the official code.
    6. Random 80/20 split, then every row divided by the largest training-row
       norm so ||x|| <= 1 (Theorem 1 assumption). Adult is standardized
       first. An explicit intercept column of ones is appended last.

The intercept column cancels inside the fairness penalty (it is the same for
every row) and is L2-regularized, as in the official code.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


RAW_DIR = Path("data/raw")

# Label column and sensitive-attribute (race) column per dataset.
LABELS = {"adult": "income", "compas": "is_recid", "hsls": "gradebin"}
SENSITIVE_COLUMNS = {"adult": "race", "compas": "race", "hsls": "racebin"}


# ----------------------------------------------------------------------
# Step 1: rebuild the authors' cleaned tables from the raw files
# ----------------------------------------------------------------------

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]


def prepare_adult(raw_dir=RAW_DIR / "adult"):
    """Same steps as the official prepare_adult.py."""
    raw_dir = Path(raw_dir)
    read = dict(header=None, sep=r",\s+", engine="python")
    train = pd.read_csv(raw_dir / "adult.data", **read)
    test = pd.read_csv(raw_dir / "adult.test", skiprows=1, **read)
    df = pd.concat([train, test], ignore_index=True)
    df.columns = ADULT_COLUMNS

    df = df.drop(columns=["fnlwgt", "workclass", "occupation"])
    df = df[df["race"].isin(["White", "Black"])].copy()

    df["race"] = (df["race"] == "White").astype(int)      # Black 0, White 1
    df["gender"] = (df["gender"] == "Male").astype(int)   # Female 0, Male 1
    df["income"] = df["income"].str.startswith(">").astype(int)

    df = df.drop(columns=["education", "native-country"])

    # Non-numeric columns, same as the official set(df) - numeric columns.
    # (select_dtypes works for both pandas 2 "object" and pandas 3 "str".)
    categorical = list(df.select_dtypes(exclude="number").columns)
    dummies = pd.get_dummies(df[categorical]).astype(int)
    df = pd.concat([df.drop(columns=categorical), dummies], axis=1)
    return df.reset_index(drop=True)


def prepare_compas(raw_path=RAW_DIR / "compas" / "compas-scores-two-years.csv"):
    """Same steps as the official prepare_compas.py (without its saved split)."""
    df = pd.read_csv(raw_path, index_col=0)
    df = df[[
        "age", "c_charge_degree", "race", "sex", "priors_count",
        "days_b_screening_arrest", "is_recid", "c_jail_in", "c_jail_out",
    ]]

    keep = (
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
    )
    df = df.loc[keep].copy()

    df["length_of_stay"] = (
        pd.to_datetime(df["c_jail_out"]) - pd.to_datetime(df["c_jail_in"])
    ).dt.days
    df = df.drop(columns=["c_jail_in", "c_jail_out", "days_b_screening_arrest"])

    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    df["race"] = (df["race"] == "Caucasian").astype(int)   # AA 0, Caucasian 1
    df["sex"] = (df["sex"] == "Male").astype(int)          # Female 0, Male 1
    df = df.rename(columns={"sex": "gender"})
    df["c_charge_degree"] = np.where(df["c_charge_degree"] == "F", 1, -1)
    return df.reset_index(drop=True)


def prepare_hsls(pkl_path=RAW_DIR / "hsls" / "hsls_df_knn_impute_past_v2.pkl"):
    """Same steps as the official prepare_hsls.py."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"{pkl_path} not found. It ships with this repo; see data/README.md."
        )
    df = pd.read_pickle(pkl_path)

    df = df.mask(df <= -7)        # survey codes <= -7 mean missing
    df = df.dropna()

    df["gradebin"] = df["grade9thbin"]
    # studentrace is stored scaled to [0, 1] in steps of 1/7; codes 1 and 7
    # are the White/Asian group in the official preprocessing.
    race_code = (df["studentrace"] * 7).astype(int)
    df["racebin"] = ((race_code == 7) | (race_code == 1)).astype(int)
    df["sexbin"] = df["studentgender"].astype(int)
    df = df.drop(columns=[
        "studentgender", "grade9thbin", "grade12thbin", "studentrace",
    ])

    scaled = MinMaxScaler().fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


PREPARERS = {"adult": prepare_adult, "compas": prepare_compas, "hsls": prepare_hsls}


# ----------------------------------------------------------------------
# Step 2: split, scale to ||x|| <= 1, add intercept
# ----------------------------------------------------------------------

def load_dataset(name, test_size=0.2, random_state=0):
    """
    Load one dataset exactly as the paper's experiments see it.

    random_state picks the train/test split (the paper uses one split).

    Returns a dictionary:
        X_train, y_train, sensitive_train, X_test, y_test, sensitive_test
        (float64 / long tensors; last column of X is the intercept),
        feature_names, and max_train_row_norm.
    """
    if name not in PREPARERS:
        raise ValueError(f"Unknown dataset: {name}")

    df = PREPARERS[name]()
    label_col = LABELS[name]
    sens_col = SENSITIVE_COLUMNS[name]

    y = df[label_col].to_numpy(dtype=np.float64)
    s = df[sens_col].to_numpy().astype(np.int64)
    features = df.drop(columns=[label_col])   # race stays in, like the paper
    feature_names = list(features.columns)
    X = features.to_numpy(dtype=np.float64)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=test_size, random_state=random_state,
    )

    if name == "adult":
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    max_norm = np.linalg.norm(X_train, axis=1).max()
    X_train, X_test = X_train / max_norm, X_test / max_norm

    X_train = np.hstack([X_train, np.ones((len(X_train), 1))])
    X_test = np.hstack([X_test, np.ones((len(X_test), 1))])
    feature_names.append("intercept")

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float64),
        "y_train": torch.tensor(y_train, dtype=torch.float64),
        "sensitive_train": torch.tensor(s_train, dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float64),
        "y_test": torch.tensor(y_test, dtype=torch.float64),
        "sensitive_test": torch.tensor(s_test, dtype=torch.long),
        "feature_names": np.array(feature_names),
        "max_train_row_norm": float(max_norm),
    }
