"""Data preparation: clean the registered dataset and create train/test splits.

Run from the repository root:
    python tourism_project/model_building/prep.py

Outputs are written to the current working directory so that the next
GitHub Actions job can receive them as a workflow artifact:
    Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Columns that carry no predictive value:
#   Unnamed: 0      - leftover row index from the CSV export
#   CustomerID      - unique identifier
#   ProductPitched  - maps one-to-one to Designation (e.g. Executive <-> Basic), so it is redundant
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID", "ProductPitched"]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def harmonise_labels(df: pd.DataFrame, text_cols) -> dict:
    """Merge category labels that differ only by case/spacing into the most frequent one (in place).

    Returns {column: {variant: standard_label}} for the labels that were changed.
    """
    replacements = {}
    for col in text_cols:
        groups = {}
        for label, count in df[col].value_counts().items():
            key = "".join(str(label).lower().split())
            groups.setdefault(key, []).append((label, count))
        mapping = {}
        for variants in groups.values():
            standard = max(variants, key=lambda item: item[1])[0]
            mapping.update({label: standard for label, _ in variants if label != standard})
        if mapping:
            df[col] = df[col].replace(mapping)
            replacements[col] = mapping
    return replacements


def write_step_summary(lines) -> None:
    """Inside GitHub Actions, show the results on the run's summary page."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as handle:
            handle.write("\n".join(lines) + "\n")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset and print a short cleaning report."""
    df = df.copy()
    rows_raw = len(df)

    # 1. Remove unnecessary columns
    dropped = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"Dropped columns: {dropped}")

    # 2. Standardise text columns: trim whitespace, then merge labels that differ only by
    #    case or spacing (e.g. "Fe Male" -> "Female") into the most frequent spelling
    text_cols = df.select_dtypes(exclude="number").columns
    for col in text_cols:
        df[col] = df[col].str.strip()
    replacements = harmonise_labels(df, text_cols)
    print(f"Label variants merged: {replacements if replacements else 'none'}")

    # 3. Remove rows without a target label and exact duplicate records
    df = df.dropna(subset=[TARGET])
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {n_before - len(df)}")

    # 4. Sanity checks
    assert set(df[TARGET].unique()) <= {0, 1}, "Target must be binary (0/1)"
    print(f"Missing values remaining: {int(df.isna().sum().sum())}")
    print(f"Rows: {rows_raw} -> {len(df)} | Feature columns: {df.shape[1] - 1}")
    return df


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    df = clean(raw)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified split keeps the buyer / non-buyer ratio equal in both sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"\nTrain: {Xtrain.shape} | Test: {Xtest.shape}")
    print(f"Buyer share - train: {ytrain.mean():.3f} | test: {ytest.mean():.3f}")
    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

    write_step_summary([
        "## Data preparation", "",
        f"- Rows: **{len(raw):,} -> {len(df):,}** ({len(raw) - len(df)} duplicate or unlabeled rows removed)",
        f"- Features kept: **{X.shape[1]}** (dropped: {', '.join(COLUMNS_TO_DROP)})",
        f"- Split: **{len(Xtrain):,} train / {len(Xtest):,} test**, buyer share {ytrain.mean():.1%} / {ytest.mean():.1%}",
        "- Output files `Xtrain.csv`, `Xtest.csv`, `ytrain.csv`, `ytest.csv` are passed to the training job as an artifact",
    ])


if __name__ == "__main__":
    main()
