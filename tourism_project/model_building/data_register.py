"""Data registration: validate the raw tourism dataset and print a summary.

Run from the repository root:
    python tourism_project/model_building/data_register.py
"""
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def write_step_summary(lines) -> None:
    """Inside GitHub Actions, show the results on the run's summary page."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as handle:
            handle.write("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # --- Validate columns ---
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        sys.exit(f"Validation FAILED. Missing columns: {missing}")
    print("Validation PASSED: all expected columns are present.")
    if extra:
        print(f"Note: unexpected columns (removed in data preparation): {extra}")

    # --- Print dataset summary ---
    size_kb = Path(DATA_PATH).stat().st_size / 1024
    fingerprint = hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest()[:16]
    print(f"\nFile: {DATA_PATH} ({size_kb:.0f} KB) | SHA-256 fingerprint: {fingerprint}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print(f"Missing values (total): {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).round(3).to_string())
    print("\nColumn data types:")
    print(df.dtypes.to_string())

    write_step_summary([
        "## Dataset registration", "",
        f"- Validation: **passed** ({len(EXPECTED_COLUMNS)} expected columns present)",
        f"- Rows x columns: **{df.shape[0]:,} x {df.shape[1]}**",
        f"- Buyer share (`ProdTaken` = 1): **{df['ProdTaken'].mean():.1%}**",
        f"- File fingerprint (SHA-256): `{fingerprint}`",
    ])


if __name__ == "__main__":
    main()
