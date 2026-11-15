
import pandas as pd
import numpy as np
from pathlib import Path

# Load extracted features
file_path = "data/processed/ppg_features.csv"

df = pd.read_csv(file_path)

# Dataset overview
print("\nDATASET OVERVIEW")
print("----------------")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Missing values
print("\nMISSING VALUES")
print(df.isna().sum())

print("\nTOTAL MISSING VALUES:")
print(df.isna().sum().sum())

# Feature summary
print("\nFEATURE SUMMARY")
print(df.describe())

# Class distribution
print("\nCLASS DISTRIBUTION")
print(df["Quality"].value_counts().sort_index())

print("\nCLASS PERCENTAGES")
print(
    df["Quality"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Near-constant signals
flat = df[df["std"] < 0.001]

print("\nNEAR-CONSTANT SIGNALS")
print("---------------------")
print("Total:", len(flat))

print("\nNEAR-CONSTANT RECORDINGS BY QUALITY")
print(flat["Quality"].value_counts().sort_index())

print("\nQUALITY PERCENTAGES:")
print(
    flat["Quality"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Recordings with missing statistical features
print("\nRECORDINGS WITH MISSING SKEWNESS OR KURTOSIS")

missing = df[
    df[["skewness", "kurtosis"]].isna().any(axis=1)
]

if missing.empty:
    print("None found.")
else:
    print(
        missing[
            ["record_id", "Quality", "std", "skewness", "kurtosis"]
        ].to_string(index=False)
    )

# Show sample near-constant recordings
print("\nSAMPLE OF NEAR-CONSTANT RECORDINGS")

if flat.empty:
    print("None found.")
else:
    print(
        flat[
            ["record_id", "Quality", "mean", "std", "range"]
        ].head(15).to_string(index=False)
    )
print("\nSUBJECT ANALYSIS")
print("----------------")

print("Unique subjects:", df["subject_id"].nunique())

print("\nRecordings per subject:")
print(df.groupby("subject_id").size().describe())

print("\nQuality distribution by subject:")
print(df.groupby("subject_id")["Quality"].agg(["count", "mean"]))

print("\nQUALITY BY RECORDING FORMAT")
print("---------------------------")

df["recording_format"] = np.where(
    df["record_id"].astype(str).str[:3].astype(int) <= 111,
    "Older single-channel",
    "Newer RGB"
)

print(
    df.groupby("recording_format")["Quality"]
    .agg(["count", "mean"])
)

print("\nSUBJECTS BY RECORDING FORMAT")
print("----------------------------")

print(
    df.groupby("recording_format")["subject_id"]
    .nunique()
)

print("\nRECORDING COUNTS BY SUBJECT AND FORMAT")

print(
    pd.crosstab(
        df["subject_id"],
        df["recording_format"]
    )
)

# Create a consistent-format dataset for modeling
rgb_df = df[df["recording_format"] == "Newer RGB"].copy()

rgb_output = Path("data/processed/ppg_features_rgb.csv")
rgb_df.to_csv(rgb_output, index=False)

print("\nRGB DATASET")
print("-----------")
print("Recordings:", len(rgb_df))
print("Unique subjects:", rgb_df["subject_id"].nunique())
print("Saved to:", rgb_output)