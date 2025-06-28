"""
Insert docstring here
"""

import os
import pandas as pd
from features import make_features


# Get path to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved Parquet file
parquet_path = os.path.join(script_dir, "combined_matches.parquet")
df = pd.read_parquet(parquet_path)
print(df.columns)
# Preview
print("Loaded DataFrame:")
print(df.head())
print(f"\nTotal matches: {len(df):,}")
print(df.columns.tolist())

print("Unique surface values in raw data:")
print(df["surface"].unique())
df_features = make_features(df)
print("Sample features:")
print(df_features.head())
print(df_features["surface_Hard"].value_counts())
print(df_features["surface_Clay"].value_counts())
print(df_features["surface_Grass"].value_counts())
print(df_features["target"].value_counts())
