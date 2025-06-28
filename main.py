import os
import sys
from data_getter import load_combined_data

# Add the current directory to sys.path so we can import from tennis_data
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


# Load the data
df = load_combined_data()

# Show a sample
# print("Loaded tennis match data:")
# print(df.head())
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str)
# Save to Parquet
output_path = os.path.join(script_dir, "combined_matches.parquet")
df.to_parquet(output_path, index=False)
print(f"\nSaved combined data to: {output_path}")
