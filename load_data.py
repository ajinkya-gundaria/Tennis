import pandas as pd
import os
import glob

def load_combined_data(data_folder=None):
    """
    Loads and combines ATP match CSV files from the specified folder.
    Returns a single pandas DataFrame.
    """
    if data_folder is None:
        # Default: go up one level and into tennis_raw_data/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(base_dir, '..', 'tennis_raw_data')

    # Load all files matching pattern
    pattern = os.path.join(data_folder, 'atp_matches_*.csv')
    csv_files = glob.glob(pattern)
    csv_files.sort()

    if not csv_files:
        raise FileNotFoundError(f"No match files found at: {data_folder}")

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    return df