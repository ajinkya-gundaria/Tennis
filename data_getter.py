import pandas as pd
import requests
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
        data_folder = os.path.join(base_dir, "..", "tennis_raw_data")

    # Load all files matching pattern
    pattern = os.path.join(data_folder, "atp_matches_*.csv")
    csv_files = glob.glob(pattern)
    csv_files.sort()

    if not csv_files:
        raise FileNotFoundError(f"No match files found at: {data_folder}")

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    return df


url_template = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_matches_"
output_path_template = "atp_matches_"


def fetch_data(start_year, end_year):
    for year in range(start_year, end_year + 1):
        url = f"{url_template}{year}.csv"
        output_path = f"{output_path_template}{year}.csv"
        response = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(response.content)


def clear_data(start_year, end_year):
    for year in range(start_year, end_year + 1):
        output_path = f"{output_path_template}{year}.csv"
        os.remove(output_path)


fetch_data(1968, 2024)
clear_data(1968, 2024)
