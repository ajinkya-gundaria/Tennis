import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from features import make_features

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(script_dir, 'combined_matches.parquet')
df_raw = pd.read_parquet(parquet_path)

# Feature engineering
df_features = make_features(df_raw)

# Extract year from tourney_id
df_features['year'] = df_features['tourney_id'].astype(str).str[:4].astype(int)

# Keep only matches from last 2 years
df_features = df_features[df_features['year'] >= 2023]

# Have player identification
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_players = encoder.fit_transform(df_features[['player_1', 'player_2']])

# Use only numeric features
X_numeric = df_features[[
    'player_rank', 'opponent_rank',
    'player_height', 'opponent_height',
    'surface_Hard', 'surface_Clay', 'surface_Grass',
    'best_of', 'minutes', 'bpSaved',
    'bpFaced', 'opp_bpSaved', 'opp_bpFaced', 'first_serve_win_pct',
    'second_serve_win_pct', 'first_serve_in_pct', 'ace_rate', 'df_rate', 
    'opp_first_serve_win_pct', 'opp_second_serve_win_pct', 'opp_first_serve_in_pct', 'opp_ace_rate', 'opp_df_rate',
]].values

X = np.hstack([X_players, X_numeric])
y = df_features['target'].values
# Train/test splits
n_runs = 1
accs = []
losses = []

print("\nRunning logistic regression with full dataset...\n")

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    accs.append(acc)
    losses.append(loss)

    print(f"Run {i+1}: Accuracy = {acc:.3f}, Log Loss = {loss:.3f}")

# Summary
print("\n--- Summary over splits ---")
print(f"Avg Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Avg Log Loss: {np.mean(losses):.3f} ± {np.std(losses):.3f}")
