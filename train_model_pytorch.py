import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from features import make_features

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(script_dir, 'combined_matches.parquet')
df_raw = pd.read_parquet(parquet_path)

# Feature engineering
df_features = make_features(df_raw)
df_features['year'] = df_features['tourney_id'].astype(str).str[:4].astype(int)
df_features = df_features[df_features['year'] >= 2023]

# Encode players
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_players = encoder.fit_transform(df_features[['player_1', 'player_2']])

# Numeric features
X_numeric = df_features[[
    'player_rank', 'opponent_rank',
    'player_height', 'opponent_height',
    'surface_Hard', 'surface_Clay', 'surface_Grass'
]].values

X = np.hstack([X_players, X_numeric])
y = df_features['target'].values

# Define PyTorch model
class LogisticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Training settings
n_runs = 5
accs, losses = [], []

print("\nRunning logistic regression with PyTorch...\n")

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

    model = LogisticNet(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model
    model.train()
    for epoch in range(100):  # could tune this
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = (model(X_test_tensor) > 0.5).float()
        y_proba = model(X_test_tensor)
        acc = (y_pred == y_test_tensor).float().mean().item()
        log_loss_val = nn.BCELoss()(y_proba, y_test_tensor).item()

    accs.append(acc)
    losses.append(log_loss_val)
    print(f"Run {i+1}: Accuracy = {acc:.3f}, Log Loss = {log_loss_val:.3f}")

# Summary
print("\n--- Summary over splits ---")
print(f"Avg Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"Avg Log Loss: {np.mean(losses):.3f} ± {np.std(losses):.3f}")
