import pandas as pd
def make_features(df):
    df = df.dropna(subset=['winner_rank', 'loser_rank', 'winner_ht', 'loser_ht', 'surface'])

    features = pd.DataFrame({
        'tourney_id': df['tourney_id'],
        'player_1': df['winner_name'],
        'player_2': df['loser_name'],
        'player_rank': df['winner_rank'],
        'opponent_rank': df['loser_rank'],
        'player_height': df['winner_ht'],
        'opponent_height': df['loser_ht'],
        'surface_Hard': (df['surface'] == 'Hard').astype(int),
        'surface_Clay': (df['surface'] == 'Clay').astype(int),
        'surface_Grass': (df['surface'] == 'Grass').astype(int),

        # New raw features
        'best_of': df['best_of'],
        'minutes': df['minutes'],
        'bpSaved': df['w_bpSaved'],
        'bpFaced': df['w_bpFaced'],
        'opp_bpSaved': df['l_bpSaved'],
        'opp_bpFaced': df['l_bpFaced'],

        # Normalized serve stats
        'first_serve_win_pct': df['w_1stWon'] / df['w_1stIn'].replace(0, pd.NA),
        'second_serve_win_pct': df['w_2ndWon'] / (df['w_svpt'] - df['w_1stIn']).replace(0, pd.NA),
        'first_serve_in_pct': df['w_1stIn'] / df['w_svpt'].replace(0, pd.NA),
        'ace_rate': df['w_ace'] / df['w_1stIn'].replace(0, pd.NA),
        'df_rate': df['w_df'] / (df['w_svpt'] - df['w_1stIn']).replace(0, pd.NA),

        'opp_first_serve_win_pct': df['l_1stWon'] / df['l_1stIn'].replace(0, pd.NA),
        'opp_second_serve_win_pct': df['l_2ndWon'] / (df['l_svpt'] - df['l_1stIn']).replace(0, pd.NA),
        'opp_first_serve_in_pct': df['l_1stIn'] / df['l_svpt'].replace(0, pd.NA),
        'opp_ace_rate': df['l_ace'] / df['l_1stIn'].replace(0, pd.NA),
        'opp_df_rate': df['l_df'] / (df['l_svpt'] - df['l_1stIn']).replace(0, pd.NA),

        'target': 1
    })

    mirror = features.copy()
    mirror['player_1'], mirror['player_2'] = features['player_2'], features['player_1']
    mirror['player_rank'], mirror['opponent_rank'] = features['opponent_rank'], features['player_rank']
    mirror['player_height'], mirror['opponent_height'] = features['opponent_height'], features['player_height']

    mirror['bpSaved'] = features['opp_bpSaved']
    mirror['bpFaced'] = features['opp_bpFaced']
    mirror['opp_bpSaved'] = features['bpSaved']
    mirror['opp_bpFaced'] = features['bpFaced']

    mirror['first_serve_win_pct'] = features['opp_first_serve_win_pct']
    mirror['second_serve_win_pct'] = features['opp_second_serve_win_pct']
    mirror['first_serve_in_pct'] = features['opp_first_serve_in_pct']
    mirror['ace_rate'] = features['opp_ace_rate']
    mirror['df_rate'] = features['opp_df_rate']

    mirror['opp_first_serve_win_pct'] = features['first_serve_win_pct']
    mirror['opp_second_serve_win_pct'] = features['second_serve_win_pct']
    mirror['opp_first_serve_in_pct'] = features['first_serve_in_pct']
    mirror['opp_ace_rate'] = features['ace_rate']
    mirror['opp_df_rate'] = features['df_rate']

    mirror['target'] = 0
    
    df_combined = pd.concat([features, mirror], ignore_index=True)
    stat_cols = [
    'first_serve_win_pct', 'second_serve_win_pct', 'first_serve_in_pct',
    'ace_rate', 'df_rate',
    'opp_first_serve_win_pct', 'opp_second_serve_win_pct',
    'opp_first_serve_in_pct', 'opp_ace_rate', 'opp_df_rate']

    for col in stat_cols:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    return df_combined
