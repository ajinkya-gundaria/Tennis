import pandas as pd


def make_features(df_matches: pd.DataFrame) -> pd.DataFrame:
    df = df_matches.dropna(
        subset=[
            "winner_name",
            "loser_name",
            "winner_rank",
            "loser_rank",
            "winner_ht",
            "loser_ht",
            "surface",
        ]
    ).copy()

    # Winner's perspective
    df["player_1"] = df["winner_name"]
    df["player_2"] = df["loser_name"]
    df["player_rank"] = df["winner_rank"]
    df["opponent_rank"] = df["loser_rank"]
    df["player_height"] = df["winner_ht"]
    df["opponent_height"] = df["loser_ht"]
    df["target"] = 1
    df["surface_clean"] = df["surface"]

    # Mirror perspective
    mirror = df.copy()
    mirror["player_1"] = df["loser_name"]
    mirror["player_2"] = df["winner_name"]
    mirror["player_rank"] = df["loser_rank"]
    mirror["opponent_rank"] = df["winner_rank"]
    mirror["player_height"] = df["loser_ht"]
    mirror["opponent_height"] = df["winner_ht"]
    mirror["target"] = 0

    # Combine
    df_combined = pd.concat([df, mirror], ignore_index=True)

    # Surface dummies
    df_combined["surface_Hard"] = (df_combined["surface_clean"] == "Hard").astype(int)
    df_combined["surface_Clay"] = (df_combined["surface_clean"] == "Clay").astype(int)
    df_combined["surface_Grass"] = (df_combined["surface_clean"] == "Grass").astype(int)

    return df_combined[
        [
            "tourney_id",
            "player_1",
            "player_2",
            "player_rank",
            "opponent_rank",
            "player_height",
            "opponent_height",
            "surface_Hard",
            "surface_Clay",
            "surface_Grass",
            "target",
        ]
    ]
