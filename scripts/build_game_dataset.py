"""Build game-level dataset from point-level data.

Pipeline:
1. Load dataset_clean.parquet
2. Filter to Hard court only
3. Aggregate points -> games (with indicators)
4. Integrate player stats (rank, age, ELO, player_id)
5. Add predictive features (h2h, will_serve_next, etc.)
6. Define target: p1_wins_next_game
7. Compute per-player, per-role sliding window features (W=2, 3, 5)
8. Export 3 parquet files: matches, games, game_features

Usage:
    python -m scripts.build_game_dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import (
    CLEAN_DATASET, PROCESSED_DIR,
    MATCHES_DATASET, GAMES_DATASET, GAME_FEATURES_DATASET,
    GAME_WINDOW_SIZES,
)
from src.features.game_aggregation import aggregate_games
from src.features.sliding_window import sliding_windows
from src.features.player_stats import (
    load_sackmann_data, load_player_elo, add_player_stats, fill_missing_elo,
)


# Leakage columns: in-game stats that won't be available at prediction time
LEAKAGE_COLS = [
    "svr_pts_won_pct",
    "aces",
    "double_faults",
    "first_serve_in_pct",
    "rally_avg",
    "n_points",
    "winners",
    "forced_errors",
    "unforced_errors",
    "net_approaches",
]


def build_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Extract match-level metadata from point data."""
    matches = (
        df.groupby("match_id")
        .agg(
            player_1=("Player 1", "first"),
            player_2=("Player 2", "first"),
            surface=("Surface", "first"),
            tournament=("Tournament", "first"),
            date=("Date", "first"),
            best_of=("Best of", "first"),
            round=("Round", "first"),
            p1_hand=("Pl 1 hand", "first"),
            p2_hand=("Pl 2 hand", "first"),
        )
        .reset_index()
    )
    return matches


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building Game-Level Dataset (Hard Court) ===\n")

    # 1. Load
    print("[1/7] Loading point data ...")
    df = pd.read_parquet(CLEAN_DATASET)
    print(f"  Total points: {len(df):,}")

    # 2. Filter Hard
    print("[2/7] Filtering to Hard court ...")
    df_hard = df[df["Surface"] == "Hard"].copy()
    print(f"  Hard points: {len(df_hard):,}")
    print(f"  Hard matches: {df_hard['match_id'].nunique():,}")

    # 3. Aggregate games
    print("[3/7] Aggregating to games ...")
    games = aggregate_games(df_hard)
    print(f"  Games: {len(games):,}")
    print(f"  Win rate (server): {games['won_game'].mean():.1%}")

    # 4. Player stats
    print("[4/7] Integrating player stats ...")
    matches = build_matches(df_hard)
    print(f"  Matches: {len(matches):,}")

    sackmann = load_sackmann_data()
    print(f"  Sackmann records: {len(sackmann):,}")

    elo_map = load_player_elo()
    print(f"  ELO players: {len(elo_map):,}")

    matches = add_player_stats(matches, sackmann, elo_map)

    # Fill missing ELO with rank-based estimation
    elo_before = matches["p1_elo"].notna().mean()
    matches = fill_missing_elo(matches)
    elo_after = matches["p1_elo"].notna().mean()
    print(f"  ELO before estimation: {elo_before:.1%}")
    print(f"  ELO after estimation:  {elo_after:.1%}")

    # Propagate player stats to games
    id_map = matches.set_index("match_id")[
        ["player_1", "player_2", "p1_player_id", "p2_player_id",
         "p1_rank", "p2_rank", "p1_age", "p2_age",
         "p1_elo", "p2_elo"]
    ].to_dict("index")

    for col in ["p1_player_id", "p2_player_id",
                "p1_rank", "p2_rank", "p1_age", "p2_age",
                "p1_elo", "p2_elo"]:
        games[col] = games["match_id"].map(
            lambda mid, c=col: id_map.get(mid, {}).get(c, np.nan)
        )

    # Derived features
    games["rank_diff"] = games["p1_rank"] - games["p2_rank"]
    games["age_diff"] = games["p1_age"] - games["p2_age"]
    games["elo_diff"] = games["p1_elo"] - games["p2_elo"]

    # 5. Predictive features
    print("[5/7] Adding predictive features ...")

    # will_serve_next: who serves next game (alternation)
    games["will_serve_next"] = (games["game_num"] % 2 == 0).astype(int)

    # h2h_last_5: head-to-head last 5 games (simplified)
    games = games.sort_values(["match_id", "game_num"]).reset_index(drop=True)
    h2h_p1_list = []
    h2h_p2_list = []
    h2h_total_list = []

    for mid, match_games in games.groupby("match_id"):
        match_games = match_games.sort_values("game_num")
        n = len(match_games)
        last_5 = match_games.iloc[max(0, n - 5):n]
        p1_wins = last_5["won_game"].sum()
        p2_wins = len(last_5) - p1_wins
        h2h_p1_list.append({"match_id": mid, "h2h_p1_wins_last_5": p1_wins})
        h2h_p2_list.append({"match_id": mid, "h2h_p2_wins_last_5": p2_wins})
        h2h_total_list.append({"match_id": mid, "h2h_total_last_5": len(last_5)})

    h2h_df = pd.DataFrame(h2h_p1_list)
    h2h_df = h2h_df.merge(pd.DataFrame(h2h_p2_list), on="match_id")
    h2h_df = h2h_df.merge(pd.DataFrame(h2h_total_list), on="match_id")
    games = games.merge(h2h_df, on="match_id", how="left")

    # days_since_last_served: proxy from last match date
    if "date" in games.columns:
        last_game_date = (
            games.groupby("match_id")["date"].max().reset_index()
        )
        last_game_date.columns = ["match_id", "last_match_date"]
        games = games.merge(last_game_date, on="match_id", how="left")
        games["days_since_last_served"] = (
            pd.to_datetime(games["last_match_date"]) - pd.to_datetime(games["date"])
        ).dt.days
        games.drop(columns=["last_match_date"], inplace=True)

    # 6. Target: p1_wins_next_game
    print("[6/7] Defining target p1_wins_next_game ...")

    # p1 is always the first server (game_num=1)
    games["p1_wins_game"] = (
        (games["game_num"] % 2 == 1) & (games["won_game"] == 1)
    ) | ((games["game_num"] % 2 == 0) & (games["won_game"] == 0))

    # Shift within each match to get next game outcome
    games["p1_wins_next_game"] = (
        games.groupby("match_id")["p1_wins_game"]
        .shift(-1)
        .astype("Int64")
    )

    target_coverage = games["p1_wins_next_game"].notna().mean() * 100
    print(f"  Target coverage: {target_coverage:.1f}%")

    # 7. Sliding windows (per-player, per-role)
    # NOTE: games must have leakage columns for rolling stats computation
    # They are removed from games.parquet after sliding windows are computed
    print(f"[7/7] Computing per-player sliding windows {GAME_WINDOW_SIZES} ...")
    features = sliding_windows(games, windows=GAME_WINDOW_SIZES)
    print(f"  Feature rows: {len(features):,}")

    # Remove raw leakage columns from games (rolling stats are in features)
    games_clean = games[[c for c in games.columns if c not in LEAKAGE_COLS]]

    # Note: game_features keeps all rolling stats (p1_aces_serve_W3 etc.)
    # which are DIFFERENT column names from LEAKAGE_COLS (aces, double_faults etc.)
    print(f"  Games features (no leakage): {len(games_clean.columns)}")
    print(f"  Window features per row: {len(features.columns) - 3}")

    # Export
    print("\nExporting ...")
    matches.to_parquet(MATCHES_DATASET, index=False)
    games_clean.to_parquet(GAMES_DATASET, index=False)
    features.to_parquet(GAME_FEATURES_DATASET, index=False)

    print(f"  {MATCHES_DATASET.name}: {len(matches):,} rows, {len(matches.columns)} cols")
    print(f"  {GAMES_DATASET.name}: {len(games_clean):,} rows, {len(games_clean.columns)} cols")
    print(f"  {GAME_FEATURES_DATASET.name}: {len(features):,} rows, {len(features.columns)} cols")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Target: p1_wins_next_game (coverage: {target_coverage:.1f}%)")
    print(f"Leakage removed: {LEAKAGE_COLS}")
    print(f"Window sizes: {GAME_WINDOW_SIZES}")
    print(f"Features per window: per-player per-role")
    print("Done.")


if __name__ == "__main__":
    main()
