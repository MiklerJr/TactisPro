"""Game-level aggregation from point-level data.

Groups points by (match_id, Gm#) to produce one row per game with
aggregated indicators (aces, DFs, win rates, etc.) and the binary
target won_game (1 = server won the game).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def _find_game_winner(game_df: pd.DataFrame) -> int:
    """Determine if the server won the game.

    The winner is decided by the last point before Gm# changes.
    We check isSvrWinner on that decisive point.
    """
    last_point = game_df.iloc[-1]
    return int(last_point["isSvrWinner"])


def _game_indicators(game_df: pd.DataFrame) -> dict:
    """Compute aggregated indicators for a single game."""
    n = len(game_df)
    if n == 0:
        return {}

    svr_won = game_df["isSvrWinner"].sum()

    return {
        "n_points": n,
        "won_game": int(svr_won > n / 2),
        "aces": int(game_df["isAce"].sum()),
        "double_faults": int(game_df["isDouble"].sum()),
        "first_serve_in_pct": float(game_df["1stIn"].mean()),
        "svr_pts_won_pct": float(svr_won / n),
        "rally_avg": float(game_df["rallyCount"].mean()),
        "winners": int(game_df["isRallyWinner"].sum()),
        "unforced_errors": int(game_df["isUnforced"].sum()),
        "forced_errors": int(game_df["isForced"].sum()),
        "net_approaches": int(game_df["net_approaches"].sum()),
    }


def aggregate_games(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate point-level data to game-level.

    Parameters
    ----------
    df : pd.DataFrame
        Point-level dataset (ideally filtered to a single surface).

    Returns
    -------
    pd.DataFrame
        One row per game with aggregated indicators.
    """
    # Sort to ensure correct ordering
    df = df.sort_values(["match_id", "Gm#", "Pt"]).reset_index(drop=True)

    games = []
    for (match_id, gm), group in df.groupby(["match_id", "Gm#"], sort=True):
        row = _game_indicators(group)

        # Game context (constant within a game)
        first = group.iloc[0]
        row["match_id"] = match_id
        row["game_num"] = int(gm)
        row["set_num"] = int(first["Set1"] + first["Set2"])
        row["is_tiebreak"] = bool(first.get("is_tiebreak", False))
        row["current_server"] = first["current_server"]
        row["current_returner"] = first["current_returner"]
        row["current_server_hand"] = first["current_server_hand"]
        row["current_returner_hand"] = first["current_returner_hand"]
        row["surface"] = first["Surface"]
        row["tournament"] = first["Tournament"]
        row["date"] = first["Date"]
        row["best_of"] = first["Best of"]

        # Score at end of game
        last = group.iloc[-1]
        row["set1"] = int(last["Set1"])
        row["set2"] = int(last["Set2"])
        row["gm1"] = int(last["Gm1"])
        row["gm2"] = int(last["Gm2"])

        games.append(row)

    return pd.DataFrame(games)
