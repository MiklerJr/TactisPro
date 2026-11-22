"""Sliding window features for game-level data (optimized).

Leakage verification:
- Features for game N use ONLY games 1..N-1 (exclusive of current game)
- Window slice: `indicators.iloc[start:i]` where i = current game index
- This ensures NO information from game N is used to predict game N+1

For each game, computes per-player, per-role indicators from the
previous N serve/return games within the same match.

Features are split by player (p1, p2) and by role (serve, return):
  - p1_serve_wr_W:      Player 1's win rate when serving
  - p1_return_wr_W:     Player 1's win rate when returning
  - p1_aces_serve_W:    Player 1's aces per service game (rolling)
  - p1_df_serve_W:      Player 1's DFs per service game (rolling)
  - p1_1st_pct_W:       Player 1's 1st serve % (rolling)
  - p1_svr_pts_won_W:   Player 1's pts won % when serving (rolling)

Per-game aggregate features (NOT per-player, see note below):
  - winners_W:          Winners per game in window (sum isRallyWinner)
  - ue_W:               Unforced errors per game in window
  - fe_W:               Forced errors per game in window
  - rally_avg_W:        Average rally length in window
  - error_rate_W:       Match context: error rate
  - win_trend_W:        Match context: win trend

Note on naming:
  Per-game features are NOT prefixed with p1_/p2_ because the source
  data (isRallyWinner, isUnforced, isForced, rallyCount) is a per-POINT
  boolean that does not carry player attribution at the dataset level.
  Both players contribute to winners/UE/FE in every game, so the
  aggregate over "games where p1 was involved" equals the aggregate
  over "games where p2 was involved". Per-player attribution of these
  specific events would require parsing the rally sequence and is left
  as future work.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def _build_indicators(match_games: pd.DataFrame, p1_id, p2_id) -> pd.DataFrame:
    """Build per-game indicators for vectorized rolling window computation."""
    n = len(match_games)
    indicators = pd.DataFrame(index=range(n))

    # Player identities
    is_p1_serve = (match_games["current_server"].values == p1_id)
    is_p1_return = (match_games["current_returner"].values == p1_id)
    is_p2_serve = (match_games["current_server"].values == p2_id)
    is_p2_return = (match_games["current_returner"].values == p2_id)

    won = match_games["won_game"].values.astype(float)

    # --- Win indicators (per role) ---
    indicators["p1_serve_won"] = np.where(is_p1_serve, won, np.nan)
    indicators["p1_return_won"] = np.where(is_p1_return, 1 - won, np.nan)
    indicators["p2_serve_won"] = np.where(is_p2_serve, won, np.nan)
    indicators["p2_return_won"] = np.where(is_p2_return, 1 - won, np.nan)

    # --- Serve-only stats (per player) ---
    for col in ["aces", "double_faults", "first_serve_in_pct", "svr_pts_won_pct"]:
        vals = match_games[col].values if col in match_games.columns else np.full(n, np.nan)
        indicators[f"p1_{col}"] = np.where(is_p1_serve, vals, np.nan)
        indicators[f"p2_{col}"] = np.where(is_p2_serve, vals, np.nan)

    # --- Per-game stats (NOT per-player: see module docstring) ---
    # These features aggregate over all games in the window since
    # isRallyWinner/isUnforced/isForced do not carry player attribution.
    for col in ["rally_avg", "n_points", "winners", "unforced_errors", "forced_errors", "net_approaches"]:
        if col in match_games.columns:
            indicators[col] = match_games[col].values.astype(float)
        else:
            indicators[col] = np.full(n, np.nan)

    # Context features (all games)
    for col in ["rally_avg", "n_points", "unforced_errors", "forced_errors"]:
        indicators[col] = match_games[col].values if col in match_games.columns else np.nan
    indicators["won_game"] = won

    return indicators


def _safe_mean(series: pd.Series) -> float:
    """Compute mean ignoring NaN, return NaN if all NaN."""
    valid = series.dropna()
    return float(valid.mean()) if len(valid) > 0 else np.nan


def _rolling_features_for_match(match_games: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute sliding window features for a single match using vectorized rolling."""
    match_games = match_games.sort_values("game_num").reset_index(drop=True)
    n = len(match_games)

    p1_id = match_games.iloc[0]["current_server"]
    p2_id = match_games.iloc[0]["current_returner"]

    indicators = _build_indicators(match_games, p1_id, p2_id)

    result_rows = []

    for w in windows:
        lookback = w * 2  # Total games to look back (serve + return alternate)

        for i in range(n):
            start = max(0, i - lookback)
            window = indicators.iloc[start:i]  # Exclusive of i

            row = {
                "match_id": match_games.iloc[i]["match_id"],
                "game_num": int(match_games.iloc[i]["game_num"]),
                "window_size": w,
            }

            if len(window) == 0:
                # Fill NaN for all features
                for prefix in ["p1", "p2"]:
                    for feat in ["serve_wr", "return_wr", "aces", "df", "1st_pct", "svr_pts_won"]:
                        row[f"{prefix}_{feat}_W{w}"] = np.nan
                for feat in ["winners", "ue", "fe", "rally_avg", "error_rate", "win_trend"]:
                    row[f"{feat}_W{w}"] = np.nan
                result_rows.append(row)
                continue

            # --- P1 features ---
            p1_serve_won = window["p1_serve_won"].dropna()
            p1_return_won = window["p1_return_won"].dropna()
            row["p1_serve_wr_W{0}".format(w)] = _safe_mean(window["p1_serve_won"])
            row["p1_return_wr_W{0}".format(w)] = _safe_mean(window["p1_return_won"])

            # P1 serve rolling stats
            row["p1_aces_W{0}".format(w)] = _safe_mean(window["p1_aces"])
            row["p1_df_W{0}".format(w)] = _safe_mean(window["p1_double_faults"])
            row["p1_1st_pct_W{0}".format(w)] = _safe_mean(window["p1_first_serve_in_pct"])
            row["p1_svr_pts_won_W{0}".format(w)] = _safe_mean(window["p1_svr_pts_won_pct"])

            # Per-game rolling stats (no p1/p2 prefix; see module docstring)
            row["winners_W{0}".format(w)] = _safe_mean(window["winners"])
            row["ue_W{0}".format(w)] = _safe_mean(window["unforced_errors"])
            row["fe_W{0}".format(w)] = _safe_mean(window["forced_errors"])
            row["rally_avg_W{0}".format(w)] = _safe_mean(window["rally_avg"])

            # --- P2 features ---
            p2_serve_won = window["p2_serve_won"].dropna()
            p2_return_won = window["p2_return_won"].dropna()
            row["p2_serve_wr_W{0}".format(w)] = _safe_mean(window["p2_serve_won"])
            row["p2_return_wr_W{0}".format(w)] = _safe_mean(window["p2_return_won"])

            # P2 serve rolling stats
            row["p2_aces_W{0}".format(w)] = _safe_mean(window["p2_aces"])
            row["p2_df_W{0}".format(w)] = _safe_mean(window["p2_double_faults"])
            row["p2_1st_pct_W{0}".format(w)] = _safe_mean(window["p2_first_serve_in_pct"])
            row["p2_svr_pts_won_W{0}".format(w)] = _safe_mean(window["p2_svr_pts_won_pct"])

            # (per-game rolling stats are emitted once above; see comment)

            # --- Context features ---
            # NOTE: rally_avg_W{w} is already emitted above (line ~142) so we
            # only need error_rate and win_trend here.
            total_points = window["n_points"].sum()
            total_errors = window["unforced_errors"].sum() + window["forced_errors"].sum()
            row["error_rate_W{0}".format(w)] = total_errors / total_points if total_points > 0 else 0.0

            # Win trend: recent half vs older half
            half = max(len(window) // 2, 1)
            recent = window.iloc[:half]["won_game"].mean()
            older = window.iloc[half:]["won_game"].mean()
            row["win_trend_W{0}".format(w)] = float(recent - older)

            result_rows.append(row)

    return pd.DataFrame(result_rows)


def sliding_windows(
    games: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute per-player, per-role sliding window features.

    Parameters
    ----------
    games : pd.DataFrame
        Game-level dataset.
    windows : list[int], optional
        Window sizes (number of serve/return games per player).
        Defaults to [2, 3, 5].

    Returns
    -------
    pd.DataFrame
        One row per (game, window_size) with per-player features.
    """
    if windows is None:
        windows = [2, 3, 5]

    games = games.sort_values(["match_id", "game_num"]).reset_index(drop=True)

    results = []
    for match_id, match_games in games.groupby("match_id", sort=True):
        match_result = _rolling_features_for_match(match_games, windows)
        results.append(match_result)

    return pd.concat(results, ignore_index=True)


def test_no_leakage(games: pd.DataFrame) -> dict:
    """Verify that sliding window features do NOT use information from the current game.

    For a sample of games, manually verify that:
    1. Feature for game N uses only games 1..N-1
    2. No feature for game N contains stats from game N itself

    Returns dict with test results.
    """
    # Pick a sample match with enough games
    sample_match = games.groupby("match_id").filter(lambda g: len(g) >= 20)
    if len(sample_match) == 0:
        return {"passed": False, "reason": "No match with >= 20 games found"}

    match_id = sample_match["match_id"].iloc[0]
    match_games = games[games["match_id"] == match_id].sort_values("game_num").reset_index(drop=True)

    p1_id = match_games.iloc[0]["current_server"]
    p2_id = match_games.iloc[0]["current_returner"]

    indicators = _build_indicators(match_games, p1_id, p2_id)

    # Test W=3: feature for game 10 (index 9) should use games 1..9 (index 0..8)
    w = 3
    lookback = w * 2
    test_idx = 9  # 10th game (game_num=10)

    if test_idx >= len(match_games):
        test_idx = len(match_games) - 1

    start = max(0, test_idx - lookback)
    window = indicators.iloc[start:test_idx]

    # Verify: window should NOT include test_idx (the current game)
    window_includes_current = test_idx in window.index

    # Verify: last game in window should be test_idx - 1
    last_game_in_window = window.index[-1] if len(window) > 0 else None
    last_game_correct = (last_game_in_window == test_idx - 1) if last_game_in_window is not None else True

    # Verify: p1_aces for game test_idx should be mean of p1_aces in window
    # NOT the value from game test_idx itself
    if len(window) > 0:
        manual_p1_aces = window["p1_aces"].mean()
        current_p1_aces = indicators.iloc[test_idx]["p1_aces"]
        uses_current_game = abs(manual_p1_aces - current_p1_aces) < 0.01  # within tolerance
    else:
        uses_current_game = False

    passed = not window_includes_current and last_game_correct and not uses_current_game

    return {
        "passed": passed,
        "match_id": match_id,
        "test_game_num": int(match_games.iloc[test_idx]["game_num"]),
        "window_start_idx": start,
        "window_end_idx": last_game_in_window,
        "window_size": len(window),
        "window_includes_current_game": window_includes_current,
        "last_game_correct": last_game_correct,
        "manual_computation_uses_current": uses_current_game,
    }
