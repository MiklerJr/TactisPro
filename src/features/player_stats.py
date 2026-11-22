"""Player stats integration module.

Matches player names from our dataset to:
- Sackmann ATP data: rank, age (historical, per match)
- Tennis Abstract yElo: ELO rating (current year)
- Player ID: unique numeric ID per player

Usage:
    from src.features.player_stats import load_player_elo, load_sackmann_data, add_player_stats

    elo_map = load_player_elo()
    sackmann = load_sackmann_data()
    matches = add_player_stats(matches, sackmann, elo_map)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config import DATA_DIR


EXTERNAL_DIR = DATA_DIR / "external"


def normalize_name(name: str) -> str:
    """Normalize player name for matching.

    Handles mismatches between Tennis Abstract and Sackmann naming.
    """
    if pd.isna(name):
        return ""
    name = name.strip()
    # Remove accents and special chars for matching
    replacements = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ü": "u", "ñ": "n", "Ñ": "N",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.lower().strip()


def load_sackmann_data(years: Optional[list[int]] = None) -> pd.DataFrame:
    """Load Sackmann ATP match data and extract rank/age per player per match.

    Returns DataFrame with columns:
        match_id, player_name, rank, age
    """
    if years is None:
        years = list(range(2010, 2025))

    frames = []
    for year in years:
        path = EXTERNAL_DIR / f"atp_matches_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["year"] = year

        # Extract winner info
        winners = df[["winner_name", "winner_rank", "winner_age", "year"]].copy()
        winners.columns = ["player_name", "rank", "age", "year"]
        winners["role"] = "winner"

        # Extract loser info
        losers = df[["loser_name", "loser_rank", "loser_age", "year"]].copy()
        losers.columns = ["player_name", "rank", "age", "year"]
        losers["role"] = "loser"

        frames.append(pd.concat([winners, losers], ignore_index=True))

    if not frames:
        raise FileNotFoundError("No Sackmann ATP files found")

    result = pd.concat(frames, ignore_index=True)
    result["player_name_norm"] = result["player_name"].apply(normalize_name)

    # Keep most recent rank/age per player per year
    result = result.dropna(subset=["rank"])
    result = result.sort_values(["player_name_norm", "year", "rank"])
    result = result.groupby(["player_name_norm", "year"]).first().reset_index()

    return result[["player_name", "player_name_norm", "year", "rank", "age"]]


def load_player_elo() -> dict[str, float]:
    """Load yElo ratings from Tennis Abstract CSV.

    Returns dict: normalized_name -> yelo_rating
    """
    elo_file = EXTERNAL_DIR / "atp_yelo_current.csv"
    if not elo_file.exists():
        return {}

    df = pd.read_csv(elo_file)
    elo_map = {}
    for _, row in df.iterrows():
        norm = normalize_name(row["player_name"])
        elo_map[norm] = row["yelo"]
    return elo_map


def build_player_id_map(players: pd.Series) -> dict[str, int]:
    """Build a stable player_id mapping from player names.

    Returns dict: normalized_name -> numeric_id
    """
    unique_players = players.unique()
    norm_players = [normalize_name(p) for p in unique_players]
    # Sort for deterministic ordering
    sorted_pairs = sorted(zip(norm_players, unique_players))
    return {norm: idx for idx, (norm, _) in enumerate(sorted_pairs, start=1)}


# Rank -> ELO lookup table (based on historical data analysis)
# Each tuple: (rank_lower, rank_upper, elo_estimated)
RANK_ELO_TABLE = [
    (1,   5,   1984),
    (5,   10,  1875),
    (10,  20,  1780),
    (20,  50,  1766),
    (50,  100, 1692),
    (100, 200, 1631),
    (200, 500, 1627),
    (500, 9999, 1676),
]


def estimate_elo_from_rank(rank: float) -> float:
    """Estimate ELO rating from ATP rank using historical lookup table.

    Parameters
    ----------
    rank : float
        ATP ranking of the player.

    Returns
    -------
    float
        Estimated ELO rating, or np.nan if rank is not available.
    """
    if pd.isna(rank) or rank <= 0:
        return np.nan
    for lo, hi, elo in RANK_ELO_TABLE:
        if lo <= rank < hi:
            return float(elo)
    return np.nan


def fill_missing_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ELO values using rank-based estimation.

    For players without yElo data (retired, inactive), estimates ELO
    from their ATP rank using a historical lookup table.

    Parameters
    ----------
    matches : DataFrame with p1_elo, p2_elo, p1_rank, p2_rank columns.

    Returns
    -------
    DataFrame with filled ELO values and recalculated elo_diff.
    """
    matches = matches.copy()

    # Fill player 1 ELO
    mask_p1 = matches["p1_elo"].isna() & matches["p1_rank"].notna()
    matches.loc[mask_p1, "p1_elo"] = matches.loc[mask_p1, "p1_rank"].apply(
        estimate_elo_from_rank
    )

    # Fill player 2 ELO
    mask_p2 = matches["p2_elo"].isna() & matches["p2_rank"].notna()
    matches.loc[mask_p2, "p2_elo"] = matches.loc[mask_p2, "p2_rank"].apply(
        estimate_elo_from_rank
    )

    # Recalculate elo_diff
    matches["elo_diff"] = matches["p1_elo"] - matches["p2_elo"]

    return matches


def add_player_stats(
    matches: pd.DataFrame,
    sackmann: Optional[pd.DataFrame] = None,
    elo_map: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Add player stats to matches DataFrame.

    Adds columns:
        p1_player_id, p2_player_id
        p1_rank, p2_rank, p1_age, p2_age
        rank_diff, age_diff
        p1_elo, p2_elo, elo_diff

    Parameters
    ----------
    matches : DataFrame with player_1, player_2 columns
    sackmann : DataFrame from load_sackmann_data()
    elo_map : dict from load_player_elo()
    """
    matches = matches.copy()

    # Build player IDs
    all_players = pd.concat([matches["player_1"], matches["player_2"]])
    player_id_map = build_player_id_map(all_players)

    # Assign player IDs
    matches["p1_player_id"] = matches["player_1"].apply(
        lambda x: player_id_map.get(normalize_name(x), 0)
    )
    matches["p2_player_id"] = matches["player_2"].apply(
        lambda x: player_id_map.get(normalize_name(x), 0)
    )

    # Add ELO
    if elo_map:
        matches["p1_elo"] = matches["player_1"].apply(
            lambda x: elo_map.get(normalize_name(x), np.nan)
        )
        matches["p2_elo"] = matches["player_2"].apply(
            lambda x: elo_map.get(normalize_name(x), np.nan)
        )
        matches["elo_diff"] = matches["p1_elo"] - matches["p2_elo"]
    else:
        matches["p1_elo"] = np.nan
        matches["p2_elo"] = np.nan
        matches["elo_diff"] = np.nan

    # Add rank/age from Sackmann (historical)
    if sackmann is not None and not sackmann.empty:
        # Extract year from date column
        if "date" in matches.columns:
            matches["_year"] = pd.to_datetime(matches["date"], errors="coerce").dt.year
        else:
            matches["_year"] = 2020  # fallback

        # Merge player 1 stats
        p1_stats = matches[["player_1", "_year"]].copy()
        p1_stats.columns = ["player_name", "year"]
        p1_stats["player_name_norm"] = p1_stats["player_name"].apply(normalize_name)
        p1_stats = p1_stats.merge(
            sackmann[["player_name_norm", "year", "rank", "age"]],
            on=["player_name_norm", "year"],
            how="left",
        )
        matches["p1_rank"] = p1_stats["rank"].values
        matches["p1_age"] = p1_stats["age"].values

        # Merge player 2 stats
        p2_stats = matches[["player_2", "_year"]].copy()
        p2_stats.columns = ["player_name", "year"]
        p2_stats["player_name_norm"] = p2_stats["player_name"].apply(normalize_name)
        p2_stats = p2_stats.merge(
            sackmann[["player_name_norm", "year", "rank", "age"]],
            on=["player_name_norm", "year"],
            how="left",
        )
        matches["p2_rank"] = p2_stats["rank"].values
        matches["p2_age"] = p2_stats["age"].values

        matches.drop(columns=["_year"], inplace=True)
    else:
        matches["p1_rank"] = np.nan
        matches["p2_rank"] = np.nan
        matches["p1_age"] = np.nan
        matches["p2_age"] = np.nan

    # Derived features
    matches["rank_diff"] = matches["p1_rank"] - matches["p2_rank"]
    matches["age_diff"] = matches["p1_age"] - matches["p2_age"]

    return matches
