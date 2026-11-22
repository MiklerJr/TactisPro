"""Improve coverage of player stats (rank, age) using last-observation-carried-forward.

Problem:
- Sackmann data only goes up to 2024
- 502 matches from 2025-2026 (14% of dataset) have no rank/age data
- Current coverage: rank 83.8%, age 83.8%

Strategy:
- For each player, find their last available rank/age from any year
- Use that as estimate for missing years
- Apply only if last observation is within 3 years of match date
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config import MATCHES_DATASET, GAMES_DATASET, PROCESSED_DIR
from src.features.player_stats import normalize_name


def build_player_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a history of rank/age per player across all years.

    Returns DataFrame with columns: player_name_norm, year, rank, age
    """
    matches = matches.copy()
    matches["year"] = pd.to_datetime(matches["date"]).dt.year
    matches["p1_norm"] = matches["player_1"].apply(normalize_name)
    matches["p2_norm"] = matches["player_2"].apply(normalize_name)

    # Combine p1 and p2 stats
    p1_history = matches[["p1_norm", "year", "p1_rank", "p1_age"]].copy()
    p1_history.columns = ["player_norm", "year", "rank", "age"]
    p1_history = p1_history.dropna(subset=["rank"])

    p2_history = matches[["p2_norm", "year", "p2_rank", "p2_age"]].copy()
    p2_history.columns = ["player_norm", "year", "rank", "age"]
    p2_history = p2_history.dropna(subset=["rank"])

    history = pd.concat([p1_history, p2_history], ignore_index=True)
    # Keep most recent rank per player-year
    history = history.sort_values(["player_norm", "year"])
    history = history.groupby(["player_norm", "year"]).first().reset_index()

    return history


def estimate_missing_stats(
    matches: pd.DataFrame,
    history: pd.DataFrame,
    max_years_back: int = 3,
) -> pd.DataFrame:
    """Estimate missing rank/age using last-observation-carried-forward.

    Parameters
    ----------
    matches : DataFrame with player_1, player_2, date, p1_rank, p2_rank, p1_age, p2_age
    history : DataFrame from build_player_history()
    max_years_back : int
        Max years between match and last observation (default 3)
    """
    matches = matches.copy()
    matches["year"] = pd.to_datetime(matches["date"]).dt.year
    matches["p1_norm"] = matches["player_1"].apply(normalize_name)
    matches["p2_norm"] = matches["player_2"].apply(normalize_name)

    # Build lookup: player_norm -> latest rank/age
    latest_obs = (
        history.sort_values("year")
        .groupby("player_norm")
        .last()
        .reset_index()
        [["player_norm", "year", "rank", "age"]]
    )
    latest_obs.columns = ["player_norm", "last_year", "last_rank", "last_age"]

    # Fill p1 stats
    matches = matches.merge(
        latest_obs.rename(columns={
            "player_norm": "p1_norm",
            "last_year": "p1_last_year",
            "last_rank": "p1_last_rank",
            "last_age": "p1_last_age",
        }),
        on="p1_norm",
        how="left",
    )
    matches["years_back_p1"] = matches["year"] - matches["p1_last_year"]

    mask_p1 = (
        matches["p1_rank"].isna()
        & matches["p1_last_rank"].notna()
        & (matches["years_back_p1"] <= max_years_back)
    )
    matches.loc[mask_p1, "p1_rank"] = matches.loc[mask_p1, "p1_last_rank"]
    matches.loc[mask_p1, "p1_age"] = matches.loc[mask_p1, "p1_last_age"]

    # Fill p2 stats
    matches = matches.merge(
        latest_obs.rename(columns={
            "player_norm": "p2_norm",
            "last_year": "p2_last_year",
            "last_rank": "p2_last_rank",
            "last_age": "p2_last_age",
        }),
        on="p2_norm",
        how="left",
    )
    matches["years_back_p2"] = matches["year"] - matches["p2_last_year"]

    mask_p2 = (
        matches["p2_rank"].isna()
        & matches["p2_last_rank"].notna()
        & (matches["years_back_p2"] <= max_years_back)
    )
    matches.loc[mask_p2, "p2_rank"] = matches.loc[mask_p2, "p2_last_rank"]
    matches.loc[mask_p2, "p2_age"] = matches.loc[mask_p2, "p2_last_age"]

    # Recalculate diffs
    matches["rank_diff"] = matches["p1_rank"] - matches["p2_rank"]
    matches["age_diff"] = matches["p1_age"] - matches["p2_age"]

    # Drop helper columns
    matches = matches.drop(columns=[
        "year", "p1_norm", "p2_norm",
        "p1_last_year", "p1_last_rank", "p1_last_age", "years_back_p1",
        "p2_last_year", "p2_last_rank", "p2_last_age", "years_back_p2",
    ], errors="ignore")

    return matches


def propagate_to_games(matches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Propagate updated p1/p2 stats from matches to games."""
    id_map = matches.set_index("match_id")[
        ["p1_rank", "p2_rank", "p1_age", "p2_age", "rank_diff", "age_diff"]
    ].to_dict("index")

    games = games.copy()
    for col in ["p1_rank", "p2_rank", "p1_age", "p2_age", "rank_diff", "age_diff"]:
        games[col] = games["match_id"].map(
            lambda mid, c=col: id_map.get(mid, {}).get(c, np.nan)
        )
    return games


def write_coverage_report(
    before_matches: pd.DataFrame,
    after_matches: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Write coverage improvement report."""
    if output_path is None:
        output_path = PROCESSED_DIR / "coverage_improvement.md"

    lines = [
        "# Mejora de Cobertura de Player Stats",
        "",
        f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Estrategia",
        "",
        "Para jugadores sin rank/age en Sackmann, se usa la ultima observacion",
        "disponible (LOCF - Last Observation Carried Forward), limitada a 3 anos.",
        "",
        "## Comparacion de cobertura",
        "",
        "| Campo | Antes | Despues | Mejora |",
        "|---|---|---|---|",
    ]

    for col in ["p1_rank", "p2_rank", "p1_age", "p2_age"]:
        before = before_matches[col].notna().mean() * 100
        after = after_matches[col].notna().mean() * 100
        delta = after - before
        lines.append(f"| {col} | {before:.1f}% | {after:.1f}% | +{delta:.1f}% |")

    # By year breakdown
    lines += ["", "## Cobertura por ano (despues)", ""]
    lines += ["| Ano | Matches | p1_rank cov |", "|---|---|---|"]
    after_matches["year"] = pd.to_datetime(after_matches["date"]).dt.year
    for y in sorted(after_matches["year"].unique()):
        ym = after_matches[after_matches["year"] == y]
        cov = ym["p1_rank"].notna().mean() * 100
        lines.append(f"| {y} | {len(ym)} | {cov:.1f}% |")

    lines += [
        "",
        "## Metodologia",
        "",
        "- **LOCF (Last Observation Carried Forward)**: usa la ultima rank/age conocida",
        "- **Ventana**: maximo 3 anos entre match y ultima observacion",
        "- **Justificacion**: jugadores activos mantienen rank similar ano a ano",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Reporte escrito en {output_path}")


if __name__ == "__main__":
    print("=== Coverage Improvement ===\n")

    # Load
    matches = pd.read_parquet(MATCHES_DATASET)
    games = pd.read_parquet(GAMES_DATASET)

    print(f"Before coverage:")
    for col in ["p1_rank", "p2_rank", "p1_age", "p2_age"]:
        cov = matches[col].notna().mean() * 100
        print(f"  {col}: {cov:.1f}%")

    # Build history and estimate
    history = build_player_history(matches)
    print(f"\nPlayer history: {len(history)} player-years")

    matches_filled = estimate_missing_stats(matches, history)

    print(f"\nAfter coverage:")
    for col in ["p1_rank", "p2_rank", "p1_age", "p2_age"]:
        cov = matches_filled[col].notna().mean() * 100
        print(f"  {col}: {cov:.1f}%")

    # Propagate to games
    games_filled = propagate_to_games(matches_filled, games)
    print(f"\nGames coverage:")
    for col in ["p1_rank", "p2_rank", "p1_age", "p2_age"]:
        cov = games_filled[col].notna().mean() * 100
        print(f"  {col}: {cov:.1f}%")

    # Save
    matches_filled.to_parquet(MATCHES_DATASET, index=False)
    games_filled.to_parquet(GAMES_DATASET, index=False)
    print(f"\nSaved updated {MATCHES_DATASET.name} and {GAMES_DATASET.name}")

    # Report
    write_coverage_report(matches, matches_filled)
