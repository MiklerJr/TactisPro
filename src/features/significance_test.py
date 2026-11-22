"""Test statistical significance of feature correlations.

Uses Spearman rank correlation (robust to outliers) with p-value.
Thresholds:
  - Significance: p < 0.05
  - Minimum effect size: |r| > 0.01
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from config import PROCESSED_DIR, GAME_FEATURES_RESOLVED_DATASET


def test_feature_significance(
    features_path: Path | None = None,
    alpha: float = 0.05,
    min_effect: float = 0.01,
) -> dict:
    """Test which features have statistically significant correlation with target.

    Parameters
    ----------
    features_path : Path
        Path to game_features.parquet
    alpha : float
        Significance level (default 0.05)
    min_effect : float
        Minimum |correlation| to consider meaningful (default 0.01)

    Returns
    -------
    dict with test results per feature
    """
    if features_path is None:
        features_path = GAME_FEATURES_RESOLVED_DATASET

    # Load features and merge with target
    features = pd.read_parquet(features_path)
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")

    # Merge to get target
    df = features.merge(
        games[["match_id", "game_num", "p1_wins_next_game"]],
        on=["match_id", "game_num"],
        how="inner",
    )

    # Drop rows without target
    df = df.dropna(subset=["p1_wins_next_game"])
    df["p1_wins_next_game"] = df["p1_wins_next_game"].astype(int)

    # Feature columns (exclude metadata)
    meta_cols = ["match_id", "game_num", "window_size", "p1_wins_next_game"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    results = []

    for col in feature_cols:
        # Drop NaN for this feature
        valid = df[[col, "p1_wins_next_game"]].dropna()

        if len(valid) < 100:
            continue

        x = valid[col]
        y = valid["p1_wins_next_game"]

        # Skip if constant
        if x.std() == 0:
            continue

        # Spearman correlation (robust, non-parametric)
        r, p_value = stats.spearmanr(x, y)

        # Significance and effect size
        significant = p_value < alpha
        meaningful = abs(r) > min_effect

        results.append({
            "feature": col,
            "correlation": r,
            "p_value": p_value,
            "n": len(valid),
            "significant": significant,
            "meaningful": meaningful,
            "window_size": col.split("_W")[-1] if "_W" in col else "N/A",
        })

    # Summary
    df_results = pd.DataFrame(results)

    summary = {
        "total_features": len(df_results),
        "significant": df_results["significant"].sum(),
        "meaningful": df_results["meaningful"].sum(),
        "significant_and_meaningful": (
            df_results["significant"] & df_results["meaningful"]
        ).sum(),
        "alpha": alpha,
        "min_effect": min_effect,
    }

    # Top features by |correlation|
    top = df_results[df_results["meaningful"]].sort_values(
        "correlation", key=abs, ascending=False
    ).head(10)

    return {
        "summary": summary,
        "top_features": top.to_dict("records"),
        "all_results": df_results,
    }


def write_significance_report(output_path: Path | None = None) -> None:
    """Write significance test report to markdown."""
    if output_path is None:
        output_path = PROCESSED_DIR / "significance_report.md"

    result = test_feature_significance()

    lines = [
        "# Test de Significancia de Features",
        "",
        f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Alpha**: {result['summary']['alpha']}",
        f"**Efecto minimo**: |r| > {result['summary']['min_effect']}",
        "",
        "## Resumen",
        "",
        f"- Features analizadas: {result['summary']['total_features']}",
        f"- Significativas (p < 0.05): {result['summary']['significant']}",
        f"- Con efecto (|r| > 0.01): {result['summary']['meaningful']}",
        f"- Significativas y con efecto: {result['summary']['significant_and_meaningful']}",
        "",
        "## Top 10 Features",
        "",
        "| Feature | Correlacion | p-value | N | Significativa |",
        "|---|---|---|---|---|",
    ]

    for f in result["top_features"]:
        sig = "Si" if f["significant"] else "No"
        lines.append(
            f"| {f['feature']} | {f['correlation']:+.4f} | {f['p_value']:.2e} | {f['n']:,} | {sig} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Reporte escrito en {output_path}")


if __name__ == "__main__":
    write_significance_report()
