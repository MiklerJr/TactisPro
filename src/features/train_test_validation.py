"""Train/test split validation for feature significance.

Problem:
- Correlation measured on full dataset may be inflated (data leakage in evaluation)
- Need to verify that significant features are also significant on unseen data

Approach:
- Temporal split: train = years < 2023, test = years >= 2023
- Compute Spearman correlation on each subset
- Check if top features remain significant on test set
- Report features that are significant on BOTH train and test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from config import PROCESSED_DIR


def temporal_split_validation(
    alpha: float = 0.05,
    min_effect: float = 0.01,
    split_year: int = 2023,
) -> dict:
    """Validate feature significance with temporal train/test split.

    Returns dict with:
    - train_results: DataFrame of correlations on train
    - test_results: DataFrame of correlations on test
    - stable_features: features significant on BOTH subsets
    - unstable_features: features significant only on one subset
    """
    # Load resolved features (after multicollinearity removal)
    features = pd.read_parquet(PROCESSED_DIR / "game_features_resolved.parquet")
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")

    # Merge with target and date
    df = features.merge(
        games[["match_id", "game_num", "p1_wins_next_game", "date"]],
        on=["match_id", "game_num"],
        how="inner",
    )
    df = df.dropna(subset=["p1_wins_next_game"])
    df["p1_wins_next_game"] = df["p1_wins_next_game"].astype(int)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    train = df[df["year"] < split_year]
    test = df[df["year"] >= split_year]

    print(f"Train: {len(train):,} rows (years < {split_year})")
    print(f"Test:  {len(test):,} rows (years >= {split_year})")

    # Feature columns (exclude metadata)
    meta_cols = ["match_id", "game_num", "window_size", "date", "year", "p1_wins_next_game"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    def compute_correlations(subset, label):
        results = []
        for col in feature_cols:
            valid = subset[[col, "p1_wins_next_game"]].dropna()
            if len(valid) < 100:
                continue
            if valid[col].std() == 0:
                continue
            r, p = stats.spearmanr(valid[col], valid["p1_wins_next_game"])
            results.append({
                "feature": col,
                "correlation": r,
                "p_value": p,
                "n": len(valid),
                "significant": p < alpha,
                "meaningful": abs(r) > min_effect,
            })
        df_res = pd.DataFrame(results)
        df_res["subset"] = label
        return df_res

    train_results = compute_correlations(train, "train")
    test_results = compute_correlations(test, "test")

    # Merge to compare
    comparison = train_results.merge(
        test_results,
        on="feature",
        suffixes=("_train", "_test"),
    )

    # Stable features: significant AND meaningful on BOTH
    comparison["stable"] = (
        comparison["significant_train"]
        & comparison["significant_test"]
        & comparison["meaningful_train"]
        & comparison["meaningful_test"]
    )

    # Same sign on both (consistency check)
    comparison["same_sign"] = np.sign(comparison["correlation_train"]) == np.sign(
        comparison["correlation_test"]
    )

    stable = comparison[comparison["stable"] & comparison["same_sign"]]
    unstable = comparison[comparison["significant_train"] != comparison["significant_test"]]

    return {
        "train": train_results,
        "test": test_results,
        "comparison": comparison,
        "stable_features": stable.sort_values("correlation_train", key=abs, ascending=False),
        "unstable_features": unstable,
        "split_year": split_year,
        "alpha": alpha,
        "min_effect": min_effect,
    }


def write_validation_report(output_path: Path | None = None) -> None:
    """Write train/test validation report."""
    if output_path is None:
        output_path = PROCESSED_DIR / "train_test_validation.md"

    result = temporal_split_validation()
    comp = result["comparison"]
    stable = result["stable_features"]

    lines = [
        "# Validacion Train/Test de Features",
        "",
        f"**Fecha**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Split year**: {result['split_year']}",
        f"**Train**: years < {result['split_year']}",
        f"**Test**: years >= {result['split_year']}",
        f"**Alpha**: {result['alpha']}",
        f"**Efecto minimo**: |r| > {result['min_effect']}",
        "",
        "## Resumen",
        "",
        f"- Features totales: {len(comp)}",
        f"- Significativas en train: {comp['significant_train'].sum()}",
        f"- Significativas en test: {comp['significant_test'].sum()}",
        f"- Estables (significativas en ambos + mismo signo): {len(stable)}",
        f"- Inestables (difieren entre train/test): {len(result['unstable_features'])}",
        "",
        "## Features Estables (significativas en train Y test)",
        "",
        "| Feature | r_train | p_train | r_test | p_test | N_train | N_test |",
        "|---|---|---|---|---|---|---|",
    ]

    for _, row in stable.head(20).iterrows():
        lines.append(
            f"| {row['feature']} | {row['correlation_train']:+.4f} | {row['p_value_train']:.2e} | "
            f"{row['correlation_test']:+.4f} | {row['p_value_test']:.2e} | "
            f"{row['n_train']:,} | {row['n_test']:,} |"
        )

    if len(stable) > 20:
        lines.append(f"\n*(Mostrando top 20 de {len(stable)} features estables)*")

    lines += [
        "",
        "## Features Inestables (significativas en uno solo)",
        "",
        "| Feature | r_train | p_train | r_test | p_test |",
        "|---|---|---|---|---|",
    ]

    for _, row in result["unstable_features"].head(20).iterrows():
        lines.append(
            f"| {row['feature']} | {row['correlation_train']:+.4f} | {row['p_value_train']:.2e} | "
            f"{row['correlation_test']:+.4f} | {row['p_value_test']:.2e} |"
        )

    lines += [
        "",
        "## Conclusiones",
        "",
        f"- **{len(stable)}** features son robustas (significativas y con efecto en train y test)",
        f"- **{len(result['unstable_features'])}** features son inestables (pueden ser ruido o spurias)",
        "- Features estables son candidatas para el modelo final",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Reporte escrito en {output_path}")
    print(f"\nFeatures estables: {len(stable)}")
    print(f"Features inestables: {len(result['unstable_features'])}")


if __name__ == "__main__":
    write_validation_report()
