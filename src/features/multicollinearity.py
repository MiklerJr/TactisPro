"""Resolve multicollinearity in game features.

Problem:
- p1_serve_wr_W and p2_return_wr_W have r = -1.0 (perfect inverse)
- p1_return_wr_W and p2_serve_wr_W have r = -1.0
- These duplicate information and inflate variance in linear models

Solution:
- Keep p1_* features (target = p1_wins_next_game, so p1 info is primary)
- Drop p2_serve_wr_W and p2_return_wr_W (perfectly redundant with p1)
- Keep other p2_* features (aces, df, 1st_pct, svr_pts_won, etc.)
  - These are NOT complements because they capture different playing styles

Per-game features (winners_W, ue_W, fe_W, rally_avg_W, error_rate_W,
win_trend_W) are emitted ONCE without p1/p2 prefix because the source
data does not carry player attribution (see sliding_window.py docstring).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from config import GAME_FEATURES_DATASET, PROCESSED_DIR


# Features to drop due to perfect multicollinearity (r = 1.0 or -1.0) with p1_*
# Win rates: p2_serve_wr_W = 1 - p1_serve_wr_W (complement)
REDUNDANT_FEATURES = [
    # Symmetric win rates (complementary)
    "p2_serve_wr_W2", "p2_return_wr_W2",
    "p2_serve_wr_W3", "p2_return_wr_W3",
    "p2_serve_wr_W5", "p2_return_wr_W5",
    # NOTE: p2_winners/p2_ue/p2_fe/p2_rally_avg no longer exist; per-game
    # features are emitted ONCE under winners_W / ue_W / fe_W / rally_avg_W.
    # error_rate and win_trend are also emitted ONCE under *_W.
]


def detect_multicollinearity(features: pd.DataFrame, threshold: float = 0.95) -> dict:
    """Detect highly correlated feature pairs.

    Returns dict with:
    - pairs: list of (feature_a, feature_b, correlation) where |r| > threshold
    - max_r: maximum |r| found
    """
    # Use W=3 window for detection
    f3 = features[features["window_size"] == 3]
    feat_cols = [c for c in f3.columns if c not in ["match_id", "game_num", "window_size"]]

    corr_matrix = f3[feat_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            if pd.notna(upper.loc[idx, col]) and upper.loc[idx, col] > threshold:
                pairs.append((idx, col, upper.loc[idx, col]))

    return {
        "pairs": pairs,
        "max_r": max(p[2] for p in pairs) if pairs else 0,
    }


def resolve_multicollinearity(features: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant features to reduce multicollinearity.

    Drops p2_* features that are perfectly redundant with p1_*:
    - p2_serve_wr_W, p2_return_wr_W: complementary to p1_* (r = -1.0)
    """
    before_cols = list(features.columns)
    features_clean = features.drop(columns=[c for c in REDUNDANT_FEATURES if c in features.columns])
    after_cols = list(features_clean.columns)
    dropped = set(before_cols) - set(after_cols)
    print(f"Dropped {len(dropped)} redundant features: {sorted(dropped)}")
    return features_clean


def verify_resolution(features: pd.DataFrame, threshold: float = 0.95) -> dict:
    """Verify no remaining multicollinearity above threshold."""
    f3 = features[features["window_size"] == 3]
    feat_cols = [c for c in f3.columns if c not in ["match_id", "game_num", "window_size"]]

    corr_matrix = f3[feat_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    remaining = []
    for col in upper.columns:
        for idx in upper.index:
            if pd.notna(upper.loc[idx, col]) and upper.loc[idx, col] > threshold:
                remaining.append((idx, col, upper.loc[idx, col]))

    return {
        "remaining_pairs": remaining,
        "max_r_remaining": max(p[2] for p in remaining) if remaining else 0,
    }


if __name__ == "__main__":
    print("=== Multicollinearity Resolution ===\n")

    # Load
    features = pd.read_parquet(GAME_FEATURES_DATASET)
    print(f"Loaded features: {features.shape}")

    # Detect before
    before = detect_multicollinearity(features)
    print(f"\nBefore: {len(before['pairs'])} pairs with |r| > 0.95")
    print(f"Max |r| before: {before['max_r']:.4f}")
    print("Sample pairs (top 5):")
    for a, b, r in sorted(before["pairs"], key=lambda x: -x[2])[:5]:
        print(f"  {a} <-> {b}: {r:.4f}")

    # Resolve
    features_resolved = resolve_multicollinearity(features)

    # Verify after
    after = verify_resolution(features_resolved)
    print(f"\nAfter: {len(after['remaining_pairs'])} pairs with |r| > 0.95")
    print(f"Max |r| after: {after['max_r_remaining']:.4f}")

    # Save
    output_path = PROCESSED_DIR / "game_features_resolved.parquet"
    features_resolved.to_parquet(output_path, index=False)
    print(f"\nSaved resolved features: {output_path}")
    print(f"Shape: {features_resolved.shape}")
