"""Motor principal de calculo de momentum (EWM + Fuzzy).

Genera `data/processed/momentum_series.parquet` con las series temporales
M_A(t), M_B(t) y N(t) para todos los partidos del dataset.

Pipeline:
1. Cargar features dinamicas (game_features_resolved.parquet).
2. Agrupar por partido y calcular pesos EWM (estratificado por ventana).
3. Calcular score agregado por jugador.
4. Aplicar membresia difusa para clasificar el estado.
5. Calcular momentum neto N(t) = M_A(t) - M_B(t).
6. Exponer utilidades de carga (`load_momentum_series`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from config import PROCESSED_DIR, GAME_FEATURES_RESOLVED_DATASET

from .fuzzy import fuzzy_membership, fuzzy_state_label


# Features SIMETRICOS por jugador (mismo set para M_A y M_B).
# Los features p2_serve_wr_W y p2_return_wr_W se derivan por complemento
# desde p1_* (ver funcion `_derive_p2_features`).
P1_POSITIVE = ["p1_serve_wr", "p1_return_wr", "p1_svr_pts_won", "p1_1st_pct"]
P1_NEGATIVE = ["p1_df"]

P2_POSITIVE = ["p2_serve_wr", "p2_return_wr", "p2_svr_pts_won", "p2_1st_pct"]
P2_NEGATIVE = ["p2_df"]


def _derive_p2_features(feats: pd.DataFrame, window: int) -> pd.DataFrame:
    """Deriva p2_serve_wr_W y p2_return_wr_W desde p1_* por complemento.

    Matematicamente:
      - p2_serve_wr_W = 1 - p1_return_wr_W  (en los mismos games: p2 sirve <=> p1 resta)
      - p2_return_wr_W = 1 - p1_serve_wr_W  (en los mismos games: p2 resta <=> p1 sirve)

    Esto permite que M_A y M_B usen el mismo set de features sin
    recalcular las ventanas deslizantes.
    """
    feats = feats.copy()
    feats[f"p2_serve_wr_W{window}"] = 1.0 - feats[f"p1_return_wr_W{window}"]
    feats[f"p2_return_wr_W{window}"] = 1.0 - feats[f"p1_serve_wr_W{window}"]
    return feats


def _get_feature_cols(window: int) -> dict:
    """Retorna listas de features por jugador para una ventana W."""
    return {
        "p1_pos": [f"{c}_W{window}" for c in P1_POSITIVE],
        "p1_neg": [f"{c}_W{window}" for c in P1_NEGATIVE],
        "p2_pos": [f"{c}_W{window}" for c in P2_POSITIVE],
        "p2_neg": [f"{c}_W{window}" for c in P2_NEGATIVE],
    }


def _compute_player_momentum(
    df: pd.DataFrame, pos_cols: list[str], neg_cols: list[str]
) -> pd.DataFrame:
    """Calcula el índice de cada juego con el prefijo disponible a su cierre."""
    values = df[pos_cols + neg_cols].to_numpy(dtype=float)
    scores = np.full(len(values), 0.5)
    for stop in range(2, len(values) + 1):
        prefix = values[:stop]
        valid = np.isfinite(prefix)
        lower = np.min(prefix, axis=0, where=valid, initial=np.inf)
        upper = np.max(prefix, axis=0, where=valid, initial=-np.inf)
        varying = upper > lower
        normalized = np.full_like(prefix, 0.5)
        np.divide(prefix - lower, upper - lower, out=normalized, where=varying & valid)
        normalized[~valid] = 0.0

        positive = normalized + 1e-9
        proportion = positive / positive.sum(axis=0)
        divergence = 1.0 + (proportion * np.log(proportion)).sum(axis=0) / np.log(stop)
        divergence = np.clip(divergence, 0.0, 1.0)
        weights = divergence / divergence.sum() if divergence.sum() > 1e-12 else np.full(
            len(divergence), 1.0 / len(divergence)
        )

        normalized[:, len(pos_cols):] = 1.0 - normalized[:, len(pos_cols):]
        raw = normalized @ weights
        low, high = raw.min(), raw.max()
        if high > low:
            scores[stop - 1] = (raw[-1] - low) / (high - low)

    out = pd.DataFrame({"score": scores})
    out[["mu_low", "mu_medium", "mu_high"]] = fuzzy_membership(scores)
    out["state"] = fuzzy_state_label(out[["mu_low", "mu_medium", "mu_high"]])
    return out


def compute_momentum_series(
    features_path: Path | None = None,
    output_path: Path | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Calcula la serie temporal de momentum para cada partido y jugador.

    Parameters
    ----------
    features_path : Path, optional
        Ruta al parquet de features. Default: GAME_FEATURES_RESOLVED_DATASET.
    output_path : Path, optional
        Donde escribir el parquet resultante.
    windows : list[int], optional
        Ventanas a procesar. Default: [5] (suave, estable).

    Returns
    -------
    pd.DataFrame con columnas:
        match_id, game_num, M_A, M_B, N (momentum neto), state_A, state_B
    """
    if features_path is None:
        features_path = GAME_FEATURES_RESOLVED_DATASET
    if output_path is None:
        output_path = PROCESSED_DIR / "momentum_series.parquet"
    if windows is None:
        windows = [5]

    feats = pd.read_parquet(features_path)

    # Solo ventana mas estable (W=5) para reducir ruido en la serie
    w = windows[0]
    sub = feats[feats["window_size"] == w].copy()
    sub = sub.sort_values(["match_id", "game_num"]).reset_index(drop=True)

    # Derivar features p2 faltantes por complemento desde p1_*
    sub = _derive_p2_features(sub, w)

    cols = _get_feature_cols(w)

    results = []

    for match_id, g in sub.groupby("match_id", sort=True):
        g = g.sort_values("game_num").reset_index(drop=True)

        df_p1 = g[cols["p1_pos"] + cols["p1_neg"]].copy()
        df_p2 = g[cols["p2_pos"] + cols["p2_neg"]].copy()

        m_a = _compute_player_momentum(df_p1, cols["p1_pos"], cols["p1_neg"])
        m_b = _compute_player_momentum(df_p2, cols["p2_pos"], cols["p2_neg"])

        out = pd.DataFrame(
            {
                "match_id": match_id,
                "game_num": g["game_num"].values,
                "M_A": m_a["score"].values,
                "M_B": m_b["score"].values,
                "N": m_a["score"].values - m_b["score"].values,
                "mu_A_low": m_a["mu_low"].values,
                "mu_A_medium": m_a["mu_medium"].values,
                "mu_A_high": m_a["mu_high"].values,
                "mu_B_low": m_b["mu_low"].values,
                "mu_B_medium": m_b["mu_medium"].values,
                "mu_B_high": m_b["mu_high"].values,
                "state_A": m_a["state"].values,
                "state_B": m_b["state"].values,
            }
        )
        results.append(out)

    series = pd.concat(results, ignore_index=True)
    series.to_parquet(output_path, index=False)
    print(f"[engine] Momentum series -> {output_path} ({len(series):,} filas, "
          f"{series['match_id'].nunique():,} partidos)")
    return series


def load_momentum_series(path: Path | None = None) -> pd.DataFrame:
    """Carga la serie de momentum desde disco."""
    if path is None:
        path = PROCESSED_DIR / "momentum_series.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    compute_momentum_series()