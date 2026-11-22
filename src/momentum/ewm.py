"""Entropy Weight Method (EWM) para ponderacion objetiva de features.

Referencia:
- Li & Wang (2024) - Metodo EWM para indices compuestos en analitica deportiva.
- Yang & Zhang (2025) - Ponderacion entrpica en sistemas de momentum.

Algoritmo:
1. Normalizacion min-max de la matriz de decision.
2. Para cada feature j, calcular la proporcion p_ij = x_ij / sum_i(x_ij).
3. Entropia: e_j = -k * sum_i(p_ij * ln(p_ij)), con k = 1/ln(n).
4. Peso: w_j = (1 - e_j) / sum_j(1 - e_j).

Los pesos resultantes son objetivamente derivados de la variabilidad de los
datos, eliminando la subjetividad del Analytic Hierarchy Process (AHP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_minmax(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Normaliza features al rango [0, 1] usando min-max.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con las columnas a normalizar.
    cols : list[str]
        Nombres de columnas a normalizar.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con columnas normalizadas en el mismo nombre.
    """
    out = df.copy()
    for c in cols:
        v = out[c].astype(float)
        vmin = v.min(skipna=True)
        vmax = v.max(skipna=True)
        if vmax == vmin:
            out[c] = 0.5
        else:
            out[c] = (v - vmin) / (vmax - vmin)
    return out


def entropy_weights(df: pd.DataFrame, cols: list[str]) -> dict:
    """Calcula pesos EWM para un conjunto de features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las features (sin valores negativos).
    cols : list[str]
        Columnas a ponderar.

    Returns
    -------
    dict con:
        - 'weights': dict {feature: peso}
        - 'entropies': dict {feature: entropia}
        - 'divergence': dict {feature: 1 - entropia} (intensidad de variacion)
        - 'method': 'EWM'
    """
    sub = df[cols].copy()
    # Reemplazar NaN por 0 para calculo de proporcion
    sub = sub.fillna(0.0)
    # Shift a estrictamente positivo (requerido por ln)
    sub = sub + 1e-9

    n = len(sub)
    if n < 2:
        raise ValueError("Se requieren >= 2 filas para calcular entropia.")

    k = 1.0 / np.log(n)

    weights = {}
    entropies = {}
    divergences = {}

    for c in cols:
        col = sub[c].values
        total = col.sum()
        if total <= 0:
            weights[c] = 0.0
            entropies[c] = 1.0
            divergences[c] = 0.0
            continue

        p = col / total
        # Evitar log(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(p > 0, p * np.log(p), 0.0)
        e = -k * term.sum()
        e = float(np.clip(e, 0.0, 1.0))
        entropies[c] = e
        divergences[c] = 1.0 - e

    total_div = sum(divergences.values())
    if total_div == 0:
        # Caso degenerado: todas las features constantes
        for c in cols:
            weights[c] = 1.0 / len(cols)
    else:
        for c in cols:
            weights[c] = divergences[c] / total_div

    return {
        "weights": weights,
        "entropies": entropies,
        "divergence": divergences,
        "method": "EWM",
    }


def aggregate_score(df: pd.DataFrame, cols: list[str], weights: dict) -> np.ndarray:
    """Calcula score agregado como suma ponderada de features normalizadas.

    Parameters
    ----------
    df : pd.DataFrame normalizado [0, 1].
    cols : list[str]
        Features incluidas en el score.
    weights : dict
        Pesos por feature (deben sumar 1).

    Returns
    -------
    np.ndarray con score por fila.
    """
    sub = df[cols].copy().fillna(0.0)
    score = np.zeros(len(sub))
    for c in cols:
        score += sub[c].values * weights[c]
    return score