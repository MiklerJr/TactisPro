"""Membresía difusa para clasificación descriptiva del momentum.

Referencia:
- Li & Wang (2024); Yang & Zhang (2025) - Logica difusa aplicada a estados
  psicologicos transitorios en deporte de elite.

Modelo:
Tres funciones de membresia triangulares sobre el score agregado:
  - 'low':    [0.0, 0.0, 0.4]    (momentum desfavorable)
  - 'medium': [0.3, 0.5, 0.7]    (momentum neutro)
  - 'high':   [0.6, 1.0, 1.0]    (momentum favorable)

El grado de membresia indica que tan probablemente el estado del jugador
pertenece a cada categoria. La suma de membresias por fila es 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triangular(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Funcion de membresia triangular.

    Parameters
    ----------
    x : array de valores en [0, 1].
    a, b, c : vertices del triangulo (a <= b <= c).

    Returns
    -------
    array con grado de pertenencia en [0, 1].
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    # Pendiente ascendente: a -> b
    if b > a:
        mask = (x >= a) & (x <= b)
        out[mask] = (x[mask] - a) / (b - a)

    # Pendiente descendente: b -> c
    if c > b:
        mask = (x >= b) & (x <= c)
        out[mask] = (c - x[mask]) / (c - b)

    return np.clip(out, 0.0, 1.0)


def fuzzy_membership(score: np.ndarray) -> pd.DataFrame:
    """Calcula grados de pertenencia difusa para cada fila.

    Parameters
    ----------
    score : np.ndarray
        Score agregado en [0, 1] (tipicamente output de EWM).

    Returns
    -------
    pd.DataFrame con columnas ['mu_low', 'mu_medium', 'mu_high'].
    """
    score = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    mu_low = triangular(score, 0.0, 0.0, 0.4)
    mu_medium = triangular(score, 0.3, 0.5, 0.7)
    mu_high = triangular(score, 0.6, 1.0, 1.0)

    return pd.DataFrame(
        {
            "mu_low": mu_low,
            "mu_medium": mu_medium,
            "mu_high": mu_high,
        }
    )


def fuzzy_state_label(mu: pd.DataFrame) -> np.ndarray:
    """Etiqueta discreta del estado difuso dominante.

    Returns
    -------
    array de strings: 'low' | 'medium' | 'high' por fila.
    """
    arr = mu[["mu_low", "mu_medium", "mu_high"]].values
    idx = np.argmax(arr, axis=1)
    labels = np.array(["low", "medium", "high"])[idx]
    return labels
