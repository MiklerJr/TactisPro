"""Detección de cruces observados del momentum neto."""

import numpy as np
import pandas as pd


def momentum_crossings(net: pd.Series) -> pd.Series:
    """Devuelve el signo nuevo al confirmar un cambio entre signos no nulos."""
    signs = np.sign(net)
    signs = signs[signs.notna() & signs.ne(0)]
    previous = signs.shift()
    return signs[previous.notna() & signs.ne(previous)]
