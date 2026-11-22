"""Particiones cronológicas que mantienen juntos los juegos de cada partido."""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def date_folds(dates, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Separa fechas completas para evitar mezclar partidos de un mismo día."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    unique_dates = np.unique(dates)
    return [
        (np.flatnonzero(np.isin(dates, unique_dates[train])),
         np.flatnonzero(np.isin(dates, unique_dates[validation])))
        for train, validation in TimeSeriesSplit(n_splits=n_splits).split(unique_dates)
    ]
