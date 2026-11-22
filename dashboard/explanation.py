"""Translate game-level evidence and local R5 contributions into tennis terms."""

import pandas as pd

_PLAYER_STATS = {
    "serve_wr": "Juegos ganados al saque",
    "return_wr": "Juegos ganados al resto",
    "aces": "Aces",
    "df": "Dobles faltas",
    "1st_pct": "Primeros saques dentro",
    "svr_pts_won": "Puntos ganados al saque",
}
_MATCH_STATS = {
    "winners_W5": "Golpes ganadores",
    "ue_W5": "Errores no forzados",
    "fe_W5": "Errores forzados",
    "rally_avg_W5": "Duración de los intercambios",
    "error_rate_W5": "Frecuencia de errores",
    "win_trend_W5": "Racha de juegos",
}
_MOMENTUM = {
    "M_A_t", "M_B_t", "N_t", "N_diff", "abs_N_t", "abs_N_diff", "N_sign",
    "N_rel_change", "N_volatility", "M_diff", "M_A_minus_prev", "M_B_minus_prev",
}


def forecast_signals(values: pd.Series, p1: str, p2: str, increasing: bool) -> list[str]:
    """Rank local signals supporting the forecast, grouping momentum derivatives."""
    grouped = {}
    for feature, contribution in values.dropna().items():
        label = _MATCH_STATS.get(feature)
        if feature in _MOMENTUM:
            label = "Evolución reciente del momentum"
        elif feature.startswith(("p1_", "p2_")) and feature.endswith("_W5"):
            stat = _PLAYER_STATS.get(feature[3:-3])
            if stat:
                player = p1 if feature.startswith("p1_") else p2
                label = f"{stat} · {player}"
        if label:
            grouped[label] = grouped.get(label, 0.0) + float(contribution)

    direction = 1 if increasing else -1
    ranked = sorted(grouped, key=lambda label: direction * grouped[label], reverse=True)
    return [label for label in ranked if direction * grouped[label] > 1e-8][:2]
