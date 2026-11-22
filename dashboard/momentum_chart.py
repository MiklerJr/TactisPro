"""Serialize only revealed observations and the next forecast for the chart."""

from pathlib import Path

import pandas as pd
import streamlit as st

from . import palette

_chart = st.components.v2.component(
    "momentum_replay",
    html='<svg class="court-chart" viewBox="0 0 760 390" role="img"></svg>',
    css=f"""
        :host {{ display: block; container-type: inline-size; }}
        .court-chart {{
            display: block;
            width: 100%;
            font-family: {palette.FONT};
            font-variant-numeric: tabular-nums;
        }}
        .court-chart text {{ fill: var(--chart-muted); }}
        .court-chart .tick {{ font-size: 14px; font-weight: 500; }}
        .court-chart .label {{ font-size: 15px; font-weight: 600; }}
        .court-chart .zone {{ font-size: 15px; font-weight: 600; }}
        .court-chart .message {{ font-size: 15px; font-weight: 500; }}
        .court-chart .grid {{ stroke: var(--chart-border); stroke-width: 1; }}
        .court-chart .trace {{ fill: none; stroke-linejoin: round; stroke-linecap: round; }}
        @container (max-width: 440px) {{
            .court-chart .tick {{ font-size: 18px; }}
            .court-chart .label, .court-chart .zone, .court-chart .message {{ font-size: 19px; }}
        }}
    """,
    js=Path(__file__).with_name("momentum_chart.js").read_text(encoding="utf-8"),
)


def momentum_chart(history: pd.DataFrame, predictions: pd.DataFrame, next_prediction,
                   total_games: int, completed: int, match_id: str,
                   visible: list[str], smoothing: int, show_predictions: bool, full_match: bool,
                   players: tuple[str, str], theme: palette.Palette) -> None:
    series = []
    styles = {
        "N": ("Momentum neto observado", theme.OBSERVED, 3.5),
        "M_A": (f"Momentum · {players[0]}", theme.PLAYER_A, 2.4),
        "M_B": (f"Momentum · {players[1]}", theme.PLAYER_B, 2.4),
    }
    games = history["game_num"].tolist()
    for column in visible:
        label, color, width = styles[column]
        values = history[column]
        series.append(dict(label=label, color=color, width=width,
                           points=list(zip(games, values.tolist()))))
        if smoothing > 1:
            average = values.rolling(smoothing, min_periods=1).mean()
            series.append(dict(label=f"{label} · media móvil", color=color, width=1,
                               dashed=True, points=list(zip(games, average.tolist()))))

    if show_predictions and not predictions.empty:
        series.insert(0, dict(label="Pronóstico anterior de R4", color=theme.PAST, width=1.6,
                              dashed=True, points=list(zip(
                                  (predictions["game_num"] + 1).tolist(),
                                  predictions["N_pred_next"].tolist(),
                              ))))

    if show_predictions and next_prediction is not None and not history.empty:
        series.append(dict(
            label=f"Pronóstico para el juego {completed + 1}", color=theme.FORECAST,
            width=2.7, dashed=True, forecast=True,
            points=[[completed, float(history.iloc[-1]["N"])],
                    [completed + 1, float(next_prediction["N_pred_next"])]],
        ))

    _chart(key="momentum_replay", data={
        "match": match_id, "completed": int(completed), "total": total_games,
        "series": series, "full": full_match, "players": players,
        "palette": {"background": theme.SURFACE, "a": theme.PLAYER_A,
                    "b": theme.PLAYER_B, "border": theme.BORDER, "muted": theme.MUTED},
    })
