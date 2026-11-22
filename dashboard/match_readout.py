"""Compact separation of the observed game and the model's next-game signals."""

from html import escape

import pandas as pd
import streamlit as st

from .explanation import forecast_signals
from .game_panel import player_indicators
from .palette import FONT, Palette


def _number(value: float, *, signed: bool = False) -> str:
    return format(value, "+.2f" if signed else ".2f").replace(".", ",")


def _observed(history, points, p1, p2, theme):
    game = int(history.iloc[-1]["game_num"])
    net = float(history.iloc[-1]["N"])
    leader = p1 if net > 0 else p2
    lead = f"Ventaja de {leader}" if net else "Índice equilibrado"
    movement = "Primer registro"
    delta = ""
    if len(history) > 1:
        change = net - float(history.iloc[-2]["N"])
        movement = "Sin cambio apreciable"
        if abs(change) >= 0.005:
            movement = f"{p1 if change > 0 else p2} gana terreno"
            delta = f'<strong>{_number(abs(change), signed=True)}</strong>'
    counts_a = player_indicators(points, p1, p2)
    counts_b = player_indicators(points, p2, p1)
    rows = []
    for i, label in enumerate(("Puntos ganados", "Winners de rally", "Errores no forzados")):
        first, second = (str(c[i][0]) if c[i][1] else "—" for c in (counts_a, counts_b))
        rows.append(f'<tr><td>{first}</td><th scope="row">{label}</th><td>{second}</td></tr>')
    serve_a, serve_b = (f"{c[3][0] / c[3][1]:.0%}".replace("%", " %") if c[3][1] else "—"
                        for c in (counts_a, counts_b))
    rows.append(f'<tr><td>{serve_a}</td><th scope="row">Primer saque dentro</th><td>{serve_b}</td></tr>')
    return f"""
      <section class="r8-read-card">
        <small>Observado · Juego {game}</small>
        <div class="r8-read-line"><strong>{escape(lead)}</strong><span>Índice {_number(net, signed=True)}</span></div>
        <div class="r8-read-line"><span>{escape(movement)}</span>{delta}</div>
        <table aria-label="Estadísticas del juego {game}">
          <thead><tr><th scope="col" style="color:{theme.PLAYER_A}">{escape(p1)}</th>
            <th scope="col">En este juego</th>
            <th scope="col" style="color:{theme.PLAYER_B}">{escape(p2)}</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
      </section>
    """


def _forecast(response, p1, p2):
    if response is None:
        return ""
    prediction = response["prediction"]
    probability = float(prediction["prob_up"])
    increasing = probability >= 0.5
    direction = f"{p1} gana terreno" if increasing else f"{p1} no gana terreno"
    probability_text = f"{(probability if increasing else 1 - probability):.1%}".replace(".", ",").replace("%", " %")
    contributions = pd.Series(response["contributions"])
    signals = forecast_signals(contributions, p1, p2, increasing)
    if signals:
        items = "".join(f"<li>{escape(signal)}</li>" for signal in signals)
        signal_block = (
            '<div class="r8-read-signals" title="Factores del pronóstico según el modelo">'
            f'<small>Señales del modelo</small><ul>{items}</ul></div>'
        )
    else:
        signal_block = '<small class="r8-read-signals">El modelo no destaca una señal clara.</small>'
    horizon = prediction["games_to_crossing"]
    crossing = ""
    if pd.notna(horizon):
        crossing = (f'<span title="Estimación condicionada a que ocurra otro cruce">'
                    f'Posible cruce en <b>{horizon:.1f}'.replace(".", ",") + ' juegos</b></span>')
    projected = _number(prediction["N_pred_next"], signed=True)
    return f"""
      <section class="r8-read-card r8-read-forecast">
        <small>PRONÓSTICO · JUEGO {int(prediction["game_num"]) + 1}</small>
        <div class="r8-read-line"><strong>{escape(direction)}</strong>
          <strong class="r8-read-prob" aria-label="Probabilidad estimada {probability_text}">{probability_text}</strong></div>
        <div class="r8-read-line r8-read-numbers"><span>Índice previsto · R4 <b>{projected}</b></span>{crossing}</div>
        {signal_block}
      </section>
    """


def match_readout(history, points, response, crossings, p1: str, p2: str, theme: Palette) -> None:
    if history.empty:
        st.caption("La lectura aparece al cerrar el primer juego.")
        return
    alert = ""
    if not crossings.empty:
        crossing = crossings.iloc[-1]
        player = p1 if crossing["direction"] > 0 else p2
        game = int(crossing["game_num"])
        alert = (
            f'<div class="r8-read-alert" role="status">↔ Último cruce · Juego {game} · '
            f'A favor de {escape(player)}</div>'
        )
    st.html(f"""
      <style>
        .r8-read-stack {{ display:grid; gap:6px; color:{theme.TEXT}; font:12px {FONT}; }}
        .r8-read-card {{
          display:grid; gap:5px; padding:8px 10px; background:{theme.SURFACE};
          border:1px solid {theme.BORDER}; border-radius:10px;
        }}
        .r8-read-card small {{ color:{theme.MUTED}; font-size:10px; }}
        .r8-read-line {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:baseline; gap:4px 8px; }}
        .r8-read-line strong {{ font-size:13px; }}
        .r8-read-line span {{ color:{theme.MUTED}; }}
        .r8-read-card table {{ width:100%; table-layout:fixed; text-align:center; border-collapse:collapse; font-size:11px; }}
        .r8-read-card table th,.r8-read-card table td {{ padding:3px 2px; border-top:1px solid {theme.BORDER}; }}
        .r8-read-card tbody th {{ color:{theme.MUTED}; font-weight:500; }}
        .r8-read-prob {{ color:{theme.FORECAST}; white-space:nowrap; }}
        .r8-read-numbers {{ font-size:10px; }}
        .r8-read-signals {{ padding-top:4px; border-top:1px solid {theme.BORDER}; }}
        .r8-read-signals ul {{ display:flex; flex-wrap:wrap; gap:4px; list-style:none; padding:0; margin:3px 0 0; }}
        .r8-read-signals li {{
          border:1px solid {theme.BORDER}; border-radius:5px; padding:2px 5px; color:{theme.TEXT}; font-size:10px;
        }}
        .r8-read-alert {{ padding:4px 7px; border-left:2px solid {theme.FORECAST}; color:{theme.FORECAST}; font-size:11px; }}
      </style>
      <div class="r8-read-stack">
        {_observed(history, points, p1, p2, theme)}
        {_forecast(response, p1, p2)}
        {alert}
      </div>
    """)
