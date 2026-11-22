"""Comparison and player radars through the latest completed game."""

from base64 import b64encode
from html import escape
from math import cos, pi, sin

import pandas as pd
import streamlit as st

from . import palette


AXES = (
    ("Puntos", "Puntos ganados"),
    ("Winners", "Golpes ganadores sin aces"),
    ("Errores NF", "Errores no forzados"),
    ("1.er saque", "Primeros saques dentro"),
    ("Aces", "Aces"),
    ("Dobles faltas", "Dobles faltas"),
)
SERVE_AXES = ("1stIn", "isAce", "isDouble")
IndicatorCounts = list[tuple[int, int]]


def player_indicators(points: pd.DataFrame, player: str, opponent: str) -> IndicatorCounts:
    """Return the player's successful/recorded counts in radar-axis order."""
    server = points["current_server"]
    receiver = points["current_returner"]
    is_server = server.eq(player) & receiver.eq(opponent)
    is_receiver = server.eq(opponent) & receiver.eq(player)
    result = points["isSvrWinner"]
    known = (is_server | is_receiver) & result.isin([0, 1])
    won = (is_server & result.eq(1)) | (is_receiver & result.eq(0))
    lost = known & ~won
    counts = [(int((known & won).sum()), int(known.sum()))]
    for column, attributed in (("isRallyWinner", won), ("isUnforced", lost)):
        flag = points[column]
        valid = known & flag.isin([0, 1])
        counts.append((int((valid & attributed & flag.eq(1)).sum()), int(valid.sum())))
    for column in SERVE_AXES:
        flag = points[column]
        valid = is_server & flag.isin([0, 1])
        counts.append((int((valid & flag.eq(1)).sum()), int(valid.sum())))
    return counts


def _position(index: int, scale: float) -> tuple[float, float]:
    angle = index * pi / 3 - pi / 2
    return 130 + 54 * scale * cos(angle), 109 + 54 * scale * sin(angle)


def _xy(position: tuple[float, float]) -> str:
    return f"{position[0]:.2f},{position[1]:.2f}"


def _percent(count: int, total: int) -> str:
    if not total:
        return "—"
    return f"{100 * count / total:.1f}".replace(".", ",").removesuffix(",0") + " %"


def _radar(counts: IndicatorCounts, player: str, color: str, theme: palette.Palette) -> str:
    values = [_position(i, count / total) if total else None for i, (count, total) in enumerate(counts)]
    spiders = []
    for scale in (.25, .50, .75, 1):
        ring = " ".join(_xy(_position(i, scale)) for i in range(len(AXES)))
        spiders.append(f'<polygon points="{ring}" class="r8-radar-grid"/>')

    # Fixed positions leave room for two lines of labels at all six axes.
    labels = [(130, 21), (216, 58), (216, 152), (130, 182), (44, 152), (44, 58)]
    for i, ((label, full_label), (count, total)) in enumerate(zip(AXES, counts)):
        axis_x, axis_y = _position(i, 1)
        lx, ly = labels[i]
        missing = " · Sin datos válidos" if not total else ""
        tooltip = escape(f"{full_label}: {_percent(count, total)}" + (f" ({count} de {total})" if total else missing))
        spiders.append(f'<line x1="130" y1="109" x2="{axis_x:.2f}" y2="{axis_y:.2f}" class="r8-radar-grid"/>')
        spiders.append(
            f'<g class="r8-radar-axis"><title>{tooltip}</title>'
            f'<text x="{lx}" y="{ly}">{label}</text>'
            f'<text x="{lx}" y="{ly+14}" class="r8-radar-value">{_percent(count, total)}</text></g>'
        )

    if all(v is not None for v in values):
        polygon = " ".join(_xy(v) for v in values)
        spiders.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity=".16"/>')
    for i, first in enumerate(values):
        second = values[(i + 1) % len(values)]
        if first is not None and second is not None:
            spiders.append(f'<polyline points="{_xy(first)} {_xy(second)}" class="r8-radar-trace"/>')
        if first is not None:
            x, y = first
            spiders.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.25" class="r8-radar-dot"/>')

    status = "Puntos al saque registrados" if any(total for _, total in counts[3:]) else "Aún sin datos al saque"
    values_text = ", ".join(
        f"{label} {_percent(count, total)}" for (_, label), (count, total) in zip(AXES, counts)
    )
    svg = f"""
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 204">
        <style>
          .r8-radar-grid {{ fill:none; stroke:{theme.BORDER}; stroke-width:1; }}
          .r8-radar-axis text {{ text-anchor:middle; fill:{theme.TEXT}; font:11px system-ui,sans-serif; }}
          .r8-radar-axis .r8-radar-value {{ fill:{color}; font-weight:700; }}
          .r8-radar-trace {{ fill:none; stroke:{color}; stroke-width:2; }}
          .r8-radar-dot {{ fill:{color}; }}
        </style>
        {"".join(spiders)}
      </svg>
    """
    encoded = b64encode(svg.encode()).decode("ascii")
    return f"""
      <section class="r8-radar-card" style="--player-color:{color}">
        <strong class="r8-radar-name">{escape(player)}</strong>
        <small>{status}</small>
        <img src="data:image/svg+xml;base64,{encoded}" alt="{escape(f'{player}. {values_text}')}">
      </section>
    """


def _comparison_cell(index: int, count: int, total: int) -> str:
    if not total:
        return '<td aria-label="Sin datos">—</td>'
    value = f"{count:,}".replace(",", " ")
    if index == 3:
        value = f"{count / total:.0%}".replace("%", " %") + f" <small>({count}/{total})</small>"
    bar = ""
    if index in (0, 3):
        bar = f'<div class="r8-stat-bar" aria-hidden="true"><span style="width:{count / total:.2%}"></span></div>'
    return f'<td>{value}{bar}</td>'


def comparison_panel(game_num: int, p1: str, p2: str, a: IndicatorCounts, b: IndicatorCounts,
                     theme: palette.Palette) -> None:
    st.markdown(f"**Acumulado hasta el juego {game_num}**")
    rows = []
    for i, ((_, label), first, second) in enumerate(zip(AXES, a, b)):
        if i == 1:
            label = "Winners de rally"
        rows.append(
            f'<tr>{_comparison_cell(i, *first)}<th scope="row">{label}</th>'
            f'{_comparison_cell(i, *second)}</tr>'
        )
    st.html(f"""
      <style>
        .r8-stat-table {{
          width:100%; table-layout:fixed; border-collapse:collapse; text-align:center;
          background:{theme.SURFACE}; color:{theme.TEXT}; font:12px {palette.FONT};
          font-variant-numeric:tabular-nums;
        }}
        .r8-stat-table th,.r8-stat-table td {{ padding:5px 6px; border-bottom:1px solid {theme.BORDER}; }}
        .r8-stat-table thead th:first-child {{ color:{theme.PLAYER_A}; }}
        .r8-stat-table thead th:last-child {{ color:{theme.PLAYER_B}; }}
        .r8-stat-table tbody th,.r8-stat-table small {{ color:{theme.MUTED}; font-weight:500; }}
        .r8-stat-bar {{
          height:3px; margin:3px auto 0; max-width:92px; background:{theme.BORDER};
          display:flex; border-radius:4px; overflow:hidden;
        }}
        .r8-stat-bar span {{ background:{theme.PLAYER_B}; }}
        .r8-stat-table td:first-child .r8-stat-bar {{ justify-content:flex-end; }}
        .r8-stat-table td:first-child .r8-stat-bar span {{ background:{theme.PLAYER_A}; }}
      </style>
      <table class="r8-stat-table" aria-label="Comparación acumulada por jugador">
        <thead><tr><th scope="col">{escape(p1)}</th><th scope="col">Indicador</th><th scope="col">{escape(p2)}</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    """)
    st.caption("Datos del registro punto a punto del proyecto. Los winners de rally excluyen los aces.")


def _render_radars(panels: str, label: str, theme: palette.Palette) -> None:
    st.html(f"""
      <style>
        .r8-radar-wrap {{ container-type:inline-size; }}
        .r8-radars {{
          display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px;
          max-width:480px; margin:0 auto; font-family:{palette.FONT};
          font-variant-numeric:tabular-nums;
        }}
        .r8-radar-card {{
          min-width:0; text-align:center; padding:5px 2px 0; background:{theme.SURFACE};
          border:1px solid {theme.BORDER}; border-radius:12px; color:{theme.MUTED};
        }}
        .r8-radar-name {{ display:block; font-size:13px; color:var(--player-color); }}
        .r8-radar-card small {{ font-size:10px; }}
        .r8-radar-card img {{ display:block; width:100%; max-width:225px; margin:auto; }}
        @container(max-width:340px) {{ .r8-radars {{ grid-template-columns:1fr; }} }}
      </style>
      <div class="r8-radar-wrap">
        <div class="r8-radars" role="group" aria-label="{label}">{panels}</div>
      </div>
    """)
    st.caption(
        "Escala común de 0 a 100 %. Puntos, winners y errores se dividen por los puntos válidos "
        "acumulados. Los indicadores de saque usan solo los puntos al saque de cada jugador. "
        "Los winners excluyen aces. El guion indica que la métrica no está disponible."
    )


def radar_panel(game_num: int, p1: str, p2: str, a: IndicatorCounts, b: IndicatorCounts,
                theme: palette.Palette) -> None:
    label = f"Radares acumulados hasta el juego {game_num}" if game_num else "Radares por jugador"
    st.markdown(f"**{label}**")
    if not game_num:
        st.caption("Los indicadores aparecen al cerrar el primer juego.")
    panels = _radar(a, p1, theme.PLAYER_A, theme) + _radar(b, p2, theme.PLAYER_B, theme)
    _render_radars(panels, label, theme)
