"""R8 - Dashboard funcional: simulacion de partido en vivo.

Ejecutar desde la raiz del proyecto:
    streamlit run dashboard/app.py

La aplicacion simula la transmision de un partido en vivo. En un escenario real,
camaras de vision por computadora generarian los datos punto a punto; en esta
prueba de concepto los puntos se reproducen desde el dataset historico y llegan
uno a uno. El marcador se actualiza con cada punto y, al cerrar cada game, el
sistema entra en pausa para mostrar la prediccion (R4/R5) y el historico.

Principio metodologico: R4 y R5 fueron entrenados con granularidad de game, por
lo que la prediccion se actualiza al cerrar cada game, nunca con informacion
parcial del game en curso.
"""
from html import escape
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import palette
from config import CLEAN_DATASET, GAMES_DATASET, MATCHES_DATASET, MOMENTUM_DATASET
from dashboard.match_readout import match_readout
from dashboard.backend_client import fetch_game, fetch_metrics
from dashboard.game_panel import comparison_panel, player_indicators, radar_panel
from dashboard.replay import resolve_live_state
from src.momentum.crossings import momentum_crossings
from dashboard.momentum_chart import momentum_chart


st.set_page_config(
    page_title="TactisPro | R8",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MIN_GAMES = 12          # partidos con al menos este numero de games
MAX_LISTED_MATCHES = 300  # limite del selector para mantener la interfaz fluida
TRAIN_CUTOFF_YEAR = 2023


def dashboard_heading(theme: palette.Palette, light: bool) -> None:
    st.html("""
        <style>
          .stMainBlockContainer { padding: 3.5rem 2.25rem 1rem; max-width: 1320px; }
          .stMainBlockContainer .stVerticalBlock { gap: .48rem; }
          .stMainBlockContainer p, [data-testid="stSidebar"] p { font-size: .83rem; }
          .stMainBlockContainer [data-testid="stCaptionContainer"] p {
            font-size: .76rem; line-height: 1.45;
          }
          .stMainBlockContainer .tactis-title {
            font-family: Georgia, serif; font-size: clamp(1.65rem, 2.5vw, 2.25rem);
            font-weight: 650; line-height: 1.16; letter-spacing: -.025em;
          }
          .stMainBlockContainer button { min-height: 2rem; }
          @media (max-width: 680px) {
            .stMainBlockContainer { padding-inline: 1rem; }
          }
          .stMainBlockContainer h1, .stMainBlockContainer h2, .stMainBlockContainer h3 {
            padding: .2rem 0;
          }
        </style>
    """)
    if light:
        st.html(f"""
          <style>
            :root {{
              color-scheme:light;
              --tactis-bg:{theme.BACKGROUND}; --tactis-surface:{theme.SURFACE};
              --tactis-text:{theme.TEXT}; --tactis-muted:{theme.MUTED}; --tactis-border:{theme.BORDER};
            }}
            [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"],
            [data-testid="stBottom"], [data-testid="stApp"] {{
              background:var(--tactis-bg) !important; color:var(--tactis-text);
            }}
            [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
              background:var(--tactis-surface) !important; color:var(--tactis-text);
            }}
            [data-testid="stApp"] :is(h1,h2,h3,p,label,summary, [data-testid="stMetricValue"]),
            [data-testid="stHeader"] button {{ color:var(--tactis-text); }}
            [data-testid="stCaptionContainer"] p {{ color:var(--tactis-muted); }}
            [data-testid^="stBaseButton-"], [data-baseweb="select"] > div,
            [data-testid="stSelectbox"] button {{
              background:var(--tactis-surface) !important;
              color:var(--tactis-text) !important; border-color:var(--tactis-border) !important;
            }}
            [data-baseweb="select"] input, [data-baseweb="select"] svg,
            [data-testid="stSliderTickBar"] p, [data-testid="stSliderThumbValue"] p {{
              color:var(--tactis-text) !important;
            }}
            [data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="option"] {{
              background:var(--tactis-surface); color:var(--tactis-text);
            }}
            [data-baseweb="popover"] [role="option"]:hover {{ background:var(--tactis-bg); }}
            [data-testid="stExpander"] details {{ border-color:var(--tactis-border); }}
            [data-testid="stTooltipIcon"] svg {{ color:var(--tactis-muted); }}
            [data-testid="stExpander"] summary:hover {{ background:var(--tactis-surface); }}
            [data-testid="stAlertContainer"] {{ background:#fff1df !important; }}
          </style>
        """)
    st.html('<h1 class="tactis-title">TactisPro</h1>')


def _toggle_theme() -> None:
    st.session_state.tactis_light = not st.session_state.get("tactis_light", False)


# ---------------------------------------------------------------------------
# Carga de datos y modelos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga la serie de momentum, los games y la identidad fija por posicion.

    La identidad de P1 y P2 proviene de `matches.parquet`, donde `player_1` y
    `player_2` son constantes dentro del partido. No debe obtenerse desde
    `current_server` ni `current_returner`, porque esos roles alternan en cada
    game mientras que las series de momentum son fijas por posicion.
    """
    momentum = pd.read_parquet(MOMENTUM_DATASET)
    games = pd.read_parquet(GAMES_DATASET)
    matches = pd.read_parquet(MATCHES_DATASET)
    identity = matches[[
        "match_id", "player_1", "player_2", "tournament", "round",
        "p1_rank", "p2_rank", "p1_age", "p2_age", "p1_hand", "p2_hand",
    ]].copy()
    return (
        momentum.sort_values(["match_id", "game_num"]).reset_index(drop=True),
        games.sort_values(["match_id", "game_num"]).reset_index(drop=True),
        identity.drop_duplicates("match_id").reset_index(drop=True),
    )


@st.cache_data(show_spinner=False)
def load_match_points(match_id: str) -> pd.DataFrame:
    """Puntos del partido ordenados por game y punto (fuente: dataset_clean).

    Se conservan solo las columnas necesarias para el marcador en vivo. La
    columna `Pt` es el indice global del punto dentro del partido; `Gm#` es el
    numero global de game y coincide con `game_num` de games/momentum/modelos.
    """
    points = pd.read_parquet(CLEAN_DATASET)
    cols = [
        "match_id", "Gm#", "Pt", "Pts", "Svr", "Set1", "Set2", "Gm1", "Gm2",
        "current_server", "current_returner", "pts_srv_raw", "pts_ret_raw",
        "is_tiebreak", "deuce", "ad_srv", "ad_ret", "is_break_point",
        "isSvrWinner", "PtWinner", "1stIn", "isAce", "isDouble", "isRallyWinner", "isUnforced",
    ]
    match_points = points.loc[points["match_id"] == match_id, cols].copy()
    match_points = match_points.sort_values(["Gm#", "Pt"]).reset_index(drop=True)
    return match_points


@st.cache_data(show_spinner=False)
def identity_report(games: pd.DataFrame, identity: pd.DataFrame) -> str:
    """Comprueba que la identidad de P1/P2 sea estable dentro de cada partido.

    El tablero etiqueta a los jugadores con `player_1` y `player_2`, que son
    fijos. Esta verificacion confirma que `p1_player_id` no cambia a lo largo del
    partido y que P1 coincide con el servidor del primer game.
    """
    uniq = games.groupby("match_id")["p1_player_id"].nunique()
    stable_pct = float((uniq == 1).mean() * 100)

    first = games.sort_values(["match_id", "game_num"]).groupby("match_id").first()
    ref = identity.set_index("match_id")
    aligned = ref.reindex(first.index)
    first_is_p1 = float((first["current_server"] == aligned["player_1"]).mean() * 100)

    return (f"p1_player_id unico en {stable_pct:.1f}% de los partidos · "
            f"P1 coincide con el servidor del game 1 en {first_is_p1:.1f}%")


@st.cache_data(show_spinner=False)
def get_match_catalog(momentum: pd.DataFrame) -> pd.DataFrame:
    counts = momentum.groupby("match_id")["game_num"].count().rename("n_games")
    catalog = counts[counts >= MIN_GAMES].reset_index()
    catalog = catalog.sort_values("n_games", ascending=False)
    return catalog.head(MAX_LISTED_MATCHES).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Estado en vivo (punto -> game)
# ---------------------------------------------------------------------------
def _point_scores(state: pd.Series) -> tuple[str, str]:
    """Devuelve el marcador de punto de P1 y P2 en el estado actual.

    `pts_srv_raw`/`pts_ret_raw` vienen en orden servidor-receptor. Se reasignan
    a P1/P2 segun quien sirve (`Svr`: 1 = P1, 2 = P2) para mantener coherencia
    con la identidad fija por posicion.
    """
    svr = int(state.get("Svr", 1) or 1)
    srv_score = str(state.get("pts_srv_raw", "0"))
    ret_score = str(state.get("pts_ret_raw", "0"))
    if svr == 1:
        return srv_score, ret_score
    return ret_score, srv_score


def _situation_flags(state: pd.Series) -> list[str]:
    """Banderas de situacion del punto en curso."""
    flags: list[str] = []
    if bool(state.get("is_tiebreak", False)):
        flags.append("TIEBREAK")
    if bool(state.get("deuce", False)):
        flags.append("DEUCE")
    if bool(state.get("ad_srv", False)) or bool(state.get("ad_ret", False)):
        flags.append("VENTAJA")
    if bool(state.get("is_break_point", False)):
        flags.append("BREAK POINT")
    return flags


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
def _short_name(name: str) -> str:
    text = str(name).replace("_", " ").strip()
    if len(text) > 26:
        text = text[:24] + "…"
    return text


def _match_title(row: pd.Series) -> str:
    """Nombre legible del partido: torneo · ronda · jugadores.

    La fecha del `match_id` no se usa; se prefiere el torneo y la ronda, que
    aportan contexto tactico. Se omiten los campos vacios.
    """
    tournament = str(row.get("tournament") or "").replace("_", " ").strip()
    round_ = str(row.get("round") or "").strip()
    p1 = _short_name(row.get("player_1") or "Jugador 1")
    p2 = _short_name(row.get("player_2") or "Jugador 2")

    parts = [p for p in (tournament, round_) if p]
    prefix = " · ".join(parts)
    players = f"{p1} vs {p2}"
    return f"{prefix} · {players}" if prefix else players


def _hand_label(value: object) -> str:
    """Normaliza el codigo de mano ('R', 'R ', 'L') a texto legible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    code = str(value).strip().upper()
    return {"R": "diestro", "L": "zurdo"}.get(code, code)


def _meta_text(row: pd.Series, prefix: str) -> str:
    """Linea de metadatos del jugador: ranking, edad y mano.

    Ranking y edad tienen ~17 % de cobertura nula, por lo que se omiten los
    campos no disponibles en lugar de mostrar valores vacios o 'nan'.
    """
    parts: list[str] = []
    rank = row.get(f"{prefix}_rank")
    if rank is not None and not (isinstance(rank, float) and np.isnan(rank)):
        parts.append(f"Ranking #{int(rank)}")
    age = row.get(f"{prefix}_age")
    if age is not None and not (isinstance(age, float) and np.isnan(age)):
        parts.append(f"{int(age)} años")
    hand = _hand_label(row.get(f"{prefix}_hand"))
    if hand:
        parts.append(hand)
    return " · ".join(parts)


def scoreboard(live: dict, p1_name: str, p2_name: str, match_row: pd.Series,
               momentum_row: pd.Series | None, theme: palette.Palette) -> None:
    state = live["state"]
    server = str(live["last"].get("current_server", "") or "").strip()
    points = _point_scores(state)
    rows = []
    for i, (name, point, color) in enumerate(zip(
        (p1_name, p2_name), points, (theme.PLAYER_A, theme.PLAYER_B),
    ), start=1):
        serve = '<span style="opacity:.7;font-size:10px"> · Saca</span>' if name == server else ""
        meta = _meta_text(match_row, f"p{i}")
        if momentum_row is not None:
            momentum = float(momentum_row["M_A" if i == 1 else "M_B"])
            meta = " · ".join(filter(None, (meta, f"Momentum {momentum:.2f}".replace(".", ","))))
        detail = f'<small class="replay-player-meta">{escape(meta)}</small>' if meta else ""
        rows.append(
            f'<tr><td style="color:{color};text-align:left;font-weight:650">'
            f'{escape(_short_name(name))}{serve}{detail}</td>'
            f'<td>{int(state.get(f"Set{i}", 0))}</td>'
            f'<td>{int(state.get(f"Gm{i}", 0))}</td>'
            f'<td style="font-size:17px;font-weight:750;color:{color}">{escape(point)}</td></tr>'
        )
    flags = " · ".join(_situation_flags(state))
    st.html(f"""
        <section style="background:{theme.SURFACE};border:1px solid {theme.BORDER};border-radius:12px;padding:3px 11px">
          <style>
            .replay-score {{width:100%;font-size:12px;color:{theme.TEXT};border-collapse:collapse}}
            .replay-score th,.replay-score td {{text-align:right;padding:1px 8px}}
            .replay-score th {{color:{theme.MUTED};font-size:10px;font-weight:550}}
            .replay-player-meta {{display:block;color:{theme.MUTED};font-size:10px;font-weight:450;line-height:1.2;padding-bottom:2px}}
            .replay-score tbody tr + tr {{border-top:1px solid {theme.BORDER}}}
          </style>
          <table class="replay-score" aria-label="Marcador del partido">
            <thead><tr><th style="text-align:left">Jugador</th><th>Sets</th><th>Juegos</th><th>Punto</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
          {f'<small style="color:{theme.PLAYER_B}">{escape(flags)}</small>' if flags else ''}
        </section>
    """)


def historial(evaluated: pd.DataFrame) -> None:
    """Compara el error del valor previsto y el acierto de la dirección."""
    if evaluated.empty:
        st.info("El historial aparece cuando se cierra el juego al que apuntaba el pronóstico.")
        return

    mae = float(evaluated["error_abs"].mean())
    dir_acc = float(evaluated["dir_correct"].mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("Juegos evaluados", len(evaluated))
    c2.metric("Error medio del valor · R4", f"{mae:.4f}")
    c3.metric("Acierto de dirección · R5", f"{dir_acc:.1%}")

    table = evaluated[
        ["game_num", "N_t", "N_pred_next", "N_observed_next", "error_abs",
         "dir_pred", "dir_observed", "dir_correct"]
    ].copy()
    table["game_num"] += 1
    table.columns = ["Juego", "Índice previo", "Valor previsto R4", "Valor observado",
                     "Error absoluto R4", "Dirección R5", "Dirección observada", "Acierto R5"]
    table["Acierto R5"] = np.where(table["Acierto R5"], "Sí", "No")
    st.dataframe(
        table.round(3), width="stretch", hide_index=True, height=260,
    )
    st.caption(
        "R4 estima el valor que se traza en la curva. El error absoluto mide su distancia al valor observado. "
        "R5 estima si el índice aumenta respecto al juego anterior. «Acierto R5» compara esa dirección con la observada."
    )


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------
def main() -> None:
    light = st.session_state.get("tactis_light", False)
    theme = palette.LIGHT if light else palette.DARK
    dashboard_heading(theme, light)

    try:
        momentum, games, identity = load_datasets()
        metrics = fetch_metrics()
    except Exception as exc:
        st.error(f"No fue posible cargar los datos o conectar con el backend: {exc}")
        st.stop()

    catalog = get_match_catalog(momentum)

    with st.sidebar:
        st.button(
            "Modo oscuro" if light else "Modo claro",
            icon=":material/dark_mode:" if light else ":material/light_mode:",
            key="theme_toggle", on_click=_toggle_theme,
        )
        st.header("Selección del partido")
        only_oos = st.checkbox(
            "Solo partidos fuera de muestra (año ≥ 2023)",
            value=True,
            help="Los partidos anteriores a 2023 formaron parte del entrenamiento de R4/R5.",
        )

        game_dates = games.drop_duplicates("match_id").set_index("match_id")["date"]
        match_year = pd.to_datetime(game_dates.reindex(catalog["match_id"]).values).year
        catalog = catalog.assign(year=match_year.values)

        visible = catalog[catalog["year"] >= TRAIN_CUTOFF_YEAR] if only_oos else catalog

        identity_by_match = identity.set_index("match_id")
        options = {}
        for _, row in visible.iterrows():
            match_id = row["match_id"]
            meta_row = (identity_by_match.loc[match_id]
                        if match_id in identity_by_match.index else pd.Series(dtype=object))
            title = _match_title(meta_row) if len(meta_row) else match_id
            label = f"{title} · {int(row['n_games'])} juegos"
            options[label] = match_id

        if not options:
            st.warning("No hay partidos disponibles con el filtro aplicado.")
            st.stop()

        selected_label = st.selectbox(
            "Partido", list(options), index=0,
            help=f"Se listan los {len(options)} partidos con mayor número de games.",
        )
        selected_match = options[selected_label]

        st.divider()
        speed = st.slider("Segundos por punto", min_value=0.2, max_value=3.0,
                          value=0.8, step=0.1)

        with st.expander("Ajustes de la gráfica"):
            show_a = st.checkbox("Momentum del jugador A", value=False)
            show_b = st.checkbox("Momentum del jugador B", value=False)
            show_n = st.checkbox("Momentum neto", value=True)
            show_predictions = st.checkbox("Predicción en la curva", value=True)
            full_match_chart = st.checkbox("Mostrar todo el partido", value=False)
            smooth_window = st.select_slider("Suavizado", options=[1, 3, 5, 7], value=1)
        with st.expander("Validación de identidad"):
            st.caption(identity_report(games, identity))

    try:
        points = load_match_points(selected_match)
    except Exception as exc:
        st.error(f"No fue posible cargar el partido: {exc}")
        st.stop()

    if points.empty:
        st.error("El partido seleccionado no tiene datos punto a punto disponibles.")
        st.stop()

    match_momentum = momentum[momentum["match_id"] == selected_match].sort_values("game_num")
    total_points = len(points)
    total_games = int(points["Gm#"].max())
    if st.session_state.get("live_match") != selected_match:
        st.session_state.update(
            live_match=selected_match, live_point=1, replay_cursor=1,
            live_playing=False, game_forecasts={},
        )

    live = resolve_live_state(points, st.session_state["live_point"])
    completed = live["completed"]
    match_row = identity[identity["match_id"] == selected_match]
    match_row = match_row.iloc[0] if not match_row.empty else pd.Series(dtype=object)
    p1_name = str(match_row.get("player_1") or "Jugador 1")
    p2_name = str(match_row.get("player_2") or "Jugador 2")
    match_title = _match_title(match_row) if len(match_row) else selected_match
    timer_running = st.session_state["live_playing"]

    def restart() -> None:
        st.session_state.game_forecasts = {}
        seek(1)

    def seek(point: int) -> None:
        st.session_state.live_point = point
        st.session_state.live_playing = False

    def toggle_playback() -> None:
        state = st.session_state
        state.live_playing = not state.live_playing
        if state.live_playing and resolve_live_state(points, state.live_point)["game_closed"]:
            state.live_point = min(state.live_point + 1, total_points)
        state.last_tick = time.monotonic()

    @st.fragment(run_every=speed if timer_running else None)
    def replay_controls() -> None:
        state = st.session_state
        point = state.live_point
        if state.live_playing and time.monotonic() - state.last_tick >= speed * 0.9:
            point = min(point + 1, total_points)
            state.live_point = point
            state.last_tick = time.monotonic()
            if point >= total_points:
                state.live_playing = False

        current = resolve_live_state(points, point)
        if current["completed"] != completed or state.live_playing != timer_running:
            st.rerun()

        st.caption(f"{match_title} · Juego {current['current_game']} de {total_games}")
        score_col, replay_col = st.columns([1, 1.2])
        with score_col:
            visible_momentum = match_momentum.loc[match_momentum["game_num"].le(current["completed"])]
            last_momentum = None if visible_momentum.empty else visible_momentum.iloc[-1]
            scoreboard(current, p1_name, p2_name, match_row, last_momentum, theme)
        controls = replay_col.columns([1, 1.15, 1.15, 1.4])
        controls[0].button("Inicio", on_click=restart, width="stretch")
        controls[1].button("Anterior", on_click=seek, args=(max(1, point - 1),),
                           disabled=point <= 1, width="stretch")
        controls[2].button("Siguiente", on_click=seek, args=(min(total_points, point + 1),),
                           disabled=point >= total_points, width="stretch")
        label = "Pausa" if state.live_playing else "Reproducir"
        controls[3].button(label, key="replay_toggle", on_click=toggle_playback,
                           disabled=point >= total_points, width="stretch")
        state.replay_cursor = point
        replay_col.slider("Punto visible", 1, total_points, key="replay_cursor",
                           on_change=lambda: seek(state.replay_cursor))

    replay_controls()

    revealed = match_momentum[match_momentum["game_num"] <= completed]
    forecasts = st.session_state.setdefault("game_forecasts", {})
    if 0 < completed < total_games and completed not in forecasts:
        try:
            forecasts[completed] = fetch_game(selected_match, completed)
        except RuntimeError as exc:
            st.warning(f"Pronóstico del juego {completed} no disponible: {exc}")

    response = forecasts.get(completed)
    next_pred = pd.Series(response["prediction"]) if response is not None else None

    # Un pronóstico solo se puntúa después de observar el juego al que se refiere.
    observed = revealed.set_index("game_num")["N"]
    evaluated_rows = []
    for game, result in sorted(forecasts.items()):
        if game + 1 not in observed.index:
            continue
        row = result["prediction"].copy()
        row["N_observed_next"] = float(observed.loc[game + 1])
        row["error_abs"] = abs(row["N_pred_next"] - row["N_observed_next"])
        row["dir_observed"] = "Aumenta" if row["N_observed_next"] > row["N_t"] else "No aumenta"
        row["dir_correct"] = row["dir_pred"] == row["dir_observed"]
        evaluated_rows.append(row)
    evaluated = pd.DataFrame(evaluated_rows)
    crossings = momentum_crossings(observed).rename("direction").reset_index()

    last_game = int(revealed.iloc[-1]["game_num"]) if not revealed.empty else 0
    recorded = points.loc[points["Gm#"].le(last_game)]
    indicators_a = player_indicators(recorded, p1_name, p2_name)
    indicators_b = player_indicators(recorded, p2_name, p1_name)

    chart_col, summary_col = st.columns([1.85, 1])
    with chart_col:
        st.markdown("**Momentum y pronóstico**")
        st.caption(f"Por encima de cero favorece a {p1_name}. Por debajo de cero favorece a "
                   f"{p2_name}. La línea discontinua muestra el valor que R4 estima para el próximo juego.")
        visible = [column for column, show in [("N", show_n), ("M_A", show_a),
                                               ("M_B", show_b)] if show]
        momentum_chart(revealed, evaluated, next_pred, total_games, completed,
                       selected_match, visible, smooth_window, show_predictions, full_match_chart,
                       (_short_name(p1_name), _short_name(p2_name)), theme)
        legend = [
            (theme.OBSERVED, "Observado", show_n, "solid"),
            (theme.PLAYER_A, p1_name, show_a, "solid"),
            (theme.PLAYER_B, p2_name, show_b, "solid"),
            (theme.PAST, "Valores previstos por R4", show_predictions, "dashed"),
            (theme.FORECAST, "Próximo juego · R4", show_predictions, "dashed"),
        ]
        st.html(
            f'<div style="display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px;color:{theme.TEXT}">'
            + "".join(
                f'<span><i aria-hidden="true" style="display:inline-block;width:16px;'
                f'border-top:3px {style} {color};vertical-align:middle;margin-right:5px"></i>{escape(label)}</span>'
                for color, label, enabled, style in legend if enabled
            )
            + "</div>"
        )
    with summary_col:
        st.markdown("**Lectura del partido**")
        last_points = recorded.loc[recorded["Gm#"].eq(last_game)]
        match_readout(revealed, last_points, response, crossings, p1_name, p2_name, theme)
        radar_panel(last_game, p1_name, p2_name, indicators_a, indicators_b, theme)

    with chart_col:
        if last_game:
            comparison_panel(last_game, p1_name, p2_name, indicators_a, indicators_b, theme)
        st.markdown("**Análisis del partido**")
        with st.expander("Historial de pronósticos"):
            historial(evaluated)

        # --- Performance del modelo ---------------------------------------------
        with st.expander("Indicadores de los modelos (R4 y R5)"):
            r4 = metrics["r4"]
            r5 = metrics["r5"]
            reg = r4["metrics"]
            st.dataframe(
                pd.DataFrame([
                    [f"R4 · {r4['model']}", "N(t+1)", f"MAE {reg['test_mae']:.4f}",
                     f"RMSE {reg['test_rmse']:.4f}", f"R² {reg['test_r2']:.4f}"],
                    ["R5 · XGBoost GridSearch", "Dirección del cambio",
                     f"Precisión {r5['precision']:.4f}", f"F1 {r5['f1']:.4f}",
                     f"AUC-ROC {r5['roc_auc']:.4f}"],
                ], columns=["Modelo", "Salida", "Métrica 1", "Métrica 2", "Métrica 3"]),
                width="stretch", hide_index=True,
            )
            st.caption(
                "Metricas obtenidas en la evaluacion temporal documentada en R4 y R5 "
                "(train < 2023, test >= 2023)."
            )
            if int(catalog[catalog["match_id"] == selected_match]["year"].iloc[0]) < TRAIN_CUTOFF_YEAR:
                st.markdown(
                    f'<div style="background:#FEE2E2;border-left:2px solid #EF4444;padding:10px 14px;'
                    f'border-radius:6px;color:#7F1D1D;font-family:{palette.FONT};font-size:12px">'
                    "<b>Atencion.</b> Este partido es anterior a 2023 y forma parte del periodo de "
                    "entrenamiento de R4 y R5. Las predicciones son in-sample y pueden mostrar un "
                    "desempeno optimista. Para una evaluacion out-of-sample, active el filtro de "
                    "partidos de 2023 en adelante.</div>",
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
