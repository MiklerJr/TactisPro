"""Estado de la reproducción de un partido histórico."""

import pandas as pd


def resolve_live_state(points: pd.DataFrame, live_point: int) -> dict:
    """Resuelve el estado del partido tras revelar `live_point` puntos.

    `Pts`, `Set1/2` y `Gm1/2` en el dataset punto a punto describen el estado
    ANTES de jugarse cada punto. Por tanto, el estado tras el ultimo punto
    revelado se obtiene de la fila siguiente (el punto que aun no se juega).

    Devuelve un diccionario con:
        state        : fila que representa el marcador actual
        last         : ultimo punto revelado
        game_closed  : True si el game en curso acaba de cerrarse
        current_game : numero de game en curso (o recien cerrado)
        completed    : games completamente revelados
    """
    total = len(points)
    live_point = max(1, min(int(live_point), total))
    last = points.iloc[live_point - 1]

    if live_point < total:
        nxt = points.iloc[live_point]
        game_closed = int(nxt["Gm#"]) != int(last["Gm#"])
        state = nxt
    else:
        game_closed = True
        state = last

    current_game = int(last["Gm#"])
    completed = current_game if game_closed else current_game - 1
    return {
        "state": state,
        "last": last,
        "game_closed": game_closed,
        "current_game": current_game,
        "completed": completed,
    }
