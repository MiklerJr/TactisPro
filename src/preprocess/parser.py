"""

Parser de los códigos del Match Charting Project (1st/2nd).

Convierte las cadenas de códigos punto a punto de Jeff Sackmann en variables
estructuradas, siguiendo la leyenda de la pestaña "Instructions" del
spreadsheet MatchChart 0.3.2 (proyecto datos/).

Referencia principal de códigos:
  - Dirección de saque: 4 = ancho, 5 = al cuerpo, 6 = T, 0 = desconocida.
  - Tipos de saque-falta: n/w/d/x/g/e/!
  - Golpes: f/b/r/s/v/z/o/p/u/y/l/m/h/i/j/k/t/q.
  - Dirección de golpe: 1/2/3 (relativo a un diestro).
  - Profundidad de devolución: 7/8/9.
  - Finalizadores: * = winner, # = error forzado, @ = error no forzado.
  - Posición: + = subida a la red, - = en la red, = = en la línea de fondo.
  - ; = pelota que roza la red, ^ = stop volley/drop volley, c = let.
  - Especiales: S/R (punto cedido al server/returner), P/Q (penalización),
    V (time violation), C (challenge incorrecto).
"""

from __future__ import annotations

SERVE_DIR_MAP = {"4": "wide", "5": "body", "6": "t", "0": "unknown"}

SERVE_FAULTS = {"n", "w", "d", "x", "g", "e", "!", "V"}

ERROR_TYPES = {
    "n": "net",
    "w": "wide",
    "d": "deep",
    "x": "wide_and_deep",
    "g": "foot_fault",
    "!": "shank",
    "e": "unknown",
    "V": "time_violation",
    "C": "challenge",
}

SHOT_TYPES = {
    "f": "forehand",
    "b": "backhand",
    "r": "forehand_slice",
    "s": "backhand_slice",
    "v": "forehand_volley",
    "z": "backhand_volley",
    "o": "overhead",
    "p": "backhand_overhead",
    "u": "forehand_drop",
    "y": "backhand_drop",
    "l": "forehand_lob",
    "m": "backhand_lob",
    "h": "forehand_half_volley",
    "i": "backhand_half_volley",
    "j": "forehand_swinging_volley",
    "k": "backhand_swinging_volley",
    "t": "trick_shot",
    "q": "unknown_shot",
}

SHOT_TYPE_SET = set(SHOT_TYPES)
SERVE_FAULT_SET = SERVE_FAULTS

ENDING_WINNER = "winner"
ENDING_FORCED = "forced"
ENDING_UNFORCED = "unforced"


def parse_cell(raw: str) -> dict:
    """Parsea una celda de servicio (1st o 2nd) y devuelve su estructura."""
    if raw is None:
        raw = ""
    raw = raw.strip()
    out = {
        "lets": 0,
        "serve_dir": None,
        "svp": False,
        "fault": False,
        "fault_type": None,
        "ace": False,
        "unret": False,
        "ending": None,
        "error_type": None,
        "shots": [],
        "rally_count": 0,
        "special": None,
        "stopped": False,
        "unparsed": "",
    }

    i = 0
    n = len(raw)
    while i < n and raw[i] == "c":
        out["lets"] += 1
        i += 1
    while i < n and raw[i] == " ":
        i += 1

    if i >= n:
        return out

    first = raw[i]

    if first in "SRPQ":
        out["special"] = first
        return out

    if first in "4560":
        out["serve_dir"] = SERVE_DIR_MAP.get(first, "unknown")
        i += 1
        if i < n and raw[i] == "+":
            out["svp"] = True
            i += 1
        if i >= n:
            out["rally_count"] = 0
            return out
        nxt = raw[i]
        if nxt == "*":
            out["ace"] = True
            out["ending"] = ENDING_WINNER
            out["rally_count"] = 0
            return out
        if nxt == "#":
            out["unret"] = True
            out["ending"] = ENDING_WINNER
            out["rally_count"] = 0
            return out
        if nxt in SERVE_FAULT_SET:
            out["fault"] = True
            out["fault_type"] = nxt
            return out
        shots, ending, err, stopped, rally_shots, unparsed = _parse_rally(raw[i:])
        out["shots"] = shots
        out["ending"] = ending
        out["error_type"] = err
        out["stopped"] = stopped
        out["rally_count"] = rally_shots
        out["unparsed"] = unparsed
        return out

    if first in SERVE_FAULT_SET:
        out["serve_dir"] = "unknown"
        out["fault"] = True
        out["fault_type"] = first
        return out

    shots, ending, err, stopped, rally_shots, unparsed = _parse_rally(raw[i:])
    out["shots"] = shots
    out["ending"] = ending
    out["error_type"] = err
    out["stopped"] = stopped
    out["rally_count"] = rally_shots
    out["unparsed"] = unparsed
    return out


def _parse_rally(text: str):
    """Parsea la secuencia de golpes de una celda.

    text: parte de la cadena a partir del primer golpe de devolución.
    Devuelve (shots, ending, error_type, stopped, rally_shots, unparsed).
    """
    shots = []
    current = None
    ending = None
    error_type = None
    stopped = False
    unparsed = ""

    for ch in text:
        if ch in SHOT_TYPE_SET:
            if current is not None:
                shots.append(current)
            current = {"type": ch, "dir": None, "depth": None,
                       "pos": None, "net_cord": False, "stop_volley": False,
                       "error": None, "ending_marker": None}
            continue

        if ch in "*@#":
            if ch == "*":
                ending = ENDING_WINNER
            elif ch == "@":
                ending = ENDING_UNFORCED
            else:
                ending = ENDING_FORCED
            if current is not None:
                current["ending_marker"] = ending
            break

        if ch in SERVE_FAULT_SET or ch == "C":
            if current is not None:
                current["error"] = ch
                error_type = ch
            if ch == "C":
                stopped = True
            continue

        if ch in "1230":
            if current is not None and current["dir"] is None:
                current["dir"] = ch
            continue
        if ch in "789":
            if current is not None and current["depth"] is None:
                current["depth"] = ch
            continue

        if ch in "+-=":
            if current is not None:
                current["pos"] = ch
            continue
        if ch == ";":
            if current is not None:
                current["net_cord"] = True
            continue
        if ch == "^":
            if current is not None:
                current["stop_volley"] = True
            continue

        unparsed += ch

    if current is not None:
        shots.append(current)

    rally_shots = len(shots)

    return shots, ending, error_type, stopped, rally_shots, unparsed


def _serve_outcome_effective(cell: dict) -> dict:
    """Resume una celda de servicio a un dict plano con métricas derivadas."""
    shots = cell.get("shots") or []

    winner_ending = cell.get("ending") == ENDING_WINNER
    error_ending = cell.get("ending") in (ENDING_FORCED, ENDING_UNFORCED)

    rally_count = cell.get("rally_count", 0) + 1  # incluye el saque
    if error_ending:
        rally_count = max(0, rally_count - 1)

    first_shot = shots[0] if shots else None

    return {
        "serve_dir": cell.get("serve_dir"),
        "svp": cell.get("svp", False),
        "serve_in": not cell.get("fault", True) and cell.get("serve_dir") is not None,
        "fault": cell.get("fault", False),
        "fault_type": cell.get("fault_type"),
        "is_ace": cell.get("ace", False),
        "is_unret": cell.get("unret", False),
        "is_double_fault": False,
        "ending": cell.get("ending"),
        "error_type": cell.get("error_type"),
        "stopped": cell.get("stopped", False),
        "rally_count": rally_count,
        "return_shot_type": (first_shot or {}).get("type"),
        "return_dir": (first_shot or {}).get("dir"),
        "return_depth": (first_shot or {}).get("depth"),
        "n_shots": len(shots),
        "net_approaches": sum(1 for s in shots if s.get("pos") == "+"),
    }


def parse_point(one: str, two: str) -> dict:
    """Combina 1st y 2nd y devuelve las métricas derivadas del punto."""
    p1 = parse_cell(one)
    p2 = parse_cell(two if two else "")

    special = p1.get("special") or p2.get("special")

    res = _serve_outcome_effective(p1)
    res2 = _serve_outcome_effective(p2)

    out = {
        "lets_1st": p1.get("lets", 0),
        "lets_2nd": p2.get("lets", 0),
        "special": special,
        "1stIn": False,
        "2ndIn": False,
        "1stSV": False,
        "2ndSV": False,
        "serve_dir": None,
        "serve_rank": None,
        "isAce": False,
        "isUnret": False,
        "isDouble": False,
        "isRallyWinner": False,
        "isForced": False,
        "isUnforced": False,
        "ending": None,
        "error_type": None,
        "rallyCount": 0,
        "return_shot_type": None,
        "return_dir": None,
        "return_depth": None,
        "net_approaches": 0,
        "n_shots": 0,
        "parse_issue": None,
    }

    out["1stIn"] = res["serve_in"]

    if res["fault"]:
        # Primer servicio fuera → depende del segundo.
        if res2["fault"] or (res2["serve_dir"] is None and not res2["is_ace"] and not res2["ending"]):
            # Doble falta (o segundo servicio sin registrar → se asume doble falta).
            out["isDouble"] = True
            out["2ndIn"] = False
            out["serve_dir"] = res2["serve_dir"] or "unknown"
            if res2["serve_dir"] is None and not res2["fault_type"]:
                out["parse_issue"] = "unresolved_double_fault"
        else:
            out["2ndIn"] = True
            out["serve_dir"] = res2["serve_dir"]
            out["serve_rank"] = 2
            out["2ndSV"] = res2["svp"]
            out["isAce"] = res2["is_ace"]
            out["isUnret"] = res2["is_unret"]
            out["ending"] = res2["ending"]
            out["error_type"] = res2["error_type"]
            out["rallyCount"] = res2["rally_count"]
            out["return_shot_type"] = res2["return_shot_type"]
            out["return_dir"] = res2["return_dir"]
            out["return_depth"] = res2["return_depth"]
            out["net_approaches"] = res2["net_approaches"]
            out["n_shots"] = res2["n_shots"]
            if res2["stopped"]:
                out["parse_issue"] = "stopped_challenge"
    else:
        # Primer servicio adentro (o celda sin dirección registrada).
        out["1stIn"] = res["serve_dir"] is not None
        out["serve_dir"] = res["serve_dir"]
        out["serve_rank"] = 1
        out["1stSV"] = res["svp"]
        out["isAce"] = res["is_ace"]
        out["isUnret"] = res["is_unret"]
        out["ending"] = res["ending"]
        out["error_type"] = res["error_type"]
        out["rallyCount"] = res["rally_count"]
        out["return_shot_type"] = res["return_shot_type"]
        out["return_dir"] = res["return_dir"]
        out["return_depth"] = res["return_depth"]
        out["net_approaches"] = res["net_approaches"]
        out["n_shots"] = res["n_shots"]
        if res["stopped"]:
            out["parse_issue"] = "stopped_challenge"

    if out["isAce"] or out["isUnret"]:
        out["isRallyWinner"] = False
    elif out["ending"] == ENDING_WINNER:
        out["isRallyWinner"] = True
    elif out["ending"] == ENDING_FORCED:
        out["isForced"] = True
    elif out["ending"] == ENDING_UNFORCED:
        out["isUnforced"] = True

    if special in ("S", "R", "Q", "P"):
        out["parse_issue"] = "special_point"
        # El finalizador no existe; se deduce con PtWinner en el pipeline.
    return out


def apply_parser_to_frame(df) -> None:
    """Aplica parse_point sobre un DataFrame con columnas 1st y 2nd (in-place)."""
    parsed = df.apply(
        lambda r: parse_point(r.get("1st"), r.get("2nd")), axis=1
    )
    return parsed