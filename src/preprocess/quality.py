"""Reporte de calidad de datos (R1) y diccionario de datos."""

from __future__ import annotations

import pandas as pd

import json
from config import CLEAN_DATASET, REPORTS_DIR, PROCESSED_DIR

# Column classification: critical (no nulls allowed) vs optional
CRITICAL_COLS = [
    "match_id", "Pt", "Gm#", "Set1", "Set2", "Player 1", "Player 2",
    "Surface", "current_server", "current_returner",
    "isSvrWinner", "isAce", "isDouble", "isRallyWinner",
    "isUnforced", "isForced", "rallyCount", "net_approaches",
]

DATA_DICT = [
    # --- Critical columns ---
    ("match_id", "CRITICAL", "Identificador único del partido."),
    ("Pt", "CRITICAL", "Número de punto dentro del partido, desde 1."),
    ("Gm#", "CRITICAL", "Número de juego del partido, desde 1."),
    ("Set1", "CRITICAL", "Sets ganados por el Player 1 hasta el momento."),
    ("Set2", "CRITICAL", "Sets ganados por el Player 2 hasta el momento."),
    ("Player 1", "CRITICAL", "Nombre del jugador 1 (definido por el primer servicio)."),
    ("Player 2", "CRITICAL", "Nombre del jugador 2."),
    ("Surface", "CRITICAL", "Superficie: Hard, Clay, Grass, Unknown."),
    ("current_server", "CRITICAL", "Nombre del jugador que sirve en este punto."),
    ("current_returner", "CRITICAL", "Nombre del jugador que recibe en este punto."),
    ("isSvrWinner", "CRITICAL", "1 si ganó el punto el sacador."),
    ("isAce", "CRITICAL", "True si el punto terminó en ace."),
    ("isDouble", "CRITICAL", "True si el punto fue doble falta."),
    ("isRallyWinner", "CRITICAL", "True si el punto terminó en winner (excluye aces)."),
    ("isUnforced", "CRITICAL", "True si el punto terminó en error no forzado."),
    ("isForced", "CRITICAL", "True si el punto terminó en error forzado."),
    ("rallyCount", "CRITICAL", "Número de golpes del rally (incluye el saque)."),
    ("net_approaches", "CRITICAL", "Número de subidas a la red del sacador."),
    # --- Optional columns ---
    ("Pts", "OPTIONAL", "Marcador de puntos del juego actual."),
    ("TbSet", "OPTIONAL", "True si el set se decide por tiebreak."),
    ("Svr", "OPTIONAL", "1 si el Player 1 sirve, 2 si sirve el Player 2."),
    ("1st", "OPTIONAL", "Código del primer servicio."),
    ("2nd", "OPTIONAL", "Código del segundo servicio."),
    ("Notes", "OPTIONAL", "Notas del chartista."),
    ("PtWinner", "OPTIONAL", "1 si ganó el Player 1, 2 si ganó el Player 2."),
    ("lets_1st", "OPTIONAL", "Lets en el primer servicio."),
    ("lets_2nd", "OPTIONAL", "Lets en el segundo servicio."),
    ("special", "OPTIONAL", "Código especial: S/R (punto cedido), P/Q (penalización)."),
    ("1stIn", "OPTIONAL", "True si el primer servicio entró."),
    ("2ndIn", "OPTIONAL", "True si el segundo servicio entró."),
    ("1stSV", "OPTIONAL", "True si el primer servicio fue serve-and-volley."),
    ("2ndSV", "OPTIONAL", "True si el segundo servicio fue serve-and-volley."),
    ("serve_dir", "OPTIONAL", "Dirección del servicio: wide, body, t, unknown."),
    ("serve_rank", "OPTIONAL", "Servicio en juego: 1 o 2."),
    ("isUnret", "OPTIONAL", "True si el restador no devolvió el saque."),
    ("ending", "OPTIONAL", "Finalizador: winner, forced, unforced."),
    ("error_type", "OPTIONAL", "Tipo de error final."),
    ("return_shot_type", "OPTIONAL", "Tipo de golpe de la devolución."),
    ("return_dir", "OPTIONAL", "Dirección de la devolución."),
    ("return_depth", "OPTIONAL", "Profundidad de la devolución."),
    ("n_shots", "OPTIONAL", "Número de golpes registrados tras el saque."),
    ("parse_issue", "OPTIONAL", "Incidencia del parser: ok, special_point, etc."),
    ("ending_inferred", "OPTIONAL", "1 si el finalizador se infirió desde PtWinner."),
    ("pts_srv_raw", "OPTIONAL", "Componente del marcador del sacador."),
    ("pts_ret_raw", "OPTIONAL", "Componente del marcador del restador."),
    ("is_tiebreak", "OPTIONAL", "True si el punto pertenece a un tiebreak."),
    ("pt_srv", "OPTIONAL", "Índice de puntos del sacador (0/1/2/3/4+)."),
    ("pt_ret", "OPTIONAL", "Índice de puntos del restador."),
    ("deuce", "OPTIONAL", "True si el juego está en deuce."),
    ("ad_srv", "OPTIONAL", "True si el sacador tiene ventaja."),
    ("ad_ret", "OPTIONAL", "True si el restador tiene ventaja (break point)."),
    ("is_break_point", "OPTIONAL", "True si el punto es de quiebre."),
    ("current_server_hand", "OPTIONAL", "Mano del sacador (R/L)."),
    ("current_returner_hand", "OPTIONAL", "Mano del restador (R/L)."),
    ("p1_player_id", "OPTIONAL", "ID numérico único del Player 1."),
    ("p2_player_id", "OPTIONAL", "ID numérico único del Player 2."),
    ("isRtrWinner", "OPTIONAL", "1 si ganó el punto el restador."),
    ("Pl 1 hand", "OPTIONAL", "Mano del jugador 1."),
    ("Pl 2 hand", "OPTIONAL", "Mano del jugador 2."),
    ("Date", "OPTIONAL", "Fecha del partido (datetime)."),
    ("Tournament", "OPTIONAL", "Torneo."),
    ("Round", "OPTIONAL", "Ronda."),
    ("Court", "OPTIONAL", "Pista."),
    ("Umpire", "OPTIONAL", "Árbitro."),
    ("Best of", "OPTIONAL", "Partido a 3 o 5 sets."),
    ("Final TB?", "OPTIONAL", "Regla del set final para el tiebreak."),
    ("Charted by", "OPTIONAL", "Chartista que registró el partido."),
]


def write_data_dictionary():
    lines = [
        "# Diccionario de datos", "",
        "Dataset: `data/processed/dataset_clean.parquet` (punto a punto).", "",
        "## Columnas críticas (0% nulos requerido)",
        "",
        "| Campo | Descripción |",
        "|---|---|",
    ]
    for name, crit, desc in DATA_DICT:
        if crit == "CRITICAL":
            lines.append(f"| `{name}` | {desc} |")
    lines += [
        "",
        "## Columnas opcionales (aceptan nulos)",
        "",
        "| Campo | Descripción |",
        "|---|---|",
    ]
    for name, crit, desc in DATA_DICT:
        if crit == "OPTIONAL":
            lines.append(f"| `{name}` | {desc} |")
    (REPORTS_DIR / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")


def test_integrity(df: pd.DataFrame) -> list:
    """Run integrity checks on point-level data.

    Returns list of (test_name, passed, details) tuples.
    """
    tests = []

    # 1. Critical columns: zero nulls required
    critical_cols = [
        "match_id", "Pt", "Gm#", "Set1", "Set2", "Player 1", "Player 2",
        "Surface", "current_server", "current_returner",
        "isSvrWinner", "isAce", "isDouble", "isRallyWinner",
        "isUnforced", "isForced", "rallyCount", "net_approaches",
    ]
    for col in critical_cols:
        if col not in df.columns:
            tests.append((f"critical_col_exists:{col}", False, "Column missing"))
            continue
        n_null = df[col].isna().sum()
        passed = n_null == 0
        tests.append((
            f"critical_col_no_nulls:{col}",
            passed,
            f"{n_null} nulls" if n_null > 0 else "0 nulls",
        ))

    # 2. Gm# consistency: each game number should appear exactly once per match
    # and should form a contiguous sequence starting from 1
    all_seq_ok = True
    seq_details = ""
    for mid, mg in df.groupby("match_id"):
        game_nums = sorted(mg["Gm#"].unique())
        expected = list(range(1, len(game_nums) + 1))
        if game_nums != expected:
            all_seq_ok = False
            seq_details = f"match {mid}: expected {expected}, got {game_nums}"
            break
    tests.append(("game_num_contiguous", all_seq_ok, seq_details or "All matches have contiguous game numbers"))

    # 3. won_game consistency: verify via game-level aggregation
    # For most games, won_game should match the last point's isSvrWinner.
    # Tolerance: allow <1% mismatches (tiebreaks, incomplete data).
    games_path = PROCESSED_DIR / "games.parquet"
    if games_path.exists():
        games = pd.read_parquet(games_path)
        last_points = (
            df[["match_id", "Gm#", "Pt", "isSvrWinner"]]
            .sort_values("Pt").drop_duplicates(["match_id", "Gm#"], keep="last")
            .rename(columns={"Gm#": "game_num"})
        )
        matched = games[["match_id", "game_num", "won_game"]].merge(
            last_points, on=["match_id", "game_num"], validate="one_to_one"
        )
        total_games = len(matched)
        won_game_mismatch = int(matched["won_game"].ne(matched["isSvrWinner"]).sum())
        mismatch_pct = won_game_mismatch / total_games * 100 if total_games > 0 else 0
        tests.append((
            "won_game_consistent",
            mismatch_pct < 1.0,
            f"{won_game_mismatch}/{total_games} games mismatch ({mismatch_pct:.2f}%)",
        ))
    else:
        tests.append(("won_game_consistent", True, "games.parquet not found, skipped"))

    # 4. Date format: should be datetime
    if "Date" in df.columns:
        date_col = pd.to_datetime(df["Date"], errors="coerce")
        n_bad = date_col.isna().sum()
        tests.append((
            "date_parseable",
            n_bad < len(df) * 0.01,  # <1% failures acceptable
            f"{n_bad} unparseable dates ({n_bad/len(df)*100:.2f}%)",
        ))

    # 5. Surface: only Hard/Clay/Grass/Unknown allowed
    if "Surface" in df.columns:
        valid_surfaces = {"Hard", "Clay", "Grass", "Unknown"}
        invalid = df[~df["Surface"].isin(valid_surfaces)]["Surface"].unique()
        tests.append((
            "surface_valid",
            len(invalid) == 0,
            f"invalid surfaces: {list(invalid)}" if len(invalid) > 0 else "All valid",
        ))

    # 6. isAce + isDouble mutual exclusivity (can't be both on same point)
    if "isAce" in df.columns and "isDouble" in df.columns:
        both = (df["isAce"] & df["isDouble"]).sum()
        tests.append((
            "ace_and_double_exclusive",
            both == 0,
            f"{both} points with both ace and double fault",
        ))

    # 7. rallyCount: should be >= 0
    if "rallyCount" in df.columns:
        neg = (df["rallyCount"] < 0).sum()
        tests.append((
            "rally_count_non_negative",
            neg == 0,
            f"{neg} negative rally counts",
        ))

    return tests


def record_quality(df: pd.DataFrame) -> dict:
    """Cuenta registros que cumplen a la vez el esquema y las reglas de integridad."""
    required = df.reindex(columns=CRITICAL_COLS)
    complete = required.notna().all(axis=1)
    for column in ["match_id", "Player 1", "Player 2", "current_server", "current_returner"]:
        complete &= required[column].astype("string").str.strip().ne("").fillna(False)

    games = required[["match_id", "Gm#"]].dropna().drop_duplicates()
    groups = games.groupby("match_id")["Gm#"]
    valid_matches = groups.agg(["min", "max", "size"])
    valid = valid_matches["min"].eq(1) & valid_matches["max"].eq(valid_matches["size"])
    valid_sequence = required["match_id"].isin(valid_matches.index[valid])

    p1, p2 = required["Player 1"], required["Player 2"]
    server, returner = required["current_server"], required["current_returner"]
    rules = {
        "Clave partido–punto única": ~required.duplicated(["match_id", "Pt"], keep=False),
        "Punto y juego positivos, enteros": required["Pt"].ge(1) & required["Gm#"].ge(1)
        & required["Pt"].mod(1).eq(0) & required["Gm#"].mod(1).eq(0),
        "Juegos consecutivos dentro del partido": valid_sequence,
        "Roles de saque y devolución coherentes": (
            (server.eq(p1) & returner.eq(p2)) | (server.eq(p2) & returner.eq(p1))
        ) & p1.ne(p2),
        "Superficie admitida": required["Surface"].isin(["Hard", "Clay", "Grass", "Unknown"]),
        "Aces y dobles faltas excluyentes": ~(required["isAce"].eq(1) & required["isDouble"].eq(1)),
        "Contadores no negativos": required[["Set1", "Set2", "rallyCount", "net_approaches"]]
        .ge(0).all(axis=1),
        "Indicadores binarios": required[["isSvrWinner", "isAce", "isDouble", "isRallyWinner",
                                          "isUnforced", "isForced"]].isin([0, 1]).all(axis=1),
    }
    consistent = pd.DataFrame(rules, index=df.index).fillna(False).all(axis=1)
    return {
        "total": len(df),
        "complete_required": int(complete.sum()),
        "conforming": int((complete & consistent).sum()),
        "violations": {name: int((~mask.fillna(False)).sum()) for name, mask in rules.items()},
    }


def build_report(df: pd.DataFrame) -> None:
    n = len(df)
    n_matches = df["match_id"].nunique()

    out = ["# Reporte de Calidad de Datos (R1)", "",
           f"Puntos analizados: **{n:,}** en **{n_matches:,}** partidos.", ""]

    # 1. Estructura
    per_surface = df.groupby("Surface", dropna=False)["Pt"].count()
    out += ["## 1. Cobertura", "", "Distribución por superficie:", ""]
    for s, c in per_surface.sort_values(ascending=False).items():
        out.append(f"- {s}: {c:,} puntos ({c / n * 100:.1f}%)")
    out.append("")

    out += ["## 2. Incidencias del parser", ""]
    issues = df["parse_issue"].value_counts()
    ok_rate = issues.get("ok", 0) / n * 100
    out.append(f"Puntos parseados sin incidencias: **{ok_rate:.1f}%**")
    for k, v in issues.items():
        out.append(f"- `{k}`: {v:,} ({v / n * 100:.1f}%)")
    out.append("")

    out += ["## 3. Indicadores derivados clave", ""]
    ok = df[df["parse_issue"] == "ok"]
    stats = [
        ("% de puntos ganados por el sacador", ok["isSvrWinner"].mean() * 100),
        ("% de aces", ok["isAce"].mean() * 100),
        ("% de dobles faltas", ok["isDouble"].mean() * 100),
        ("% de errores no forzados", ok["isUnforced"].mean() * 100),
        ("% de errores forzados", ok["isForced"].mean() * 100),
        ("% de winners de rally", ok["isRallyWinner"].mean() * 100),
        ("% de puntos de quiebre", df["is_break_point"].mean() * 100),
        ("Golpes promedio por punto", ok["rallyCount"].mean()),
        ("1er servicio dentro (%)", ok["1stIn"].mean() * 100),
        ("% de puntos con finalizador inferido", df["ending_inferred"].mean() * 100),
    ]
    out += [f"- **{label}**: {val:.2f}{'%' if '%' in label or label.endswith('(%)') else ''}" for label, val in stats]
    out.append("")

    # 4. Nulos e inconsistencias
    out += ["## 4. Nulos e inconsistencias", ""]
    null_count = int(df.isna().sum().sum())
    out.append(f"Celdas nulas totales: **{null_count:,}** ({null_count / (n * df.shape[1]) * 100:.2f}%)")
    ncols_null = df.columns[df.isna().any()].tolist()
    out.append(f"Columnas con al menos un nulo ({len(ncols_null)}): `{'`, `'.join(ncols_null)}`")
    out.append("")

    quality = record_quality(df)
    out += [
        "### Cumplimiento por registro", "",
        f"- Campos obligatorios completos: **{quality['complete_required']:,}/{n:,}** "
        f"(**{quality['complete_required'] / n:.2%}**).",
        f"- Registros que también cumplen todas las reglas: **{quality['conforming']:,}/{n:,}** "
        f"(**{quality['conforming'] / n:.2%}**).",
        "- Los campos opcionales pueden contener nulos. Las incidencias de decodificación "
        "se presentan por separado en la sección 2.", "",
        "| Regla | Registros que incumplen |", "|---|---:|",
    ]
    out.extend(f"| {rule} | {count:,} |" for rule, count in quality["violations"].items())
    out.append("")
    (PROCESSED_DIR / "r1_record_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 5. Integrity tests
    out += ["## 5. Tests de integridad", ""]
    tests = test_integrity(df)
    all_passed = True
    for test_name, passed, details in tests:
        status = "PASS" if passed else "FAIL"
        out.append(f"- [{status}] `{test_name}`: {details}")
        if not passed:
            all_passed = False
    out.append("")
    out.append(f"**Resultado general: {'TODOS LOS TESTS PASARON' if all_passed else 'HAY TESTS QUE FALLARON'}**")
    out.append("")

    out += ["## 6. Notas metodológicas", ""]
    try:
        meta = json.loads((PROCESSED_DIR / "metadata.json").read_text())
        out += [
            f"- Se eliminaron **{meta['duplicates_removed']:,}** filas duplicadas de ",
            "  `(match_id, Pt)` (artefactos de puntos repetidos por el chartista).", ""]
    except FileNotFoundError:
        out += [""]
    out += [
            "- Los CSV de puntos de Sackmann **no** traen las columnas derivadas "
            "(isAce, rallyCount, etc.); estas se reconstruyeron parseando `1st`/`2nd` "
            "con `src/preprocess/parser.py` (Fase 2.1).",
            "- `special_point` (S/R/P/Q) indica puntos censados sin códigos; "
            "su ganador se toma de `PtWinner`.",
            "- `no_ending_*` indica finalizador ausente; el resultado se infiere de `PtWinner`.",
            "- `1stIn` se considera True si el primer servicio entró y el rally se jugó.",
            ""]

    md = "\n".join(out)
    (REPORTS_DIR / "01_calidad_datos.md").write_text(md, encoding="utf-8")
    print(f"Reporte escrito en reports/01_calidad_datos.md ({len(md):,} chars)")


def run():
    df = pd.read_parquet(CLEAN_DATASET)
    write_data_dictionary()
    build_report(df)


if __name__ == "__main__":
    run()